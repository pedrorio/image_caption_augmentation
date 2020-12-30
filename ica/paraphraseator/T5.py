import torch
# import torch_xla
# import torch_xla.core.xla_model as xm
from pytorch_lightning import LightningModule, seed_everything, Trainer
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, T5Config, PretrainedConfig, \
    get_linear_schedule_with_warmup
from ica.paraphraseator.datamodules.ImageCaptionsDataModule import ImageCaptionsDataModule
from ica.paraphraseator.callbacks.checkpoint_callback import checkpoint_callback
from ica.paraphraseator.callbacks.LoggerCallback import LoggerCallback
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.transforms import Compose

from pytorch_lightning.loggers import TensorBoardLogger

import pdb

seed_everything(42)


# todo: clean padding tokens
# todo: fix accumulate_grad_batches
class T5(LightningModule):
    # required
    def __init__(
            self,

            # model params
            data_dir: str = 'data/raw',
            logs_dir: str = 'data/logs',
            cache_dir: str = 'data/cache',
            checkpoints_dir: str = 'data/checkpoints',
            model_name_or_path: str = 'ramsrigouthamg/t5_paraphraser',
            batch_size: int = 6,
            learning_rate: float = 3e-4,
            adam_epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            warmup_steps: int = 0,

            # training params
            accumulate_grad_batches: int = 1,
            # accumulate_grad_batches: int = 16,
            max_epochs: int = 2,
            gpus: int = 0,
            num_workers: int = 8,
            fp_16: bool = False,
            amp_level: str = 'O2',
            gradient_clip_val: float = 1.0,
            **config_kwargs
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.logs_dir = logs_dir
        self.checkpoints_dir = checkpoints_dir

        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay

        self.batch_size = batch_size

        # training
        self.warmup_steps = warmup_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.num_workers = num_workers
        self.fp_16 = fp_16
        self.amp_level = amp_level
        self.gradient_clip_val = gradient_clip_val

        self.logger = TensorBoardLogger(
            save_dir=self.logs_dir,
            name=self.model_name_or_path,
            version=""
        )

        # self.config = T5Config.from_pretrained(
        #     pretrained_model_name_or_path=self.model_name_or_path,
        # cache_dir=self.cache_dir
        # )

        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            # cache_dir=self.cache_dir
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            # config=self.config,
            # cache_dir=self.cache_dir
        )

        self.datamodule = ImageCaptionsDataModule(
            batch_size=self.batch_size,
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            num_workers=self.num_workers
        )

        self.automatic_optimization = False

        self.save_hyperparameters()

    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=decoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

    # required
    def training_step(self, batch, batch_idx, dataloader_idx=None, hiddens=None):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # required
    def validation_step(self, batch, batch_idx, dataloader_idx=None, hiddens=None):
        loss = self.shared_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # todo: check accuracy?
    def shared_step(self, batch):
        # print(batch)
        # labels = batch.target_ids
        # labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["x_inputs"],
            attention_mask=batch["x_attention"],
            labels=batch["y_inputs"],
            decoder_input_ids=batch["y_inputs"],
            decoder_attention_mask=batch["y_attention"],
        )
        # loss = outputs[0]
        loss = outputs.loss
        return loss

    def setup(self, stage):
        print(f'stage: {stage}')
        data_per_device = len(self.datamodule.stage_len(stage)) // (self.batch_size * max(1, self.gpus))
        total_number_of_batches = self.accumulate_grad_batches * float(self.max_epochs)
        self.total_steps = data_per_device // total_number_of_batches
        print(f'total_steps: {self.total_steps}')

    # required
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        parameters = [
            {
                "params": [
                    param for name, param in self.model.named_parameters() if
                    not any(no_decay_name in name for no_decay_name in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param for name, param in self.model.named_parameters() if
                           any(no_decay_name in name for no_decay_name in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon
        )
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': self.lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False
    ):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def generate(self, sentence):
        annotated_sentence = f'paraphrase: {sentence}'
        encoded_sentence = self.tokenizer.batch_encode_plus(
            [annotated_sentence],
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True
        )
        generated_sample = self.model.generate(
            input_ids=encoded_sentence.input_ids,
            attention_mask=encoded_sentence.attention_masks,
            do_sample=True,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=1
        )[0]
        return self.tokenizer.decode(
            generated_sample,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    def train_model(self):
        self.trainer = Trainer(
            accumulate_grad_batches=self.accumulate_grad_batches,
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            precision=16 if self.fp_16 else 32,
            amp_level=self.amp_level,
            gradient_clip_val=self.gradient_clip_val,
            callbacks=[
                # checkpoint_callback(checkpoints_dir=self.checkpoints_dir),
                # LoggerCallback()
            ],
            default_root_dir=self.logs_dir,
            logger=self.logger
        )
        self.trainer.fit(model=self, datamodule=self.datamodule)
        self.model.save_pretrained(self.model_name_or_path)

    def test_model(self, dataset):
        self.trainer = Trainer(
            accumulate_grad_batches=self.accumulate_grad_batches,
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            precision=16 if self.fp_16 else 32,
            amp_level=self.amp_level,
            gradient_clip_val=self.gradient_clip_val,
            callbacks=[
                # checkpoint_callback(checkpoints_dir=self.checkpoints_dir),
                # LoggerCallback()
            ],
            default_root_dir=self.logs_dir,
            logger=self.logger
        )
        self.trainer.test(model=self, datamodule=self.datamodule)


def main():
    t5 = T5()
    t5.train_model()


if __name__ == "__main__":
    main()
