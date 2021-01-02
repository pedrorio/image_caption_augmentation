import torch
# import torch_xla
# import torch_xla.core.xla_model as xm
from pytorch_lightning import LightningModule, seed_everything, Trainer
# from transformers import T5Config
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqModelOutput
from ica.paraphraseator.datamodules.ImageCaptionsDataModule import ImageCaptionsDataModule
from ica.paraphraseator.callbacks.checkpoint_callback import checkpoint_callback
from ica.paraphraseator.callbacks.LoggerCallback import LoggerCallback
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.transforms import Compose
import os
from pytorch_lightning.loggers import TensorBoardLogger
from ica.utils.helpers import clean_sentence, annotate_sentence, encode_sentence, decode_sequence_of_tokens, \
    generate_sequence_of_tokens
from ica.utils.Encoding import Encoding
from ica.paraphraseator.measures.GLEAU import GLEAU
from dataclasses import dataclass
from torch.tensor import Tensor

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
            batch_size: int = 16,
            learning_rate: float = 3e-4,
            adam_epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            warmup_steps: int = 0,

            # training params
            accumulate_grad_batches: int = 16,
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

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

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
            name="T5",
            # name=self.model_name_or_path,
            version="",
            default_hp_metric=False,
            log_graph=False
        )

        # self.config = T5Config.from_pretrained(
        #     pretrained_model_name_or_path=self.model_name_or_path,
        # cache_dir=self.cache_dir
        # )

        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            cache_dir=self.cache_dir
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            # config=self.config,
            cache_dir=self.cache_dir
        )

        self.datamodule = ImageCaptionsDataModule(
            batch_size=self.batch_size,
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            num_workers=self.num_workers
        )

        self.save_hyperparameters()
        self.logger.log_hyperparams(params=self.hparams)

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None,
                ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=decoder_input_ids,
        )

    # required
    def training_step(self, batch, batch_idx, dataloader_idx=None, hiddens=None):
        loss, gleau = self.shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_gleau', gleau, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # required
    def validation_step(self, batch, batch_idx, dataloader_idx=None, hiddens=None):
        loss, gleau = self.shared_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_gleau', gleau, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def shared_step(self, batch) -> Seq2SeqModelOutput:

        encoded_y = Encoding(**batch["encoded_y"])
        encoded_x = Encoding(**batch["encoded_x"])
        cleaned_y = batch["clean_y"]

        outputs = self.forward(
            input_ids=encoded_x.input_ids.squeeze(),
            attention_mask=encoded_x.attention_mask.squeeze(),
            # labels=encoded_y.input_ids.squeeze(),
            decoder_input_ids=encoded_y.input_ids.squeeze(),
            decoder_attention_mask=encoded_y.attention_mask.squeeze(),
        )

        gleaus = []

        def encoded_element_from_batch(i: int, encoded_collection):
            encoded = Encoding(**dict(
                input_ids=encoded_collection.input_ids[i],
                attention_mask=encoded_collection.attention_mask[i]
            ))
            return encoded

        for i in range(self.batch_size):
            target_sentence = cleaned_y[i]
            encoded = encoded_element_from_batch(i, encoded_x)
            predicted_sentence = self.generate_from_encoded(encoded)

            self.logger.experiment.add_text(
                "target  \n predicted",
                f'{target_sentence}  \n {predicted_sentence}'
            )

            gleau = GLEAU(target_sentence, predicted_sentence)
            gleaus.append(gleau)

        gleau = Tensor(gleaus).mean()
        loss = outputs.loss

        return loss, gleau

    def setup(self, stage):
        print(f'stage: {stage}')
        data_per_device = self.datamodule.stage_len(stage) // (self.batch_size * max(1, self.gpus))
        total_number_of_batches = self.accumulate_grad_batches * float(self.max_epochs)
        self.total_steps = data_per_device // total_number_of_batches
        print(f'total_steps: {self.total_steps}')

    @property
    def automatic_optimization(self) -> bool:
        return False

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
        self.optimizer = AdamW(
            parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon
        )
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [self.optimizer], [self.scheduler]

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
        self.scheduler.step()

    def generate_from_encoded(self, encoded):
        sequence_of_tokens = generate_sequence_of_tokens(self.model, encoded)
        return decode_sequence_of_tokens(self.tokenizer, sequence_of_tokens)

    def generate_from_sentence(self, sentence):
        cleaned = clean_sentence(sentence)
        annotated = annotate_sentence(cleaned)
        encoded = encode_sentence(self.tokenizer, annotated)
        return self.generate_from_encoded(encoded)

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
            log_every_n_steps=1,
            logger=self.logger,
        )
        self.trainer.fit(model=self, datamodule=self.datamodule)
        self.model.save_pretrained(self.model_name_or_path)
        self.tokenizer.save_pretrained(self.model_name_or_path)

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
            log_every_n_steps=20,
            logger=self.logger,
            progress_bar_refresh_rate=60,
            val_check_interval=0.25
        )
        self.trainer.test(model=self, datamodule=self.datamodule)


def main():
    t5 = T5(
        gpus=1,
        num_workers=4,
        batch_size=5,
        data_dir="/content/drive/MyDrive/ica/data/raw",
        logs_dir="/content/drive/MyDrive/ica/data/logs",
        cache_dir="/content/drive/MyDrive/ica/data/cache",
        checkpoints_dir="/content/drive/MyDrive/ica/data/checkpoints"
    )
    t5.train_model()

if __name__ == "__main__":
    main()
