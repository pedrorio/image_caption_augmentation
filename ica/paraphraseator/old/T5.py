import torch
# import torch_xla
# import torch_xla.core.xla_model as xm
from pytorch_lightning import LightningModule, seed_everything
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

seed_everything(42)


class T5(LightningModule):

    def __init__(self, data_dir: str, hparams: dict):
        print("ParaphraseGenerator init")
        super().__init__()

        self.hparams = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)

        print("ParaphraseGenerator init")

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask
        )

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        return {
            "loss": loss,
            "log": {
                "train_loss": loss
            }
        }

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {
            "avg_train_loss": avg_train_loss,
            "log": {
                "avg_train_loss": avg_train_loss
            },
            'progress_bar': {
                "avg_train_loss": avg_train_loss
            }
        }

    def validation_step(self, batch, batch_idx):
        return {
            "val_loss": self.common_step(batch)
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {
            "avg_val_loss": avg_loss,
            "log": {
                "val_loss": avg_loss
            },
            'progress_bar': {
                "val_loss": avg_loss
            }
        }

    def common_step(self, batch):
        labels = batch["target_ids"]
        # labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            labels=labels,
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]
        return loss

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            self.total_steps = (
                    (
                            len(
                                self.train_dataloader().dataset
                            ) // (
                                    self.hparams.train_batch_size * max(1, self.hparams.gpus)
                            )
                    ) // (
                            self.hparams.accumulate_grad_batches * float(self.hparams.max_epochs)
                    )
            )

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "hparams": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "hparams": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        print("Model configure_optimizers before optimizer")
        self.opt = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        print("Model configure_optimizers after optimizer")
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': self.lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }
        print("ParaphraseGenerator configure_optimizers")
        return [self.opt], [scheduler]
        # return [self.optimizer], [scheduler]

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            second_order_closure=None,
            using_native_amp=None
    ):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        # return {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": "{:.3f}".format(self.trainer.avg_loss)}

    def is_logger(self):
        return self.trainer.proc_rank <= 0
