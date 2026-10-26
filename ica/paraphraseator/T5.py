"""T5 paraphraser training and sampling decode."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from ica.config import TrainConfig, config_from_args
from ica.paraphraseator.callbacks.CheckpointEveryNSteps import CheckpointEveryNSteps
from ica.paraphraseator.datamodules.ImageCaptionsDataModule import ImageCaptionsDataModule
from ica.paraphraseator.measures.GLEAU import GLEAU
from ica.utils.Encoding import Encoding
from ica.utils.helpers import (
    annotate_sentence,
    clean_sentence,
    decode_sequence_of_tokens,
    encode_sentence,
    generate_kwargs_from_settings,
    generate_sequence_of_tokens,
)


@dataclass
class TrainResult:
    global_step: int
    train_loss: float
    model_dir: str
    captions: list[str]


@dataclass
class ParaphraseBatch:
    texts: list[str]
    token_ids: torch.Tensor


def _as_float(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def metric_value(metrics, name: str) -> float:
    for key in (name, f"{name}_step", f"{name}_epoch"):
        if key in metrics:
            return _as_float(metrics[key])
    raise KeyError(name)


def paraphrase_sentences(model, tokenizer, sentences: list[str], cfg: TrainConfig, device=None) -> ParaphraseBatch:
    """Sample one paraphrase per sentence. token_ids has shape [N * num_return_sequences, L]."""
    if not sentences:
        return ParaphraseBatch(texts=[], token_ids=torch.empty(0, 0, dtype=torch.long))
    annotated = [annotate_sentence(clean_sentence(sentence)) for sentence in sentences]
    encoded = tokenizer(
        annotated,
        max_length=cfg.max_source_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    settings = generate_kwargs_from_settings(
        do_sample=cfg.do_sample,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        early_stopping=cfg.early_stopping,
        num_return_sequences=cfg.num_return_sequences,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
    )
    # T5's generation config sets max_length; passing max_new_tokens alongside it errors.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and settings.get("max_new_tokens") is not None:
        generation_config.max_length = None
    token_ids = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        **settings,
    )
    texts = tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return ParaphraseBatch(texts=list(texts), token_ids=token_ids.detach().cpu())


def paraphrase_saved(model_dir: str, sentences: list[str], cfg: TrainConfig) -> ParaphraseBatch:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    return paraphrase_sentences(model, tokenizer, sentences, cfg, device=torch.device("cpu"))


class T5(LightningModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(
            {
                "model_name_or_path": cfg.model_name_or_path,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "max_epochs": cfg.max_epochs,
                "max_steps": cfg.max_steps,
                "seed": cfg.seed,
                "max_examples": cfg.max_examples,
            }
        )
        os.makedirs(cfg.data_dir, exist_ok=True)
        os.makedirs(cfg.cache_dir, exist_ok=True)
        os.makedirs(cfg.logs_dir, exist_ok=True)
        os.makedirs(cfg.checkpoints_dir, exist_ok=True)
        os.makedirs(cfg.output_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            cache_dir=cfg.cache_dir,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_name_or_path,
            cache_dir=cfg.cache_dir,
        )
        self.datamodule = ImageCaptionsDataModule(
            tokenizer=self.tokenizer,
            batch_size=cfg.batch_size,
            data_dir=cfg.data_dir,
            num_workers=cfg.num_workers,
            max_examples=cfg.max_examples,
            max_source_length=cfg.max_source_length,
            max_target_length=cfg.max_target_length,
            datasets=list(cfg.datasets or []),
            seed=cfg.seed,
            train_fraction=cfg.train_fraction,
            val_fraction=cfg.val_fraction,
        )
        safe_name = cfg.model_name_or_path.replace("/", "--")
        self.save_model_path = str(Path(cfg.checkpoints_dir) / safe_name)
        self.total_steps = 1
        self.tb_logger = TensorBoardLogger(
            save_dir=cfg.logs_dir,
            name="paraphraser",
            default_hp_metric=False,
        )
        self.tb_logger.log_hyperparams(params=self.hparams)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def _loss(self, batch):
        input_ids = batch["encoded_x"]["input_ids"]
        attention_mask = batch["encoded_x"]["attention_mask"]
        labels = batch["encoded_y"]["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    def _gleau(self, batch) -> torch.Tensor:
        encoded_x = batch["encoded_x"]
        cleaned_y = batch["clean_y"]
        scores = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            batch_size = encoded_x["input_ids"].size(0)
            for index in range(batch_size):
                encoded = Encoding(
                    input_ids=encoded_x["input_ids"][index],
                    attention_mask=encoded_x["attention_mask"][index],
                )
                predicted = self.generate_from_encoded(encoded)
                target = cleaned_y[index]
                scores.append(GLEAU(target, predicted))
                if index == 0 and self.logger is not None:
                    self.logger.experiment.add_text(
                        "target / predicted",
                        f"{target} / {predicted}",
                    )
        if was_training:
            self.train()
        return torch.tensor(scores, dtype=torch.float32).mean()

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        batch_size = batch["encoded_x"]["input_ids"].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if self.cfg.compute_train_gleau:
            gleau = self._gleau(batch)
            self.log("train_gleau", gleau, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        batch_size = batch["encoded_x"]["input_ids"].size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if self.cfg.compute_train_gleau:
            gleau = self._gleau(batch)
            self.log("val_gleau", gleau, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._loss(batch)
        batch_size = batch["encoded_x"]["input_ids"].size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def setup(self, stage):
        if stage in (None, "fit"):
            self.total_steps = self.estimate_total_steps()

    def estimate_total_steps(self) -> int:
        if self.cfg.max_steps and self.cfg.max_steps > 0:
            return int(self.cfg.max_steps)
        train_len = self.datamodule.stage_len("fit")
        devices = max(1, self.cfg.gpus)
        steps_per_epoch = max(1, train_len // max(1, self.cfg.batch_size * devices))
        accum = max(1, self.cfg.accumulate_grad_batches)
        return max(1, (steps_per_epoch // accum) * max(1, self.cfg.max_epochs))

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        named = list(self.model.named_parameters())
        decay = [param for name, param in named if not any(skip in name for skip in no_decay)]
        no_decay_params = [param for name, param in named if any(skip in name for skip in no_decay)]
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": self.cfg.weight_decay})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        optimizer = AdamW(groups, lr=self.cfg.learning_rate, eps=self.cfg.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=max(1, self.total_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def generate_from_encoded(self, encoded: Encoding) -> str:
        settings = generate_kwargs_from_settings(
            do_sample=self.cfg.do_sample,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            early_stopping=self.cfg.early_stopping,
            num_return_sequences=1,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )
        sequence = generate_sequence_of_tokens(self.model, encoded, **settings)
        return decode_sequence_of_tokens(self.tokenizer, sequence)

    def generate_from_sentence(self, sentence: str) -> str:
        cleaned = clean_sentence(sentence)
        annotated = annotate_sentence(cleaned)
        encoded_dict = encode_sentence(self.tokenizer, annotated, max_length=self.cfg.max_source_length)
        encoded = Encoding(**encoded_dict)
        return self.generate_from_encoded(encoded)

    def paraphrase(self, sentences: list[str]) -> ParaphraseBatch:
        self.eval()
        return paraphrase_sentences(self.model, self.tokenizer, sentences, self.cfg, device=self.device)

    def train_model(self) -> TrainResult:
        accelerator = "gpu" if self.cfg.gpus and self.cfg.gpus > 0 else "cpu"
        devices = self.cfg.gpus if accelerator == "gpu" else 1
        ckpt_path = None
        if self.cfg.checkpoint_to_load:
            ckpt_path = str(Path(self.cfg.checkpoints_dir) / self.cfg.checkpoint_to_load)
        limit_val_batches = self.cfg.limit_val_batches
        sanity_steps = self.cfg.num_sanity_val_steps
        trainer = Trainer(
            max_epochs=self.cfg.max_epochs,
            max_steps=self.cfg.max_steps if self.cfg.max_steps and self.cfg.max_steps > 0 else -1,
            accelerator=accelerator,
            devices=devices,
            precision="16-mixed" if self.cfg.fp_16 else "32-true",
            accumulate_grad_batches=self.cfg.accumulate_grad_batches,
            gradient_clip_val=self.cfg.gradient_clip_val,
            callbacks=[
                CheckpointEveryNSteps(
                    save_step_frequency=self.cfg.accumulate_grad_batches * self.cfg.batch_size,
                    checkpoints_dir=self.cfg.checkpoints_dir,
                )
            ],
            default_root_dir=self.cfg.logs_dir,
            log_every_n_steps=1,
            logger=self.tb_logger,
            enable_checkpointing=False,
            enable_progress_bar=self.cfg.enable_progress_bar,
            limit_val_batches=limit_val_batches,
            num_sanity_val_steps=sanity_steps,
            enable_model_summary=False,
        )
        trainer.fit(model=self, datamodule=self.datamodule, ckpt_path=ckpt_path)
        self.model.save_pretrained(self.save_model_path)
        self.tokenizer.save_pretrained(self.save_model_path)
        loss = metric_value(trainer.callback_metrics, "train_loss")
        return TrainResult(
            global_step=int(trainer.global_step),
            train_loss=loss,
            model_dir=self.save_model_path,
            captions=list(self.datamodule.captions),
        )


def train_paraphraser(cfg: TrainConfig) -> TrainResult:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    seed_everything(cfg.seed, workers=True)
    for path in (cfg.data_dir, cfg.logs_dir, cfg.cache_dir, cfg.checkpoints_dir, cfg.output_dir):
        os.makedirs(path, exist_ok=True)
    return T5(cfg).train_model()


def main():
    cfg = config_from_args()
    result = train_paraphraser(cfg)
    print(
        f"global_step={result.global_step} train_loss={result.train_loss:.6f} model_dir={result.model_dir}"
    )


if __name__ == "__main__":
    main()
