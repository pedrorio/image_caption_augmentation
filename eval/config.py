from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

from ica.config import TrainConfig


@dataclass
class ParaphraserJob:
    model_name_or_path: str
    max_steps: int
    max_epochs: int
    batch_size: int
    learning_rate: float
    max_source_length: int
    max_target_length: int
    max_new_tokens: int
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    early_stopping: bool
    num_return_sequences: int
    seed: int
    gpus: int


@dataclass
class CaptionerJob:
    epochs: int
    batch_size: int
    learning_rate: float
    embed_dim: int
    hidden_dim: int
    max_decode_len: int
    seeds: list[int]
    device: str


@dataclass
class EvalConfig:
    scale: str
    seed: int
    captions_json: str
    hf_dataset: str
    image_cache_dir: str
    output_json: str
    max_images: Optional[int]
    min_captions_per_image: int
    train_fraction: float
    val_fraction: float
    test_fraction: float
    near_copy_jaccard: float
    near_copy_bleu: float
    meaning_model: str
    decision_alpha: float
    paraphraser: ParaphraserJob
    captioner: CaptionerJob

    def to_train_config(self, data_dir: str, work_dir: str) -> TrainConfig:
        job = self.paraphraser
        root = Path(work_dir)
        return TrainConfig(
            data_dir=data_dir,
            logs_dir=str(root / "logs"),
            cache_dir="data/cache",
            checkpoints_dir=str(root / "checkpoints"),
            output_dir=str(root / "output"),
            model_name_or_path=job.model_name_or_path,
            batch_size=job.batch_size,
            learning_rate=job.learning_rate,
            max_epochs=job.max_epochs,
            max_steps=job.max_steps,
            gpus=job.gpus,
            num_workers=0,
            seed=job.seed,
            max_examples=None,
            max_source_length=job.max_source_length,
            max_target_length=job.max_target_length,
            compute_train_gleau=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            datasets=["rsicd"],
            do_sample=job.do_sample,
            top_k=job.top_k,
            top_p=job.top_p,
            early_stopping=job.early_stopping,
            num_return_sequences=job.num_return_sequences,
            max_new_tokens=job.max_new_tokens,
            temperature=job.temperature,
            enable_progress_bar=False,
            accumulate_grad_batches=1,
        )


def load_eval_config(path: Union[str, Path]) -> EvalConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    paraphraser = ParaphraserJob(**raw.pop("paraphraser"))
    captioner = CaptionerJob(**raw.pop("captioner"))
    return EvalConfig(paraphraser=paraphraser, captioner=captioner, **raw)
