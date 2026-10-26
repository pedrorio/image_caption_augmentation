"""YAML and CLI configuration for paraphraser training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import yaml

LimitBatches = Union[int, float]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    local = Path("configs/default.yaml")
    if local.is_file():
        return local
    return repo_root() / "configs" / "default.yaml"


@dataclass
class TrainConfig:
    """Training settings. Paths default to the repository, not Colab Drive."""

    data_dir: str = "data/raw"
    logs_dir: str = "data/logs"
    cache_dir: str = "data/cache"
    checkpoints_dir: str = "data/checkpoints"
    output_dir: str = "data/output"
    model_name_or_path: str = "ramsrigouthamg/t5_paraphraser"
    batch_size: int = 4
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    warmup_steps: int = 0
    accumulate_grad_batches: int = 1
    max_epochs: int = 2
    max_steps: int = -1
    gpus: int = 0
    num_workers: int = 0
    fp_16: bool = False
    gradient_clip_val: float = 1.0
    seed: int = 42
    max_examples: Optional[int] = None
    max_source_length: int = 64
    max_target_length: int = 64
    compute_train_gleau: bool = False
    limit_val_batches: LimitBatches = 1
    num_sanity_val_steps: int = 0
    datasets: Optional[list[str]] = None
    do_sample: bool = True
    top_k: int = 120
    top_p: float = 0.98
    early_stopping: bool = True
    num_return_sequences: int = 1
    max_new_tokens: int = 20
    temperature: float = 1.0
    checkpoint_to_load: Optional[str] = None
    train_fraction: float = 0.7
    val_fraction: float = 0.2
    enable_progress_bar: bool = True

    def __post_init__(self) -> None:
        if self.datasets is None:
            self.datasets = ["rsicd", "sydney", "ucm"]
        self.datasets = [name.lower() for name in self.datasets]
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError("max_examples must be positive when set")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gpus < 0:
            raise ValueError("gpus must be >= 0")


def load_config(path: Union[str, Path]) -> TrainConfig:
    """Load a YAML config. Unknown keys are an error so typos fail loudly."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must be a mapping")
    known = {item.name for item in fields(TrainConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown}")
    return TrainConfig(**raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a T5 model to paraphrase remote-sensing image captions."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. Defaults to configs/default.yaml.",
    )
    parser.add_argument("--data-dir", default=None, help="Directory of caption JSON files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory. Logs, cache, and checkpoints are placed inside it unless overridden.",
    )
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--model-name", default=None, help="Hugging Face model id or local path.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of source captions. Whole images are kept until this budget is reached.",
    )
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def apply_cli(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Apply CLI overrides. Explicit directory flags beat --output-dir."""
    updates: dict[str, Any] = {}
    if args.output_dir:
        root = Path(args.output_dir)
        updates["output_dir"] = str(root)
        if args.logs_dir is None:
            updates["logs_dir"] = str(root / "logs")
        if args.cache_dir is None:
            updates["cache_dir"] = str(root / "cache")
        if args.checkpoints_dir is None:
            updates["checkpoints_dir"] = str(root / "checkpoints")
    if args.data_dir is not None:
        updates["data_dir"] = args.data_dir
    if args.logs_dir is not None:
        updates["logs_dir"] = args.logs_dir
    if args.cache_dir is not None:
        updates["cache_dir"] = args.cache_dir
    if args.checkpoints_dir is not None:
        updates["checkpoints_dir"] = args.checkpoints_dir
    if args.model_name is not None:
        updates["model_name_or_path"] = args.model_name
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.epochs is not None:
        updates["max_epochs"] = args.epochs
    if args.max_steps is not None:
        updates["max_steps"] = args.max_steps
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.max_examples is not None:
        updates["max_examples"] = args.max_examples
    if args.gpus is not None:
        updates["gpus"] = args.gpus
    if args.num_workers is not None:
        updates["num_workers"] = args.num_workers
    return replace(cfg, **updates)


def config_from_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    path = Path(args.config) if args.config else default_config_path()
    return apply_cli(load_config(path), args)
