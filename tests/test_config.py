from pathlib import Path

import pytest

from ica.config import TrainConfig, apply_cli, build_parser, load_config
from ica.data_split import split_sizes

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_local_and_cpu():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.data_dir == "data/raw"
    assert cfg.output_dir == "data/output"
    assert cfg.gpus == 0
    assert cfg.num_workers == 0
    assert cfg.model_name_or_path == "ramsrigouthamg/t5_paraphraser"
    assert cfg.seed == 42
    assert cfg.max_examples is None
    assert not str(cfg.data_dir).startswith("/content/drive")


def test_colab_config_reproduces_previous_entry_point():
    cfg = load_config(ROOT / "configs" / "colab.yaml")
    assert cfg.data_dir == "/content/drive/MyDrive/ica/data/raw"
    assert cfg.logs_dir == "/content/drive/MyDrive/ica/data/logs"
    assert cfg.cache_dir == "/content/drive/MyDrive/ica/data/cache"
    assert cfg.checkpoints_dir == "/content/drive/MyDrive/ica/data/checkpoints"
    assert cfg.batch_size == 5
    assert cfg.gpus == 1
    assert cfg.num_workers == 4
    assert cfg.max_epochs == 2
    assert cfg.accumulate_grad_batches == 16
    assert cfg.max_source_length == 512
    assert cfg.max_target_length == 512
    assert cfg.compute_train_gleau is True
    assert cfg.model_name_or_path == "ramsrigouthamg/t5_paraphraser"


def test_cli_overrides_replace_yaml_values():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    args = build_parser().parse_args(
        [
            "--data-dir",
            "custom/data",
            "--output-dir",
            "custom/out",
            "--model-name",
            "t5-small",
            "--batch-size",
            "2",
            "--epochs",
            "3",
            "--seed",
            "7",
            "--max-examples",
            "20",
            "--gpus",
            "0",
        ]
    )
    merged = apply_cli(cfg, args)
    assert merged.data_dir == "custom/data"
    assert merged.output_dir == "custom/out"
    assert merged.logs_dir == "custom/out/logs"
    assert merged.cache_dir == "custom/out/cache"
    assert merged.checkpoints_dir == "custom/out/checkpoints"
    assert merged.model_name_or_path == "t5-small"
    assert merged.batch_size == 2
    assert merged.max_epochs == 3
    assert merged.seed == 7
    assert merged.max_examples == 20
    assert merged.gpus == 0


def test_specific_directory_flags_win_over_output_dir():
    cfg = TrainConfig()
    args = build_parser().parse_args(
        ["--output-dir", "custom/out", "--logs-dir", "custom/logs"]
    )
    merged = apply_cli(cfg, args)
    assert merged.logs_dir == "custom/logs"
    assert merged.checkpoints_dir == "custom/out/checkpoints"


def test_unknown_config_key_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("data_dir: data/raw\nnot_a_real_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_pair_split_fractions_sum_to_n():
    assert split_sizes(100) == (70, 20, 10)
    assert split_sizes(10) == (7, 2, 1)
    assert sum(split_sizes(1)) == 1
    assert sum(split_sizes(273000)) == 273000
