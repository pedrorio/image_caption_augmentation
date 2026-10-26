import math
from pathlib import Path

from ica.config import load_config
from ica.paraphraseator.T5 import paraphrase_saved, train_paraphraser

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_is_one_cpu_step():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    assert cfg.model_name_or_path == "t5-small"
    assert cfg.gpus == 0
    assert cfg.max_steps == 1
    assert cfg.max_examples is not None
    assert 10 <= cfg.max_examples <= 20
    assert not str(cfg.data_dir).startswith("/content/drive")


def test_smoke_train_and_paraphrase_shapes(tmp_path: Path):
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg.logs_dir = str(tmp_path / "logs")
    cfg.checkpoints_dir = str(tmp_path / "checkpoints")
    cfg.output_dir = str(tmp_path / "output")
    result = train_paraphraser(cfg)
    assert result.global_step == 1
    assert math.isfinite(result.train_loss)
    assert 10 <= len(result.captions) <= 20
    batch = paraphrase_saved(result.model_dir, result.captions, cfg)
    assert len(batch.texts) == len(result.captions)
    assert all(isinstance(text, str) for text in batch.texts)
    assert batch.token_ids.ndim == 2
    assert tuple(batch.token_ids.shape)[0] == len(result.captions)
    assert batch.token_ids.shape[1] >= 1
    assert (Path(result.model_dir) / "config.json").is_file()
