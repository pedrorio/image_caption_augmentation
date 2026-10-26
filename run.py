"""Fine-tune the caption paraphraser from a YAML config and optional CLI overrides."""

from ica.config import config_from_args
from ica.paraphraseator.T5 import train_paraphraser


def main(argv=None):
    cfg = config_from_args(argv)
    result = train_paraphraser(cfg)
    print(
        f"global_step={result.global_step} train_loss={result.train_loss:.6f} model_dir={result.model_dir}"
    )
    return result


if __name__ == "__main__":
    main()
