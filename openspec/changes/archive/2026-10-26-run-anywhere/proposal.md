# Run anywhere

## Why

The entry script hard-codes Google Drive paths, dependencies are unpinned (including the deprecated `sklearn` name), and bytecode is tracked. A checkout should install and start on a local CPU machine.

## What Changes

- YAML config and CLI overrides for data dir, output dir, model name, batch size, epochs, seed, and max examples. Defaults are repository-local and use zero GPUs.
- `configs/colab.yaml` keeps `/content/drive/MyDrive/ica/data` paths, batch size 5, 1 GPU, and 4 workers.
- Pin dependencies with `pyproject.toml` and `uv.lock`. Depend on `scikit-learn`, not `sklearn`.
- Stop tracking `*.pyc` and ignore `__pycache__/` plus the usual Python artifacts.
- Training settings that a local CPU run needs: finite cross-entropy, optional per-step GLEU, explicit sampling temperature and `max_new_tokens`.

## Capabilities

### New Capabilities

- `dependency-lock`: pinned installs and the scikit-learn package name.
- `repo-hygiene`: no tracked bytecode, and a Python gitignore.

### Modified Capabilities

- `paraphraser`: local entry paths, Colab file, sequence length, sampling, and the training step.

## Impact

- `run.py`, `ica/config.py`, the T5 trainer and datamodule, `configs/default.yaml`, `configs/colab.yaml`.
- `pyproject.toml`, `uv.lock`, `requirements.txt`, `setup.py`, `.python-version`, `.gitignore`.
- Removes tracked `ica/__init__.pyc` and `ica/paraphraseator/__init__.pyc` from the index without rewriting history.
- `hyper` and `googletrans` are not pinned. Nothing in the training or eval path imports `hyper`. `googletrans==4.0.0rc1` pins `httpx==0.13.3`, which breaks `datasets` (no `httpx.RequestError`). `ica/translator/Translator.py` still imports it for the old back-translation path; that extra is not installed by the lock.
