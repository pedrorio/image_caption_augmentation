# Tasks

## 1. Configuration

- [x] 1.1 Add YAML loading and CLI overrides and verify `tests/test_config.py` against `configs/default.yaml` and `configs/colab.yaml`

## 2. Dependencies and hygiene

- [x] 2.1 Pin dependencies with uv, depend on scikit-learn, and verify `uv.lock` exists and `pyproject.toml` has no sklearn requirement
- [x] 2.2 Stop tracking bytecode, ignore `__pycache__/`, and verify `git ls-files '*.pyc'` is empty

## Workflow follow-up

- Archive this change after the config and hygiene checks pass.
