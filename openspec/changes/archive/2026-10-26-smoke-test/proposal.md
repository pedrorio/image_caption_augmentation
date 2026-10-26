# Smoke test

## Why

A broken import, a missing path, or a trainer that cannot take a step should fail in CI in minutes, not during a full fine-tune.

## What Changes

- Add `configs/smoke.yaml`: `t5-small`, CPU, about 10 to 20 source captions, one training step.
- Pytest trains and paraphrases with that config and checks that a loss and a token batch with the right shape were produced.
- GitHub Actions runs that pytest on CPU and caches Hugging Face downloads.

## Capabilities

### New Capabilities

- `smoke-test`: the tiny CPU train-and-paraphrase check.

### Modified Capabilities

- None.

## Impact

- `tests/test_smoke.py`, `configs/smoke.yaml`, `.github/workflows/ci.yml`.
- CI does not run the downstream A/B.
