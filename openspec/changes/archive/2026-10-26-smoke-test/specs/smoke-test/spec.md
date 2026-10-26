# smoke-test Specification

## Purpose

Runs a one-step CPU paraphrase train and a short inference so a broken import or path fails before any full experiment.

## ADDED Requirements

### Requirement: Tiny CPU config

The repository SHALL include a smoke config that selects `t5-small`, zero GPUs, one training step, and a source-caption budget from 10 to 20 inclusive.

#### Scenario: Smoke settings

- **WHEN** `configs/smoke.yaml` is loaded
- **THEN** the model is `t5-small`, the GPU count is 0, `max_steps` is 1, and `max_examples` is between 10 and 20

### Requirement: Train and paraphrase shape

The smoke test SHALL run that config on CPU, SHALL finish at least one training step with a finite `train_loss`, and SHALL paraphrase the loaded source captions. The token-id tensor SHALL have shape `[N, L]` where `N` equals the number of input captions and `L` is at least 1. The decoded output SHALL be a list of `N` strings. The saved model directory SHALL contain a config file.

#### Scenario: One step and a batch of paraphrases

- **WHEN** the smoke test runs
- **THEN** the recorded global step is 1, `train_loss` is finite, and paraphrase token ids have one row per input caption

### Requirement: CI runs the smoke test

GitHub Actions SHALL run the smoke pytest on a CPU runner using the locked dependencies and SHALL cache Hugging Face downloads.

#### Scenario: Workflow file

- **WHEN** `.github/workflows/ci.yml` is read
- **THEN** it runs pytest and caches the Hugging Face directory
