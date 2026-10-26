# Design

## Context

See proposal.md. RSICD records in this repo already carry an image-level `split` field. The A/B must use that partition before augmentation.

## Goals / Non-Goals

- Goals: no test image is paraphrased, and a test proves inserted test text is detected.
- Non-Goals: changing the paraphraser datamodule's pair split. Evaluation finetuning writes only training images into its temporary data directory.

## Decisions

- Use the `split` value stored on each image (`train`, `val`, `test`). A toy run may subsample inside those splits, but it does not reshuffle images across them.
- Paraphrase generation for augmentation receives only training image ids. Validation captions may be paraphrased for the quality report and are not added to captioner training. Test captions are not sent to the paraphraser.
- The guard fails if any paraphrase map key is outside the training ids, or if a normalized test caption or a normalized paraphrase of a test caption appears in the training texts.
- The unit test builds unique captions so the string check is decisive. Real RSICD repeats sentences such as "many planes are parked in an airport" across images; those shared sentences are not treated as leakage of a test image.

## Risks / Trade-offs

- [Shared generic captions look like string leakage] → The test uses unique strings. The real-data guard is image-id based plus "paraphrases of test ids must not be inserted".

## Migration Plan

- `pytest tests/test_leakage.py` is part of CI.
- `python -m eval all` runs the guard before captioner training.

## Open Questions

- None.
