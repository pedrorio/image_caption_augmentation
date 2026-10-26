# Downstream effect

## Why

The project asks whether T5 paraphrases help a captioning model. That is an A/B on the same images, hyperparameters, and held-out test split, not a paraphraser loss.

## What Changes

- Train the same small captioner on original train captions (A) and on original train captions plus paraphrases (B).
- Use at least three seeds. Report mean ± sample standard deviation per arm and the per-seed B-minus-A difference for BLEU-4, ROUGE-L, and CIDEr.
- If a full-scale CIDEr difference is inside seed noise, say the augmentation did not help and that the repository should be archived.
- Toy-scale runs use this code but are labeled pipeline checks, not results.

## Capabilities

### New Capabilities

- `downstream-ab`: paired captioner training and the archive decision rule.

### Modified Capabilities

- None.

## Impact

- `eval` captioner, metrics, and decision code. `configs/toy_ab.yaml` and `configs/full_ab.yaml`.
- This session runs only the toy config on CPU.
