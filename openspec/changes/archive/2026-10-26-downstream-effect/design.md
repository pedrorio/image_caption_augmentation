# Design

## Context

See proposal.md. The downstream model has to be trainable on CPU for the pipeline check and identical across arms.

## Goals / Non-Goals

- Goals: same images, same captioner hyperparameters, same test split, at least three seeds, per-seed differences, and an explicit noise rule for full-scale CIDEr.
- Non-Goals: a full RSICD GPU run in this session. `configs/full_ab.yaml` is the command for that run.

## Decisions

- Captioner: frozen ImageNet ResNet-18 (classification head removed, 512-d features) plus a one-layer GRU decoder. A VisionEncoderDecoder such as ViT-GPT2 is a different, heavier model and cannot be the same system we both smoke-check on CPU and configure for the full run. Freezing the encoder keeps the A/B about the captions.
- A and B share the vocabulary built from train originals plus paraphrases, the same learning rate, hidden size, embedding size, epochs, and batch size. B's training set adds non-empty paraphrases of training captions. Test references stay the original test captions.
- Metrics: BLEU-4, ROUGE-L, and CIDEr from pycocoevalcap when its tokenizer runs, otherwise the same n-gram definitions with a documented whitespace tokenizer.
- Decision rule, full scale only: augmentation did not help if the absolute mean of per-seed CIDEr (B minus A) is less than one sample standard deviation of those differences, or a two-sided paired test of the differences against zero has p > 0.05. The repository should then be archived. Toy runs compute the statistic and mark it not applicable.

## Risks / Trade-offs

- [Three seeds have little power] → The rule treats an insignificant difference as "did not help", which is the requested bias.
- [More captions give B more updates per epoch] → Epoch count is the shared hyperparameter, so B seeing more caption variants is the augmentation under test.

## Migration Plan

- Toy: `python -m eval all --config configs/toy_ab.yaml`.
- Full: the Colab cell in the README with `configs/full_ab.yaml`.

## Open Questions

- None.
