# Design

## Context

The paraphraser is a PyTorch Lightning T5 module. The only entry script passed absolute Colab paths and ignored the class-level local defaults. See proposal.md for why this change exists.

## Goals / Non-Goals

- Goals: one config object for paths, model id, batch size, epochs, seed, and caption budget; a Colab file that still points at Drive; a CPU training step that returns a finite loss.
- Non-Goals: the downstream captioning A/B, dependency pinning, and CI. Those are separate changes.

## Decisions

- YAML plus argparse, not a new framework. Overrides are explicit flags so a notebook cell can change one field without editing the file.
- `--output-dir` relocates logs, cache, and checkpoints unless a more specific flag is set. `--data-dir` stays independent because captions are inputs, not run artifacts.
- `max_examples` counts source captions and keeps whole images. Stopping mid-image would mix one image's captions across the budget boundary.
- Per-step GLEU stays available but defaults off. It decodes every target, which dominates CPU time. `configs/colab.yaml` sets `compute_train_gleau: true`.
- Loss uses the target token ids with padding set to -100 and Lightning's automatic optimization. The previous manual-optimization flag returned a loss the trainer did not apply, so a "running" job did not learn.
- Pair split fractions stay 70/20/10. The seed now feeds `torch.randperm` so the split is repeatable. The split is still over pairs, not images.

## Risks / Trade-offs

- [Shorter default sequences drop tokens] → Colab config keeps length 512. Local and smoke runs set the length they can afford.
- [Disabling train-time GLEU hides a logged metric] → The metric remains implemented and is on in the Colab config.
- [Pair-level split can leak an image across subsets] → Left as the paraphraser's current behavior. The evaluation harness, in a later change, splits by image before augmentation.

## Migration Plan

- `python run.py` uses `configs/default.yaml`.
- The previous Colab command becomes `python run.py --config configs/colab.yaml`.

## Open Questions

- None.
