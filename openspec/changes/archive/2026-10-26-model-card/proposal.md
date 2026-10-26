# Model card

## Why

The README is a title. Readers need the question, the dataset, the protocol, how to run the checks, and a model card, without numbers that were not produced in a run.

## What Changes

- Document the project, the A/B question, RSICD, the paraphraser, the leakage guard, and the commands for smoke, toy eval, and the full run.
- State that the full-scale A/B has not been run. Quote only toy-scale numbers from a run in this session, labeled as a pipeline check.
- Add a model card: base T5 checkpoint, training captions, sampling settings (not beam search), temperature, hallucinated objects and copies as failure modes, and the compute the full run needs.
- Repeat the archive decision rule next to the results section.

## Capabilities

### New Capabilities

- `model-card`: README model card and the honest results section.

### Modified Capabilities

- None.

## Impact

- `README.md` only, plus the toy-scale JSON it cites.
