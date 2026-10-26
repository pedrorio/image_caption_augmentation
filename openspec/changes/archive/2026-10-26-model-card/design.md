# Design

## Context

See proposal.md. Numbers in the README have to be copied from `results/toy_scale_pipeline_check.json` after that file is produced.

## Goals / Non-Goals

- Goals: a README that answers what the experiment is and refuses to present the toy check as a finding.
- Non-Goals: a paper-style full result table.

## Decisions

- The model card describes `ramsrigouthamg/t5_paraphraser` as the full-run checkpoint and `t5-small` as the smoke and toy checkpoint. Decoding is sampling with top-k, top-p, and temperature, not beam search.
- Failure modes called out: copied inputs, hallucinated objects that were not in the source caption, and remote-sensing phrases the base T5 checkpoint was not trained on.
- Compute: toy and smoke are CPU; the full protocol needs a GPU and is not claimed to have been run.

## Risks / Trade-offs

- [A later edit could paste untidy numbers] → The results section points at the JSON label `toy-scale pipeline check`.

## Migration Plan

- Replace the one-line README when the toy JSON exists.

## Open Questions

- None.
