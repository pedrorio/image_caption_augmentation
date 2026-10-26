# Paraphrase quality

## Why

A paraphraser that copies the source keeps meaning and adds no captions. Meaning and diversity have to be reported together or the copy looks like a success.

## What Changes

- Score held-out source captions (validation images) with embedding cosine to the source and with diversity against that same source.
- Diversity is self-BLEU-4 versus the source, mean unigram Jaccard overlap, exact-copy rate, and near-copy rate.
- The report always includes both groups. One without the other is not a result.

## Capabilities

### New Capabilities

- `paraphrase-quality`: paired meaning and diversity scores on held-out captions.

### Modified Capabilities

- None.

## Impact

- New `eval` quality report, used by `python -m eval`.
- The toy run records these scores and labels them as a pipeline check.
