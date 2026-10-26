# Design

## Context

See proposal.md. The paraphraser can copy remote-sensing captions. A single similarity score would hide that.

## Goals / Non-Goals

- Goals: one report that always contains a meaning score and diversity scores against the source caption.
- Non-Goals: claiming those toy-scale scores are a quality result. Full-scale scoring is the same code with `configs/full_ab.yaml`.

## Decisions

- Meaning is cosine similarity of `sentence-transformers/all-MiniLM-L6-v2` embeddings, not BERTScore. MiniLM is small enough to run on CPU beside the toy pipeline. BERTScore would add a second large encoder for the same kind of score.
- Diversity is computed against the source, not between paraphrases: self-BLEU-4 (smoothed sentence BLEU), unigram Jaccard, exact normalized copy rate, and near-copy rate.
- A near copy is an exact normalized match, or token Jaccard at least 0.8, or smoothed BLEU-4 at least 0.75.
- Held-out sources are validation-image captions. Test captions are not paraphrased for this report.

## Risks / Trade-offs

- [MiniLM cosine is not BLEU and not human judgment] → The report name says embedding cosine, and diversity is printed beside it.
- [Short captions make unsmoothed BLEU zero] → Use NLTK smoothing method 1 and document it.

## Migration Plan

- `python -m eval all --config configs/toy_ab.yaml` writes the paired report.
- The full command is the same with `configs/full_ab.yaml`.

## Open Questions

- None.
