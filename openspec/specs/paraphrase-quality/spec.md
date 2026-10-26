# paraphrase-quality Specification

## Purpose
Reports whether paraphrases of held-out captions keep the source meaning and whether they are more than copies of that source.

## Requirements

### Requirement: Paired meaning and diversity

On a held-out set of source captions, the system SHALL report meaning and diversity in the same result. Meaning SHALL be the mean cosine similarity between sentence embeddings of each paraphrase and its source. Diversity SHALL include mean self-BLEU-4 of each paraphrase against its source, mean unigram Jaccard overlap with the source, the fraction of exact normalized copies, and the fraction of near copies. A result that omits either group SHALL be rejected.

#### Scenario: Copy is not a quality win

- **WHEN** every paraphrase is identical to its source after normalization
- **THEN** the exact-copy rate is 1 and the meaning score is present in the same report

#### Scenario: Both groups are required

- **WHEN** a quality report is built
- **THEN** it contains `meaning_cosine`, `self_bleu4_vs_source`, `source_unigram_jaccard_mean`, `exact_copy_rate`, and `near_copy_rate`

### Requirement: Near copy definition

A paraphrase SHALL count as a near copy when its normalized text equals the source, or its token Jaccard with the source is at least 0.8, or its smoothed sentence BLEU-4 against the source is at least 0.75.

#### Scenario: High overlap

- **WHEN** a paraphrase shares enough tokens with the source to meet either threshold
- **THEN** it increments the near-copy count

### Requirement: Held-out sources

Quality sources SHALL be captions from validation images. Test-image captions SHALL NOT be paraphrased for this report.

#### Scenario: Validation only

- **WHEN** the quality set is chosen
- **THEN** every source caption belongs to a validation image and none belongs to a test image
