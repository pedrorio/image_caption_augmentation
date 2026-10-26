# downstream-ab Specification

## Purpose
Measures whether training captions plus T5 paraphrases improve a captioner over the original captions, on a shared held-out test split.

## Requirements

### Requirement: Paired captioner training

The system SHALL train one captioner on original training captions only (arm A) and one on those captions plus paraphrases of training captions (arm B). Both arms SHALL use the same images, the same hyperparameters, and the same held-out test images. The run SHALL use at least three seeds. Test references SHALL be the original test captions.

#### Scenario: Shared protocol

- **WHEN** an A/B run finishes
- **THEN** both arms list the same seeds, the same test image ids, and the same captioner hyperparameters

#### Scenario: At least three seeds

- **WHEN** the configured seed list is shorter than three
- **THEN** the run refuses to start

### Requirement: Reported differences

For BLEU-4, ROUGE-L, and CIDEr, the system SHALL report the per-seed score of each arm, the per-seed B-minus-A difference, and the mean and sample standard deviation across seeds for each arm and for the difference.

#### Scenario: Per-seed CIDEr difference

- **WHEN** three seeds finish
- **THEN** the report contains three CIDEr differences equal to that seed's B score minus that seed's A score

### Requirement: Archive decision rule

On a full-scale run, if the absolute mean of the per-seed CIDEr differences (B minus A) is less than one sample standard deviation of those differences, or a two-sided paired test of those differences against zero has p greater than 0.05, the system SHALL report that augmentation did not help and that the repository should be archived. A toy-scale run SHALL compute the same statistic, SHALL label it as a pipeline check, and SHALL NOT apply the archive action.

#### Scenario: Difference inside seed noise

- **WHEN** a full-scale absolute mean CIDEr difference is smaller than the sample standard deviation of the per-seed differences
- **THEN** the decision says augmentation did not help and the repository should be archived

#### Scenario: Insignificant paired test

- **WHEN** a full-scale paired test of the CIDEr differences against zero has p greater than 0.05
- **THEN** the decision says augmentation did not help and the repository should be archived

#### Scenario: Toy scale does not archive

- **WHEN** the same statistic is computed for a run whose scale is toy
- **THEN** the decision is marked not applicable and is labeled a toy-scale pipeline check
