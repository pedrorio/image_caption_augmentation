# leakage-guard Specification

## Purpose
Stops test-image captions and their paraphrases from entering captioner training by splitting images before augmentation.

## Requirements

### Requirement: Split before augmentation

The system SHALL assign every image to train, validation, or test before generating paraphrases. The assignment SHALL follow the image `split` field in the caption file when that field is present. Only training images SHALL be paraphrased for augmentation.

#### Scenario: Test images are not paraphrased

- **WHEN** paraphrases for the captioner training set are generated
- **THEN** every paraphrase is tied to a training image id and no test image id is present

#### Scenario: Order of operations

- **WHEN** an A/B run prepares data
- **THEN** the split exists before the first training paraphrase is produced

### Requirement: Leakage test

An automated test SHALL build a split with distinct caption strings, SHALL pass when training texts contain only training captions and paraphrases of training captions, and SHALL fail if a test caption or a paraphrase of a test caption is added to those texts.

#### Scenario: Clean split passes

- **WHEN** training texts are the training captions plus paraphrases of those captions
- **THEN** the leakage check returns without error

#### Scenario: Inserted test caption fails

- **WHEN** a test image's caption is added to the training texts
- **THEN** the leakage check raises an error

#### Scenario: Inserted test paraphrase fails

- **WHEN** a paraphrase generated from a test caption is added to the training texts
- **THEN** the leakage check raises an error
