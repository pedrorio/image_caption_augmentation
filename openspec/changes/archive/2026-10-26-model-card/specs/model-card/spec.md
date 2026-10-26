# model-card Specification

## Purpose

Tells a reader which T5 checkpoint is used, how it decodes, what it is known to get wrong, and what compute a real A/B needs.

## ADDED Requirements

### Requirement: Model card contents

The README SHALL name the base T5 checkpoint used for a full run, the caption files used as training data, and the decoding settings, including that decoding is sampling rather than beam search, plus top-k, top-p, and temperature. It SHALL name known failure modes, including copied inputs and hallucinated objects, and it SHALL state the compute a full run needs versus the CPU toy check.

#### Scenario: Decoding is described

- **WHEN** the README model card is read
- **THEN** it states sampling, temperature, top-k, and top-p, and it names `ramsrigouthamg/t5_paraphraser`

#### Scenario: Failure modes and compute

- **WHEN** the README model card is read
- **THEN** it mentions hallucinated objects and states that the full A/B is a GPU run that has not been executed in the reported session

### Requirement: Honest results section

The README results section SHALL say the full-scale A/B has not been run. Any number from the toy pipeline SHALL be labeled toy-scale and SHALL match the committed toy-scale JSON. The section SHALL include the archive decision rule: if the full-scale absolute mean CIDEr difference (B minus A) is less than one standard deviation of the per-seed differences, or the paired test has p greater than 0.05, augmentation did not help and the repository should be archived.

#### Scenario: Full run not claimed

- **WHEN** the results section is read
- **THEN** it states that the full-scale A/B has not been run and labels toy numbers as a pipeline check

#### Scenario: Decision rule is visible

- **WHEN** the results section is read
- **THEN** it states the CIDEr noise rule and that a full-scale failure of that rule means the repository should be archived
