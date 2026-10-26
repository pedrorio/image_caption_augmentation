# No leakage

## Why

The paraphraser datamodule splits caption pairs at random, so captions from one image can land in train and test. Augmenting after that split would train the captioner on paraphrases of the test set.

## What Changes

- Split images into train, validation, and test before any paraphrase is generated.
- Paraphrase training captions only. Do not paraphrase test captions.
- Fail the run if a test image id is in training or if a test caption or a paraphrase of a test caption is added to the training texts.
- Cover that failure with an automated test on controlled captions.

## Capabilities

### New Capabilities

- `leakage-guard`: image split before augmentation, and a test that test text stays out of training.

### Modified Capabilities

- None.

## Impact

- `eval` split and leakage checks. The existing pair split inside paraphraser pretraining is unchanged and is not the A/B split.
- Repeated generic RSICD sentences can occur on both a train image and a test image. The guard keys paraphrases by image id. The automated test uses unique strings so a copied test sentence fails.
