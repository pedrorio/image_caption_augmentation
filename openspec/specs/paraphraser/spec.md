# paraphraser Specification

## Purpose

Describes the caption paraphraser as it behaves today: which remote-sensing caption files it reads, how those captions become T5 training pairs, how training and decoding run, and which artifacts the run writes.

## Requirements

### Requirement: Remote-sensing caption files

The system SHALL read `dataset_rsicd_modified.json` (RSICD), `dataset_sydney_modified.json` (Sydney), and `dataset_ucm_modified.json` (UCM) from the data directory. Each file SHALL provide an `images` array whose items include `imgid`, `filename`, `split`, and `sentences` with a `raw` string.

#### Scenario: Files present in the data directory

- **WHEN** the data directory contains the three JSON files
- **THEN** the loader builds one caption collection per file and keeps the dataset name stored in that file

#### Scenario: Missing caption file

- **WHEN** a caption JSON is not on disk
- **THEN** the loader downloads that repo-relative path from the image_caption_augmentation master branch and parses the JSON

### Requirement: Same-image caption pairs

The system SHALL build ordered training pairs by taking every permutation of two distinct non-empty raw captions from the same image, then concatenate RSICD, Sydney, and UCM pairs into one collection.

#### Scenario: Five captions on one image

- **WHEN** an image has five distinct non-empty raw captions
- **THEN** that image contributes 20 ordered source-target pairs

### Requirement: Pair-level random split

The system SHALL split concatenated pairs at random into about 70 percent train, 20 percent validation, and 10 percent test. This split SHALL ignore the image `split` field, so pairs from one image MAY fall into different subsets.

#### Scenario: Fractions cover every pair

- **WHEN** the concatenated collection is split for training
- **THEN** the three subset sizes are the rounded 70/20/10 fractions of the pair count and sum to that count

### Requirement: T5 paraphrase model

The system SHALL fine-tune a Hugging Face T5 encoder-decoder, default checkpoint `ramsrigouthamg/t5_paraphraser`, with the matching T5 tokenizer. A source caption SHALL be lowercased, limited to alphabetic tokens, and prefixed with `paraphrase: ` before encoding. Sequences SHALL be padded or truncated to length 512.

#### Scenario: Default checkpoint

- **WHEN** training starts with no checkpoint override
- **THEN** the model weights and tokenizer are loaded from `ramsrigouthamg/t5_paraphraser`

#### Scenario: Source prefix

- **WHEN** a source caption is encoded for the model
- **THEN** the encoded text starts with the prefix `paraphrase: `

### Requirement: Training step

The system SHALL train with PyTorch Lightning and AdamW (learning rate 3e-4, epsilon 1e-8, weight decay 0), a linear warmup schedule, gradient clipping 1.0, and a default of 2 epochs. A step SHALL compute the generation loss and a sentence GLEU between a decoded hypothesis and the cleaned target. The import-time seed SHALL be 42. The entry script SHALL request 1 GPU, batch size 5, and 4 loader workers.

#### Scenario: Logged train metrics

- **WHEN** a training step finishes
- **THEN** `train_loss` and `train_gleau` are logged

#### Scenario: Entry-script batch and device

- **WHEN** `run.py` builds the trainer
- **THEN** it passes batch size 5, `gpus` 1, and `num_workers` 4

### Requirement: Sampling decode

The system SHALL decode with sampling (`do_sample` true, `top_k` 120, `top_p` 0.98, `early_stopping` true, one return sequence). Temperature and the new-token cap SHALL be whatever the installed library uses by default.

#### Scenario: One decoded string

- **WHEN** a sentence is paraphrased
- **THEN** generation returns one token sequence and the tokenizer decodes it with special tokens removed

### Requirement: Training outputs

The system SHALL write TensorBoard event files under the logs directory, step checkpoints under the checkpoints directory, and a `save_pretrained` model and tokenizer directory named from the checkpoint id.

#### Scenario: Fit finishes

- **WHEN** `train_model` returns after fit
- **THEN** the model and tokenizer directories exist under the checkpoints directory

### Requirement: Colab entry paths

The runnable entry script SHALL point data, logs, cache, and checkpoints at `/content/drive/MyDrive/ica/data/raw`, `/content/drive/MyDrive/ica/data/logs`, `/content/drive/MyDrive/ica/data/cache`, and `/content/drive/MyDrive/ica/data/checkpoints`.

#### Scenario: run.py paths

- **WHEN** `run.py` is executed with no arguments
- **THEN** those four `/content/drive/MyDrive/ica/data` paths are passed to the trainer
