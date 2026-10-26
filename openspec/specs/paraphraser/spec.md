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

The system SHALL fine-tune a Hugging Face T5 encoder-decoder, default checkpoint `ramsrigouthamg/t5_paraphraser`, with a matching tokenizer. A source caption SHALL be lowercased, limited to alphabetic tokens, and prefixed with `paraphrase: ` before encoding. Source and target lengths SHALL come from the config. Local defaults SHALL use length 64. `configs/colab.yaml` SHALL use length 512.

#### Scenario: Default checkpoint

- **WHEN** training starts with no checkpoint override
- **THEN** the model weights and tokenizer are loaded from `ramsrigouthamg/t5_paraphraser`

#### Scenario: Source prefix

- **WHEN** a source caption is encoded for the model
- **THEN** the encoded text starts with the prefix `paraphrase: `

#### Scenario: Colab sequence length

- **WHEN** `configs/colab.yaml` is loaded
- **THEN** the source and target lengths are 512

### Requirement: Training step

The system SHALL optimize token cross-entropy on the target caption with AdamW, a linear warmup, gradient clipping, and the configured epoch count. Padding positions SHALL be ignored in the loss. The seed SHALL come from the config. Sentence GLEU SHALL be logged only when `compute_train_gleau` is true. The default config SHALL request zero GPUs.

#### Scenario: Logged train metrics

- **WHEN** a training step finishes
- **THEN** `train_loss` is logged as a finite float, `train_gleau` is logged only when `compute_train_gleau` is true, AdamW uses learning rate 3e-4, epsilon 1e-8, and weight decay 0, gradient clipping is 1.0, and the seed defaults to 42

#### Scenario: Entry-script batch and device

- **WHEN** the default config builds the trainer
- **THEN** it requests 0 GPUs, and `configs/colab.yaml` still requests batch size 5, 1 GPU, and GLEU

### Requirement: Sampling decode

The system SHALL decode with the configured sampler. Defaults SHALL be sampling on, `top_k` 120, `top_p` 0.98, temperature 1.0, early stopping on, one return sequence, and `max_new_tokens` 20.

#### Scenario: One decoded string

- **WHEN** a sentence is paraphrased with one return sequence
- **THEN** generation returns one string and one token row for that sentence, with special tokens removed from the string

#### Scenario: Batch token shape

- **WHEN** N sentences are paraphrased with one return sequence
- **THEN** the token-id tensor has shape `[N, L]` with `L` at least 1

### Requirement: Training outputs

The system SHALL write TensorBoard event files under the logs directory and a `save_pretrained` model plus tokenizer under the checkpoints directory. The saved directory name SHALL be the model id with `/` replaced by `--`.

#### Scenario: Fit finishes

- **WHEN** `train_model` returns after fit
- **THEN** the saved directory contains the model config and the tokenizer

### Requirement: Colab entry paths

Drive paths SHALL be selected by `configs/colab.yaml`. `run.py` with no arguments SHALL load `configs/default.yaml` and SHALL NOT pass `/content/drive/MyDrive` paths.

#### Scenario: run.py paths

- **WHEN** `run.py` is executed with no arguments
- **THEN** it loads `configs/default.yaml` and the data directory is `data/raw`, not a `/content/drive` path

### Requirement: Run configuration

The system SHALL load training settings from a YAML file and SHALL accept CLI overrides for the data directory, output directory, model name, batch size, epochs, seed, and maximum number of source captions. The default file SHALL use repository-local paths and zero GPUs. Unknown YAML keys SHALL be rejected.

#### Scenario: Default file is local

- **WHEN** `configs/default.yaml` is loaded with no CLI overrides
- **THEN** the data directory is `data/raw`, the GPU count is 0, and no path starts with `/content/drive`

#### Scenario: CLI overrides YAML

- **WHEN** the user passes `--data-dir`, `--output-dir`, `--model-name`, `--batch-size`, `--epochs`, `--seed`, and `--max-examples`
- **THEN** each of those values replaces the YAML value

#### Scenario: Output directory relocates artifacts

- **WHEN** the user passes `--output-dir` and does not pass a logs or checkpoints flag
- **THEN** logs and checkpoints are created under that output directory

#### Scenario: Unknown key

- **WHEN** a YAML file contains a key that is not a training setting
- **THEN** loading the file fails with an error that names the key

### Requirement: Colab config file

The system SHALL ship `configs/colab.yaml` with data, logs, cache, and checkpoints under `/content/drive/MyDrive/ica/data`, batch size 5, 1 GPU, 4 loader workers, 2 epochs, and gradient accumulation 16.

#### Scenario: Colab paths and device

- **WHEN** `configs/colab.yaml` is loaded
- **THEN** those four Drive paths, batch size 5, 1 GPU, and 4 workers are the resolved settings

### Requirement: Caption budget

`max_examples` SHALL limit how many source captions are read. The loader SHALL keep every caption of an included image and SHALL stop before adding an image that would pass the budget, once at least one image has been included.

#### Scenario: Budget stops on an image boundary

- **WHEN** `max_examples` is 20 and each image has 5 captions
- **THEN** the loaded caption count is at most 20 and is a multiple of the per-image caption count
