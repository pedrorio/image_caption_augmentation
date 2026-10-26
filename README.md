# Image Caption Augmentation

This repository asks whether T5 paraphrases of remote-sensing captions help a downstream captioning model. The paraphraser fine-tunes T5 on caption pairs from the same image. The evaluation then trains a small captioner twice, on the same images and the same held-out test split: arm A uses the original training captions, arm B uses those captions plus paraphrases of training captions only.

The full-scale A/B has not been run. Numbers below come from one CPU toy-scale pipeline check and are not a result.

## Data

Caption files already in this repo:

- `data/raw/dataset_rsicd_modified.json` (RSICD; image ids such as `airport_1.jpg`; official `split` field train 8734 / val 1093 / test 1094)
- `data/raw/dataset_sydney_modified.json`
- `data/raw/dataset_ucm_modified.json`

The paraphraser can train on all three. The captioner A/B uses the RSICD file only. Captions are the strings in that JSON, not the captions shipped with the image download.

Images for the A/B are the public Hugging Face dataset [`arampacha/rsicd`](https://huggingface.co/datasets/arampacha/rsicd) (224×224). Files are matched to the JSON by basename and cached under `data/cache/rsicd_images`.

RSICD often repeats the same generic sentence on different images. The leakage guard therefore keys off image ids. A string that appears on both a train image and a test image is counted as `shared_generic_caption_strings` and does not by itself fail the run. The unit test uses captions that are unique to one image and fails if a test caption or a paraphrase of a test caption is inserted into training text.

## Method

1. Read the image `split` field and assign train, validation, and test **before** any paraphrase is generated. A toy run draws a stratified subset inside those official splits.
2. Fine-tune the paraphraser on pairs of distinct captions from the same **training** image. Validation captions are paraphrased only for the quality report. Test captions are not paraphrased.
3. Score paraphrase quality on the validation captions. Meaning and diversity are always written together.
4. Train the captioner. The vision encoder is a frozen ImageNet ResNet-18 (`fc` removed, 512-d). The decoder is a one-layer GRU. Arm A and arm B share images, vocabulary (built from arm B text), hyperparameters, and test references (the original test captions). Decoding at test time is greedy. At least three seeds are required. The report stores each seed's scores and the per-seed B−A difference, plus mean and sample standard deviation.

Meaning is the mean cosine similarity of `sentence-transformers/all-MiniLM-L6-v2` embeddings of each paraphrase and its source. Diversity is smoothed sentence BLEU-4 against the source, mean unigram Jaccard with the source, the fraction of exact normalized copies, and the fraction of near copies. A near copy is an exact normalized match, or token Jaccard ≥ 0.8, or smoothed sentence BLEU-4 ≥ 0.75.

Downstream metrics are BLEU-4, ROUGE-L, and CIDEr from `pycocoevalcap` after its Stanford PTB tokenizer (Java).

## Decision rule

On a **full-scale** run, augmentation did not help and the repository should be archived if the absolute mean CIDEr difference (B minus A) is less than one sample standard deviation of the per-seed differences, or a paired test of those differences against zero has p > 0.05.

A toy-scale run computes the same statistic, labels it a pipeline check, and does not apply the archive action.

## How to run

Python 3.12. Dependencies are pinned in `uv.lock` (CPU PyTorch). `scikit-learn` is the dependency name.

```bash
uv sync --frozen --all-groups
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Smoke test (about 20 RSICD captions, `t5-small`, 1 CPU step). This is what GitHub Actions runs, together with the other pytest modules. Java is required for the CIDEr tokenizer.

```bash
uv run pytest -q
uv run python run.py --config configs/smoke.yaml
```

Toy-scale pipeline check (CPU, 12 images, 1 paraphraser step, 1 captioner epoch, seeds 0, 1, 2). This command wrote `results/toy_scale_pipeline_check.json` in the session that added that file.

```bash
uv run python -m eval all --config configs/toy_ab.yaml
```

`python -m eval leakage --config configs/toy_ab.yaml` only checks the image split.

Local paraphraser training uses `configs/default.yaml` (`data/raw`, 0 GPUs). `configs/colab.yaml` restores the old Drive paths under `/content/drive/MyDrive/ica/data`, batch size 5, 1 GPU, 4 workers, and gradient accumulation 16.

```bash
uv run python run.py --config configs/default.yaml
uv run python run.py --data-dir data/raw --output-dir data/output --model-name t5-small --batch-size 4 --epochs 1 --seed 0 --max-examples 20
```

### Full-scale GPU run

`configs/full_ab.yaml` is the full protocol: every RSICD image in its official split, paraphraser `ramsrigouthamg/t5_paraphraser` for 2 epochs, captioner for 5 epochs, seeds 0–4, CUDA. It has not been run. The committed lock installs CPU wheels (`torch==2.14.1+cpu`). On a GPU machine, install a CUDA build of `torch` and `torchvision` before this cell. The run refuses to start if CUDA was requested and is missing.

```python
# Colab (or any GPU host), from the repo root, after dependencies are installed.
# Full-scale A/B. Not run in the session that added this README.
!python -m eval all --config configs/full_ab.yaml
```

## Results

The full-scale A/B has not been run.

The following figures are a **toy-scale pipeline check, not a result**. They are copied from `results/toy_scale_pipeline_check.json`.

Setup recorded in that file: scale `toy`, 8 train / 2 val / 2 test images (`min_captions_per_image` 2, seed 0), paraphraser `t5-small` for 1 step (sampling, top-k 50, top-p 0.95, temperature 1.0, `max_new_tokens` 12), captioner frozen ResNet-18 + GRU for 1 epoch on CPU, seeds `[0, 1, 2]`, vocab size 205. Train image ids: `square_279.jpg`, `00801.jpg`, `farmland_11.jpg`, `desert_267.jpg`, `00554.jpg`, `00466.jpg`, `port_339.jpg`, `viaduct_339.jpg`. Validation: `storagetanks_358.jpg`, `baseballfield_61.jpg`. Test: `storagetanks_99.jpg`, `00723.jpg`. `train_test_image_overlap` is 0, `test_images_paraphrased_for_training` is 0, and `shared_generic_caption_strings` is 0 on this subset.

Paraphrase quality on 7 validation captions (`n` = 7):

| metric | value |
| --- | --- |
| meaning_cosine | 0.7393929362297058 |
| self_bleu4_vs_source | 0.3233471121699671 |
| source_unigram_jaccard_mean | 0.43934589220303505 |
| exact_copy_rate | 0.14285714285714285 |
| near_copy_rate | 0.14285714285714285 |

Two of those outputs, as stored in the JSON: the source `there is a factory beside the river .` came back as `There is a factory beside the river.` (a near copy), and `there are many storage tanks in the factory .` came back as `Es sind viele Lager- tanks in die Fab.` (a non-English fragment from one step of `t5-small`).

Per-seed caption metrics:

| seed | A BLEU-4 | A ROUGE-L | A CIDEr | B BLEU-4 | B ROUGE-L | B CIDEr | B−A CIDEr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.1533208540230568e-13 | 0.16038111567123878 | 0.03261535952197393 | 1.544449777793687e-05 | 0.30056620038399745 | 0.2591989211116662 | 0.22658356158969228 |
| 1 | 1.2803722402439778e-09 | 0.16038111567123878 | 0.04904812775268767 | 1.8255137118388308e-09 | 0.20853786816269287 | 0.18585041155048915 | 0.13680228379780146 |
| 2 | 5.93420260935575e-13 | 0.1918238993710692 | 0.14176420721646513 | 1.3400825781415383e-09 | 0.16038111567123878 | 0.04003564490566845 | -0.10172856231079667 |

Mean ± sample standard deviation:

| | BLEU-4 | ROUGE-L | CIDEr |
| --- | --- | --- | --- |
| A | 4.2706033086343857e-10 ± 7.389898150554331e-10 | 0.17086204357118226 ± 0.018153499633168268 | 0.0744758981637089 ± 0.058849777988668014 |
| B | 5.1492211247422835e-06 ± 8.915971123959016e-06 | 0.22316172807264303 ± 0.07122750489087917 | 0.16169499252260794 ± 0.11156051546423586 |
| B−A | 5.14879406441142e-06 ± 8.916340788113719e-06 | 0.05229968450146078 ± 0.08588890625797899 | 0.08721909435889902 ± 0.16967935107494095 |

On this toy check the CIDEr mean difference is 0.08721909435889902 and the sample standard deviation of the per-seed differences is 0.16967935107494095. The absolute mean is smaller than that standard deviation (`within_noise` true). The paired test p-value is 0.46723676320198565, which is greater than 0.05. The gain is within seed noise. Because the scale is toy, `applicable` is false and `action` is `not applied`. That is not a decision to archive the repository.

## Model card

| | |
| --- | --- |
| Base checkpoint (full run) | `ramsrigouthamg/t5_paraphraser` |
| Checkpoint used in the smoke test and the toy check | `t5-small` |
| Training data | Caption JSON in `data/raw`: RSICD, Sydney, and UCM. The A/B fine-tune sees only RSICD training images. Sources are lowercased alphabetic tokens prefixed with `paraphrase: `. |
| Decoding | Sampling, not beam search. Full config: `do_sample` true, top-k 120, top-p 0.98, temperature 1.0, `max_new_tokens` 20, one return sequence. The config also sets `early_stopping: true`; current Transformers ignores that flag for this generate call. Toy decoding used top-k 50, top-p 0.95, temperature 1.0, `max_new_tokens` 12. |
| Known failure modes | Exact and near copies of the input. Hallucinated objects that were not in the source caption. On the toy `t5-small` check, outputs also drifted into non-English fragments. A one-epoch GRU on 8 images mostly repeated a few tokens, so BLEU-4 on that check is about zero. |
| Compute | The toy check is a CPU run: 12 images, 1 paraphraser step, 1 captioner epoch, 3 seeds. The full A/B is a GPU run (paraphraser `gpus: 1`, captioner `device: cuda`): all RSICD images, 2 paraphraser epochs, 5 captioner epochs, 5 seeds. The full A/B has not been executed in the reported session. `uv.lock` pins the CPU PyTorch wheel for CI. |

## Layout

- `run.py` trains the paraphraser from YAML plus CLI overrides.
- `eval/` splits images, paraphrases, scores quality, and runs the captioner A/B.
- `configs/smoke.yaml`, `configs/toy_ab.yaml`, `configs/full_ab.yaml` are the three protocols.
- `openspec/` records the baseline behavior and the changes archived in this work.
