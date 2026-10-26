"""Image-level split, then paraphrase, then captioner A/B."""

import gc
import json
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from eval.captioner import Vocab, extract_features, greedy_caption, train_decoder
from eval.config import EvalConfig, load_eval_config
from eval.data import group_by_official_split, load_caption_records
from eval.decision import cider_decision
from eval.images import ensure_images
from eval.leakage import LeakageError, assert_image_split
from eval.metrics import corpus_scores
from eval.quality import quality_report
from eval.records import CaptionRecord
from eval.text import normalize_caption
from ica.paraphraseator.T5 import paraphrase_sentences, train_paraphraser


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _write_train_json(images: list[CaptionRecord], directory: Path) -> None:
    payload = {"dataset": "RSICD", "images": [image.raw for image in images]}
    (directory / "dataset_rsicd_modified.json").write_text(json.dumps(payload), encoding="utf-8")


def _paraphrase_records(model, tokenizer, records: list[CaptionRecord], cfg: EvalConfig) -> dict[str, list[str]]:
    train_cfg = cfg.to_train_config(data_dir=".", work_dir=".")
    pairs: list[tuple[str, str]] = []
    for record in records:
        for caption in record.captions:
            pairs.append((record.image_id, caption))
    texts: list[str] = []
    batch_size = max(1, cfg.paraphraser.batch_size)
    for start in range(0, len(pairs), batch_size):
        chunk = [caption for _image_id, caption in pairs[start : start + batch_size]]
        decoded = paraphrase_sentences(model, tokenizer, chunk, train_cfg, device=torch.device("cpu"))
        texts.extend(decoded.texts)
    by_image: dict[str, list[str]] = defaultdict(list)
    for (image_id, _source), text in zip(pairs, texts):
        by_image[image_id].append(text)
    return by_image


def _training_texts(records: list[CaptionRecord], paraphrases: dict[str, list[str]], arm: str) -> list[str]:
    texts: list[str] = []
    for record in records:
        texts.extend(record.captions)
        if arm == "B":
            texts.extend(text for text in paraphrases.get(record.image_id, []) if text.strip())
    return texts


def _examples(records, paraphrases, arm: str, vocab: Vocab, index_of: dict[str, int], max_len: int):
    examples = []
    for record in records:
        texts = _training_texts([record], paraphrases, arm)
        for text in texts:
            if not text.strip():
                continue
            examples.append((index_of[record.image_id], vocab.encode(text, max_len)))
    if not examples:
        raise RuntimeError(f"arm {arm} has no training captions")
    return examples


def _score_arm(hypotheses: list[str], references: list[list[str]]) -> dict[str, float]:
    scores = corpus_scores(hypotheses, references)
    return {key: float(scores[key]) for key in ("bleu4", "rouge_l", "cider")}


def run_eval(cfg: EvalConfig) -> dict:
    if cfg.scale not in ("toy", "full"):
        raise ValueError(f"scale must be toy or full, got {cfg.scale!r}")
    if len(cfg.captioner.seeds) < 3:
        raise ValueError("the A/B requires at least 3 seeds")
    if cfg.captioner.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("this config requests cuda but no GPU is available")
    device = torch.device(cfg.captioner.device)

    records = load_caption_records(cfg.captions_json)
    groups = group_by_official_split(
        records,
        max_images=cfg.max_images,
        seed=cfg.seed,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        min_captions_per_image=cfg.min_captions_per_image,
    )
    # Split exists before any paraphrase is generated.
    assert_image_split(groups["train"], groups["test"], {})

    with tempfile.TemporaryDirectory(prefix="ica-paraphrase-") as temporary:
        data_dir = Path(temporary) / "raw"
        data_dir.mkdir()
        _write_train_json(groups["train"], data_dir)
        train_cfg = cfg.to_train_config(str(data_dir), temporary)
        trained = train_paraphraser(train_cfg)
        tokenizer = AutoTokenizer.from_pretrained(trained.model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(trained.model_dir)
        model.eval()
        train_paraphrases = _paraphrase_records(model, tokenizer, groups["train"], cfg)
        val_paraphrases = _paraphrase_records(model, tokenizer, groups["val"], cfg)
        del model
        del tokenizer
        gc.collect()

    assert_image_split(groups["train"], groups["test"], train_paraphrases)
    test_ids = {record.image_id for record in groups["test"]}
    if set(train_paraphrases) & test_ids:
        raise LeakageError("training paraphrases include a test image")

    val_sources = [caption for record in groups["val"] for caption in record.captions]
    val_outputs = [text for record in groups["val"] for text in val_paraphrases.get(record.image_id, [])]
    if len(val_sources) != len(val_outputs):
        raise RuntimeError("validation paraphrase count does not match source captions")
    quality = quality_report(
        val_sources,
        val_outputs,
        cfg.meaning_model,
        cfg.near_copy_jaccard,
        cfg.near_copy_bleu,
    )
    gc.collect()

    ordered = groups["train"] + groups["val"] + groups["test"]
    ensure_images(ordered, cfg.image_cache_dir, cfg.hf_dataset)
    features = extract_features(ordered, device)
    index_of = {record.image_id: index for index, record in enumerate(ordered)}
    vocab_sentences = _training_texts(groups["train"], train_paraphrases, "B")
    vocab = Vocab.build(vocab_sentences)

    references = [list(record.captions) for record in groups["test"]]
    per_seed = []
    for seed in cfg.captioner.seeds:
        arm_scores = {}
        for arm in ("A", "B"):
            examples = _examples(
                groups["train"],
                train_paraphrases,
                arm,
                vocab,
                index_of,
                cfg.captioner.max_decode_len,
            )
            decoder = train_decoder(
                features,
                examples,
                vocab,
                embed_dim=cfg.captioner.embed_dim,
                hidden_dim=cfg.captioner.hidden_dim,
                epochs=cfg.captioner.epochs,
                batch_size=cfg.captioner.batch_size,
                learning_rate=cfg.captioner.learning_rate,
                seed=seed,
                device=device,
            )
            hypotheses = [
                greedy_caption(
                    decoder,
                    features[index_of[record.image_id]],
                    vocab,
                    cfg.captioner.max_decode_len,
                    device,
                )
                for record in groups["test"]
            ]
            arm_scores[arm] = {
                "metrics": _score_arm(hypotheses, references),
                "hypotheses": hypotheses,
            }
            del decoder
        difference = {
            metric: arm_scores["B"]["metrics"][metric] - arm_scores["A"]["metrics"][metric]
            for metric in ("bleu4", "rouge_l", "cider")
        }
        per_seed.append(
            {
                "seed": seed,
                "A": arm_scores["A"]["metrics"],
                "B": arm_scores["B"]["metrics"],
                "B_minus_A": difference,
                "hypotheses": {"A": arm_scores["A"]["hypotheses"], "B": arm_scores["B"]["hypotheses"]},
            }
        )

    summary = {}
    for arm in ("A", "B"):
        summary[arm] = {
            metric: _mean_sd([row[arm][metric] for row in per_seed])
            for metric in ("bleu4", "rouge_l", "cider")
        }
    summary["B_minus_A"] = {
        metric: _mean_sd([row["B_minus_A"][metric] for row in per_seed])
        for metric in ("bleu4", "rouge_l", "cider")
    }
    decision = cider_decision(
        [row["B_minus_A"]["cider"] for row in per_seed],
        scale=cfg.scale,
        alpha=cfg.decision_alpha,
    )
    train_norms = {normalize_caption(text) for text in _training_texts(groups["train"], train_paraphrases, "B")}
    test_norms = {normalize_caption(caption) for record in groups["test"] for caption in record.captions}
    shared_generic = len(train_norms & test_norms)
    label = (
        "toy-scale pipeline check, not a result"
        if cfg.scale == "toy"
        else "full-scale A/B"
    )
    report = {
        "scale": cfg.scale,
        "label": label,
        "dataset": {
            "captions_json": cfg.captions_json,
            "hf_images": cfg.hf_dataset,
            "split_source": "image split field in the caption JSON, before paraphrasing",
            "n_train_images": len(groups["train"]),
            "n_val_images": len(groups["val"]),
            "n_test_images": len(groups["test"]),
            "train_image_ids": [record.image_id for record in groups["train"]],
            "val_image_ids": [record.image_id for record in groups["val"]],
            "test_image_ids": [record.image_id for record in groups["test"]],
            "min_captions_per_image": cfg.min_captions_per_image,
        },
        "paraphraser": {
            "model_name_or_path": cfg.paraphraser.model_name_or_path,
            "max_steps": cfg.paraphraser.max_steps,
            "max_epochs": cfg.paraphraser.max_epochs,
            "do_sample": cfg.paraphraser.do_sample,
            "top_k": cfg.paraphraser.top_k,
            "top_p": cfg.paraphraser.top_p,
            "temperature": cfg.paraphraser.temperature,
            "max_new_tokens": cfg.paraphraser.max_new_tokens,
            "num_return_sequences": cfg.paraphraser.num_return_sequences,
        },
        "captioner": {
            "architecture": "frozen ImageNet ResNet-18 (fc removed) + one-layer GRU",
            "epochs": cfg.captioner.epochs,
            "batch_size": cfg.captioner.batch_size,
            "learning_rate": cfg.captioner.learning_rate,
            "embed_dim": cfg.captioner.embed_dim,
            "hidden_dim": cfg.captioner.hidden_dim,
            "max_decode_len": cfg.captioner.max_decode_len,
            "seeds": list(cfg.captioner.seeds),
            "device": cfg.captioner.device,
            "vocab_size": len(vocab),
        },
        "leakage": {
            "train_test_image_overlap": 0,
            "test_images_paraphrased_for_training": 0,
            "shared_generic_caption_strings": shared_generic,
            "note": (
                "RSICD repeats generic sentences across images. "
                "shared_generic_caption_strings counts those overlaps. "
                "The guard fails if a test image id is paraphrased into training."
            ),
        },
        "paraphrase_quality": quality,
        "quality_examples": [
            {"source": source, "paraphrase": paraphrase}
            for source, paraphrase in list(zip(val_sources, val_outputs))[:8]
        ],
        "downstream": {"per_seed": per_seed, "summary": summary, "decision": decision},
        "test_references": references,
    }
    destination = Path(cfg.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_from_path(path: str) -> dict:
    return run_eval(load_eval_config(path))
