import json
import random
from collections import defaultdict
from pathlib import Path

from eval.records import CaptionRecord


def load_caption_records(path: str | Path) -> list[CaptionRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for image in payload["images"]:
        seen: list[str] = []
        bag: set[str] = set()
        for sentence in image["sentences"]:
            raw = sentence["raw"].strip()
            if not raw or raw in bag:
                continue
            bag.add(raw)
            seen.append(raw)
        if not seen:
            continue
        filename = str(image["filename"])
        records.append(
            CaptionRecord(
                image_id=Path(filename).name,
                filename=filename,
                split=str(image.get("split") or ""),
                captions=seen,
                raw=image,
            )
        )
    return records


def group_by_official_split(
    records: list[CaptionRecord],
    max_images: int | None,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    min_captions_per_image: int,
) -> dict[str, list[CaptionRecord]]:
    """Subsample inside the image split stored in the caption file.

    Images are not moved between splits. ``max_images`` draws a stratified
    subset so a toy run still has a train, validation, and test image.
    """
    eligible = [record for record in records if len(record.captions) >= min_captions_per_image]
    by_split: dict[str, list[CaptionRecord]] = defaultdict(list)
    for record in eligible:
        key = record.split if record.split in ("train", "val", "test") else "train"
        by_split[key].append(record)
    for group in by_split.values():
        group.sort(key=lambda record: record.image_id)

    if max_images is not None and max_images < sum(len(group) for group in by_split.values()):
        rng = random.Random(seed)
        for group in by_split.values():
            rng.shuffle(group)
        n_test = max(1, int(round(max_images * test_fraction)))
        n_val = max(1, int(round(max_images * val_fraction)))
        n_train = max_images - n_val - n_test
        if n_train < 1:
            n_train = 1
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
        n_train = min(n_train, len(by_split["train"]))
        n_val = min(n_val, len(by_split["val"]))
        n_test = min(n_test, len(by_split["test"]))
        chosen = {
            "train": by_split["train"][:n_train],
            "val": by_split["val"][:n_val],
            "test": by_split["test"][:n_test],
        }
    else:
        chosen = {name: list(by_split[name]) for name in ("train", "val", "test")}

    missing = [name for name, group in chosen.items() if not group]
    if missing:
        raise ValueError(f"split is missing images for: {missing}")
    return chosen
