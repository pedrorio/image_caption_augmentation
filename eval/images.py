from pathlib import Path

from datasets import load_dataset
from PIL import Image

from eval.records import CaptionRecord


def ensure_images(records: list[CaptionRecord], cache_dir: str | Path, hf_dataset: str) -> None:
    """Download RSICD images from Hugging Face and match them by filename."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    wanted: dict[str, CaptionRecord] = {}
    for record in records:
        destination = cache / Path(record.filename).name
        record.image_path = destination
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        wanted[Path(record.filename).name] = record
    if not wanted:
        return
    found: set[str] = set()
    loaded = load_dataset(hf_dataset, streaming=True)
    splits = list(loaded.keys()) if hasattr(loaded, "keys") else [None]
    for split_name in splits:
        stream = loaded[split_name] if split_name is not None else loaded
        for row in stream:
            name = Path(str(row["filename"])).name
            if name not in wanted or name in found:
                continue
            image = row["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image.convert("RGB").save(cache / name)
            found.add(name)
            if len(found) == len(wanted):
                return
    missing = sorted(set(wanted) - found)
    raise FileNotFoundError(
        f"Could not find {len(missing)} RSICD images in {hf_dataset}. Examples: {missing[:5]}"
    )
