import json
from pathlib import Path

from ica.paraphraseator.datamodules.datasets.ImageCaptionsDataset import ImageCaptionsDataset


def _write_dataset(path: Path, captions_per_image: int, n_images: int) -> None:
    images = []
    for image_index in range(n_images):
        sentences = []
        for caption_index in range(captions_per_image):
            sentences.append(
                {
                    "tokens": ["caption"],
                    "raw": f"Image {image_index} caption {caption_index}.",
                    "imgid": image_index,
                    "sentid": caption_index,
                }
            )
        images.append(
            {
                "sentids": list(range(captions_per_image)),
                "imgid": image_index,
                "sentences": sentences,
                "split": "train",
                "filename": f"{image_index}.jpg",
            }
        )
    path.write_text(json.dumps({"dataset": "RSICD", "images": images}), encoding="utf-8")


def test_five_distinct_captions_make_twenty_pairs(tmp_path: Path):
    file_path = tmp_path / "dataset_rsicd_modified.json"
    _write_dataset(file_path, captions_per_image=5, n_images=1)
    dataset = ImageCaptionsDataset(str(file_path), max_captions=None)
    assert dataset.n_captions == 5
    assert dataset.n_samples == 20


def test_caption_budget_keeps_whole_images(tmp_path: Path):
    file_path = tmp_path / "dataset_rsicd_modified.json"
    _write_dataset(file_path, captions_per_image=5, n_images=6)
    dataset = ImageCaptionsDataset(str(file_path), max_captions=20)
    assert dataset.n_captions == 20
    assert dataset.n_captions % 5 == 0
    assert dataset.n_samples == 80
