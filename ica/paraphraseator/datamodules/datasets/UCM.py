import os

from .ImageCaptionsDataset import ImageCaptionsDataset


def UCM(data_dir: str, transform=None, max_captions: int | None = None):
    file_path = os.path.join(data_dir, "dataset_ucm_modified.json")
    return ImageCaptionsDataset(file_path=file_path, transform=transform, max_captions=max_captions)
