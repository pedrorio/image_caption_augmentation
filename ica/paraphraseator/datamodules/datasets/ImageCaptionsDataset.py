import itertools
import json
import os

import requests
from numpy import array, load, savez_compressed
from torch.utils.data import Dataset


class ImageCaptionsDataset(Dataset):
    """Caption pairs from one remote-sensing JSON file.

    ``max_captions`` limits how many source captions are read. Images are
    taken whole so a caption is never separated from the rest of its image,
    and loading stops once the budget is met.
    """

    def __init__(self, file_path: str, transform=None, max_captions: int | None = None):
        self.transform = transform
        self.file_path = file_path
        self.max_captions = max_captions
        self.captions: list[str] = []

        cached = f"{self.file_path}.npz"
        # A capped read must not reuse or overwrite the full-file cache.
        if max_captions is None and os.path.exists(cached):
            self.unpickle_data()
        elif os.path.exists(self.file_path):
            self.import_data()
            if max_captions is None:
                self.pickle_data()
        else:
            self.download_data()
            self.import_data()
            if max_captions is None:
                self.pickle_data()

        print(f"{self.name}: {self.n_samples} pairs from {self.n_captions} captions")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def download_data(self):
        request = requests.get(
            f"https://raw.githubusercontent.com/pedrorio/image_caption_augmentation/master/{self.file_path}",
            timeout=60,
        )
        request.raise_for_status()
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as fp:
            json.dump(request.json(), fp)

    def _usable_captions(self, image: dict) -> list[str]:
        seen = set()
        captions = []
        for sentence in image["sentences"]:
            raw = sentence["raw"].strip()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            captions.append(raw)
        return captions

    def process_data(self):
        pairs: list[tuple[str, str]] = []
        used_captions: list[str] = []
        with open(self.file_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        name = data["dataset"]
        for image in data["images"]:
            captions = self._usable_captions(image)
            if not captions:
                continue
            if self.max_captions is not None and used_captions:
                if len(used_captions) >= self.max_captions:
                    break
                if len(used_captions) + len(captions) > self.max_captions:
                    break
            used_captions.extend(captions)
            if len(captions) >= 2:
                pairs.extend(itertools.permutations(captions, 2))
            if self.max_captions is not None and len(used_captions) >= self.max_captions:
                break
        return pairs, used_captions, name

    def import_data(self):
        pairs, captions, name = self.process_data()
        self.name = name
        self.captions = captions
        self.n_captions = len(captions)
        self.n_samples = len(pairs)
        self.x = array([pair[0] for pair in pairs], dtype=object)
        self.y = array([pair[1] for pair in pairs], dtype=object)

    def unpickle_data(self):
        data = load(f"{self.file_path}.npz", allow_pickle=True)
        self.x = data["x"]
        self.y = data["y"]
        self.name = str(data["name"])
        self.n_samples = int(data["n_samples"])
        if "captions" in data.files:
            self.captions = list(data["captions"])
        else:
            self.captions = list(dict.fromkeys(list(self.x) + list(self.y)))
        self.n_captions = len(self.captions)

    def pickle_data(self):
        savez_compressed(
            f"{self.file_path}.npz",
            x=self.x,
            y=self.y,
            name=self.name,
            n_samples=self.n_samples,
            captions=array(self.captions, dtype=object),
        )
