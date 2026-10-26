from pytorch_lightning import LightningDataModule
from torch import Generator, randperm
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.transforms import Compose

from ica.data_split import split_sizes

from .datasets.RSICD import RSICD
from .datasets.Sydney import Sydney
from .datasets.UCM import UCM
from .transforms.EncodeTransform import EncodeTransform

FACTORIES = {
    "rsicd": RSICD,
    "sydney": Sydney,
    "ucm": UCM,
}


class ImageCaptionsDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        batch_size: int = 4,
        data_dir: str = "data/raw",
        num_workers: int = 0,
        max_examples: int | None = None,
        max_source_length: int = 64,
        max_target_length: int = 64,
        datasets: list[str] | None = None,
        seed: int = 42,
        train_fraction: float = 0.7,
        val_fraction: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = Compose(
            [
                EncodeTransform(
                    tokenizer=self.tokenizer,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                )
            ]
        )
        self.num_workers = num_workers
        self.max_examples = max_examples
        self.dataset_names = datasets or ["rsicd", "sydney", "ucm"]
        self.seed = seed
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.captions: list[str] = []
        self.train_len = 0
        self.val_len = 0
        self.test_len = 0

    def setup(self, stage=None):
        remaining = self.max_examples
        parts = []
        self.captions = []
        for name in self.dataset_names:
            if remaining is not None and remaining <= 0:
                break
            try:
                factory = FACTORIES[name]
            except KeyError as exc:
                known = ", ".join(sorted(FACTORIES))
                raise ValueError(f"Unknown dataset {name!r}. Expected one of: {known}") from exc
            dataset = factory(
                data_dir=self.data_dir,
                transform=self.transform,
                max_captions=remaining,
            )
            parts.append(dataset)
            self.captions.extend(dataset.captions)
            if remaining is not None:
                remaining -= dataset.n_captions

        if not parts:
            raise ValueError("No caption datasets were loaded")

        full_dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        full_len = len(full_dataset)
        self.train_len, self.val_len, self.test_len = split_sizes(
            full_len, self.train_fraction, self.val_fraction
        )
        generator = Generator().manual_seed(self.seed)
        # randperm keeps a zero-length split from crashing random_split.
        order = randperm(full_len, generator=generator).tolist() if full_len else []
        train_end = self.train_len
        val_end = self.train_len + self.val_len
        self.train_dataset = Subset(full_dataset, order[:train_end])
        self.val_dataset = Subset(full_dataset, order[train_end:val_end])
        self.test_dataset = Subset(full_dataset, order[val_end:])

    def _loader(self, dataset, length: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=length >= self.batch_size,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.train_len, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, self.val_len, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, self.test_len, shuffle=False)

    def stage_len(self, stage) -> int:
        if stage == "fit":
            return self.train_len
        if stage == "val":
            return self.val_len
        if stage == "test":
            return self.test_len
        raise ValueError(f"Unknown stage {stage!r}")
