from pytorch_lightning import LightningDataModule
from .datasets.RSICD import RSICD
from .datasets.Sydney import Sydney
from .datasets.UCM import UCM
from .transforms.ToTensorTransform import ToTensorTransform
from .transforms.EncodeTransform import EncodeTransform
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.transforms.transforms import Compose
from transformers import T5Tokenizer


class ImageCaptionsDataModule(LightningDataModule):

    def __init__(
            self,
            tokenizer: T5Tokenizer,
            batch_size: int = 6,
            data_dir: str = 'data/raw',
            num_workers: int = 8
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = Compose([
            EncodeTransform(tokenizer=self.tokenizer),
            # ToTensorTransform()
        ])
        self.num_workers = num_workers

        self.prepare_data()
        self.setup()

        # max_seq_length = 512,

    # def prepare_data(self):
    # #     download, tokenize and store
    # #     cannot assign state (i.e. self.x = y)
    #     pass

    def setup(self, stage=None):
        # import, split, transform
        # in this case, the implementation uses transforms as the tokenizer (not canonical)
        rsicd = RSICD(data_dir=self.data_dir, transform=self.transform)
        sydney = Sydney(data_dir=self.data_dir, transform=self.transform)
        ucm = UCM(data_dir=self.data_dir, transform=self.transform)

        full_dataset = ConcatDataset([rsicd, sydney, ucm])

        full_len = len(full_dataset)  # size of first dim
        self.train_len = round(full_len * 0.7)
        self.val_len = round(full_len * 0.2)
        self.test_len = round(full_len - self.val_len - self.train_len)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [self.train_len, self.val_len, self.test_len]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False
        )

    def stage_dataloader(self, stage) -> DataLoader:
        if stage == 'fit':
            return self.train_dataloader()
        elif stage == 'val':
            return self.val_dataloader()
        elif stage == 'test':
            return self.test_dataloader()

    def stage_len(self, stage) -> int:
        if stage == 'fit':
            return self.train_len
        elif stage == 'val':
            return self.val_len
        elif stage == 'test':
            return self.test_len
