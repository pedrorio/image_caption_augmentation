from pytorch_lightning import seed_everything
from torch.utils.data import Dataset as TorchDataset

seed_everything(42)


class Dataset(TorchDataset):

    def __init__(self, hparams):
        # hparams is tokenizer, data, source_column, target_column
        self.hparams = hparams
        self.source_column = hparams.source_column
        self.target_column = hparams.target_column

        self.max_len = 512

    def __getitem__(self, index):
        source_ids = self.input_ids(self.inputs, index)
        target_ids = self.input_ids(self.targets, index)
        src_mask = self.attention_mask(self.inputs, index)
        target_mask = self.attention_mask(self.targets, index)
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    @staticmethod
    def input_ids(collection, index):
        return collection[index]["input_ids"].squeeze()

    @staticmethod
    def attention_mask(collection, index):
        return collection[index]["attention_mask"].squeeze()
