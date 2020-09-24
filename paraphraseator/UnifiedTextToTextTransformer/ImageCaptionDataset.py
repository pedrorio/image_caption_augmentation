import itertools
import json
from common.index import set_seed
from pandas import DataFrame
from torch.utils.data import Dataset

set_seed(42)


class ImageCaptionDataset(Dataset):

    def __init__(self, tokenizer, files, max_len=256):
        self.source_column = "phrase1"
        self.target_column = "phrase2"
        self.data = DataFrame(columns=(self.source_column, self.target_column))
        num_pairs = 0
        for dataset in files:
            auxdata = json.load(open(dataset))
            for image in auxdata["images"]:
                sentences = set()
                for sentence in image["sentences"]:
                    text = sentence["raw"].strip().lower()
                    if len(text) <= 1:
                        continue
                    if text.endswith('.'): text = text[:-1].strip()
                    sentences.add(text)

                for s1, s2 in itertools.permutations(sentences, 2):
                    self.data.loc[num_pairs] = [s1, s2]
                    num_pairs += 1
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx, self.source_column], self.data.loc[idx, self.target_column]
            input_ = "UnifiedTextToTextTransformer: " + input_ + ' </s>'
            target = target + " </s>"

            tokenized_inputs = self.tokenizer.batch_encode_plus([input_], max_length=self.max_len,
                                                                pad_to_max_length=True, return_tensors="pt",
                                                                truncation=True)
            tokenized_targets = self.tokenizer.batch_encode_plus([target], max_length=self.max_len,
                                                                 pad_to_max_length=True, return_tensors="pt",
                                                                 truncation=True)
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
