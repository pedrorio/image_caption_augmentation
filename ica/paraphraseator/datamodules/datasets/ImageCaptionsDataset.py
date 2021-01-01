import os
import json
import itertools
from torch.utils.data import Dataset
from pandas import DataFrame
from numpy import savez_compressed, load
import requests


class ImageCaptionsDataset(Dataset):
    def __init__(self, file_path: str, transform=None):
        """
        Data loading

        Loads the json file
        Adds image caption permutations for each specific image
        :param file_path:
        """

        self.transform = transform
        self.file_path = file_path

        if os.path.exists(f'{self.file_path}.npz'):
            self.unpickle_data()
        elif os.path.exists(self.file_path):
            self.import_data()
            self.pickle_data()
        else:
            self.download_data()
            self.import_data()
            self.pickle_data()

        print(f'{self.name}: {self.n_samples}')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def download_data(self):
        request = requests.get(
            f'https://raw.githubusercontent.com/pedrorio/image_caption_augmentation/master/{self.file_path}'
        )
        with open(self.file_path, 'w') as fp:
            json.dump(request.json(), fp, indent=4)

    def process_data(self):
        xy = DataFrame(columns=("x", "y"))
        with open(f'{self.file_path}', "r") as fp:
            data = json.load(fp)
            xy.index.name = data["dataset"]
            number_of_pairs = 0

            for image in data["images"]:
                sentences = set()
                for sentence in image["sentences"]:
                    sentence = sentence["raw"]
                    sentences.add(sentence)
                for x, y in itertools.permutations(sentences, 2):
                    xy.loc[number_of_pairs] = [x, y]
                    number_of_pairs += 1
        return xy

    def import_data(self):
        xy = self.process_data()
        self.name = xy.index.name
        self.n_samples = len(xy.index)
        self.x = xy["x"].to_numpy()
        self.y = xy["y"].to_numpy()

    def unpickle_data(self):
        data = load(f'{self.file_path}.npz', allow_pickle=True)
        # print(data.files)
        self.x = data["x"]
        self.y = data["y"]
        self.name = data["name"]
        self.n_samples = data["n_samples"]

    def pickle_data(self):
        savez_compressed(f'{self.file_path}.npz', x=self.x, y=self.y, name=self.name, n_samples=self.n_samples)
