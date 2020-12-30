import json
from nltk.tokenize import word_tokenize
from typing import Set, Dict, TypedDict, List, Union
from dataclasses import dataclass
from .Image import Image
from .Sentence import Sentence


class Augmentator:
    """
    Augments the list of datasets.
    """
    DatasetNames = Union[
        'dataset_sydney_modified',
        'dataset_ucm_modified',
        'dataset_rsicd_modified'
    ]
    Augmentations = Union[
        'translated',
        'paraphraseated'
    ]
    DATASET_NAMES: List[DatasetNames] = [
        'dataset_sydney_modified',
        'dataset_ucm_modified',
        'dataset_rsicd_modified'
    ]

    def __init__(self, name: DatasetNames, augmentation_type: Augmentations, recursive_levels: int = 1):

        os.makedirs('data/raw') if not os.path.exists('data/raw') else None

        self.name: str = name
        self.data: Dict = self.load_json(f'data/raw/{self.name}.json')
        self.output_data: Dict = self.data
        self.sentences: Dict[int, Set[str]] = {}
        self.new_sentences: Dict[int, Set[str]] = {}
        self.augmentation_type: str = augmentation_type
        self.recursive_levels = recursive_levels

    def call(self) -> None:
        self.process_sentences()
        self.augment_new_sentences()
        self.save_augmented_dataset()

    def augment_new_sentences(self) -> None:
        try:
            self.new_sentences = {
                int(k): v for k, v in self.load_json(
                    f'data/{self.augmentation_type}/{self.name}.json'
                ).items()
            }
        except FileNotFoundError:
            map(self.augment_image, self.data["images"])
            self.store_json(
                f'data/{self.augmentation_type}/{self.name}.json',
                {k: list(v) for k, v in self.new_sentences.items()}
            )

    def augment_image(self, image: Image) -> None:
        img_id = int(image["imgid"])
        print(f'[{self.name}] Captions for image: {img_id}')
        try:
            self.new_sentences[img_id] = self.load_json(
                f'data/{self.augmentation_type}/{self.name}/{img_id}.json'
            )
        except FileNotFoundError:
            print(f"[{self.name}] Image {img_id} not found")
            if img_id not in self.new_sentences:
                self.new_sentences[img_id] = set()
            self.augment_sentences(img_id, self.recursive_levels)
            self.store_json(
                f'data/{self.augmentation_type}/{self.name}/{img_id}.json',
                list(self.new_sentences[img_id])
            )

    def augment_sentences(self, img_id: int, levels: int) -> Set[str]:
        if levels >= 1:
            for sentence in self.sentences[img_id]:
                print(f'[{self.name}] Sentence from image: {img_id}')
                self.generate_augmentations(sentence, img_id)
            levels -= 1
            self.augment_sentences(img_id, levels)

        return self.new_sentences[img_id]

    def generate_augmentations(self, sentence: str, img_id: int):
        pass

    def process_sentences(self) -> None:
        sentences = map(self.create_image_sentences, self.data["images"])
        ids = map(lambda image: int(image["imgid"]), self.data["images"])
        self.sentences = dict(zip(ids, sentences))

    def create_image_sentences(self, image: Image) -> Set[str]:
        return {
            self.clean_sentence(sentence) for sentence in image["sentences"] if self.is_usable(sentence)
        }

    @staticmethod
    def clean_new_sentence(new_sentence: str) -> str:
        new_sentence = new_sentence.strip().lower().capitalize()
        return new_sentence if new_sentence.endswith('.') else new_sentence + '.'

    @staticmethod
    def clean_sentence(sentence: Sentence) -> str:
        return sentence["raw"].replace(".", "").strip().lower()

    @staticmethod
    def is_usable(sentence: Sentence):
        return sentence["raw"] != ''

    @staticmethod
    def store_json(file_name: str, container: Union[List, Dict]) -> None:
        json.dump(container, open(file_name, 'w'), indent=4)

    @staticmethod
    def load_json(file_name: str):
        return json.load(open(file_name))

    def encode_sentence(self, sentence: str, image_id: int) -> Sentence:
        sentence = self.clean_new_sentence(sentence)
        return {
            "tokens": word_tokenize(sentence), "raw": sentence, "imgid": image_id, "sentid": -1
        }

    def save_augmented_dataset(self) -> None:
        try:
            self.output_data = self.load_json(
                f'data/augmented/{self.augmentation_type}/{self.name}.json'
            )
        except FileNotFoundError:
            for image in self.output_data["images"]:
                img_id = int(image["imgid"])
                self.output_data["images"][img_id]["sentences"] += map(
                    lambda sentence: self.encode_sentence(sentence, img_id), self.new_sentences[img_id]
                )
            self.store_json(
                f'data/augmented/{self.augmentation_type}/{self.name}.json', self.output_data
            )

    def filter_sentences(self, sentence: str, img_id: int) -> bool:
        return sentence not in self.sentences[img_id] and paraphrase not in self.new_sentences[img_id]