import json
from nltk.tokenize import word_tokenize
from typing import Set, Dict, TypedDict, List, Union, Literal


class DatasetAugmentator:
    """
    Augments the list of datasets.
    """

    DatasetNames = Literal['dataset_sydney_modified', 'dataset_ucm_modified', 'dataset_rsicd_modified']
    Augmentations = Literal['translated', 'paraphraseated']

    Sentence = TypedDict('Sentence', {'tokens': List[str], 'raw': str, 'imgid': int, "sentid": int})
    Image = TypedDict(
        'Image',
        {
            'sentids': List[int], 'imgid': int, "sentences": List[Sentence],
            "split": Literal["train", "test"], "filename": str
        }
    )

    DATASET_NAMES: List[DatasetNames] = ['dataset_sydney_modified', 'dataset_ucm_modified', 'dataset_rsicd_modified']

    def __init__(self, name: DatasetNames, augmentation_type: Augmentations, recursive_levels: int = 1):
        self.name: str = name
        self.data: Dict = json.load(open(f'data/raw/{self.name}.json'))
        self.output_data: Dict = self.data
        self.sentences: Dict[int, Set[str]] = {}
        self.new_sentences: Dict[int, Set[str]] = {}
        self.augmentation_type: str = augmentation_type
        self.recursive_levels = recursive_levels

    def process(self) -> None:
        self.process_sentences()
        self.augment_new_sentences()
        self.save_augmented_dataset()

    def augment_new_sentences(self) -> None:
        try:
            self.new_sentences = {int(k): v for k, v in
                                  json.load(open(f'data/{self.augmentation_type}/{self.name}.json')).items()}
        except FileNotFoundError:
            [*map(self.augment_image, self.data["images"])]
            self.store_json(f'data/{self.augmentation_type}/{self.name}.json',
                            {k: list(v) for k, v in self.new_sentences.items()})

    def augment_image(self, image: Image) -> None:
        img_id = int(image["imgid"])
        print(f'[{self.name}] Captions for image: {img_id}')
        try:
            self.new_sentences[img_id] = json.load(open(f'data/{self.augmentation_type}/{self.name}/{img_id}.json'))
        except FileNotFoundError:
            print(f"[{self.name}] Image {img_id} not found")
            if img_id not in self.new_sentences:
                self.new_sentences[img_id] = set()
            self.augment_sentences(img_id, self.recursive_levels)
            self.store_json(f'data/{self.augmentation_type}/{self.name}/{img_id}.json',
                            list(self.new_sentences[img_id]))

    def augment_sentences(self, img_id: int, levels: int) -> Set[str]:
        if levels >= 1:
            for sentence in self.sentences[img_id]:
                if sentence == '':
                    continue
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
        image_sentences = set()
        texts = [*map(self.clean_sentence, image["sentences"])]
        [*map(image_sentences.add, texts)]
        return image_sentences

    @staticmethod
    def clean_new_sentence(new_sentence: str) -> str:
        new_sentence = new_sentence.strip().lower().capitalize()
        return new_sentence if new_sentence.endswith('.') else new_sentence + '.'

    @staticmethod
    def clean_sentence(sentence: Sentence) -> str:
        sentence = sentence["raw"].replace(".", "").strip().lower()
        return sentence

    @staticmethod
    def store_json(file_name: str, container: Union[List, Dict]) -> None:
        json.dump(container, open(file_name, 'w'), indent=4)

    def encode_sentence(self, sentence: str, image_id: int) -> Sentence:
        sentence = self.clean_new_sentence(sentence)
        return {"tokens": word_tokenize(sentence), "raw": sentence, "imgid": image_id, "sentid": -1}

    def save_augmented_dataset(self) -> None:
        try:
            self.output_data = json.load(open(f'data/augmented/{self.augmentation_type}/{self.name}.json'))
        except FileNotFoundError:
            for image in self.output_data["images"]:
                img_id = int(image["imgid"])
                self.output_data["images"][img_id]["sentences"] += [
                    *map(lambda sentence: self.encode_sentence(sentence, img_id), self.new_sentences[img_id])]
            self.store_json(f'data/augmented/{self.augmentation_type}/{self.name}.json', self.output_data)
