from augmentator.DatasetAugmentator import DatasetAugmentator
from googletrans import Translator
from typing import Literal, List
import time


class DatasetTranslator(DatasetAugmentator):
    """
    Back translates the list of datasets in the list of languages.
    """

    Language = Literal['pt', 'es', 'it', 'ko', 'zh-CN', 'ru', 'fr', 'de', 'ja', 'ar']
    LANGUAGES: List[Language] = ['pt', 'es', 'it', 'ko', 'zh-CN', 'ru', 'fr', 'de', 'ja', 'ar']

    Augmentation: DatasetAugmentator.Augmentations = "translated"

    def __init__(self, name: DatasetAugmentator.DatasetNames, augmentation_type=Augmentation,
                 recursive_levels: int = 1) -> None:
        super().__init__(name, augmentation_type, recursive_levels)
        self.translator = Translator()

    def generate_augmentations(self, sentence: str, img_id: int) -> None:
        print(f'[{self.name}] {sentence}')
        translations = [*map(lambda lang: self.back_translation(lang, sentence), self.LANGUAGES)]
        translations = [*filter(lambda translation: translation not in self.sentences[img_id], translations)]
        [*map(self.new_sentences[img_id].add, translations)]
        time.sleep(2)

    def back_translation(self, lang: Language, sentence: str) -> str:
        to_foreign = self.translator.translate(sentence, dest=lang, src='en').text
        to_english = self.translator.translate(to_foreign, dest='en', src=lang).text
        print(f'bt: {lang}->en')
        return to_english if to_foreign != to_english else self.back_translation(lang, sentence)