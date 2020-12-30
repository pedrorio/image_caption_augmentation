from ..utils import Augmentator
from googletrans import Translator as GTranslator
from typing import Literal, List
import time


class Translator(Augmentator):
    """
    Back translates the list of datasets in the list of languages.
    """
    Language = Literal['pt', 'es', 'it', 'ko', 'zh-CN', 'ru', 'fr', 'de', 'ja', 'ar']
    LANGUAGES: List[Language] = ['pt', 'es', 'it', 'ko', 'zh-CN', 'ru', 'fr', 'de', 'ja', 'ar']
    Augmentation: Augmentator.Augmentations = "translated"

    def __init__(
            self,
            name: Augmentator.DatasetNames,
            augmentation_type=Augmentation,
            recursive_levels: int = 1
    ) -> None:
        super().__init__(name, augmentation_type, recursive_levels)
        self.translator = GTranslator()

    def generate_augmentations(self, sentence: str, img_id: int) -> None:
        print(f'[{self.name}] {sentence}')
        self.new_sentences[img_id] = {
            translation for translation in
            [self.back_translation(lang, sentence) for lang in self.LANGUAGES] if
            self.filter_sentences(translation, img_id)
        }
        time.sleep(2)

    def back_translation(self, lang: Language, sentence: str) -> str:
        to_foreign = self.translator.translate(sentence, dest=lang, src='en').text
        to_english = self.translator.translate(to_foreign, dest='en', src=lang).text
        print(f'bt: {lang}->en')
        return to_english if to_foreign != to_english else self.back_translation(lang, sentence)
