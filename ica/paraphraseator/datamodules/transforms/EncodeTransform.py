from nltk.tokenize import word_tokenize
from ....utils.helpers import clean_sentence, annotate_sentence, encode_sentence
import pdb


class EncodeTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sample):
        x, y = sample

        clean_x = clean_sentence(x)
        annotated_x = annotate_sentence(clean_x)
        encoded_x = encode_sentence(self.tokenizer, annotated_x)

        clean_y = clean_sentence(y)
        encoded_y = encode_sentence(self.tokenizer, y)

        sample = dict(
            x=x, clean_x=clean_x, annotated_x=annotated_x, encoded_x=encoded_x,
            y=y, clean_y=clean_y, encoded_y=encoded_y,
        )
        return sample
