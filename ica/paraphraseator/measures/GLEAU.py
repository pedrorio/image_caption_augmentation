from nltk.translate.gleu_score import sentence_gleu
from ...utils.helpers import clean_words
import pdb


def GLEAU(target, prediction):

    target_words = clean_words(target)
    prediction_words = clean_words(prediction)

    return sentence_gleu(references=target_words, hypothesis=prediction_words, min_len=1, max_len=4)

