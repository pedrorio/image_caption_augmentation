from nltk.translate.chrf_score import sentence_chrf
from ...utils.helpers import clean_words

def CHRF(x, y):
    x_words = clean_words(x)
    y_words = clean_words(y)

    return sentence_chrf(references=x_words, hypothesis=y_words, min_len=1, max_len=4)
