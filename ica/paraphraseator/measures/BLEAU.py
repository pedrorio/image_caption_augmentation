from nltk.translate.bleu_score import sentence_bleu
from ...utils.helpers import clean_words

def BLEAU(x, y):
    x_words = clean_words(x)
    y_words = clean_words(y)

    return sentence_bleu(references=[x_words], hypothesis=y_words, weights=(0.25, 0.25, 0.25, 0.25))

