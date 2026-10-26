from nltk.translate.gleu_score import sentence_gleu

from ica.utils.helpers import clean_words


def GLEAU(target, prediction) -> float:
    target_words = clean_words(target)
    prediction_words = clean_words(prediction)
    if not target_words or not prediction_words:
        return 0.0
    return float(
        sentence_gleu(
            references=[target_words],
            hypothesis=prediction_words,
            min_len=1,
            max_len=4,
        )
    )
