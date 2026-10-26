import re

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

_TOKEN = re.compile(r"[a-z0-9]+")
_SMOOTHER = SmoothingFunction().method1


def normalize_caption(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))


def tokenize_caption(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize_caption(left))
    right_tokens = set(tokenize_caption(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sentence_bleu4(reference: str, hypothesis: str) -> float:
    ref_tokens = tokenize_caption(reference)
    hyp_tokens = tokenize_caption(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return float(
        sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=_SMOOTHER,
        )
    )
