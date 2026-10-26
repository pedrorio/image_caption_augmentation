import math

from eval.metrics import corpus_scores


def test_pycocoevalcap_scores_are_finite():
    scores = corpus_scores(
        ["a cat sits on a mat", "a dog runs"],
        [
            ["a cat sits on the mat", "the cat is on a mat"],
            ["a dog runs outside"],
        ],
    )
    assert scores["tokenizer"] == "pycocoevalcap.PTBTokenizer"
    for key in ("bleu4", "rouge_l", "cider"):
        assert math.isfinite(scores[key])
