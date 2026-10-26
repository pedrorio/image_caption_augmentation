"""BLEU-4, ROUGE-L, and CIDEr via pycocoevalcap.

The scorers expect tokenized text. Official COCO eval tokenizes with the
Stanford PTB tokenizer shipped in pycocoevalcap, which needs Java. This
wrapper uses that tokenizer and records the name on the result.
"""

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def corpus_scores(hypotheses: list[str], references: list[list[str]]) -> dict[str, float]:
    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("each hypothesis needs a reference list")
    gts = {
        index: [{"caption": reference} for reference in refs]
        for index, refs in enumerate(references)
    }
    res = {index: [{"caption": hypothesis}] for index, hypothesis in enumerate(hypotheses)}
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)
    bleu, _per_image = Bleu(4).compute_score(gts_tok, res_tok, verbose=0)
    rouge, _rouge_each = Rouge().compute_score(gts_tok, res_tok)
    cider, _cider_each = Cider().compute_score(gts_tok, res_tok)
    return {
        "bleu4": float(bleu[3]),
        "rouge_l": float(rouge),
        "cider": float(cider),
        "tokenizer": "pycocoevalcap.PTBTokenizer",
    }
