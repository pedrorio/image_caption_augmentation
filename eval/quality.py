from sentence_transformers import SentenceTransformer

from eval.text import normalize_caption, sentence_bleu4, token_jaccard

REPORT_KEYS = (
    "meaning_cosine",
    "self_bleu4_vs_source",
    "source_unigram_jaccard_mean",
    "exact_copy_rate",
    "near_copy_rate",
)


def diversity_against_source(
    sources: list[str],
    paraphrases: list[str],
    jaccard_threshold: float,
    bleu_threshold: float,
) -> dict[str, float]:
    if len(sources) != len(paraphrases) or not sources:
        raise ValueError("sources and paraphrases must be paired and non-empty")
    bleus = []
    overlaps = []
    exact = 0
    near = 0
    for source, paraphrase in zip(sources, paraphrases):
        bleu = sentence_bleu4(source, paraphrase)
        overlap = token_jaccard(source, paraphrase)
        is_exact = normalize_caption(source) == normalize_caption(paraphrase) and normalize_caption(source) != ""
        is_near = is_exact or overlap >= jaccard_threshold or bleu >= bleu_threshold
        bleus.append(bleu)
        overlaps.append(overlap)
        exact += int(is_exact)
        near += int(is_near)
    count = len(sources)
    return {
        "self_bleu4_vs_source": sum(bleus) / count,
        "source_unigram_jaccard_mean": sum(overlaps) / count,
        "exact_copy_rate": exact / count,
        "near_copy_rate": near / count,
        "n": float(count),
    }


def meaning_cosine(sources: list[str], paraphrases: list[str], model_name: str) -> float:
    if len(sources) != len(paraphrases) or not sources:
        raise ValueError("sources and paraphrases must be paired and non-empty")
    encoder = SentenceTransformer(model_name)
    left = encoder.encode(list(sources), normalize_embeddings=True)
    right = encoder.encode(list(paraphrases), normalize_embeddings=True)
    return float((left * right).sum(axis=1).mean())


def assemble_quality_report(meaning: float, diversity: dict[str, float]) -> dict[str, float]:
    report = {"meaning_cosine": float(meaning), **diversity}
    missing = [key for key in REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"quality report is missing {missing}; meaning and diversity are reported together")
    return {key: float(report[key]) for key in list(REPORT_KEYS) + ["n"]}


def quality_report(
    sources: list[str],
    paraphrases: list[str],
    meaning_model: str,
    jaccard_threshold: float,
    bleu_threshold: float,
) -> dict[str, float]:
    diversity = diversity_against_source(sources, paraphrases, jaccard_threshold, bleu_threshold)
    cosine = meaning_cosine(sources, paraphrases, meaning_model)
    return assemble_quality_report(cosine, diversity)
