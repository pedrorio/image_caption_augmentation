import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_model_card_and_toy_numbers():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = json.loads((ROOT / "results" / "toy_scale_pipeline_check.json").read_text(encoding="utf-8"))
    lowered = readme.lower()
    assert "the full-scale a/b has not been run" in lowered
    assert "toy-scale pipeline check, not a result" in lowered
    assert "ramsrigouthamg/t5_paraphraser" in readme
    assert "sampling" in lowered
    assert "temperature" in lowered
    assert "hallucinated objects" in lowered
    assert "should be archived" in lowered
    assert "p > 0.05" in readme
    quality = report["paraphrase_quality"]
    for key in (
        "meaning_cosine",
        "self_bleu4_vs_source",
        "source_unigram_jaccard_mean",
        "exact_copy_rate",
        "near_copy_rate",
    ):
        assert str(quality[key]) in readme
    cider = report["downstream"]["summary"]["B_minus_A"]["cider"]
    assert str(cider["mean"]) in readme
    assert str(cider["sd"]) in readme
    assert str(report["downstream"]["decision"]["p_value"]) in readme
    for row in report["downstream"]["per_seed"]:
        assert str(row["B_minus_A"]["cider"]) in readme
