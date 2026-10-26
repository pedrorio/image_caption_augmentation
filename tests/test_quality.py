import pytest

from eval.quality import REPORT_KEYS, assemble_quality_report, diversity_against_source


def test_exact_copies_are_reported_with_meaning_slot():
    diversity = diversity_against_source(
        ["Many planes are parked at the airport."],
        ["many planes are parked at the airport"],
        jaccard_threshold=0.8,
        bleu_threshold=0.75,
    )
    assert diversity["exact_copy_rate"] == 1.0
    assert diversity["near_copy_rate"] == 1.0
    assert diversity["source_unigram_jaccard_mean"] == 1.0
    report = assemble_quality_report(1.0, diversity)
    assert report["meaning_cosine"] == 1.0
    for key in REPORT_KEYS:
        assert key in report


def test_unrelated_paraphrase_is_not_a_copy():
    diversity = diversity_against_source(
        ["several boats float in the harbor"],
        ["a dense residential block beside a highway"],
        jaccard_threshold=0.8,
        bleu_threshold=0.75,
    )
    assert diversity["exact_copy_rate"] == 0.0
    assert diversity["near_copy_rate"] == 0.0


def test_report_refuses_diversity_without_meaning():
    diversity = diversity_against_source(["a cat"], ["a cat"], 0.8, 0.75)
    diversity.pop("self_bleu4_vs_source")
    with pytest.raises(ValueError, match="meaning and diversity"):
        assemble_quality_report(0.5, diversity)
