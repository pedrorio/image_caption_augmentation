import math
import statistics
from typing import Literal, Never

from scipy.stats import ttest_1samp

Scale = Literal["toy", "full"]


def _exhaustive_scale(scale: str) -> Never:
    raise AssertionError(f"unknown scale {scale!r}")


def cider_decision(diffs: list[float], scale: str, alpha: float = 0.05) -> dict:
    """Apply the archive rule to per-seed CIDEr differences (B minus A).

    Full scale: augmentation did not help when the absolute mean difference is
    below one sample standard deviation of the differences, or the paired test
    against zero has p > alpha. Toy scale computes the same numbers and does
    not apply the archive action.
    """
    if scale not in ("toy", "full"):
        _exhaustive_scale(scale)
    if len(diffs) < 3:
        raise ValueError("the decision needs at least 3 per-seed differences")
    mean = float(statistics.mean(diffs))
    sd = float(statistics.stdev(diffs))
    if sd == 0.0:
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        p_value = float(ttest_1samp(diffs, popmean=0.0).pvalue)
        if math.isnan(p_value):
            p_value = 1.0
    within_noise = abs(mean) < sd if sd > 0.0 else mean == 0.0
    insignificant = p_value > alpha
    did_not_help = bool(within_noise or insignificant)
    applicable = scale == "full"
    if applicable and did_not_help:
        action = "archive the repository; augmentation did not help"
    elif applicable:
        action = "full-scale difference is outside the noise rule"
    else:
        action = "not applied"
    note = None
    if scale == "toy":
        note = "toy-scale pipeline check; the archive rule applies only to a full-scale run"
    elif scale == "full":
        note = None
    else:
        _exhaustive_scale(scale)
    return {
        "mean_diff": mean,
        "sd_diff": sd,
        "p_value": p_value,
        "alpha": alpha,
        "within_noise": within_noise,
        "p_greater_than_alpha": insignificant,
        "augmentation_did_not_help": did_not_help,
        "applicable": applicable,
        "action": action,
        "note": note,
        "rule": (
            "On a full-scale run, augmentation did not help and the repository "
            "should be archived if |mean CIDEr B-A| < 1 sample sd of the "
            "per-seed differences, or a paired test of those differences "
            "against zero has p > 0.05."
        ),
    }
