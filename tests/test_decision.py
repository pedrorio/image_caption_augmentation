from eval.decision import cider_decision


def test_difference_inside_seed_noise_archives_on_full_scale():
    decision = cider_decision([0.2, -0.1, 0.0], scale="full", alpha=0.05)
    assert decision["within_noise"] is True
    assert decision["augmentation_did_not_help"] is True
    assert decision["applicable"] is True
    assert "archive" in decision["action"]


def test_toy_scale_does_not_apply_the_archive_action():
    decision = cider_decision([0.2, -0.1, 0.0], scale="toy", alpha=0.05)
    assert decision["augmentation_did_not_help"] is True
    assert decision["applicable"] is False
    assert decision["action"] == "not applied"
    assert decision["note"] is not None and "toy-scale" in decision["note"]


def test_consistent_gain_is_outside_the_noise_rule():
    decision = cider_decision([2.0, 2.1, 1.9], scale="full", alpha=0.05)
    assert decision["within_noise"] is False
    assert decision["p_greater_than_alpha"] is False
    assert decision["augmentation_did_not_help"] is False
