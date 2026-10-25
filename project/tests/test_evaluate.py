"""Metric arithmetic, cutoff and short-result denominator conventions."""

import pytest

from project.evaluate import metrics_at_k


def test_precision_recall_f1_hand_example():
    result = metrics_at_k(["a", "x", "b"], ["a", "b", "c", "d"], k=3)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == .5
    assert result["f1"] == pytest.approx(4 / 7)
    assert result["hits"] == 2


def test_short_result_list_still_uses_k_denominator():
    result = metrics_at_k(["a"], ["a", "b"], k=10)
    assert result["precision"] == .1 and result["recall"] == .5


def test_cutoff_unknown_retrieval_and_zero_hits():
    result = metrics_at_k(["x", "a"], ["a"], k=1)
    assert result == {"precision": 0, "recall": 0, "f1": 0, "hits": 0}
    assert metrics_at_k([], [], 10)["f1"] == 0


def test_invalid_cutoff_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        metrics_at_k([], [], 0)
