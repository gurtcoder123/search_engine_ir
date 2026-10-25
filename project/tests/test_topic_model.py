"""Hand-calculated topic counts, smoothing and query Bayes probabilities."""

import numpy as np
import pytest

from project.ranking.topic_model import build_topic_model, query_topic_weights


@pytest.fixture
def model():
    documents = {"a": {"topic": "technology"}, "b": {"topic": "science"},
                 "c": {"topic": "technology"}}
    counts = {"a": {"chip": 3}, "b": {"atom": 3}, "c": {"chip": 1}}
    vocabulary = {"chip": 0, "atom": 1}  # Deliberately not alphabetical.
    log_probs, metadata = build_topic_model(documents, counts, vocabulary,
                                            topics=("technology", "science"))
    return vocabulary, log_probs, metadata


def test_raw_counts_smoothing_and_uniform_priors(model):
    _, log_probs, metadata = model
    np.testing.assert_allclose(np.exp(log_probs), [[5 / 6, 1 / 6], [1 / 5, 4 / 5]])
    np.testing.assert_allclose(np.exp(log_probs).sum(axis=1), 1)
    assert metadata["doc_counts"] == [2, 1]
    assert metadata["word_totals"] == [4, 3]
    assert metadata["priors"] == [0.5, 0.5]  # Not proportional to doc counts.


def test_query_posterior_matches_hand_calculation(model):
    vocabulary, log_probs, metadata = model
    weights = query_topic_weights("CHIPS", vocabulary, log_probs, metadata["priors"])
    expected = np.array([5 / 6, 1 / 5])
    np.testing.assert_allclose(weights, expected / expected.sum())


def test_repeated_query_words_count_and_unknowns_do_not_change_weights(model):
    vocabulary, log_probs, metadata = model
    single = query_topic_weights("chip", vocabulary, log_probs, metadata["priors"])
    repeated = query_topic_weights("chip chip", vocabulary, log_probs, metadata["priors"])
    expected = np.array([(5 / 6) ** 2, (1 / 5) ** 2])
    np.testing.assert_allclose(repeated, expected / expected.sum())
    assert repeated[0] > single[0]
    np.testing.assert_allclose(
        query_topic_weights("chip zzunknown", vocabulary, log_probs, metadata["priors"]), single)


@pytest.mark.parametrize("query", ["", "the and", "zzunknown"])
def test_no_known_query_words_returns_prior(model, query):
    vocabulary, log_probs, metadata = model
    np.testing.assert_allclose(query_topic_weights(query, vocabulary, log_probs, metadata["priors"]),
                               metadata["priors"])


def test_long_query_does_not_underflow(model):
    vocabulary, log_probs, metadata = model
    weights = query_topic_weights("chip " * 1000, vocabulary, log_probs, metadata["priors"])
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1)
    assert weights[0] > 0.999


def test_missing_topic_and_invalid_label_need_manual_review():
    with pytest.raises(ValueError, match="Missing: science"):
        build_topic_model({"a": {"topic": "technology"}}, {"a": {"chip": 1}},
                          {"chip": 0}, topics=("technology", "science"))
    with pytest.raises(ValueError, match="manual topic"):
        build_topic_model({"a": {"topic": None}}, {"a": {"chip": 1}}, {"chip": 0})


def test_empty_vocabulary_is_explained():
    with pytest.raises(ValueError, match="no searchable words"):
        build_topic_model({"a": {"topic": "technology"}}, {"a": {}}, {}, topics=("technology",))
