"""Tiny directed graphs checked against hand solutions and a dense solve."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, isspmatrix_csr

from project.ranking.tspr import build_link_matrix, build_rankings, pagerank


def test_all_internal_edges_not_just_bfs_tree_and_binary_duplicate_links():
    documents = {
        "b": {"title": "Article B", "outgoing_links": ["article_A", "Article A", "Article B", "Outside"]},
        "a": {"title": "Article A", "outgoing_links": ["Article B", "Rejected"]},
        "empty": {"title": "Empty", "outgoing_links": []},
    }
    links = build_link_matrix(documents, ["b", "empty", "a"])
    assert isspmatrix_csr(links)
    np.testing.assert_array_equal(links.toarray(), [[0, 0, 1], [0, 0, 0], [1, 0, 0]])


def test_dangling_mass_follows_topic_teleport_hand_solution():
    # A -> B, B is dangling. Every restart goes to A.
    links = csr_matrix([[0., 1.], [0., 0.]])
    rank, info = pagerank(links, [1, 0], tolerance=1e-12)
    np.testing.assert_allclose(rank, [1 / 1.85, .85 / 1.85], atol=1e-11)
    assert info["converged"]
    assert rank.sum() == pytest.approx(1)


def test_sparse_iteration_matches_independent_dense_linear_solve():
    adjacency = np.array([[0., 1., 1.], [1., 0., 0.], [0., 0., 0.]])
    teleport = np.array([.7, .3, 0])
    transition = np.array([[0, .5, .5], [1, 0, 0], teleport])
    expected = np.linalg.solve(np.eye(3) - .85 * transition.T, .15 * teleport)
    actual, info = pagerank(csr_matrix(adjacency), teleport, tolerance=1e-12)
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    assert info["converged"] and (actual >= 0).all()


def test_all_dangling_graph_equals_teleport():
    rank, info = pagerank(csr_matrix((3, 3)), [.2, .8, 0])
    np.testing.assert_allclose(rank, [.2, .8, 0])
    assert info["converged"] and info["iterations"] == 1


def test_topic_vectors_are_distinct_and_can_have_zero_scores():
    documents = {
        "a": {"title": "A", "topic": "technology", "outgoing_links": ["B"]},
        "b": {"title": "B", "topic": "science", "outgoing_links": ["A"]},
        "c": {"title": "C", "topic": "science", "outgoing_links": []},
    }
    ranks, global_rank, metadata = build_rankings(documents, ["b", "c", "a"],
                                                 topics=("technology", "science"))
    assert ranks.shape == (2, 3)
    np.testing.assert_allclose(ranks.sum(axis=1), 1)
    assert ranks[0, 1] == 0  # C is unreachable from technology.
    assert ranks[1, 1] > 0
    assert global_rank.sum() == pytest.approx(1) and (global_rank > 0).all()
    assert metadata["doc_ids"] == ["b", "c", "a"]
    assert metadata["labels"] == ["science", "science", "technology"]
    assert metadata["internal_links"] == 2 and metadata["dangling_documents"] == 1


def test_nonconvergence_is_reported_not_silently_accepted():
    _, info = pagerank(csr_matrix([[0., 1.], [1., 0.]]), [1, 0], max_iterations=1)
    assert not info["converged"] and info["iterations"] == 1


def test_missing_topic_empty_corpus_and_invalid_settings():
    documents = {"a": {"title": "A", "topic": "technology"}}
    with pytest.raises(ValueError, match="science"):
        build_rankings(documents, ["a"], topics=("technology", "science"))
    with pytest.raises(ValueError, match="No approved"):
        build_rankings({}, [])
    with pytest.raises(ValueError, match="damping"):
        pagerank(csr_matrix((1, 1)), [1], damping=1)
