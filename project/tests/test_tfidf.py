"""Hand-calculated weights, sparse zero rows and save/load alignment."""

import json

import numpy as np
from scipy.sparse import isspmatrix_csr, load_npz

from project.indexing.indexer import build_inverted_index, load_tokens, save_index
from project.indexing.tfidf import build_tfidf_matrix, save_tfidf_matrix


def test_exact_log_tf_idf_and_l2_normalization():
    # Deliberately use a different dictionary order from the required row order.
    counts = {
        "doc-a": {"common": 1, "apple": 1},
        "doc-b": {"common": 2, "apple": 10, "banana": 1},
        "doc-c": {"common": 1},
    }
    matrix, vocabulary, df, idf = build_tfidf_matrix(
        counts, {"common": 3, "banana": 1, "apple": 2}, ["doc-b", "doc-a", "doc-c"],
    )
    apple_idf, banana_idf = np.log10(3 / 2), np.log10(3)
    norm = np.sqrt((2 * apple_idf) ** 2 + banana_idf ** 2)
    expected = [[2 * apple_idf / norm, banana_idf / norm, 0], [1, 0, 0], [0, 0, 0]]
    assert isspmatrix_csr(matrix)
    assert vocabulary == {"apple": 0, "banana": 1, "common": 2}
    np.testing.assert_array_equal(df, [2, 1, 3])
    np.testing.assert_allclose(idf, [apple_idf, banana_idf, 0])
    np.testing.assert_allclose(matrix.toarray(), expected)
    assert matrix.nnz == 3  # Common terms do not create stored zero entries.
    np.testing.assert_allclose(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel(), [1, 1, 0])


def test_empty_approved_documents_still_count_towards_n():
    _, df, counts = build_inverted_index({"empty": [], "full": ["chip", "chip"]})
    matrix, vocabulary, _, idf = build_tfidf_matrix(counts, df, ["empty", "full"])
    assert vocabulary == {"chip": 0}
    np.testing.assert_allclose(idf, [np.log10(2)])
    np.testing.assert_array_equal(matrix.toarray(), [[0], [1]])


def test_all_terms_in_all_documents_give_zero_vectors_not_nan():
    _, df, counts = build_inverted_index({"a": ["chip"], "b": ["chip", "chip"]})
    matrix, _, _, idf = build_tfidf_matrix(counts, df, ["a", "b"])
    assert matrix.shape == (2, 1) and matrix.nnz == 0
    np.testing.assert_array_equal(idf, [0])
    assert np.isfinite(matrix.toarray()).all()


def test_empty_vocabulary_and_empty_corpus_have_valid_shapes():
    matrix, vocabulary, df, idf = build_tfidf_matrix({"a": {}, "b": {}}, {}, ["a", "b"])
    assert matrix.shape == (2, 0) and matrix.nnz == 0
    assert vocabulary == {} and df.size == idf.size == 0
    matrix, _, _, _ = build_tfidf_matrix({}, {}, [])
    assert matrix.shape == (0, 0)


def test_save_reload_preserves_sparse_values_and_mappings(tmp_path):
    doc_ids = ["b", "a", "empty"]
    _, df, counts = build_inverted_index({"b": ["chip", "chip"], "a": ["softwar"], "empty": []})
    matrix, vocabulary, df_vector, idf = build_tfidf_matrix(counts, df, doc_ids)
    save_tfidf_matrix(matrix, vocabulary, df_vector, idf, doc_ids, tmp_path)
    loaded = load_npz(tmp_path / "tfidf_matrix.npz")
    assert isspmatrix_csr(loaded)
    assert loaded.shape == matrix.shape and (loaded != matrix).nnz == 0
    assert json.loads((tmp_path / "term_to_col.json").read_text()) == vocabulary
    assert json.loads((tmp_path / "doc_ids.json").read_text()) == doc_ids
    assert json.loads((tmp_path / "doc_to_row.json").read_text()) == {"b": 0, "a": 1, "empty": 2}
    np.testing.assert_array_equal(np.load(tmp_path / "df.npy", allow_pickle=False), df_vector)
    np.testing.assert_array_equal(np.load(tmp_path / "idf.npy", allow_pickle=False), idf)


def test_saved_approved_tokens_to_index_to_tfidf(tmp_path):
    (tmp_path / "Tokens").mkdir()
    (tmp_path / "Document_Collection.json").write_text(json.dumps([
        {"doc_id": "b", "approved": True},
        {"doc_id": "rejected", "approved": False},
        {"doc_id": "a", "approved": True},
    ]), encoding="utf-8")
    for doc_id, tokens in {"b": ["chip", "chip"], "a": ["softwar"], "rejected": ["spam"]}.items():
        (tmp_path / "Tokens" / f"{doc_id}.json").write_text(
            json.dumps({"doc_id": doc_id, "tokens": tokens}), encoding="utf-8",
        )
    postings, df, counts = build_inverted_index(load_tokens(tmp_path))
    index_dir = tmp_path / "Index"
    save_index(postings, df, counts, index_dir)
    doc_ids = json.loads((index_dir / "doc_ids.json").read_text())
    counts = json.loads((index_dir / "term_counts.json").read_text())
    df = json.loads((index_dir / "df.json").read_text())
    matrix, vocabulary, df_vector, idf = build_tfidf_matrix(counts, df, doc_ids)
    save_tfidf_matrix(matrix, vocabulary, df_vector, idf, doc_ids, index_dir)
    assert doc_ids == ["b", "a"]
    assert "spam" not in vocabulary
    np.testing.assert_array_equal(load_npz(index_dir / "tfidf_matrix.npz").toarray(), [[1, 0], [0, 1]])
