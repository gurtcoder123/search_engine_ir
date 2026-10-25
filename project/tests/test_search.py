"""Saved offline artifacts to real query scoring and the interactive CLI."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from project.collection.crawler import Document, DocumentCollection
from project.indexing.indexer import build_inverted_index, save_index
from project.indexing.normalize import tokenize
from project.search import load_data, make_query_vector, search
from project.indexing.tfidf import build_tfidf_matrix, save_tfidf_matrix


@pytest.fixture
def corpus(tmp_path):
    collection = DocumentCollection()
    texts = ["Computer chip chip", "Software", "Chip science", "The and", "Rejected computer"]
    for number, text in enumerate(texts, start=1):
        collection.add_document(Document(number, f"Article {number}", f"https://example.org/{number}",
                                         f"doc-{number}", "now", text, text=text,
                                         topic="technology", approved=number < 5))
    records = [asdict(d) for d in collection.documents]
    (tmp_path / "Document_Collection.json").write_text(json.dumps(records), encoding="utf-8")
    tokens = {d.doc_id: tokenize(d.text) for d in collection.documents if d.approved}
    postings, df, counts = build_inverted_index(tokens)
    index_dir = tmp_path / "Index"
    save_index(postings, df, counts, index_dir)
    doc_ids = list(tokens)
    matrix, vocabulary, df_vector, idf = build_tfidf_matrix(counts, df, doc_ids)
    save_tfidf_matrix(matrix, vocabulary, df_vector, idf, doc_ids, index_dir)
    return tmp_path


def test_single_term_query_returns_matching_metadata_and_uses_stemming(corpus):
    data = load_data(corpus)
    results = search("COMPUTERS", data)
    assert [r["doc_id"] for r in results] == ["doc-1"]
    assert results[0]["title"] == "Article 1"
    assert results[0]["url"] == "https://example.org/1"
    assert results[0]["preview"] == "Computer chip chip"
    assert results[0]["score"] > 0


def test_candidate_union_not_intersection(corpus):
    results = search("computer software", load_data(corpus))
    assert {r["doc_id"] for r in results} == {"doc-1", "doc-2"}
    assert results[0]["doc_id"] == "doc-2"
    assert results[0]["score"] == pytest.approx(1 / np.sqrt(2))


def test_query_log_tf_idf_and_l2_match_hand_calculation():
    vocabulary = {"chip": 0, "comput": 1}
    idf = np.array([0.3, 0.6])
    vector, terms = make_query_vector("chip chip computer", vocabulary, idf)
    expected = np.array([(1 + np.log10(2)) * 0.3, 0.6])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(vector.toarray()[0], expected)
    assert terms == ["chip", "comput"]


@pytest.mark.parametrize("query", ["", "the and of", "zzzzunknownterm"])
def test_empty_stopword_and_unknown_queries_return_no_results(corpus, query):
    assert search(query, load_data(corpus)) == []


def test_unknown_terms_do_not_dilute_known_query_weights(corpus):
    data = load_data(corpus)
    assert search("computer zzzzunknownterm", data) == search("computer", data)


def test_zero_idf_query_has_no_nonzero_weights():
    vector, _ = make_query_vector("common", {"common": 0}, np.array([0.0]))
    assert vector.shape == (1, 1) and vector.nnz == 0


def test_top_k_and_repeated_queries_use_loaded_data_only(corpus):
    data = load_data(corpus)
    with patch.object(Path, "read_text", side_effect=AssertionError("Query must not reload files")):
        assert len(search("chip", data, top_k=1)) == 1
        assert len(search("chip", data)) == 2
        assert search("chip", data, top_k=0) == []


def test_loader_refuses_changed_approval_selection(corpus):
    path = corpus / "Document_Collection.json"
    records = json.loads(path.read_text())
    records[0]["approved"] = False
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="selection changed"):
        load_data(corpus)


def test_loader_explains_when_no_documents_are_approved(tmp_path):
    (tmp_path / "Document_Collection.json").write_text(json.dumps([
        {"doc_id": "a", "approved": None},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="Run reviewer.py"):
        load_data(tmp_path)


@pytest.mark.parametrize("as_module", [False, True])
def test_interactive_cli_handles_repeated_queries_then_quits(corpus, as_module):
    command = [sys.executable, "-B"]
    if as_module:
        command += ["-m", "project.search"]
    else:
        command += [str(Path(__file__).resolve().parents[1] / "search.py")]
    result = subprocess.run(command + ["--data-dir", str(corpus)],
                            input="software\nzzzzunknownterm\nq\n", text=True,
                            capture_output=True, check=True)
    assert result.stdout.count("Loaded 4 documents") == 1
    assert "Article 2 | cosine=1.0000" in result.stdout
    assert "https://example.org/2" in result.stdout
    assert "No matching documents." in result.stdout
    assert "Article 5" not in result.stdout
