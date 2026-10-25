"""Exact small examples for postings, TF counts and DF. No live data needed."""

from dataclasses import asdict
import json

import pytest

from project.collection.crawler import Document, DocumentCollection
from project.indexing.indexer import build_inverted_index, load_tokens, save_index
from project.indexing.normalize import load_document


def test_raw_counts_and_df_are_different():
    postings, df, counts = build_inverted_index({
        "doc-1": ["comput", "comput", "chip"],
        "doc-2": ["comput", "softwar", "softwar", "softwar"],
    })
    assert postings == {
        "comput": [("doc-1", 2), ("doc-2", 1)],
        "chip": [("doc-1", 1)],
        "softwar": [("doc-2", 3)],
    }
    assert df == {"comput": 2, "chip": 1, "softwar": 1}
    assert counts == {"doc-1": {"comput": 2, "chip": 1},
                      "doc-2": {"comput": 1, "softwar": 3}}


def test_empty_documents_are_kept_in_document_count():
    postings, df, counts = build_inverted_index({"empty": [], "full": ["chip"]})
    assert counts == {"empty": {}, "full": {"chip": 1}}
    assert postings == {"chip": [("full", 1)]}
    assert df == {"chip": 1}
    assert len(counts) == 2  # Future IDF uses N=2, not N=1.
    assert build_inverted_index({}) == ({}, {}, {})


def test_index_save_reload_preserves_counts_and_order(tmp_path):
    postings, df, counts = build_inverted_index({"b": ["chip", "chip"], "a": []})
    save_index(postings, df, counts, tmp_path)
    assert json.loads((tmp_path / "inverted_index.json").read_text()) == {"chip": [["b", 2]]}
    assert json.loads((tmp_path / "df.json").read_text()) == df
    assert json.loads((tmp_path / "term_counts.json").read_text()) == counts
    assert json.loads((tmp_path / "doc_ids.json").read_text()) == ["b", "a"]


def test_load_tokens_uses_current_approval_not_folder_contents(tmp_path):
    (tmp_path / "Tokens").mkdir()
    (tmp_path / "Document_Collection.json").write_text(json.dumps([
        {"doc_id": "accepted", "approved": True},
        {"doc_id": "rejected", "approved": False},
        {"doc_id": "pending", "approved": None},
    ]), encoding="utf-8")
    for doc_id in ["accepted", "rejected", "pending", "old-file"]:
        (tmp_path / "Tokens" / f"{doc_id}.json").write_text(
            json.dumps({"doc_id": doc_id, "tokens": [doc_id]}), encoding="utf-8",
        )
    assert load_tokens(tmp_path) == {"accepted": ["accepted"]}


def test_missing_approved_token_file_is_not_silently_excluded(tmp_path):
    (tmp_path / "Document_Collection.json").write_text(
        json.dumps([{"doc_id": "accepted", "approved": True}]), encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_tokens(tmp_path)


def test_approved_documents_to_stemmed_tokens_to_saved_index(tmp_path):
    collection = DocumentCollection()
    for page_id, text, approved in [
        (1, "Computer computers chip", True),
        (2, "Computing software", True),
        (3, "Rejected computer", False),
    ]:
        collection.add_document(Document(page_id, "Title", "url", f"doc-{page_id}",
                                         "now", "Preview", text=text, approved=approved))
    collection_path = tmp_path / "Document_Collection.json"
    collection_path.write_text(json.dumps([asdict(d) for d in collection.documents]), encoding="utf-8")
    original = collection_path.read_bytes()
    load_document(collection, tmp_path / "Tokens")
    tokens = load_tokens(tmp_path)
    postings, df, counts = build_inverted_index(tokens)
    save_index(postings, df, counts, tmp_path / "Index")
    assert postings["comput"] == [("doc-1", 2), ("doc-2", 1)]
    assert df["comput"] == 2
    assert "reject" not in postings
    assert json.loads((tmp_path / "Index/doc_ids.json").read_text()) == ["doc-1", "doc-2"]
    assert collection_path.read_bytes() == original
