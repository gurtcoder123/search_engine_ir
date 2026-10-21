"""Review/finalization and CLI integration without network requests."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import load_npz

from project.crawler import Document
from project.graph import build_incidence_matrix
from project.reviewer import finalize_documents, review_documents


def example_documents():
    return [Document(number, title, f"https://en.wikipedia.org/wiki/{title}", title, "now", f"Preview {title}", text=f"Full text {title}")
            for number, title in enumerate(["A", "B", "C"], start=1)]


def test_rejection_removes_source_and_incoming_edges_without_changing_raw_documents():
    documents = example_documents()
    lists = {"A": ["B", "C"], "B": ["C"], "C": ["A", "B"]}
    decisions = {
        "A": {"status": "accepted", "topic": "technology"},
        "B": {"status": "rejected", "topic": None},
        "C": {"status": "accepted", "topic": "science"},
    }
    final, filtered = finalize_documents(documents, lists, decisions)
    assert [document.doc_id for document in final] == ["A", "C"]
    assert [document.topic for document in final] == ["technology", "science"]
    assert filtered == {"A": ["C"], "C": ["A"]}
    assert all(document.topic is None for document in documents)
    assert lists["A"] == ["B", "C"]
    np.testing.assert_array_equal(build_incidence_matrix(final, filtered).toarray(), [[0, 1], [1, 0]])


def test_review_saves_each_decision_and_resumes_after_quit(tmp_path):
    documents = example_documents()
    decisions_file = tmp_path / "review_decisions.json"
    with patch("builtins.input", side_effect=["1", "q"]):
        assert review_documents(documents, decisions_file) is None
    assert json.loads(decisions_file.read_text())["A"]["topic"] == "technology"
    with patch("builtins.input", side_effect=["r", "2"]) as choose:
        decisions = review_documents(documents, decisions_file)
    assert choose.call_count == 2
    assert decisions["B"]["status"] == "rejected"
    assert decisions["C"]["topic"] == "science"


def test_skip_remains_pending_and_view_does_not_make_a_decision(tmp_path, capsys):
    documents = example_documents()[:1]
    with patch("builtins.input", side_effect=["invalid", "v", "s"]):
        decisions = review_documents(documents, tmp_path / "decisions.json")
    assert decisions == {}
    assert "Full text A" in capsys.readouterr().out
    with pytest.raises(ValueError, match="needs review"):
        finalize_documents(documents, {"A": []}, decisions)


def test_review_again_can_reverse_a_rejection(tmp_path):
    document = example_documents()[:1]
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps({"A": {"status": "rejected", "topic": None}}))
    with patch("builtins.input", return_value="5"):
        decisions = review_documents(document, path, review_again=True)
    final, lists = finalize_documents(document, {"A": []}, decisions)
    assert final[0].topic == "politics"
    assert lists == {"A": []}


def test_invalid_topic_and_empty_corpus_are_not_finalized():
    documents = example_documents()[:1]
    with pytest.raises(ValueError, match="Invalid topic"):
        finalize_documents(documents, {}, {"A": {"status": "accepted", "topic": "unknown"}})
    with pytest.raises(ValueError, match="empty"):
        finalize_documents(documents, {}, {"A": {"status": "rejected", "topic": None}})


def test_matrix_preserves_final_document_order_zero_rows_and_binary_edges():
    documents = list(reversed(example_documents()))
    lists = {"A": ["B", "B", "A", "Outside"], "B": [], "C": []}
    matrix = build_incidence_matrix(documents, lists)
    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(matrix.toarray(), [[0, 0, 0], [0, 0, 0], [0, 1, 0]])


def test_reviewer_and_graph_cli_end_to_end(tmp_path):
    candidate_dir = tmp_path / "candidates"
    reviewed_dir = tmp_path / "reviewed"
    candidate_dir.mkdir()
    documents_path = candidate_dir / "documents.jsonl"
    documents_path.write_text("".join(json.dumps(asdict(doc)) + "\n" for doc in example_documents()), encoding="utf-8")
    lists_path = candidate_dir / "adjacency_lists.json"
    lists_path.write_text(json.dumps({"A": ["B", "C"], "B": ["C"], "C": ["A"]}))
    original = documents_path.read_bytes(), lists_path.read_bytes()
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(project / "reviewer.py"), "--input-dir", str(candidate_dir),
        "--output-dir", str(reviewed_dir),
    ], input="1\nr\n2\n", text=True, capture_output=True, check=True)
    assert "Saved 2 accepted documents" in result.stdout
    assert not (reviewed_dir / "incidence_matrix.npz").exists()
    assert (documents_path.read_bytes(), lists_path.read_bytes()) == original
    subprocess.run([
        sys.executable, str(project / "graph.py"), "--input-dir", str(reviewed_dir),
    ], text=True, capture_output=True, check=True)
    matrix = load_npz(reviewed_dir / "incidence_matrix.npz")
    manifest = json.loads((reviewed_dir / "graph_manifest.json").read_text())
    assert manifest["doc_to_row"] == {"A": 0, "C": 1}
    np.testing.assert_array_equal(matrix.toarray(), [[0, 1], [1, 0]])
