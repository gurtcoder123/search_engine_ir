"""Offline review -> saved models -> query ranking, without a live crawl."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from project.config import TOPICS
from project.collection.crawler import Document
from project.preprocess import build_search_data
from project.search import load_data, search
from project.ranking.topic_model import query_topic_weights


@pytest.fixture
def corpus(tmp_path):
    records = []
    words = ["chip computer", "atom physics", "doctor medicine", "football sport", "election vote"]
    for i, (topic, text) in enumerate(zip(TOPICS, words)):
        records.append(asdict(Document(i, f"Article {i}", f"https://example.org/{i}",
                                       f"doc-{i}", "now", text, text=text + " article",
                                       topic=topic, approved=True,
                                       outgoing_links=[f"Article {(i + 1) % 5}"])))
    records.append(asdict(Document(99, "Rejected", "https://example.org/99", "doc-99",
                                   "now", "spam", text="spam", topic=None, approved=False)))
    (tmp_path / "Document_Collection.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp_path


def test_full_offline_build_and_query_load_does_not_change_review(corpus):
    collection_before = (corpus / "Document_Collection.json").read_bytes()
    info = build_search_data(corpus)
    assert info["internal_links"] == 5
    data = load_data(corpus, mode="tspr")
    assert data["topic_pagerank"].shape == (5, 5)
    assert "spam" not in data["term_to_col"]
    assert (corpus / "Document_Collection.json").read_bytes() == collection_before
    assert not (corpus / "Tokens" / "doc-99.json").exists()
    model = data["topic_model"]
    weights = query_topic_weights("computer", data["term_to_col"], data["topic_log_probs"], model["priors"])
    assert model["topics"][weights.argmax()] == "technology"
    with patch.object(Path, "read_text", side_effect=AssertionError("Query must use memory")):
        for mode in ("tfidf", "pagerank", "tspr"):
            result = search("computer football", data, mode=mode)
            assert {r["doc_id"] for r in result} == {"doc-0", "doc-3"}
            for record in result:
                row = data["doc_to_row"][record["doc_id"]]
                if mode == "pagerank":
                    assert record["score"] == pytest.approx(data["global_pagerank"][row])
                elif mode == "tfidf":
                    assert record["score"] == record["cosine"]
    weights = query_topic_weights("computer football", data["term_to_col"], data["topic_log_probs"], model["priors"])
    for record in search("computer football", data, mode="tspr"):
        row = data["doc_to_row"][record["doc_id"]]
        assert record["score"] == pytest.approx(weights @ data["topic_pagerank"][:, row])


def test_topic_ranking_changes_order_and_does_not_run_pagerank_at_query_time(corpus):
    build_search_data(corpus)
    data = load_data(corpus, mode="tspr")
    # Control the vectors to check mode selection, independently of graph structure.
    data["topic_pagerank"][:] = 0
    data["topic_pagerank"][:, 3] = 1
    data["global_pagerank"][:] = [1, 0, 0, 0, 0]
    with patch("project.ranking.tspr.pagerank", side_effect=AssertionError("No online power iteration")):
        assert search("computer football", data, mode="tspr")[0]["doc_id"] == "doc-3"
        assert search("computer football", data, mode="pagerank")[0]["doc_id"] == "doc-0"


def test_changed_manual_topic_requires_rank_rebuild(corpus):
    build_search_data(corpus)
    path = corpus / "Document_Collection.json"
    records = json.loads(path.read_text())
    records[0]["topic"] = "science"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="labels changed"):
        load_data(corpus, mode="tspr")


def test_no_approved_documents_explains_review_first(corpus):
    path = corpus / "Document_Collection.json"
    records = json.loads(path.read_text())
    for record in records:
        record["approved"] = None
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="Run reviewer.py"):
        build_search_data(corpus)
    assert not (corpus / "Index").exists()


@pytest.mark.parametrize("as_module", [False, True])
def test_preprocess_and_tspr_search_entrypoints(corpus, as_module):
    for module in ("preprocess", "search"):
        command = [sys.executable, "-B"]
        if as_module:
            command += ["-m", f"project.{module}"]
        else:
            command += [str(Path(__file__).resolve().parents[1] / f"{module}.py")]
        command += ["--data-dir", str(corpus)]
        if module == "search":
            command += ["--mode", "tspr"]
        result = subprocess.run(command, input="computer\nq\n", text=True,
                                capture_output=True, check=True)
        if module == "preprocess":
            assert "Internal links: 5" in result.stdout
        else:
            assert "Ranking mode: tspr" in result.stdout
            assert "Query topics:" in result.stdout and "Article 0 | tspr=" in result.stdout
