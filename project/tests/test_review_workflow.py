"""Crawl -> save -> human review -> reload, without any graph code."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from project.config import TOPICS
from project.collection.reviewer import review_documents
from project.tests.test_crawler import make_crawler


@pytest.fixture
def saved_corpus(tmp_path):
    crawler, _ = make_crawler(depth=1)
    crawler.crawl()
    crawler.save(tmp_path)
    # Topic folders may already exist, as on the user's machine.
    for topic in TOPICS:
        (tmp_path / "Documents" / topic).mkdir()
    return tmp_path


def load_records(directory):
    return json.loads((directory / "Document_Collection.json").read_text())


def assert_matching_files(directory, records):
    for record in records:
        assert json.loads((directory / record["file_path"]).read_text()) == record


def test_accept_reject_updates_both_copies_and_keeps_rejected_article(saved_corpus, capsys):
    before = load_records(saved_corpus)
    with patch("builtins.input", side_effect=["1", "r", "2"]):
        documents = review_documents(saved_corpus)
    records = load_records(saved_corpus)
    assert records == [asdict(d) for d in documents]
    assert [r["approved"] for r in records] == [True, False, True]
    assert [r["topic"] for r in records] == ["technology", None, "science"]
    assert records[0]["file_path"] == "Documents/technology/doc-1.json"
    assert records[1]["file_path"] == "Documents/doc-2.json"
    assert records[2]["file_path"] == "Documents/science/doc-3.json"
    assert len(list((saved_corpus / "Documents").rglob("*.json"))) == 3
    for old, new in zip(before, records):
        assert old["text"] == new["text"]
        assert old["outgoing_links"] == new["outgoing_links"]
    assert_matching_files(saved_corpus, records)
    assert "Incoming from saved pages (1): B" in capsys.readouterr().out


def test_quit_saves_previous_choices_and_resume_skips_completed(saved_corpus):
    with patch("builtins.input", side_effect=["1", "q"]):
        review_documents(saved_corpus)
    assert [r["approved"] for r in load_records(saved_corpus)] == [True, None, None]
    with patch("builtins.input", side_effect=["r", "2"]) as choose:
        review_documents(saved_corpus)
    assert choose.call_count == 2
    assert_matching_files(saved_corpus, load_records(saved_corpus))


def test_skip_view_and_invalid_choice_do_not_review_a_document(saved_corpus):
    before = (saved_corpus / "Document_Collection.json").read_bytes()
    with patch("builtins.input", side_effect=["invalid", "v", "l", "s", "q"]):
        review_documents(saved_corpus)
    assert (saved_corpus / "Document_Collection.json").read_bytes() == before


def test_review_again_can_change_topic_and_reject_without_losing_text(saved_corpus):
    with patch("builtins.input", side_effect=["1", "r", "s"]):
        review_documents(saved_corpus)
    original_text = load_records(saved_corpus)[0]["text"]
    with patch("builtins.input", side_effect=["5", "2", "s"]):
        review_documents(saved_corpus, review_again=True)
    records = load_records(saved_corpus)
    assert records[0]["file_path"] == "Documents/politics/doc-1.json"
    assert records[1]["file_path"] == "Documents/science/doc-2.json"
    assert records[0]["text"] == original_text
    with patch("builtins.input", side_effect=["r", "s", "s"]):
        review_documents(saved_corpus, review_again=True)
    records = load_records(saved_corpus)
    assert records[0]["file_path"] == "Documents/doc-1.json"
    assert records[0]["approved"] is False and records[0]["topic"] is None
    assert len(records) == 3
    assert_matching_files(saved_corpus, records)


def test_title_matching_incoming_uses_normalized_titles(saved_corpus, capsys):
    records = load_records(saved_corpus)
    records[1]["outgoing_links"] = ["a"]
    (saved_corpus / "Document_Collection.json").write_text(json.dumps(records), encoding="utf-8")
    with patch("builtins.input", return_value="q"):
        review_documents(saved_corpus)
    assert "Incoming from saved pages (1): B" in capsys.readouterr().out


def test_reviewer_refuses_to_overwrite_an_unrelated_file(saved_corpus):
    path = saved_corpus / "Documents/technology/doc-1.json"
    path.write_text("existing article", encoding="utf-8")
    before = (saved_corpus / "Document_Collection.json").read_bytes()
    with patch("builtins.input", return_value="1"), pytest.raises(FileExistsError):
        review_documents(saved_corpus)
    assert path.read_text() == "existing article"
    assert before == (saved_corpus / "Document_Collection.json").read_bytes()


def test_reviewer_refuses_paths_outside_the_collection(saved_corpus):
    records = load_records(saved_corpus)
    records[0]["file_path"] = "../outside.json"
    (saved_corpus / "Document_Collection.json").write_text(json.dumps(records), encoding="utf-8")
    with patch("builtins.input", return_value="1"), pytest.raises(ValueError, match="inside"):
        review_documents(saved_corpus)


@pytest.mark.parametrize("module", [False, True])
def test_reviewer_cli_loads_saved_corpus_and_reports_approval_count(saved_corpus, module):
    if module:
        command = [sys.executable, "-B", "-m", "project.collection.reviewer"]
    else:
        command = [sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "collection" / "reviewer.py")]
    result = subprocess.run(command + ["--data-dir", str(saved_corpus)],
                            input="1\nr\n2\n", text=True, capture_output=True, check=True)
    assert "Approved: 2; rejected: 1; unreviewed: 0" in result.stdout
    assert "at least 100" in result.stdout
    assert_matching_files(saved_corpus, load_records(saved_corpus))


def test_no_review_decision_deletes_or_shortlists_away_raw_records(saved_corpus):
    with patch("builtins.input", side_effect=["1", "r", "2"]):
        review_documents(saved_corpus)
    records = load_records(saved_corpus)
    accepted = [r for r in records if r["approved"] is True]
    assert len(records) == 3 and len(accepted) == 2
    assert {p.name for p in saved_corpus.iterdir()} == {"Documents", "Document_Collection.json"}
