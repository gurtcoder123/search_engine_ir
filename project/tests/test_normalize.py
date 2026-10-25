"""Small, offline examples for the shared tokenisation rules."""

from dataclasses import asdict
import json

from project.collection.crawler import Document, DocumentCollection
from project.indexing.normalize import load_document, normalize_word, save_document, tokenize


def test_case_unicode_and_numbers():
    assert normalize_word("ＣＯＭＰＵＴＥＲ") == "comput"
    assert tokenize("THE ＣＯＭＰＵＴＥＲ costs 2026 dollars") == ["comput", "cost", "2026", "dollar"]


def test_simple_dotted_initials_without_expansion():
    assert tokenize("U.S.A. USA U.S.A U.S. US") == ["usa", "usa", "usa", "us", "us"]


def test_possessives_and_curly_apostrophes():
    assert tokenize("Computer’s computer's teachers' tools") == ["comput", "comput", "teacher", "tool"]


def test_hyphens_kept_but_other_punctuation_separates():
    assert tokenize("COVID-19, state-of-the-art; email_address / science+tech") == [
        "covid-19", "state-of-the-art", "email", "address", "scienc", "tech",
    ]


def test_repetitions_order_short_words_and_stop_words():
    assert tokenize("the chip and chip x AI 7 don't") == ["chip", "chip", "x", "ai", "7"]
    assert tokenize("") == []
    assert tokenize("... !!!") == []
    assert tokenize("the and of") == []


def test_porter_stemming_and_stopword_removal_before_stemming():
    assert tokenize("computer computers computing") == ["comput", "comput", "comput"]
    assert tokenize("was this does computers") == ["comput"]


def test_numbers_compounds_and_non_english_letters_are_not_stemmed():
    assert tokenize("2026 covid-19 café x") == ["2026", "covid-19", "café", "x"]


def test_save_tokens_is_separate_from_original_document(tmp_path):
    document = Document(1, "A", "url", "doc-1", "now", "Preview", text="Raw text", approved=True)
    original = asdict(document)
    path = save_document(document, ["chip", "chip"], tmp_path)
    assert json.loads(path.read_text()) == {"doc_id": "doc-1", "tokens": ["chip", "chip"]}
    assert asdict(document) == original


def test_load_only_approved_documents_and_keep_original_text(tmp_path):
    collection = DocumentCollection()
    for page_id, approved in enumerate([True, False, None], start=1):
        collection.add_document(Document(page_id, "Title", "url", f"doc-{page_id}", "now", "Preview",
                                         text="The computer and computers", approved=approved))
    original = [asdict(d) for d in collection.documents]
    result = load_document(collection, tmp_path)
    assert result == {"doc-1": ["comput", "comput"]}
    assert json.loads((tmp_path / "doc-1.json").read_text())["tokens"] == ["comput", "comput"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["doc-1.json"]
    assert [asdict(d) for d in collection.documents] == original


def test_no_approved_documents_creates_no_token_files(tmp_path):
    target = tmp_path / "Tokens"
    assert load_document(DocumentCollection(), target) == {}
    assert not target.exists()
