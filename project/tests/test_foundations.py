"""The small shared contracts, without graph-building dependencies."""

import dataclasses
import inspect

from project.config import TOPICS
from project.collection.crawler import Document, DocumentCollection, WebCrawler


def test_topic_order():
    assert TOPICS == ("technology", "science", "health_and_medicine", "sports", "politics")


def test_document_review_state_has_no_discovery_label():
    document = Document(1, "A", "url", "doc-1", "now", "preview")
    assert document.topic is None and document.approved is None
    assert "discovered_topic" not in {f.name for f in dataclasses.fields(Document)}


def test_shared_collection_deduplicates_page_ids():
    collection = DocumentCollection()
    a = Document(1, "A", "url", "doc-1", "now", "preview")
    assert collection.add_document(a)
    assert not collection.add_document(a)
    assert collection.get_documents() == [a]


def test_crawler_keeps_the_requested_method_structure():
    assert {name for name, value in vars(WebCrawler).items()
            if inspect.isfunction(value)} == {"__init__", "request", "crawl", "reset", "save"}
