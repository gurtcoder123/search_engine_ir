"""The shared contracts every later stage depends on: topics, IDs, imports."""

import dataclasses

from project import config, crawler, graph, reviewer
from project.crawler import Document, TOPICS


def test_topic_names_and_order_are_fixed():
    assert config.TOPICS == (
        "technology", "science", "health_and_medicine", "sports", "politics",
    )


def test_every_module_shares_one_topic_list():
    # One authoritative definition, so a stage cannot drift onto its own copy.
    assert crawler.TOPICS is config.TOPICS
    assert reviewer.TOPICS is config.TOPICS
    assert graph.TOPICS is config.TOPICS


def test_partner_import_from_crawler_still_works():
    # PARTNER_HANDOFF.md documents `from project.crawler import Document, TOPICS`.
    assert TOPICS is config.TOPICS
    assert dataclasses.is_dataclass(Document)


def test_document_carries_the_agreed_fields():
    assert {field.name for field in dataclasses.fields(Document)} == {
        "page_id", "doc_id", "title", "url", "text", "preview",
        "topic", "discovered_topic", "revision_id", "collected_at",
        "outgoing_links",
    }


def test_new_documents_are_unlabelled_and_keep_provenance_separate():
    document = Document(
        page_id=101, title="A", url="https://en.wikipedia.org/wiki/A",
        doc_id=config.doc_id_for(101), collected_at="now", preview="Preview",
        discovered_topic="technology",
    )
    assert document.topic is None
    assert document.discovered_topic == "technology"
    assert document.outgoing_links == []


def test_doc_ids_are_opaque_and_unique_per_page():
    assert config.doc_id_for(12345) == "doc-12345"
    assert config.doc_id_for(1) != config.doc_id_for(2)
    assert isinstance(config.doc_id_for(1), str)


def test_collection_deduplicates_by_page_id_without_clearing_itself():
    collection = crawler.DocumentCollection()
    first = Document(101, "A", "url", config.doc_id_for(101), "now", "Preview")
    duplicate = Document(101, "A (again)", "url", config.doc_id_for(101), "now", "Preview")
    assert collection.add_document(first) is True
    assert collection.add_document(duplicate) is False
    assert collection.get_documents() == [first]


def test_data_layout_is_under_one_ignored_directory():
    for directory in (config.CANDIDATES_DIR, config.REVIEWED_DIR, config.INDEX_DIR):
        assert directory.parent == config.DATA_DIR
    assert config.DATA_DIR.parent == config.PROJECT_DIR
