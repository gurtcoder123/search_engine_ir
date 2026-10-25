"""Offline tests for the simple crawler, including API continuation."""

from dataclasses import asdict
import json
from unittest.mock import patch

import pytest
import requests

from project.collection.crawler import Document, DocumentCollection, WebCrawler


class FakeResponse:
    def __init__(self, data, status_code=200, headers=None):
        self.data, self.status_code, self.headers = data, status_code, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

    def json(self):
        return self.data


class FakeWikipedia:
    def __init__(self):
        self.calls = []
        self.pages = {"A": 1, "B": 2, "C": 3, "D": 4}
        self.links = {"A": ["B", "C", "A"], "B": ["A", "C", "D"], "C": [], "D": []}

    def get(self, url, *, params, timeout):
        self.calls.append(dict(params))
        title = params["titles"]
        title = "A" if title == "Alias A" else title
        if title not in self.pages:
            return FakeResponse({"query": {"pages": [{"title": title, "missing": True, "ns": 0}]}})
        return FakeResponse({"query": {"pages": [{
            "pageid": self.pages[title], "ns": 0, "title": title,
            "fullurl": f"https://en.wikipedia.org/wiki/{title}", "lastrevid": 42,
            "extract": "First paragraph has words.\n\nSecond paragraph has words.\n\nThird paragraph has words.\n\nFourth paragraph has words.",
            "links": [{"ns": 0, "title": link} for link in self.links[title]],
        }]}})


def make_crawler(**kwargs):
    kwargs.setdefault("depth", 2)
    kwargs.setdefault("min_words", 0)
    kwargs.setdefault("request_delay", 0)
    kwargs.setdefault("max_links_per_page", None)
    collection = kwargs.pop("document_collection", DocumentCollection())
    topics = kwargs.pop("topics", ["technology"])
    seeds = kwargs.pop("seed_topics", [["A"]])
    crawler = WebCrawler(collection, topics, seeds, **kwargs)
    wiki = FakeWikipedia()
    crawler.session.get = wiki.get
    return crawler, wiki


def test_bfs_retains_all_links_including_backlinks_and_cycles():
    crawler, wiki = make_crawler()
    documents = crawler.crawl()
    assert documents is crawler.document_collection.documents
    assert [d.title for d in documents] == ["A", "B", "C", "D"]
    assert documents[1].outgoing_links == ["A", "C", "D"]
    assert [c["titles"] for c in wiki.calls] == ["A", "B", "C", "D"]
    assert all(d.approved is None and d.topic is None for d in documents)
    assert not hasattr(crawler, "adjacency_lists")


def test_global_visits_skip_repeated_seed_without_expanding_again():
    crawler, wiki = make_crawler(topics=["technology", "science"],
                                 seed_topics=[["A"], ["A", "D"]], max_pages_per_topic=2)
    assert [d.title for d in crawler.crawl()] == ["A", "B", "D"]
    assert [c["titles"] for c in wiki.calls] == ["A", "B", "D"]
    assert crawler.counts == {"technology": 2, "science": 1}


def test_unvisited_links_are_chosen_after_excluding_already_collected_pages():
    collection = DocumentCollection()
    for i, title in enumerate(["X", "Y", "Z"], start=100):
        collection.add_document(Document(i, title, "url", f"doc-{i}", "now", "preview"))
    crawler, wiki = make_crawler(document_collection=collection, seed_topics=[["B"]],
                                 max_links_per_page=1, depth=1)
    wiki.links["B"] = ["X", "Y", "Z", "C"]
    crawler.crawl()
    assert [c["titles"] for c in wiki.calls] == ["B", "C"]
    assert collection.documents[3].outgoing_links == ["X", "Y", "Z", "C"]


def test_redirect_page_id_deduplication():
    crawler, _ = make_crawler(seed_topics=[["A", "Alias A"]], depth=0)
    assert len(crawler.crawl()) == 1


def test_depth_zero_and_zero_delay_are_not_replaced_with_defaults():
    crawler, wiki = make_crawler(depth=0)
    with patch("project.collection.crawler.time.sleep") as sleep:
        assert len(crawler.crawl()) == 1
    assert crawler.depth == 0 and crawler.request_delay == 0
    assert len(wiki.calls) == 1
    sleep.assert_not_called()


def test_caps_limit_new_pages():
    crawler, _ = make_crawler(max_pages_per_topic=2)
    assert len(crawler.crawl()) == 2


def test_discovery_limit_does_not_truncate_saved_outgoing_links():
    crawler, _ = make_crawler(depth=1, max_links_per_page=1)
    documents = crawler.crawl()
    assert len(documents) == 2
    assert documents[0].outgoing_links == ["B", "C", "A"]


def test_sampling_is_reproducible():
    a, _ = make_crawler(max_links_per_page=1)
    b, _ = make_crawler(max_links_per_page=1)
    assert [d.page_id for d in a.crawl()] == [d.page_id for d in b.crawl()]


@pytest.mark.parametrize("paragraphs,chars,expected", [
    (2, None, "First paragraph has words.\n\nSecond paragraph has words."),
    (None, 5, "First"),
    (1, 5, "First"),
    (None, None, "First paragraph has words.\n\nSecond paragraph has words.\n\nThird paragraph has words.\n\nFourth paragraph has words."),
])
def test_text_limits_control_both_saved_text_and_preview(paragraphs, chars, expected):
    crawler, _ = make_crawler(depth=0, max_paragraphs=paragraphs, max_chars=chars)
    document = crawler.crawl()[0]
    assert document.text == document.preview == expected


def test_default_text_is_three_paragraphs():
    crawler, _ = make_crawler(depth=0)
    document = crawler.crawl()[0]
    assert "Third paragraph" in document.text and "Fourth paragraph" not in document.text
    assert document.revision_id == 42


def test_spacing_includes_missing_pages():
    crawler, _ = make_crawler(seed_topics=[["Missing", "A"]], depth=0, request_delay=0.25)
    with patch("project.collection.crawler.time.sleep") as sleep:
        assert len(crawler.crawl()) == 1
    assert [c.args for c in sleep.call_args_list] == [(0.25,), (0.25,)]


def test_link_pagination_is_merged_and_spaced():
    crawler, _ = make_crawler(request_delay=0.25)
    replies = [
        FakeResponse({"query": {"pages": [{"pageid": 1, "ns": 0, "title": "A", "extract": "Text",
                                          "links": [{"ns": 0, "title": "B"}]}]},
                      "continue": {"plcontinue": "next", "continue": "||"}}),
        FakeResponse({"query": {"pages": [{"pageid": 1, "links": [{"ns": 0, "title": "C"}]}]}}),
    ]
    with patch.object(crawler.session, "get", side_effect=replies) as get, patch("project.collection.crawler.time.sleep") as sleep:
        page = crawler.request("A")
    assert [link["title"] for link in page["links"]] == ["B", "C"]
    assert get.call_args_list[1].kwargs["params"]["plcontinue"] == "next"
    assert get.call_args_list[1].kwargs["params"]["pageids"] == 1
    assert sleep.call_count == 2


@pytest.mark.parametrize("status", [403, 429, 500, 503])
def test_http_failure_skips_article_without_retry_and_continues(status):
    crawler, wiki = make_crawler(seed_topics=[["A", "B"]], depth=0, request_delay=0.25)
    valid = wiki.get("", params={"titles": "B"}, timeout=1)
    with patch.object(crawler.session, "get", side_effect=[
        FakeResponse({}, status, {"Retry-After": "120"}), valid,
    ]) as get, patch("project.collection.crawler.time.sleep") as sleep:
        assert [d.title for d in crawler.crawl()] == ["B"]
    assert [call.kwargs["params"]["titles"] for call in get.call_args_list] == ["A", "B"]
    assert [call.args for call in sleep.call_args_list] == [(0.25,), (0.25,)]


@pytest.mark.parametrize("code", ["maxlag", "ratelimited", "badvalue"])
def test_api_errors_are_not_retried(code):
    crawler, _ = make_crawler()
    with patch.object(crawler.session, "get", return_value=FakeResponse(
        {"error": {"code": code, "info": "failed"}},
    )) as get:
        assert crawler.request("A") is None
    assert get.call_count == 1


def test_invalid_json_is_skipped_without_retry():
    crawler, _ = make_crawler()
    reply = FakeResponse({})
    with patch.object(reply, "json", side_effect=ValueError("bad JSON")), \
            patch.object(crawler.session, "get", return_value=reply) as get:
        assert crawler.request("A") is None
    assert get.call_count == 1


def test_failed_link_batch_skips_article_instead_of_saving_partial_links():
    crawler, _ = make_crawler()
    first = FakeResponse({"query": {"pages": [{
        "pageid": 1, "ns": 0, "title": "A", "extract": "Text",
        "links": [{"ns": 0, "title": "B"}],
    }]}, "continue": {"plcontinue": "next", "continue": "||"}})
    with patch.object(crawler.session, "get", side_effect=[first, FakeResponse({}, 503)]) as get:
        assert crawler.request("A") is None
    assert get.call_count == 2


def test_ordinary_network_failure_skips_page():
    crawler, wiki = make_crawler(seed_topics=[["A", "B"]], depth=0)
    original = wiki.get
    def get(url, *, params, timeout):
        if params["titles"] == "A":
            raise requests.ConnectionError("offline")
        return original(url, params=params, timeout=timeout)
    crawler.session.get = get
    assert [d.title for d in crawler.crawl()] == ["B"]


def test_disambiguation_and_non_article_responses_are_skipped():
    crawler, _ = make_crawler()
    for page in [
        {"pageid": 1, "ns": 0, "title": "A", "pageprops": {"disambiguation": ""}},
        {"pageid": 2, "ns": 14, "title": "Category:A"},
    ]:
        with patch.object(crawler.session, "get", return_value=FakeResponse({"query": {"pages": [page]}})):
            assert crawler.request("A") is None


def test_save_uses_collection_array_and_article_files_and_reset_leaves_disk(tmp_path):
    crawler, _ = make_crawler(depth=0)
    document = crawler.crawl()[0]
    crawler.save(tmp_path)
    records = json.loads((tmp_path / "Document_Collection.json").read_text())
    article = tmp_path / "Documents/doc-1.json"
    assert records == [asdict(document)]
    assert json.loads(article.read_text()) == records[0]
    with pytest.raises(FileExistsError):
        crawler.save(tmp_path)
    crawler.reset()
    assert crawler.document_collection.documents == []
    assert article.exists() and (tmp_path / "Document_Collection.json").exists()


def test_constructor_keeps_seed_lists_and_assigns_settings_directly():
    seeds = [["A", "B"], ["C"]]
    crawler, _ = make_crawler(topics=["technology", "science"], seed_topics=seeds,
                              depth=0, request_delay=0, max_paragraphs=2, max_chars=500)
    assert crawler.seed_topics is seeds
    assert crawler.topics == ["technology", "science"]
    assert crawler.depth == 0 and crawler.request_delay == 0
    assert crawler.max_paragraphs == 2 and crawler.max_chars == 500


def test_progress_reports_fetch_collection_and_short_page_skip(caplog):
    crawler, _ = make_crawler(seed_topics=[["A", "B"]], depth=0, min_words=15)
    # A is too short; B has sufficient text. No requests reach the network.
    original_request = crawler.request

    def request(title):
        page = original_request(title)
        if title == "B":
            page["extract"] = "word " * 20
        return page

    crawler.request = request
    with caplog.at_level("INFO", logger="project.collection.crawler"):
        crawler.crawl()
    assert "Starting seed group: technology" in caplog.text
    assert "Fetching: A" in caplog.text
    assert "Fetching: B" in caplog.text
    assert "Skipping A: insufficient text" in caplog.text
    assert "Collected: B - technology 1/40" in caplog.text
