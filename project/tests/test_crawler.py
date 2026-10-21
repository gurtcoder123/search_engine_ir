"""Offline API fixtures exercise crawling, redirect resolution and graph edges."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import requests
from scipy.sparse import csr_matrix

from project.crawler import Document, WebCrawler, WikipediaAPIError
from project.graph import build_incidence_matrix


class FakeResponse:
    def __init__(self, data, *, status_code=200, headers=None):
        self.data = data
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self.data


class FakeWikipedia:
    def __init__(self):
        self.headers = {}
        self.calls = []
        self.aliases = {"Alias A": "A", "Bee": "B"}
        self.pages = {"A": 101, "B": 202, "C": 303, "Outside": 404, "Ambiguous": 505, "Short": 606}
        self.text = {
            101: "Article A has substantive text about our graph.",
            202: "Article B contains enough text for the test.",
            303: "Article C is a dangling page without outgoing links.",
            404: "Outside is a valid article not always selected.",
            505: "A disambiguation page should be skipped.",
            606: "Tiny.",
        }
        self.links = {101: ["Bee", "Bee", "C", "Outside", "A"], 202: ["A", "C"]}

    def get(self, url, *, params, timeout):
        self.calls.append(dict(params))
        assert timeout > 0
        assert params["action"] == "query", "Only structured query responses should be used"
        pages, redirects = {}, []
        for title in params["titles"].split("|"):
            canonical = self.aliases.get(title, title)
            if canonical != title:
                redirects.append({"from": title, "to": canonical})
            if canonical not in self.pages:
                pages[canonical] = {"title": canonical, "ns": 0, "missing": True}
                continue
            page_id = self.pages[canonical]
            page = {
                "pageid": page_id, "title": canonical, "ns": 0,
                "fullurl": "https://en.wikipedia.org/wiki/" + canonical,
                "revisions": [{"revid": page_id * 10}],
            }
            if "extracts" in params.get("prop", ""):
                page["extract"] = self.text[page_id]
                page["links"] = [{"ns": 0, "title": title} for title in self.links.get(page_id, [])]
            if canonical == "Ambiguous":
                page["pageprops"] = {"disambiguation": ""}
            pages[canonical] = page
        return FakeResponse({"query": {"pages": list(pages.values()), "redirects": redirects}})


def make_crawler(seeds, **kwargs):
    return WebCrawler(
        seed_urls=seeds, session=FakeWikipedia(), request_delay=0,
        min_words=kwargs.pop("min_words", 0), **kwargs,
    )


def test_bfs_returns_documents_and_adjacency_lists_before_matrix_construction():
    crawler = make_crawler({"technology": ["A"]}, max_links_per_page=None)
    documents, adjacency_lists = crawler.crawl(depth=1, max_pages_per_topic=3)
    assert adjacency_lists == {"wiki_101": ["wiki_202", "wiki_303"], "wiki_202": ["wiki_101", "wiki_303"], "wiki_303": []}
    assert not hasattr(crawler, "incidence_matrix")
    matrix = build_incidence_matrix(documents, adjacency_lists)
    assert isinstance(matrix, csr_matrix)
    assert [document.title for document in documents] == ["A", "B", "C"]
    np.testing.assert_array_equal(matrix.toarray(), [[0, 1, 1], [1, 0, 1], [0, 0, 0]])
    assert matrix.dtype == np.uint8
    assert [document.doc_id for document in documents] == ["wiki_101", "wiki_202", "wiki_303"]
    assert all(document.topic is None for document in documents)
    assert all(document.discovered_topic == "technology" for document in documents)
    assert documents[0].revision_id == 1010
    assert documents[0].preview == documents[0].text


def test_depth_zero_resolves_unvisited_alias_and_keeps_dangling_rows():
    crawler = make_crawler(["A", "B", "C"], max_links_per_page=1)
    documents, adjacency_lists = crawler.crawl(depth=0, max_pages_per_topic=3)
    matrix = build_incidence_matrix(documents, adjacency_lists)
    assert len(documents) == 3
    np.testing.assert_array_equal(matrix.toarray(), [[0, 1, 1], [1, 0, 1], [0, 0, 0]])
    assert crawler._resolved_page_ids["Bee"] == 202
    assert "Outside" in documents[0].outgoing_links
    assert matrix.shape == (3, 3)  # Outside was resolved but never added.


def test_redirect_seeds_are_deduplicated_by_page_id():
    crawler = make_crawler(["Alias A", "A", "B"])
    documents, adjacency_lists = crawler.crawl(depth=0, max_pages_per_topic=10)
    matrix = build_incidence_matrix(documents, adjacency_lists)
    assert [document.page_id for document in documents] == [101, 202]
    np.testing.assert_array_equal(matrix.toarray(), [[0, 1], [1, 0]])
    fetched = [call["titles"] for call in crawler.session.calls if "extracts" in call.get("prop", "")]
    assert fetched == ["Alias A", "B"]


def test_page_limit_does_not_add_link_targets_to_the_corpus():
    crawler = make_crawler(["A"])
    documents, adjacency_lists = crawler.crawl(depth=2, max_pages_per_topic=1)
    matrix = build_incidence_matrix(documents, adjacency_lists)
    assert len(documents) == 1
    assert matrix.shape == (1, 1)
    assert matrix.nnz == 0


def test_plain_text_and_link_titles_are_read_directly_from_api_fields():
    crawler = make_crawler(["A"])
    crawler.session.text[101] = "First paragraph.\n\nSecond paragraph with more detail."
    crawler.session.links[101] = ["B", "B", "Category:Science", "List of scientists", "C (disambiguation)"]
    documents, _ = crawler.crawl(depth=0, max_pages_per_topic=1)
    assert documents[0].text == crawler.session.text[101]
    assert documents[0].preview == "First paragraph."
    assert documents[0].outgoing_links == ["B"]
    request = crawler.session.calls[0]
    assert request["explaintext"] == 1
    assert "exintro" not in request and "exchars" not in request
    assert request["plnamespace"] == 0 and request["pllimit"] == "max"
    assert all(call["action"] == "query" for call in crawler.session.calls)


def test_missing_disambiguation_and_short_pages_are_skipped():
    crawler = make_crawler(["Missing", "Ambiguous", "Short", "C"], min_words=5)
    documents, adjacency_lists = crawler.crawl(depth=0, max_pages_per_topic=5)
    matrix = build_incidence_matrix(documents, adjacency_lists)
    assert [document.title for document in documents] == ["C"]
    assert matrix.shape == (1, 1)
    assert all(call["action"] == "query" for call in crawler.session.calls)


def test_empty_crawl_is_an_explicit_failure():
    crawler = make_crawler(["Missing"])
    with pytest.raises(RuntimeError, match="No articles collected"):
        crawler.crawl(depth=0, max_pages_per_topic=2)


def test_crawl_resets_previous_collection():
    crawler = make_crawler(["A", "C"])
    first, first_lists = crawler.crawl(depth=0, max_pages_per_topic=2)
    second, second_lists = crawler.crawl(depth=0, max_pages_per_topic=2)
    assert first is not second
    assert len(second) == 2
    assert first_lists == second_lists


def test_snapshots_and_lists_round_trip_without_exporting_a_matrix(tmp_path):
    crawler = make_crawler(["A", "B", "C"])
    documents, adjacency_lists = crawler.crawl(depth=0, max_pages_per_topic=3)
    crawler.save(tmp_path)
    saved_documents = [json.loads(line) for line in (tmp_path / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    saved_lists = json.loads((tmp_path / "adjacency_lists.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [document["doc_id"] for document in saved_documents] == manifest["doc_ids"]
    assert saved_documents[0]["text"] == documents[0].text
    assert saved_documents[0]["revision_id"] == 1010
    assert manifest["topic_labels_reviewed"] is False
    assert saved_lists == adjacency_lists
    assert not (tmp_path / "incidence_matrix.npz").exists()


def test_manual_graph_keeps_isolates_and_deduplicates_links():
    crawler = make_crawler(["A"])
    for page_id, title, links in [(1, "A", ["B", "B", "A"]), (2, "B", []), (3, "C", [])]:
        crawler.add_document(Document(page_id, title, "url", f"wiki_{page_id}", "now", "preview", outgoing_links=links))
    matrix = build_incidence_matrix(crawler.get_collected_documents(), crawler.build_adjacency_lists())
    np.testing.assert_array_equal(matrix.toarray(), [[0, 1, 0], [0, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize("kwargs", [
    {"seed_urls": []}, {"seed_urls": ["https://example.com/wiki/A"]},
    {"seed_urls": ["Category:Science"]}, {"seed_urls": ["A"], "min_words": -1},
    {"seed_urls": ["A"], "topics": ["one", "two"]},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        WebCrawler(**kwargs)


def test_urls_and_topic_hints_are_normalized():
    crawler = make_crawler({"science": ["https://en.wikipedia.org/wiki/A#History", "B"]})
    documents, _ = crawler.crawl(depth=0, max_pages_per_topic=2)
    assert [document.title for document in documents] == ["A", "B"]
    assert all(document.discovered_topic == "science" for document in documents)


def test_api_maxlag_retries_and_nonretryable_errors():
    crawler = make_crawler(["A"])
    with patch.object(crawler.session, "get", side_effect=[
        FakeResponse({"error": {"code": "maxlag", "info": "busy"}}),
        FakeResponse({"query": {"pages": []}}),
    ]), patch("project.crawler.time.sleep"):
        assert crawler._request(action="query") == {"query": {"pages": []}}
    with patch.object(crawler.session, "get", return_value=FakeResponse({
        "error": {"code": "badvalue", "info": "invalid input"},
    })):
        with pytest.raises(WikipediaAPIError, match="badvalue"):
            crawler._request(action="query")


def test_http_failures_have_bounded_retries():
    crawler = make_crawler(["A"], max_retries=1)
    with patch.object(crawler.session, "get", side_effect=requests.ConnectionError("offline")) as get:
        with patch("project.crawler.time.sleep"):
            with pytest.raises(WikipediaAPIError, match="offline"):
                crawler._request(action="query")
        assert get.call_count == 2


def test_resolution_failure_does_not_return_a_silently_incomplete_graph():
    crawler = make_crawler(["A"])
    original_query = crawler._query_pages

    def fail_graph_resolution(titles, *, details=False):
        if not details:
            raise WikipediaAPIError("resolution unavailable")
        return original_query(titles, details=details)

    with patch.object(crawler, "_query_pages", side_effect=fail_graph_resolution):
        with pytest.raises(WikipediaAPIError, match="resolution unavailable"):
            crawler.crawl(depth=0, max_pages_per_topic=1)


def test_http_rate_limit_honors_retry_after():
    crawler = make_crawler(["A"])
    with patch.object(crawler.session, "get", side_effect=[
        FakeResponse({}, status_code=429, headers={"Retry-After": "12"}),
        FakeResponse({"query": {"pages": []}}),
    ]), patch("project.crawler.time.sleep") as sleep:
        crawler._request(action="query")
        sleep.assert_called_once_with(12.0)


def test_long_server_cooldown_stops_instead_of_retrying_early():
    crawler = make_crawler(["A"])
    with patch.object(crawler.session, "get", return_value=FakeResponse(
        {}, status_code=429, headers={"Retry-After": "120"},
    )), patch("project.crawler.time.sleep") as sleep:
        with pytest.raises(WikipediaAPIError, match="cooldown"):
            crawler._request(action="query")
        sleep.assert_not_called()


def test_checkpoint_preserves_articles_when_graph_resolution_fails(tmp_path):
    crawler = make_crawler(["A"], checkpoint_dir=tmp_path)
    with patch.object(crawler, "_resolve_outgoing_links", side_effect=WikipediaAPIError("offline")):
        with pytest.raises(WikipediaAPIError, match="offline"):
            crawler.crawl(depth=0, max_pages_per_topic=1)
    snapshot = json.loads((tmp_path / "wiki_101.json").read_text(encoding="utf-8"))
    assert snapshot["revision_id"] == 1010
    assert "Article A" in snapshot["text"]


def test_each_group_gets_its_own_quota_and_shared_pages_can_bridge_groups():
    crawler = make_crawler({"technology": ["A"], "science": ["B"]}, max_links_per_page=None)
    documents, adjacency_lists = crawler.crawl(depth=2, max_pages_per_topic=2)
    assert [document.title for document in documents] == ["A", "B", "C", "Outside"]
    assert crawler.topic_counts == {"technology": 2, "science": 2}
    assert len({document.page_id for document in documents}) == 4
    assert all(document.topic is None for document in documents)
    assert [document.discovered_topic for document in documents] == ["technology", "technology", "science", "science"]
    assert "wiki_404" in adjacency_lists["wiki_101"]
    fetched = [call["titles"] for call in crawler.session.calls if "extracts" in call.get("prop", "")]
    assert len(fetched) == len(set(fetched)) == 4


def test_five_groups_default_to_40_candidates_each_and_200_total():
    groups = {f"group{group}": [f"Group{group} root"] for group in range(5)}
    crawler = make_crawler(groups, max_links_per_page=None)
    api = crawler.session
    api.pages, api.text, api.links = {}, {}, {}
    for group in range(5):
        root = f"Group{group} root"
        titles = [root] + [f"Group{group} article{number}" for number in range(1, 40)]
        for number, title in enumerate(titles):
            page_id = group * 100 + number + 1
            api.pages[title] = page_id
            api.text[page_id] = "Article text."
            api.links[page_id] = titles[1:] if number == 0 else []
    documents, adjacency_lists = crawler.crawl()
    assert len(documents) == len(adjacency_lists) == 200
    assert list(crawler.topic_counts.values()) == [40] * 5
    assert all(document.topic is None for document in documents)


def test_all_link_batches_are_collected_before_bfs_or_graph_filtering():
    crawler = make_crawler(["A"])
    replies = [
        FakeResponse({
            "query": {"pages": [{
                "pageid": 101, "title": "A", "ns": 0,
                "revisions": [{"revid": 1010}], "extract": "The entire plain text.\n\nAnother paragraph.",
                "links": [{"ns": 0, "title": "B"}],
            }]},
            "continue": {"plcontinue": "batch_two", "continue": "||"},
        }),
        FakeResponse({
            "query": {"pages": [{"pageid": 101, "links": [{"ns": 0, "title": "C"}]}]},
            "continue": {"plcontinue": "batch_three", "continue": "||"},
        }),
        FakeResponse({
            "query": {"pages": [{"pageid": 101, "links": [{"ns": 0, "title": "B"}, {"ns": 0, "title": "Outside"}]}]},
        }),
    ]
    with patch.object(crawler.session, "get", side_effect=replies) as get:
        document = crawler._fetch_document("A", "technology")
    assert document.outgoing_links == ["B", "C", "Outside"]
    assert document.preview == "The entire plain text."
    assert get.call_args_list[1].kwargs["params"]["plcontinue"] == "batch_two"
    assert get.call_args_list[2].kwargs["params"]["plcontinue"] == "batch_three"
    assert get.call_args_list[1].kwargs["params"]["pageids"] == "101"


def test_empty_api_extract_is_skipped_instead_of_creating_an_empty_document():
    crawler = make_crawler(["A", "C"])
    crawler.session.text[101] = ""
    documents, _ = crawler.crawl(depth=0, max_pages_per_topic=2)
    assert [document.title for document in documents] == ["C"]
