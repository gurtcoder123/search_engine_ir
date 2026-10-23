"""Offline API fixtures exercise crawling, redirects, pagination and edges."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import requests

from project.crawler import (
    Document, DocumentCollection, WebCrawler, WikipediaAPIError,
)
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
    """A tiny Wikipedia: six articles, two redirect aliases, one bad page each."""

    def __init__(self):
        self.headers = {}
        self.calls = []
        self.aliases = {"Alias A": "A", "Bee": "B"}
        self.pages = {"A": 101, "B": 202, "C": 303, "Outside": 404,
                      "Ambiguous": 505, "Short": 606}
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
        assert params["action"] == "query", "Only structured JSON queries, never HTML"
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
                page["links"] = [{"ns": 0, "title": t} for t in self.links.get(page_id, [])]
            if canonical == "Ambiguous":
                page["pageprops"] = {"disambiguation": ""}
            pages[canonical] = page
        return FakeResponse({"query": {"pages": list(pages.values()), "redirects": redirects}})


def make_crawler(seeds, **kwargs):
    """seeds is {group: [titles]}. Defaults here keep the tests fast and offline."""
    collection = kwargs.pop("document_collection", DocumentCollection())
    kwargs.setdefault("min_words", 0)
    kwargs.setdefault("request_delay", 0)
    kwargs.setdefault("max_links_per_page", None)
    return WebCrawler(
        document_collection=collection,
        topics=list(seeds), seed_topics=list(seeds.values()),
        session=FakeWikipedia(), **kwargs,
    )


# --------------------------------------------------------------- BFS behaviour

def test_crawl_fills_the_shared_collection_and_returns_adjacency_lists():
    collection = DocumentCollection()
    crawler = make_crawler({"technology": ["A"]}, document_collection=collection, depth=1)
    documents, adjacency_lists = crawler.crawl()

    assert documents is collection.get_documents()
    assert [d.title for d in documents] == ["A", "B", "C", "Outside"]
    assert [d.doc_id for d in documents] == ["doc-101", "doc-202", "doc-303", "doc-404"]
    assert adjacency_lists == {
        "doc-101": ["doc-202", "doc-303", "doc-404"],  # A's self-link is dropped
        "doc-202": ["doc-101", "doc-303"],
        "doc-303": [],                                  # dangling
        "doc-404": [],
    }
    assert documents[0].revision_id == 1010
    # No matrix is built during crawling; that waits for manual review.
    assert not hasattr(crawler, "incidence_matrix")


def test_crawling_never_assigns_a_topic_and_keeps_provenance_separate():
    crawler = make_crawler({"technology": ["A"]}, depth=1)
    documents, _ = crawler.crawl()
    assert all(d.topic is None for d in documents)
    assert all(d.discovered_topic == "technology" for d in documents)


def test_depth_zero_collects_only_the_seeds():
    crawler = make_crawler({"technology": ["A", "B", "C"]}, depth=0)
    documents, adjacency_lists = crawler.crawl()
    assert [d.title for d in documents] == ["A", "B", "C"]
    # "Outside" is linked from A but was never collected, so it is not a node.
    assert "Outside" in documents[0].outgoing_links
    assert build_incidence_matrix(documents, adjacency_lists).shape == (3, 3)


def test_depth_one_follows_links_but_depth_zero_does_not():
    shallow = make_crawler({"technology": ["A"]}, depth=0)
    deep = make_crawler({"technology": ["A"]}, depth=1)
    assert len(shallow.crawl()[0]) == 1
    assert len(deep.crawl()[0]) == 4


def test_group_stops_at_its_new_document_cap():
    crawler = make_crawler({"technology": ["A"]}, depth=2, max_pages_per_topic=2)
    documents, _ = crawler.crawl()
    assert len(documents) == 2
    assert crawler.topic_counts == {"technology": 2}


def test_shortfall_is_reported_rather_than_padded():
    # Only four articles are reachable, so a cap of 40 cannot be met.
    crawler = make_crawler({"technology": ["A"]}, depth=1, max_pages_per_topic=40)
    documents, _ = crawler.crawl()
    assert len(documents) == 4
    assert crawler.topic_counts["technology"] == 4


# ------------------------------------------------- dedup, bridging, redirects

def test_pages_are_deduplicated_globally_by_page_id():
    crawler = make_crawler({"technology": ["A", "Alias A", "A"]}, depth=0)
    documents, _ = crawler.crawl()
    assert [d.page_id for d in documents] == [101]


def test_each_group_has_its_own_quota_and_shared_pages_bridge_groups():
    crawler = make_crawler({"technology": ["A"], "science": ["B"]},
                           depth=2, max_pages_per_topic=2)
    documents, adjacency_lists = crawler.crawl()

    assert [d.title for d in documents] == ["A", "B", "C", "Outside"]
    assert crawler.topic_counts == {"technology": 2, "science": 2}
    assert len({d.page_id for d in documents}) == 4
    # science reached A again as a bridge; A keeps technology's provenance and
    # did not consume a second slot.
    assert [d.discovered_topic for d in documents] == [
        "technology", "technology", "science", "science",
    ]
    assert "doc-404" in adjacency_lists["doc-101"]
    # Each article's text was downloaded exactly once.
    fetched = [c["titles"] for c in crawler.session.calls if "extracts" in c.get("prop", "")]
    assert len(fetched) == len(set(fetched))


def test_redirect_aliases_resolve_to_the_collected_article():
    crawler = make_crawler({"technology": ["A", "B"]}, depth=0)
    _, adjacency_lists = crawler.crawl()
    # A links to "Bee", which redirects to the collected article B.
    assert adjacency_lists["doc-101"] == ["doc-202"]
    assert crawler.title_to_page_id["Bee"] == 202


def test_titles_are_normalized_before_comparison():
    crawler = make_crawler({"technology": ["computer_science"]}, depth=0)
    crawler.session.pages["Computer science"] = 707
    crawler.session.text[707] = "Normalized title text."
    documents, _ = crawler.crawl()
    assert [d.title for d in documents] == ["Computer science"]


# ------------------------------------------------------------ page eligibility

def test_missing_disambiguation_and_short_pages_are_skipped():
    crawler = make_crawler({"technology": ["Missing", "Ambiguous", "Short", "C"]},
                           depth=0, min_words=5)
    documents, _ = crawler.crawl()
    assert [d.title for d in documents] == ["C"]
    assert {entry["title"] for entry in crawler.skipped} == {"Missing", "Ambiguous", "Short"}


def test_empty_extract_does_not_become_an_empty_document():
    crawler = make_crawler({"technology": ["A", "C"]}, depth=0)
    crawler.session.text[101] = ""
    documents, _ = crawler.crawl()
    assert [d.title for d in documents] == ["C"]


def test_non_article_link_targets_are_not_followed():
    crawler = make_crawler({"technology": ["A"]}, depth=1)
    crawler.session.links[101] = ["Category:Science", "List of scientists",
                                  "C (disambiguation)", "B"]
    documents, _ = crawler.crawl()
    assert documents[0].outgoing_links == ["B"]


def test_a_crawl_that_collects_nothing_fails_loudly():
    crawler = make_crawler({"technology": ["Missing"]}, depth=0)
    with pytest.raises(RuntimeError, match="No articles collected"):
        crawler.crawl()


# ------------------------------------------------------- links and pagination

def test_all_link_batches_are_merged_before_bfs_or_edge_filtering():
    crawler = make_crawler({"technology": ["A"]}, depth=0)
    replies = [
        FakeResponse({
            "query": {"pages": [{
                "pageid": 101, "title": "A", "ns": 0,
                "revisions": [{"revid": 1010}],
                "extract": "The entire plain text.\n\nAnother paragraph.",
                "links": [{"ns": 0, "title": "B"}],
            }]},
            "continue": {"plcontinue": "batch_two", "continue": "||"},
        }),
        FakeResponse({
            "query": {"pages": [{"pageid": 101, "links": [{"ns": 0, "title": "C"}]}]},
            "continue": {"plcontinue": "batch_three", "continue": "||"},
        }),
        FakeResponse({
            "query": {"pages": [{"pageid": 101, "links": [
                {"ns": 0, "title": "B"}, {"ns": 0, "title": "Outside"},
            ]}]},
        }),
    ]
    with patch.object(crawler.session, "get", side_effect=replies) as get:
        pages = crawler._fetch_pages(["A"], full=True)
    document = crawler._build_document(pages[0], "technology")

    assert document.outgoing_links == ["B", "C", "Outside"]
    assert document.preview == "The entire plain text."
    assert get.call_args_list[1].kwargs["params"]["plcontinue"] == "batch_two"
    assert get.call_args_list[2].kwargs["params"]["plcontinue"] == "batch_three"
    assert get.call_args_list[1].kwargs["params"]["pageids"] == "101"


def test_links_bfs_never_followed_are_still_saved_and_resolved():
    # One link followed per page, but every link must still reach the graph.
    crawler = make_crawler({"technology": ["A"]}, depth=1, max_links_per_page=1)
    documents, adjacency_lists = crawler.crawl()
    # Saved exactly as the API gave them: the alias "Bee" and the self-link "A"
    # are kept here and only resolved/dropped when edges are built.
    assert documents[0].outgoing_links == ["Bee", "C", "Outside", "A"]
    followed = [d.title for d in documents]
    assert len(followed) == 2, "only one link was followed from the seed"
    # The unfollowed link's target was collected by nobody, so it is not an edge,
    # but the followed-and-collected ones are, regardless of discovery order.
    assert set(adjacency_lists["doc-101"]) <= {"doc-202", "doc-303"}


def test_requested_api_fields_are_structured_not_html():
    crawler = make_crawler({"technology": ["A"]}, depth=0)
    crawler.crawl()
    request = crawler.session.calls[0]
    assert request["explaintext"] == 1
    assert request["plnamespace"] == 0 and request["pllimit"] == "max"
    assert "exintro" not in request and "exchars" not in request
    assert all(call["action"] == "query" for call in crawler.session.calls)


# -------------------------------------------------- spacing and error handling

def test_every_request_is_spaced_including_skipped_pages():
    crawler = make_crawler({"technology": ["Missing", "C"]}, depth=0, request_delay=0.5)
    # A frozen clock means the delay is always still owed, so every request
    # that waits its turn shows up as a sleep.
    with patch("project.crawler.time.sleep") as sleep:
        with patch("project.crawler.time.monotonic", return_value=0.0):
            crawler.crawl()
    # "Missing" is skipped, but its request still waited its turn first.
    assert sleep.call_count == 2
    assert sleep.call_args_list[0].args == (0.5,)


def test_maxlag_is_retried_and_a_bad_request_is_not():
    crawler = make_crawler({"technology": ["A"]})
    with patch.object(crawler.session, "get", side_effect=[
        FakeResponse({"error": {"code": "maxlag", "info": "busy"}}),
        FakeResponse({"query": {"pages": []}}),
    ]), patch("project.crawler.time.sleep"):
        assert crawler._get(action="query") == {"query": {"pages": []}}

    with patch.object(crawler.session, "get", return_value=FakeResponse(
        {"error": {"code": "badvalue", "info": "invalid input"}},
    )):
        with pytest.raises(WikipediaAPIError, match="badvalue"):
            crawler._get(action="query")


def test_connection_failures_retry_a_bounded_number_of_times():
    crawler = make_crawler({"technology": ["A"]}, max_retries=1)
    with patch.object(crawler.session, "get",
                      side_effect=requests.ConnectionError("offline")) as get:
        with patch("project.crawler.time.sleep"):
            with pytest.raises(WikipediaAPIError, match="offline"):
                crawler._get(action="query")
        assert get.call_count == 2


def test_rate_limit_honours_retry_after():
    crawler = make_crawler({"technology": ["A"]})
    with patch.object(crawler.session, "get", side_effect=[
        FakeResponse({}, status_code=429, headers={"Retry-After": "12"}),
        FakeResponse({"query": {"pages": []}}),
    ]), patch("project.crawler.time.sleep") as sleep:
        crawler._get(action="query")
        sleep.assert_called_once_with(12.0)


def test_a_long_cooldown_stops_instead_of_sleeping_for_minutes():
    crawler = make_crawler({"technology": ["A"]})
    with patch.object(crawler.session, "get", return_value=FakeResponse(
        {}, status_code=429, headers={"Retry-After": "120"},
    )), patch("project.crawler.time.sleep") as sleep:
        with pytest.raises(WikipediaAPIError, match="cooldown"):
            crawler._get(action="query")
        sleep.assert_not_called()


def test_one_failing_page_does_not_end_the_crawl():
    crawler = make_crawler({"technology": ["A", "C"]}, depth=0)
    real_fetch = crawler._fetch_pages

    def fail_on_a(titles, *, full=False):
        if titles == ["A"]:
            raise WikipediaAPIError("offline")
        return real_fetch(titles, full=full)

    with patch.object(crawler, "_fetch_pages", side_effect=fail_on_a):
        documents, _ = crawler.crawl()
    assert [d.title for d in documents] == ["C"]
    assert crawler.skipped[0]["reason"] == "offline"


# ------------------------------------------------------------ adjacency + save

def test_adjacency_excludes_self_links_duplicates_and_outside_targets():
    collection = DocumentCollection()
    crawler = make_crawler({"technology": ["A"]}, document_collection=collection)
    for page_id, title, links in [(1, "A", ["B", "B", "A", "Nowhere"]),
                                  (2, "B", []), (3, "C", [])]:
        collection.add_document(Document(
            page_id, title, "url", f"doc-{page_id}", "now", "preview",
            outgoing_links=links,
        ))
    crawler.title_to_page_id.update({"Nowhere": None})
    adjacency_lists = crawler.resolve_adjacency_lists()

    assert adjacency_lists == {"doc-1": ["doc-2"], "doc-2": [], "doc-3": []}
    matrix = build_incidence_matrix(collection.get_documents(), adjacency_lists)
    np.testing.assert_array_equal(matrix.toarray(), [[0, 1, 0], [0, 0, 0], [0, 0, 0]])


def test_save_writes_the_layout_the_reviewer_reads(tmp_path):
    crawler = make_crawler({"technology": ["A", "B", "C"]}, depth=0)
    documents, adjacency_lists = crawler.crawl()
    crawler.save(tmp_path)

    saved = [json.loads(line) for line
             in (tmp_path / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert [d["doc_id"] for d in saved] == manifest["doc_ids"]
    assert saved[0]["text"] == documents[0].text, "full text is saved, not just the preview"
    assert saved[0]["topic"] is None
    assert manifest["topic_labels_reviewed"] is False
    assert manifest["corpus_build_id"]
    assert manifest["crawl_settings"]["depth"] == 0
    assert json.loads((tmp_path / "adjacency_lists.json").read_text(encoding="utf-8")) == adjacency_lists
    # The matrix belongs to graph.py, after review.
    assert not (tmp_path / "incidence_matrix.npz").exists()


def test_save_refuses_before_a_crawl(tmp_path):
    crawler = make_crawler({"technology": ["A"]})
    with pytest.raises(RuntimeError, match="Run crawl"):
        crawler.save(tmp_path)


# ------------------------------------------------------------ input validation

@pytest.mark.parametrize("kwargs", [
    {"topics": ["technology"], "seed_topics": []},
    {"topics": ["technology"], "seed_topics": [[]]},
    {"topics": ["a", "b"], "seed_topics": [["A"]]},
    {"topics": ["a"], "seed_topics": [["A"]], "depth": -1},
    {"topics": ["a"], "seed_topics": [["A"]], "min_words": -1},
    {"topics": ["a"], "seed_topics": [["A"]], "max_pages_per_topic": 0},
    {"topics": ["a"], "seed_topics": [["A"]], "max_links_per_page": 0},
    {"topics": ["a"], "seed_topics": [["A"]], "timeout": 0},
    {"topics": ["a"], "seed_topics": [["A"]], "max_retries": -1},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        WebCrawler(document_collection=DocumentCollection(), **kwargs)


def test_zero_valued_settings_are_honoured_not_replaced_by_defaults():
    crawler = make_crawler({"technology": ["A"]}, depth=0, min_words=0, request_delay=0)
    assert crawler.depth == 0
    assert crawler.min_words == 0
    assert crawler.request_delay == 0
