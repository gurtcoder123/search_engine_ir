"""Collect Wikipedia candidates and their outgoing link titles.

Usage:
    collection = DocumentCollection()
    crawler = WebCrawler(
        document_collection=collection,
        topics=["technology", "sports"],
        seed_topics=[["Computer", "Software"], ["Association football"]],
        depth=2,
        max_pages_per_topic=40,
    )
    documents, adjacency_lists = crawler.crawl()
    crawler.save(CANDIDATES_DIR)

Text, metadata and link titles come straight from the MediaWiki JSON API; there
is no HTML parsing. Every document keeps topic=None until a human reviews it.
All outgoing titles are saved even when BFS follows only a few of them.
reviewer.py picks the final corpus; graph.py builds the matrix afterwards.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import time
from urllib.parse import quote

import requests

if __package__:
    from .config import CANDIDATES_DIR, TOPICS, USER_AGENT, corpus_build_id, doc_id_for
else:
    from config import CANDIDATES_DIR, TOPICS, USER_AGENT, corpus_build_id, doc_id_for


LOGGER = logging.getLogger(__name__)
API_URL = "https://en.wikipedia.org/w/api.php"
ARTICLE_URL = "https://en.wikipedia.org/wiki/"
DEFAULT_USER_AGENT = USER_AGENT

# The API accepts 50 titles per query, and gives links one batch at a time.
TITLES_PER_REQUEST = 50
# Worth waiting out: the server is busy rather than refusing us.
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_CODES = {"maxlag", "ratelimited", "readonly"}
# Past this, stop and let the user rerun later instead of sleeping for minutes.
MAX_COOLDOWN_SECONDS = 60

# TOPICS is re-exported so the documented partner import keeps working:
#     from project.crawler import Document, TOPICS
# config.py remains its single authoritative definition.
__all__ = ["Document", "DocumentCollection", "WebCrawler", "WikipediaAPIError", "TOPICS"]


class WikipediaAPIError(RuntimeError):
    """An HTTP or MediaWiki API request could not be completed."""


@dataclass
class Document:
    # Original constructor fields stay in the same order.
    page_id: int
    title: str
    url: str
    doc_id: str
    collected_at: str
    preview: str
    topic: str | None = None
    text: str = ""
    revision_id: int | None = None
    outgoing_links: list[str] = field(default_factory=list)
    # Which seed group first reached this page. Provenance only: it must never
    # become the final topic, which only manual review assigns.
    discovered_topic: str | None = None


class DocumentCollection:
    def __init__(self):
        self.documents: list[Document] = []
        self._by_page_id: dict[int, Document] = {}

    def add_document(self, document: Document) -> bool:
        """Return False instead of inserting a duplicate Wikipedia page."""
        if document.page_id in self._by_page_id:
            return False
        self._by_page_id[document.page_id] = document
        self.documents.append(document)
        return True

    def get_documents(self) -> list[Document]:
        return self.documents

    def get_by_page_id(self, page_id) -> Document | None:
        """The stored document for a Wikipedia page ID, or None."""
        return self._by_page_id.get(page_id)

    def clear(self) -> int:
        """Empty the collection in place, returning how many were removed."""
        count = len(self.documents)
        self.documents.clear()
        self._by_page_id.clear()
        return count


def normalize_title(title: str) -> str:
    """Spell a title the way Wikipedia does, so two spellings compare equal.

    Underscores are spaces and only the first letter is case-insensitive, so
    "computer_science" and "Computer science" mean the same article.
    """
    title = " ".join(title.replace("_", " ").split())
    return title[:1].upper() + title[1:]


def is_article_title(title: str) -> bool:
    """Skip other namespaces (Category:, File:), list pages and disambiguations."""
    lower = title.casefold()
    return bool(title) and ":" not in title and not (
        lower.startswith(("list of ", "lists of ", "outline of ", "index of "))
        or lower.endswith("(disambiguation)")
    )


def _retry_after_seconds(response):
    """The server's requested wait, when it sent a plain number of seconds."""
    try:
        return float(response.headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return None


class WebCrawler:

    def __init__(
        self,
        document_collection,
        topics,
        seed_topics,
        *,
        depth=2,
        max_pages_per_topic=40,
        max_links_per_page=30,
        min_words=100,
        request_delay=1.0,
        timeout=30.0,
        max_retries=2,
        random_seed=17,
        user_agent=DEFAULT_USER_AGENT,
        session=None,
    ):
        """seed_topics is a list of seed-title lists, parallel to topics.

        e.g. topics=["technology", "sports"]
             seed_topics=[["Computer", "Software"], ["Association football"]]

        The optional settings carry real defaults rather than None, so passing
        depth=0 or request_delay=0 means exactly that.

        max_links_per_page limits what BFS explores, NOT the links saved for the
        graph. None follows every eligible link.
        """
        if len(topics) != len(seed_topics):
            raise ValueError("topics and seed_topics must be the same length")
        if not any(seed_topics):
            raise ValueError("Provide at least one seed title")
        if depth < 0 or max_pages_per_topic < 1 or min_words < 0 or request_delay < 0:
            raise ValueError("depth/min_words/request_delay must be >= 0 and the page cap >= 1")
        if timeout <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and max_retries must be >= 0")
        if max_links_per_page is not None and max_links_per_page < 1:
            raise ValueError("max_links_per_page must be positive, or None to follow every link")

        self.document_collection = document_collection
        self.topics = list(topics)
        self.seed_topics = [list(seeds) for seeds in seed_topics]
        self.depth = depth
        self.max_pages_per_topic = max_pages_per_topic
        self.max_links_per_page = max_links_per_page
        self.min_words = min_words
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.random_seed = random_seed
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        # Every title we have resolved to a page ID (None when the page does not
        # exist). Redirect aliases land here too, so a link written as an alias
        # still matches the article we actually collected.
        self.title_to_page_id: dict[str, int | None] = {}
        self.adjacency_lists: dict[str, list[str]] = {}
        self.topic_counts: dict[str, int] = {}
        self.skipped: list[dict] = []
        self._last_request_time = 0.0

    # ---------------------------------------------------------------- requests

    def _wait_turn(self):
        """Space requests apart. Every path comes through here: first requests,
        link pagination, retries, and the pages we end up skipping."""
        remaining = self.request_delay - (time.monotonic() - self._last_request_time)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_time = time.monotonic()

    def _get(self, **params):
        """One API request with bounded retries. Returns the decoded JSON.

        Raises WikipediaAPIError when retries run out, when the problem is not
        worth retrying, or when the server asks for a long cooldown.
        """
        params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            try:
                response = self.session.get(API_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError) as error:
                response = getattr(error, "response", None)
                status = getattr(response, "status_code", None)
                if status is not None and status not in RETRY_STATUS:
                    raise WikipediaAPIError(str(error)) from error
                failure = WikipediaAPIError(str(error))
            else:
                api_error = data.get("error")
                if api_error is None:
                    return data
                failure = WikipediaAPIError(f"{api_error.get('code')}: {api_error.get('info')}")
                if api_error.get("code") not in RETRY_CODES:
                    raise failure

            if attempt == self.max_retries:
                raise failure
            delay = max(_retry_after_seconds(response) or 0.0, min(5 * 2 ** attempt, 30))
            if delay > MAX_COOLDOWN_SECONDS:
                raise WikipediaAPIError(f"Server asked for a {delay:.0f}s cooldown; rerun the crawl later")
            LOGGER.warning("API busy; retrying in %.0fs", delay)
            time.sleep(delay)

    def _fetch_pages(self, titles, *, full=False):
        """Look up titles. With full=True also fetch text, links and metadata."""
        params = {
            "action": "query", "titles": "|".join(titles),
            "redirects": 1, "prop": "info",
        }
        if full:
            params.update(
                prop="info|pageprops|revisions|extracts|links", inprop="url",
                rvprop="ids", explaintext=1, exlimit=1,
                plnamespace=0, pllimit="max",
            )
        data = self._get(**params)
        query = data.get("query", {})
        pages = query.get("pages", [])
        self._remember_page_ids(titles, query, pages)
        if full:
            self._fetch_remaining_links(pages, data.get("continue", {}))
        return pages

    def _remember_page_ids(self, requested, query, pages):
        """Record which page ID each title leads to, following redirects.

        The API reports "Bee" -> "B" under redirects and "bee" -> "Bee" under
        normalized, so a title can take two hops before reaching the article.
        """
        for page in pages:
            missing = "missing" in page or "invalid" in page
            self.title_to_page_id[normalize_title(page["title"])] = (
                None if missing else page.get("pageid")
            )

        aliases = {
            normalize_title(item["from"]): normalize_title(item["to"])
            for kind in ("normalized", "converted", "redirects")
            for item in query.get(kind, [])
        }
        for title in list(requested) + list(aliases):
            target, seen = normalize_title(title), set()
            while target in aliases and target not in seen:
                seen.add(target)
                target = aliases[target]
            self.title_to_page_id[normalize_title(title)] = self.title_to_page_id.get(target)

    def _fetch_remaining_links(self, pages, continuation):
        """pllimit="max" returns one batch, not every link. Follow the rest."""
        page_by_id = {page["pageid"]: page for page in pages if "pageid" in page}
        while page_by_id and "plcontinue" in continuation:
            data = self._get(
                action="query", pageids="|".join(str(i) for i in page_by_id),
                prop="links", plnamespace=0, pllimit="max", **continuation,
            )
            for page in data.get("query", {}).get("pages", []):
                if page.get("pageid") in page_by_id:
                    page_by_id[page["pageid"]].setdefault("links", []).extend(page.get("links", []))
            continuation = data.get("continue", {})

    # ---------------------------------------------------------------- crawling

    def crawl(self):
        """BFS from each seed group in turn. Returns (documents, adjacency_lists).

        Seeds are depth 0. A group stops at max_pages_per_topic NEW documents or
        when its queue runs dry, so 40 is a ceiling and not a promise. A page an
        earlier group already stored is not stored twice and does not use up this
        group's quota, but its links are still explored: it bridges the groups.
        """
        for group, seeds in zip(self.topics, self.seed_topics):
            self._crawl_group(group, [normalize_title(seed) for seed in seeds])

        documents = self.document_collection.get_documents()
        if not documents:
            raise RuntimeError("No articles collected. Check the seeds, min_words and crawler.skipped.")
        self.resolve_adjacency_lists()
        return documents, self.adjacency_lists

    def _crawl_group(self, group, seeds):
        """One seed group's BFS, counting only pages no other group stored."""
        queue = deque((title, 0) for title in seeds)
        queued = set(seeds)      # this group's own frontier, so groups can overlap
        visited_pages = set()
        self.topic_counts[group] = 0

        while queue and self.topic_counts[group] < self.max_pages_per_topic:
            title, level = queue.popleft()
            document = self._collect(title, group)
            if document is None or document.page_id in visited_pages:
                continue
            visited_pages.add(document.page_id)

            # False means another group already stored it: a bridge, not a new
            # candidate, so it costs this group nothing but still opens its links.
            if self.document_collection.add_document(document):
                self.topic_counts[group] += 1
                LOGGER.info("%s: %s (%d/%d)", group, document.title,
                            self.topic_counts[group], self.max_pages_per_topic)

            if level < self.depth:
                for link in self._links_to_follow(document):
                    if link not in queued:
                        queued.add(link)
                        queue.append((link, level + 1))

        if self.topic_counts[group] < self.max_pages_per_topic:
            LOGGER.warning("%s: collected %d of %d candidates; the frontier ran out",
                           group, self.topic_counts[group], self.max_pages_per_topic)

    def _collect(self, title, group):
        """The article for this title, or None when it is unusable.

        Reuses a page we already downloaded. A failed request is recorded in
        self.skipped rather than raised, so one bad page cannot end the crawl.
        """
        known = self.document_collection.get_by_page_id(
            self.title_to_page_id.get(normalize_title(title))
        )
        if known is not None:
            return known

        try:
            pages = self._fetch_pages([title], full=True)
        except WikipediaAPIError as error:
            self._skip(title, group, str(error))
            return None
        if not pages:
            self._skip(title, group, "no page returned")
            return None

        # A redirect can land on a page another group already stored.
        known = self.document_collection.get_by_page_id(pages[0].get("pageid"))
        if known is not None:
            return known

        document = self._build_document(pages[0], group)
        if document is None:
            self._skip(title, group, "missing, disambiguation, or too short")
        return document

    def _skip(self, title, group, reason):
        self.skipped.append({"title": title, "group": group, "reason": reason})
        LOGGER.info("skipped %s (%s)", title, reason)

    def _build_document(self, page, group):
        """Turn one API page into a Document, or None if it is not an article."""
        if "missing" in page or "invalid" in page or page.get("ns") != 0:
            return None
        if not is_article_title(page["title"]) or "disambiguation" in page.get("pageprops", {}):
            return None
        text = page.get("extract", "").strip()
        if not text or len(text.split()) < self.min_words:
            return None

        # Keep every eligible outgoing title, in the article's own order.
        links = list(dict.fromkeys(
            normalize_title(link["title"]) for link in page.get("links", [])
            if link.get("ns") == 0 and is_article_title(link["title"])
        ))
        revision = (page.get("revisions") or [{}])[0]
        return Document(
            page_id=page["pageid"],
            title=page["title"],
            url=page.get("canonicalurl") or page.get("fullurl")
            or ARTICLE_URL + quote(page["title"].replace(" ", "_")),
            doc_id=doc_id_for(page["pageid"]),
            collected_at=datetime.now(timezone.utc).isoformat(),
            preview=text.split("\n\n", 1)[0].strip(),
            text=text,
            revision_id=revision.get("revid"),
            outgoing_links=links,
            discovered_topic=group,
        )

    def _links_to_follow(self, document):
        """Which outgoing links BFS explores. All of them stay saved either way."""
        links = document.outgoing_links
        if self.max_links_per_page is None or len(links) <= self.max_links_per_page:
            return links
        # Sample reproducibly, keeping the article's own link order.
        chooser = random.Random(self.random_seed + document.page_id)
        picked = sorted(chooser.sample(range(len(links)), self.max_links_per_page))
        return [links[i] for i in picked]

    # --------------------------------------------------------------- adjacency

    def resolve_adjacency_lists(self):
        """Turn saved outgoing titles into doc-ID edges inside the corpus.

        Every collected document gets an entry, empty ones included. Titles we
        never looked up are resolved in batches here, so a link to a collected
        article is not lost just because BFS never followed it. Destinations
        outside the corpus, duplicate edges and self-links are dropped.
        """
        documents = self.document_collection.get_documents()
        for document in documents:
            self.title_to_page_id[normalize_title(document.title)] = document.page_id

        unresolved = sorted({
            title for document in documents for title in document.outgoing_links
            if title not in self.title_to_page_id
        })
        for start in range(0, len(unresolved), TITLES_PER_REQUEST):
            self._fetch_pages(unresolved[start:start + TITLES_PER_REQUEST])

        self.adjacency_lists = {}
        for document in documents:
            targets = set()
            for title in document.outgoing_links:
                page_id = self.title_to_page_id.get(normalize_title(title))
                target = self.document_collection.get_by_page_id(page_id)
                if target is not None and target.doc_id != document.doc_id:
                    targets.add(target.doc_id)
            self.adjacency_lists[document.doc_id] = sorted(targets)
        return self.adjacency_lists

    # ------------------------------------------------------------------ saving

    def save(self, output_dir):
        """Write the candidate set in the layout reviewer.py reads.

        Nothing is deleted: point a new crawl at a fresh directory.
        """
        documents = self.document_collection.get_documents()
        if not documents or set(self.adjacency_lists) != {d.doc_id for d in documents}:
            raise RuntimeError("Run crawl() before save()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "documents.jsonl").open("w", encoding="utf-8") as stream:
            for document in documents:
                stream.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")

        doc_ids = [document.doc_id for document in documents]
        _write_json(output_dir / "adjacency_lists.json", self.adjacency_lists)
        _write_json(output_dir / "crawl_log.json", self.skipped)
        _write_json(output_dir / "manifest.json", {
            "stage": "candidates_awaiting_review",
            "corpus_build_id": corpus_build_id(doc_ids),
            "doc_ids": doc_ids,
            "topics": list(TOPICS),
            "new_candidates_per_seed_group": self.topic_counts,
            "topic_labels_reviewed": False,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "crawl_settings": {
                "depth": self.depth,
                "max_pages_per_topic": self.max_pages_per_topic,
                "max_links_per_page": self.max_links_per_page,
                "min_words": self.min_words,
                "request_delay": self.request_delay,
                "random_seed": self.random_seed,
                "seed_topics": dict(zip(self.topics, self.seed_topics)),
            },
            "collection_method": "MediaWiki query: plain-text extracts and main-namespace links",
        })
        LOGGER.info("saved %d candidates to %s", len(documents), output_dir)


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Collect Wikipedia candidates and their outgoing links. "
                    "Nothing is labelled here: run reviewer.py next.")
    parser.add_argument("--seeds-file", type=Path, required=True,
                        help="JSON mapping each topic to its seed titles")
    parser.add_argument("--depth", type=int, default=2, help="Seeds are depth 0")
    parser.add_argument("--max-pages-per-topic", type=int, default=40)
    parser.add_argument("--max-links-per-page", type=int, default=30,
                        help="Discovery limit only; 0 follows every eligible link")
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT,
                        help="Descriptive User-Agent, preferably naming your contact")
    parser.add_argument("--output-dir", type=Path, default=CANDIDATES_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with args.seeds_file.open(encoding="utf-8") as stream:
        seeds = json.load(stream)
    if not isinstance(seeds, dict) or not any(seeds.values()):
        parser.error("The seeds file must map topic names to lists of article titles")
    unknown = [group for group in seeds if group not in TOPICS]
    if unknown:
        parser.error(f"Unknown topic(s) {unknown}; use: {', '.join(TOPICS)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"{args.output_dir} already holds a crawl; use --output-dir for a new one")

    crawler = WebCrawler(
        document_collection=DocumentCollection(),
        topics=list(seeds), seed_topics=list(seeds.values()),
        depth=args.depth, max_pages_per_topic=args.max_pages_per_topic,
        max_links_per_page=args.max_links_per_page or None,
        min_words=args.min_words, request_delay=args.delay, user_agent=args.user_agent,
    )
    documents, adjacency_lists = crawler.crawl()
    crawler.save(args.output_dir)

    print(f"\nCandidates: {len(documents)}; internal links: {sum(map(len, adjacency_lists.values()))}")
    for group in crawler.topics:
        collected = crawler.topic_counts.get(group, 0)
        shortfall = "" if collected >= args.max_pages_per_topic else "  (short of the cap)"
        print(f"  {group}: {collected}{shortfall}")
    if crawler.skipped:
        print(f"Skipped {len(crawler.skipped)} pages; see crawl_log.json")
    print(f"Saved to {args.output_dir.resolve()}")
    print("These are candidates, not a labelled corpus. Run reviewer.py next.")


if __name__ == "__main__":
    main()
