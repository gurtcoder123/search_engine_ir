"""Collect Wikipedia candidates and page-to-page adjacency lists.

Usage:
    crawler = WebCrawler(seed_urls={"technology": ["Computer"]})
    documents, adjacency_lists = crawler.crawl(depth=2, max_pages_per_topic=40)

Text, metadata and link titles come directly from the MediaWiki JSON API.
All documents have topic=None until reviewed. Outgoing titles are retained,
even if they are not followed. Adjacency lists contain only collected targets.
reviewer.py selects the corpus; graph.py builds its matrix afterwards.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
from pathlib import Path
import random
import time
from urllib.parse import parse_qs, quote, unquote, urlsplit

import requests


LOGGER = logging.getLogger(__name__)
API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "Group17InfoRetrievalCrawler/0.1 (educational Wikipedia corpus)"
TOPICS = ("technology", "science", "health_and_medicine", "sports", "politics")


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
    discovered_topic: str | None = None


class DocumentCollection:
    def __init__(self):
        self.documents: list[Document] = []
        self._page_ids: set[int] = set()

    def add_document(self, document: Document) -> bool:
        """Return False instead of inserting a duplicate Wikipedia page."""
        if document.page_id in self._page_ids:
            return False
        self._page_ids.add(document.page_id)
        self.documents.append(document)
        return True

    def get_documents(self) -> list[Document]:
        return self.documents


class WikipediaAPIError(RuntimeError):
    """An HTTP or MediaWiki API request could not be completed."""


def _title_key(title: str) -> str:
    title = " ".join(title.replace("_", " ").split())
    # English Wikipedia only ignores case for a title's first letter.
    return title[:1].upper() + title[1:]


def _article_title(value: str) -> str:
    """Accept a title or English Wikipedia URL; remove section anchors."""
    value = value.strip()
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https"}:
        if parsed.hostname not in {"en.wikipedia.org", "en.m.wikipedia.org"}:
            raise ValueError("Seeds must be English Wikipedia URLs or article titles")
        if parsed.path.startswith("/wiki/"):
            value = unquote(parsed.path[len("/wiki/"):])
        elif parsed.path == "/w/index.php":
            value = parse_qs(parsed.query).get("title", [""])[0]
        else:
            raise ValueError(f"Not a Wikipedia article URL: {value}")
    return _title_key(value.split("#", 1)[0])


def _eligible_title(title: str) -> bool:
    lower = title.casefold()
    return bool(title) and ":" not in title and not (
        lower.startswith(("list of ", "lists of ", "outline of ", "index of "))
        or lower.endswith("(disambiguation)")
    )


class WebCrawler:
    def __init__(
        self,
        topics: str | Sequence[str] | None = None,
        seed_urls: Sequence[str] | Mapping[str, Sequence[str]] | str | None = None,
        *,
        request_delay: float = 1.0,
        timeout: float = 30,
        max_retries: int = 2,
        max_links_per_page: int | None = 30,
        min_words: int = 100,
        random_seed: int = 17,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        """Seeds are titles/URLs, or {topic: [titles/URLs]}.

        For a flat list, topics may be None, one hint, or one hint per seed.
        Hints are saved as discovered_topic; topic itself stays None.
        max_links_per_page limits discovery, NOT the links saved for the graph.
        """
        if request_delay < 0 or timeout <= 0 or max_retries < 0 or min_words < 0:
            raise ValueError("Invalid request delay, timeout, retries, or minimum words")
        if max_links_per_page is not None and max_links_per_page < 1:
            raise ValueError("max_links_per_page must be positive or None")
        self.topics = [topics] if isinstance(topics, str) else list(topics or [])
        self.seed_urls = seed_urls
        self._seeds = self._prepare_seeds(seed_urls)
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_links_per_page = max_links_per_page
        self.min_words = min_words
        self.random_seed = random_seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._last_request = 0.0
        self._reset()

    def _prepare_seeds(self, seeds):
        if isinstance(seeds, Mapping):
            pairs = []
            for topic, values in seeds.items():
                values = [values] if isinstance(values, str) else values
                pairs.extend((_article_title(value), topic) for value in values)
        else:
            values = [seeds] if isinstance(seeds, str) else list(seeds or [])
            if len(self.topics) not in {0, 1, len(values)}:
                raise ValueError("Use one topic hint, one per seed, or a topic-to-seeds mapping")
            hints = self.topics * len(values) if len(self.topics) == 1 else self.topics
            pairs = [
                (_article_title(value), hints[i] if hints else None)
                for i, value in enumerate(values)
            ]
        if not pairs or any(not _eligible_title(title) for title, _ in pairs):
            raise ValueError("Provide at least one eligible Wikipedia article seed")
        return list(dict.fromkeys(pairs))

    def _reset(self):
        self.document_collection = DocumentCollection()
        self.adjacency_lists: dict[str, list[str]] = {}
        self.topic_counts: dict[str, int] = {}
        self.crawl_log: list[dict] = []
        self._resolved_page_ids: dict[str, int | None] = {}

    def _request(self, **params) -> dict:
        parameters = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        for attempt in range(self.max_retries + 1):
            response = None
            remaining = self.request_delay - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
            self._last_request = time.monotonic()
            try:
                response = self.session.get(API_URL, params=parameters, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError) as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if attempt == self.max_retries or (status and status not in {429, 500, 502, 503, 504}):
                    raise WikipediaAPIError(str(error)) from error
            else:
                error = data.get("error")
                if not error:
                    return data
                if error.get("code") not in {"maxlag", "ratelimited", "readonly"} or attempt == self.max_retries:
                    raise WikipediaAPIError(f"{error.get('code')}: {error.get('info')}")
            delay = min(5 * 2 ** attempt, 30)
            retry_after = getattr(response, "headers", {}).get("Retry-After")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    try:
                        deadline = parsedate_to_datetime(retry_after)
                        delay = max(delay, (deadline - datetime.now(timezone.utc)).total_seconds())
                    except (TypeError, ValueError):
                        pass
            if delay > 60:
                raise WikipediaAPIError(f"Server requests a {delay:.0f}s cooldown; retry the crawl later")
            LOGGER.warning("API temporarily unavailable; retrying in %.1fs", delay)
            time.sleep(delay)
        raise WikipediaAPIError("Request retries exhausted")

    def _query_pages(self, titles: list[str], *, details=False) -> list[dict]:
        params = {"action": "query", "titles": "|".join(titles), "redirects": 1, "prop": "info"}
        if details:
            params.update(
                prop="info|pageprops|revisions|extracts|links", inprop="url",
                rvprop="ids|timestamp", explaintext=1, exsectionformat="plain",
                exlimit=1, plnamespace=0, pllimit="max",
            )
        data = self._request(**params)
        query = data.get("query", {})
        pages = query.get("pages", [])
        aliases = {
            _title_key(item["from"]): _title_key(item["to"])
            for kind in ("normalized", "converted", "redirects")
            for item in query.get(kind, [])
        }
        for page in pages:
            key = _title_key(page["title"])
            page_id = page.get("pageid") if not ("missing" in page or "invalid" in page) else None
            self._resolved_page_ids[key] = page_id
        for title in [*titles, *aliases]:
            key = _title_key(title)
            target, seen = key, set()
            while target in aliases and target not in seen:
                seen.add(target)
                target = aliases[target]
            self._resolved_page_ids[key] = self._resolved_page_ids.get(target)
        # A page may have more than 500 links. Follow the API's continuation
        # tokens rather than quietly retaining only its first batch.
        continuation = data.get("continue", {})
        page_by_id = {page["pageid"]: page for page in pages if "pageid" in page}
        while details and "plcontinue" in continuation:
            data = self._request(
                action="query", pageids="|".join(map(str, page_by_id)),
                prop="links", plnamespace=0, pllimit="max", **continuation,
            )
            for page in data.get("query", {}).get("pages", []):
                if page.get("pageid") in page_by_id:
                    page_by_id[page["pageid"]].setdefault("links", []).extend(page.get("links", []))
            continuation = data.get("continue", {})
        return pages

    def _fetch_document(self, title: str, topic_hint: str | None) -> Document | None:
        # A previously collected page may still be a useful bridge for another
        # seed group's BFS. Reuse its saved text/links rather than downloading it.
        known_id = self._resolved_page_ids.get(_title_key(title))
        for document in self.get_collected_documents():
            if document.page_id == known_id:
                return document
        pages = self._query_pages([title], details=True)
        if not pages:
            return None
        page = pages[0]
        if (
            "missing" in page or "invalid" in page or page.get("ns") != 0
            or not _eligible_title(page["title"])
            or "disambiguation" in page.get("pageprops", {})
            or not page.get("revisions")
        ):
            return None
        if page["pageid"] in self.document_collection._page_ids:
            return next(document for document in self.get_collected_documents()
                        if document.page_id == page["pageid"])
        text = page.get("extract", "").strip()
        if not text or len(text.split()) < self.min_words:
            return None
        preview = text.split("\n\n", 1)[0].strip()
        links = list(dict.fromkeys(
            _title_key(link["title"]) for link in page.get("links", [])
            if link.get("ns") == 0 and _eligible_title(link["title"])
        ))
        return Document(
            page_id=page["pageid"], title=page["title"],
            url=page.get("canonicalurl") or page.get("fullurl")
            or "https://en.wikipedia.org/wiki/" + quote(page["title"].replace(" ", "_")),
            doc_id=f"wiki_{page['pageid']}",
            collected_at=datetime.now(timezone.utc).isoformat(), preview=preview,
            text=text, revision_id=page["revisions"][0]["revid"], outgoing_links=links,
            discovered_topic=topic_hint,
        )

    def add_document(self, body: Document | dict) -> Document:
        """Accept a fetched Document or a record with Document field names."""
        document = body if isinstance(body, Document) else Document(**body)
        added = self.document_collection.add_document(document)
        if added and self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            filename = self.checkpoint_dir / f"{document.doc_id}.json"
            with filename.open("w", encoding="utf-8") as stream:
                json.dump(asdict(document), stream, ensure_ascii=False, indent=2)
        return document

    def crawl(self, depth: int = 2, max_pages_per_topic: int = 40) -> tuple[list[Document], dict[str, list[str]]]:
        """BFS per seed group, counting globally unique new pages per group.

        Seeds are depth 0. Five groups at the default limit produce at most
        200 candidates. Groups can return fewer if their frontier runs out.
        Seed provenance never sets the document's actual topic label.
        """
        if depth < 0 or max_pages_per_topic < 1:
            raise ValueError("depth must be nonnegative and max_pages_per_topic must be positive")
        self._reset()
        groups = {}
        for title, topic in self._seeds:
            groups.setdefault(topic, []).append(title)
        for topic_hint, seeds in groups.items():
            group_name = topic_hint or "unassigned"
            self.topic_counts[group_name] = 0
            queue = deque((title, 0) for title in seeds)
            queued = set(seeds)
            visited_pages = set()
            while queue and self.topic_counts[group_name] < max_pages_per_topic:
                title, current_depth = queue.popleft()
                try:
                    document = self._fetch_document(title, topic_hint)
                except WikipediaAPIError as error:
                    self.crawl_log.append({"title": title, "group": group_name, "status": "error", "reason": str(error)})
                    LOGGER.warning("Could not collect %s: %s", title, error)
                    continue
                if document is None:
                    self.crawl_log.append({"title": title, "group": group_name, "status": "skipped"})
                    continue
                if document.page_id in visited_pages:
                    continue
                visited_pages.add(document.page_id)
                is_new = document.page_id not in self.document_collection._page_ids
                if is_new:
                    self.add_document(document)
                    self.topic_counts[group_name] += 1
                    LOGGER.info("%s: %s (%s/%s new candidates)", group_name, document.title,
                                self.topic_counts[group_name], max_pages_per_topic)
                self.crawl_log.append({"title": document.title, "group": group_name,
                                       "status": "collected" if is_new else "already_collected",
                                       "depth": current_depth})
                if current_depth < depth:
                    links = document.outgoing_links
                    if self.max_links_per_page is not None and len(links) > self.max_links_per_page:
                        rng = random.Random(self.random_seed + document.page_id)
                        indices = sorted(rng.sample(range(len(links)), self.max_links_per_page))
                        links = [links[i] for i in indices]
                    for target in links:
                        if target not in queued:
                            queued.add(target)
                            queue.append((target, current_depth + 1))
            if self.topic_counts[group_name] < max_pages_per_topic:
                LOGGER.warning("%s: collected %s/%s new candidates; frontier exhausted",
                               group_name, self.topic_counts[group_name], max_pages_per_topic)
        documents = self.get_collected_documents()
        if not documents:
            raise RuntimeError("No articles collected. Check seeds, API access, minimum words, and crawl_log.")
        self._resolve_outgoing_links()
        self.build_adjacency_lists()
        return documents, self.adjacency_lists

    def _resolve_outgoing_links(self):
        # Resolve ALL saved links, even aliases not visited because of limits.
        titles = sorted({
            title for document in self.get_collected_documents()
            for title in document.outgoing_links
            if title not in self._resolved_page_ids
        })
        for start in range(0, len(titles), 50):
            # Do not silently return an incomplete graph on a resolution error.
            self._query_pages(titles[start:start + 50])

    def build_adjacency_lists(self) -> dict[str, list[str]]:
        """Match every saved hyperlink to collected documents, not BFS parents."""
        documents = self.get_collected_documents()
        page_to_doc = {document.page_id: document.doc_id for document in documents}
        for document in documents:
            self._resolved_page_ids[_title_key(document.title)] = document.page_id
        adjacency_lists = {}
        for document in documents:
            targets = set()
            for title in document.outgoing_links:
                target = page_to_doc.get(self._resolved_page_ids.get(_title_key(title)))
                if target is not None and target != document.doc_id:
                    targets.add(target)
            adjacency_lists[document.doc_id] = sorted(targets)
        self.adjacency_lists = adjacency_lists
        return adjacency_lists

    def get_collected_documents(self) -> list[Document]:
        return self.document_collection.get_documents()

    def save(self, output_dir: str | Path):
        """Save candidates and lists. No matrix is built before manual review."""
        documents = self.get_collected_documents()
        if not documents or set(self.adjacency_lists) != {doc.doc_id for doc in documents}:
            raise RuntimeError("Run crawl() or build_adjacency_lists() before saving")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "documents.jsonl").open("w", encoding="utf-8") as stream:
            for document in documents:
                stream.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
        with (output / "adjacency_lists.json").open("w", encoding="utf-8") as stream:
            json.dump(self.adjacency_lists, stream, ensure_ascii=False, indent=2)
        with (output / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump({
                "stage": "candidates_awaiting_review",
                "doc_ids": [document.doc_id for document in documents],
                "topics": list(TOPICS), "new_candidates_per_seed_group": self.topic_counts,
                "topic_labels_reviewed": False,
                "collection_method": "MediaWiki query: plain-text extracts and main-namespace links",
                "revision_pinned": False,
            }, stream, indent=2, ensure_ascii=False)
        with (output / "crawl_log.json").open("w", encoding="utf-8") as stream:
            json.dump(self.crawl_log, stream, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seeds-file", type=Path, help="JSON mapping each topic to its seed titles/URLs")
    seeds.add_argument("--seed", action="append", help="Wikipedia title/URL; repeat for a single group")
    parser.add_argument("--topic", help="Discovery hint only; topic labels still need manual review")
    parser.add_argument("--depth", type=int, default=2, help="Seeds are depth 0")
    parser.add_argument("--max-pages-per-topic", type=int, default=40)
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--max-links-per-page", type=int, default=30, help="Discovery limit; 0 follows all eligible links")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Descriptive User-Agent, preferably with your contact")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data" / "candidates")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.seeds_file:
        with args.seeds_file.open(encoding="utf-8") as stream:
            seed_urls = json.load(stream)
        if not isinstance(seed_urls, dict) or any(not values for values in seed_urls.values()):
            parser.error("Seeds file must map each group to a nonempty list of your seed URLs/titles")
    else:
        seed_urls = args.seed
    crawler = WebCrawler(
        args.topic, seed_urls, request_delay=args.delay, min_words=args.min_words,
        max_links_per_page=args.max_links_per_page or None, user_agent=args.user_agent,
        checkpoint_dir=args.output_dir / "snapshots",
    )
    documents, adjacency_lists = crawler.crawl(args.depth, args.max_pages_per_topic)
    crawler.save(args.output_dir)
    print(f"Candidates: {len(documents)}; internal links: {sum(map(len, adjacency_lists.values()))}")
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
