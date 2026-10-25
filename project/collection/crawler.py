"""Small Wikipedia collector: request(), crawl(), reset() and save().

    collection = DocumentCollection()
    crawler = WebCrawler(collection, ["technology"], [["Computer"]])
    crawler.crawl()
    crawler.save()

No graph construction or automatic topic labelling.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import random
import re
import time

import requests

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import PREPROCESSING_DIR, TOPICS, USER_AGENT

LOGGER = logging.getLogger(__name__)
API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = USER_AGENT


def normalize_title(title):
    """Wikipedia treats underscores as spaces and capitalizes the first letter."""
    title = " ".join(title.replace("_", " ").split())
    return title[:1].upper() + title[1:]


@dataclass
class Document:
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
    approved: bool | None = None  # None: unreviewed; True: accepted; False: rejected
    file_path: str = ""          # Relative to the collection's folder


class DocumentCollection:
    def __init__(self):
        self.documents: list[Document] = []
        self._page_ids: set[int] = set()

    def add_document(self, document):
        if document.page_id in self._page_ids:
            return False
        self._page_ids.add(document.page_id)
        self.documents.append(document)
        return True

    def get_documents(self):
        return self.documents

    def clear(self):
        self.documents.clear()
        self._page_ids.clear()


class WebCrawler:
    def __init__(self, document_collection, topics, seed_topics, *, depth=3,
                 max_pages_per_topic=40, max_links_per_page=5, min_words=100,
                 max_paragraphs=3, max_chars=None, request_delay=1.0, timeout=30,
                 random_seed=17, user_agent=DEFAULT_USER_AGENT):
        """Assume valid inputs; seed_topics is a list of seed-title lists."""
        self.document_collection = document_collection
        self.topics, self.seed_topics = list(topics), seed_topics
        self.depth, self.max_pages_per_topic = depth, max_pages_per_topic
        self.max_links_per_page, self.min_words = max_links_per_page, min_words
        self.max_paragraphs, self.max_chars = max_paragraphs, max_chars
        self.request_delay, self.timeout = request_delay, timeout
        self.random_seed = random_seed
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.counts = {}

    def request(self, title):
        """Fetch the article and its link batches; skip on any request failure.

        There are no retries. A failed link batch skips the whole article so
        we do not save a misleading partial outgoing-link list.
        """
        params = dict(action="query", format="json", formatversion=2, maxlag=5,
                      titles=title, redirects=1, prop="extracts|links|info|pageprops",
                      explaintext=1, pllimit="max", plnamespace=0, inprop="url")
        LOGGER.info("Fetching: %s", title)
        page = None
        while True:
            if self.request_delay:
                time.sleep(self.request_delay)
            try:
                response = self.session.get(API_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError) as error:
                LOGGER.warning("Skipping %s: %s", title, error)
                return None
            if data.get("error"):
                LOGGER.warning("Skipping %s: %s", title, data["error"])
                return None
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                LOGGER.info("Skipping %s: no page returned", title)
                return None
            if page is None:
                page = dict(pages[0])
                if ("missing" in page or "invalid" in page or page.get("ns") != 0
                        or "disambiguation" in page.get("pageprops", {})):
                    LOGGER.info("Skipping %s: missing, non-article or disambiguation page", title)
                    return None
                page["links"] = list(page.get("links", []))
            else:
                page["links"].extend(pages[0].get("links", []))
            continuation = data.get("continue", {})
            if "plcontinue" not in continuation:
                return page
            LOGGER.info("Fetching more links: %s", title)
            params = dict(action="query", format="json", formatversion=2, maxlag=5,
                          pageids=page["pageid"], prop="links", plnamespace=0,
                          pllimit="max", **continuation)

    def crawl(self):
        """BFS each seed group, returning only the shared document list.

        Already-visited pages are not expanded again. All their outgoing titles
        remain saved, including links back to other collected pages.
        """
        visited = {normalize_title(d.title) for d in self.document_collection.documents}
        chooser = random.Random(self.random_seed)
        self.counts = {}
        for topic, seeds in zip(self.topics, self.seed_topics):
            LOGGER.info("Starting seed group: %s", topic)
            queue = deque((normalize_title(s), 0) for s in seeds)
            queued = {title for title, _ in queue}
            collected = 0
            while queue and collected < self.max_pages_per_topic:
                title, level = queue.popleft()
                if title in visited:
                    continue
                visited.add(title)
                page = self.request(title)
                if page is None:
                    continue
                visited.add(normalize_title(page["title"]))
                if page["pageid"] in self.document_collection._page_ids:
                    LOGGER.info("Skipping %s: already collected", page["title"])
                    continue
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page.get("extract", "")) if p.strip()]
                if self.max_paragraphs is not None:
                    paragraphs = paragraphs[:self.max_paragraphs]
                text = "\n\n".join(paragraphs)
                if self.max_chars is not None:
                    text = text[:self.max_chars].rstrip()
                word_count = len(text.split())
                if not text or word_count < self.min_words:
                    LOGGER.info("Skipping %s: insufficient text (%d words; minimum %d)",
                                page["title"], word_count, self.min_words)
                    continue
                links = list(dict.fromkeys(
                    normalize_title(link["title"]) for link in page.get("links", [])
                    if link.get("ns") == 0
                ))
                doc = Document(
                    page_id=page["pageid"], title=page["title"], url=page.get("fullurl", ""),
                    doc_id=f"doc-{page['pageid']}", collected_at=datetime.now(timezone.utc).isoformat(),
                    preview=text, text=text, revision_id=page.get("lastrevid"), outgoing_links=links,
                )
                self.document_collection.add_document(doc)
                collected += 1
                LOGGER.info("Collected: %s - %s %d/%d", doc.title, topic,
                            collected, self.max_pages_per_topic)
                if level < self.depth:
                    candidates = [link for link in links if link not in visited and link not in queued]
                    if self.max_links_per_page is not None and len(candidates) > self.max_links_per_page:
                        candidates = chooser.sample(candidates, self.max_links_per_page)
                    for link in candidates:
                        queued.add(link)
                        queue.append((link, level + 1))
            self.counts[topic] = collected
            LOGGER.info("%s: collected %d/%d new pages", topic, collected, self.max_pages_per_topic)
            if collected < self.max_pages_per_topic:
                LOGGER.warning("%s: frontier exhausted before reaching the cap", topic)
        return self.document_collection.get_documents()

    def reset(self):
        """Clear only the shared in-memory collection. Saved files are untouched."""
        self.document_collection.clear()
        self.counts.clear()

    def save(self, output_dir=PREPROCESSING_DIR):
        """One article JSON per page and one authoritative collection array.

        Use a fresh directory for a separate crawl; never overwrite a review.
        """
        output_dir = Path(output_dir)
        collection_file = output_dir / "Document_Collection.json"
        if collection_file.exists():
            raise FileExistsError("A collection already exists here; use a fresh output directory")
        documents_dir = output_dir / "Documents"
        LOGGER.info("Saving %d documents to %s", len(self.document_collection.documents), output_dir)
        documents_dir.mkdir(parents=True, exist_ok=True)
        for document in self.document_collection.documents:
            document.file_path = f"Documents/{document.doc_id}.json"
            path = output_dir / document.file_path
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            path.write_text(json.dumps(asdict(document), ensure_ascii=False, indent=2), encoding="utf-8")
        collection_file.write_text(json.dumps(
            [asdict(d) for d in self.document_collection.documents], ensure_ascii=False, indent=2,
        ), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    seeds = json.loads(Path(__file__).with_name("seeds.json").read_text(encoding="utf-8"))
    collection = DocumentCollection()
    crawler = WebCrawler(collection, list(seeds), list(seeds.values()))
    crawler.crawl()
    crawler.save()
    print(f"Saved {len(collection.documents)} candidates. Run reviewer.py to approve and label them.")
