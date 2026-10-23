# Wikipedia Search Engine — Group 17

Information Retrieval course project: implement Topic-Sensitive PageRank from
first principles and evaluate how a synthetic spam farm affects retrieval.
Core numerical work uses NumPy and SciPy; NetworkX and Scikit-Learn are not used.

## Current implementation

- `project/config.py`: the topic list, document-ID format and data layout.
- `project/crawler.py`: MediaWiki JSON API collection and adjacency lists.
- `project/reviewer.py`: console-based acceptance, rejection and manual topic labels.
- `project/graph.py`: sparse adjacency matrix and a loader that checks it matches the reviewed corpus.
- `project/tests/`: offline API fixtures and review-to-graph checks.

Preprocessing, TF-IDF indexing, query processing, PageRank/TSPR and attack
evaluation are still upcoming. This repository does not yet contain a complete
search engine or measured retrieval results.

The intended data flow is:

```text
Seed titles grouped by topic
    -> BFS collection: documents + adjacency lists
    -> manual review and topic assignment
    -> final corpus + filtered adjacency lists
    -> sparse graph and text indexes
    -> ranking, retrieval and attack evaluation
```

See [the implementation guide](project/README.md) for the commands supported
by the current code.

For the partner implementing query retrieval and PageRank/TSPR, read
[PARTNER_HANDOFF.md](PARTNER_HANDOFF.md) for the agreed inputs, data structures,
preprocessing and weighting rules, graph conventions, and pending dependencies.

## Shared document classes

The current definitions are in `project/crawler.py` and are also imported by
the reviewer and graph scripts.

### Document

One record represents one article:

| Field | Type | Meaning |
| --- | --- | --- |
| `page_id` | `int` | Wikipedia's page identifier |
| `doc_id` | `str` | Our stable, opaque ID, `doc-<page_id>`; never a row index |
| `title` | `str` | Canonical title returned by the API |
| `url` | `str` | Link to the article |
| `collected_at` | `str` | UTC collection timestamp |
| `revision_id` | `int` or `None` | Revision metadata recorded at collection time |
| `text` | `str` | Full plain text returned by the API |
| `preview` | `str` | First paragraph used for review/results |
| `outgoing_links` | `list[str]` | Outgoing article titles returned by the API |
| `topic` | `str` or `None` | Actual manual label; always `None` for new candidates |
| `discovered_topic` | `str` or `None` | Seed-group provenance, not a manual label |

### DocumentCollection

Stores the shared `documents: list[Document]`, keyed by Wikipedia page ID for
deduplication. `add_document(document)` inserts a unique page and returns
`False` if that page is already held; `get_documents()` returns the stored
list; `get_by_page_id(page_id)` looks one up.

### WebCrawler

Collects article records and preserves their outgoing links. It takes the
shared collection as a constructor argument and exposes `crawl()`, described
below.

## How collection works

`WebCrawler` receives the collection in its constructor and fills it in place,
so every component refers to the same object:

```python
collection = DocumentCollection()
crawler = WebCrawler(
    document_collection=collection,
    topics=["technology", "science"],
    seed_topics=[["your technology seed"], ["your science seed"]],
    depth=2,
    max_pages_per_topic=40,
)
documents, adjacency_lists = crawler.crawl()
```

`topics` and `seed_topics` are parallel lists, one seed list per topic. The CLI
builds them from a seeds file keyed by the five topic names.

`crawl()`:

1. Takes one seed group at a time and starts a BFS queue with its titles.
2. Requests plain article text, metadata and outgoing titles from the API,
   following link continuation tokens so no batch is lost.
3. Checks Wikipedia page IDs so the same article is never stored twice.
4. Builds documents with `topic=None`, recording the seed group separately in
   `discovered_topic`, and appends them to the supplied collection.
5. Enqueues outgoing titles while below the depth limit. Every eligible link is
   recorded whether or not BFS follows it.
6. Stops a group at 40 new unique candidates, or when its queue is exhausted.
7. Resolves every saved link against the collected documents and returns the
   documents plus adjacency lists. The sparse matrix waits for manual review.

The crawler keeps the supplied collection rather than replacing or clearing it,
so use a fresh collection for a separate experiment. A per-group frontier avoids
BFS cycles while the collection's page IDs prevent duplicates across groups. A
page an earlier group already stored does not fill a second group's quota, but
its outgoing links are still explored, so it can bridge two groups.

Forty is a ceiling, not a guarantee: a shallow depth or a small
`--max-links-per-page` can exhaust a group's frontier first. The crawler reports
what each group actually collected.

## Shared graph conventions

- An adjacency list maps a source `doc_id` to destination `doc_id`s.
- Duplicate edges and self-links are excluded; a dangling page has an empty list.
- Rejecting an article removes both its list and references to it in other lists.
- The final matrix is SciPy CSR. Entry `[i, j]` is 1 if document i links to j.
- Matrix rows/columns follow the final `documents` list. The same ordering must
  be used for document TF-IDF vectors and PageRank/TSPR arrays.
- Although the code calls it an incidence matrix, it is mathematically a
  directed, binary adjacency matrix.

## Install and check

Run from the repository root with Python 3.10 or newer:

```powershell
python -m pip install -r project/requirements.txt
python -m pytest project/tests -q -p no:cacheprovider --basetemp=tmp/project_tests
```

Install `pytest` separately to run the tests. Crawling uses the public MediaWiki
Action API; keep request delays and follow API pagination to retain all links.

## Local files

`.gitignore` excludes local `tmp/`, `output/`, downloaded data, Python caches,
Word/PDF documents, and proposal-generation utilities. These remain available
locally and are not part of the code repository.
