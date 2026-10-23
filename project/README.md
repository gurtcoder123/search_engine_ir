# Wikipedia collection, review and graph

The workflow has three simple scripts:

1. `crawler.py` gathers candidates and their adjacency lists.
2. `reviewer.py` lets you accept/reject articles and assign their actual topics.
3. `graph.py` builds a sparse matrix from the reviewed corpus.

Run the following from the workspace root. Python 3.10+ is required.

## 1. Provide your seeds and crawl

Install the dependencies and copy the empty seeds template:

```powershell
python -m pip install -r project/requirements.txt
Copy-Item project/seeds.example.json project/seeds.json
```

Edit `project/seeds.json` and put plain Wikipedia article titles (not URLs) in each topic's
list. You can provide several starting points per topic. All five lists in
the template must be filled before using it.

```powershell
python project/crawler.py --seeds-file project/seeds.json --depth 2 --max-pages-per-topic 40
```

Depth defaults to 2; seeds are depth 0. The limit defaults to 40 **new,
globally unique articles per seed group**, or at most 200 across five groups.
Groups can collect fewer if the frontier runs out. Repeated pages are stored
once and do not fill another group's quota. An already-collected page can
still be followed as a bridge when exploring a later group's seeds.

Every document has `topic=None`. The seed group is recorded separately in
`discovered_topic` and is only a suggestion, not a topic assignment.

For a small check, write a seeds file with one topic and crawl into a scratch
directory:

```powershell
python project/crawler.py --seeds-file tmp/one_seed.json --depth 1 --max-pages-per-topic 3 --output-dir tmp/crawl_check
```

Seed-file keys must be the five topic names from `config.py`. The crawler
refuses to write into a directory that already holds a crawl, so nothing is
overwritten; point `--output-dir` somewhere new instead.

Use `--user-agent` to provide a descriptive identifier including your contact.
Requests are sequential, with a configurable delay and bounded retries that
respect server backoff. A long server cooldown or exhausted rate-limit/load
retries stops the whole crawl, rather than skipping ahead to another page.
`--max-links-per-page` defaults to 30 and only limits
which links BFS follows, not which outgoing links are recorded. Set it to 0
to follow all eligible links.

The crawler takes a shared `DocumentCollection` and fills it in place:

```python
from project.crawler import DocumentCollection, WebCrawler

collection = DocumentCollection()
crawler = WebCrawler(
    document_collection=collection,
    topics=["technology", "science"],
    seed_topics=[["your first seed", "your second seed"], ["your science seed"]],
    depth=2,
    max_pages_per_topic=40,
)
documents, adjacency_lists = crawler.crawl()
crawler.save("project/data/candidates")
```

`topics` and `seed_topics` are parallel lists: one list of seed titles per
topic. The collection is never cleared for you, so use a fresh one per
experiment. Optional settings (`depth`, `request_delay`, `min_words`, ...)
carry ordinary defaults, so passing `depth=0` means exactly that.

`documents` is a list of Document objects, and the same list the collection
holds. Each includes the full article text, first-paragraph preview, title,
URL, revision ID and raw outgoing titles.

`adjacency_lists` maps source document IDs to destination document IDs:

```python
{
    "doc-101": ["doc-202", "doc-303"],
    "doc-202": [],
    "doc-303": ["doc-101"],
}
```

The lists include all API-reported eligible links whose destinations are
collected, even links BFS did not follow. Uncollected destinations do not
become graph nodes. Redirect aliases are resolved before producing these
lists. There is no sparse matrix at the crawling stage.

By default, outputs go to `project/data/candidates/`:

- `documents.jsonl`: complete candidate records.
- `adjacency_lists.json`: unique internal document-to-document links.
- `manifest.json`: candidate order, seed-group counts, crawl settings, review
  status and a `corpus_build_id` fingerprint of this candidate set.
- `crawl_log.json`: the pages that were skipped, and why.

The crawler reads structured JSON fields directly: `extract` is plain article
text (`explaintext=1`), `links` contains outgoing titles, and the other fields
provide page metadata. It follows link continuation tokens to collect every
batch. There is no BeautifulSoup, HTML parsing or `action=parse` request.
The first paragraph of the plain text becomes the preview.

Links are the API's main-namespace page links, which can include links supplied
by templates/navigation; we no longer apply a custom article-body HTML filter.
`revision_id` records the revision metadata returned at collection time, but
this simpler current-page API mode is not an exact revision-pinned fetch.
The saved local text and links are the frozen experimental snapshot.
Tokenization and stemming for retrieval happen after corpus review. A page that
cannot be fetched for an ordinary page-specific error is recorded in
`crawl_log.json` and skipped. Server cooldowns stop the run; partial documents
remain in the supplied in-memory collection, but no completed export is claimed.
Use a fresh output directory for separate experiments.

## 2. Review and label

```powershell
python project/reviewer.py
```

Each article shows its title, preview, URL and seed provenance. Choose:

- 1-5: accept and assign one of the five topics.
- r: reject.
- s: skip for later, or leave an existing decision unchanged.
- v: show the saved full article text.
- q: quit; decisions already made remain saved.

Decisions are written immediately to
`project/data/candidates/review_decisions.json`. Run the command again to
resume. Use `--review-again` to revisit earlier decisions, including rejections.

Only after all articles have an accept/reject decision does the script export
the reviewed corpus to `project/data/reviewed/`. Skipped articles remain
pending. The original candidate documents and lists are not overwritten.

Final outputs are `documents.jsonl`, `adjacency_lists.json` and
`manifest.json`. Rejected documents are removed as sources **and** destinations.
Accepted documents keep their original document IDs and order; pages with no
remaining links keep empty lists. Topic counts are displayed so you can aim
for a balanced final corpus (approximately 20 articles per topic).
The reviewer warns if fewer than 100 articles are accepted or any topic is empty.
The reviewed manifest contains a content fingerprint covering the complete
document records, order and internal links. Re-review that changes the corpus
marks an existing graph manifest stale without deleting the old matrix.

For other directories, use `--input-dir` and `--output-dir`.
Programmatically, `finalize_documents(documents, adjacency_lists, decisions)`
returns the accepted documents and filtered lists without modifying the raw
objects.

## 3. Build the final matrix

```powershell
python project/graph.py
```

This reads `project/data/reviewed/` and saves:

- `incidence_matrix.npz`: SciPy CSR matrix.
- `graph_manifest.json`: final `doc_ids`, `doc_to_row`, content fingerprint and stale status.

The project calls it an incidence matrix; mathematically it is a directed,
binary adjacency matrix. Entry [i, j] is 1 when final document i links to
final document j. Self-links, duplicate edges and external destinations are
excluded. Isolated/dangling pages remain present. The script prints edge,
dangling-page and isolated-page counts for a quick connectivity check.

```python
from project.graph import build_incidence_matrix

incidence_matrix = build_incidence_matrix(final_documents, final_adjacency_lists)
doc_ids = [document.doc_id for document in final_documents]
doc_to_row = {doc_id: row for row, doc_id in enumerate(doc_ids)}
```

Build the text vocabulary, IDF and TF-IDF matrix from this same reviewed
document collection and order. If review decisions change later, rerun graph
construction and rebuild the text indexes/rank vectors too.

Load a saved graph through the checked loader:

```python
from project.graph import load_graph

adjacency_matrix = load_graph("project/data/reviewed")
```

It rejects stale graphs, changed records/links, incompatible row mappings,
wrong matrix dimensions, and old manifests without a content fingerprint.
Run `graph.py` again to regenerate a current graph. Bare `scipy.sparse.load_npz`
does not perform these corpus checks. The full text-index loader is still pending.

## Offline tests

With pytest installed:

```powershell
python -m pytest project/tests -q -p no:cacheprovider --basetemp=tmp/review_workflow_tests
```

These tests do not access Wikipedia. They cover per-group quotas, shared-page
deduplication, full outgoing links despite discovery limits, redirect handling,
review/resume, rejection filtering, and the final matrix/save workflow.
