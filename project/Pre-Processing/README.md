# Saved Wikipedia corpus and preprocessing artifacts

This directory contains the dataset used by Group 17's search engine and
reference evaluation. It is versioned so a clone can run search immediately
after installing project/requirements.txt.

- Document_Collection.json: authoritative 200-record review store, with 162
  approved and 38 rejected articles. Only approved records enter retrieval.
- Documents/: saved article JSONs, including text extracts, article URLs,
  Wikipedia page/revision IDs, outgoing link titles and review decisions.
- Approved_Contents.json: approved titles and document IDs grouped by topic.
- Tokens/: normalized token lists for approved documents.
- Index/: postings, raw term counts, DF/IDF, sparse TF-IDF, vocabulary/document
  mappings, topic model and converged global/topic PageRank vectors.

Article text retains the licensing terms stated on its original Wikipedia
page. The saved URL and revision ID identify the source for attribution and
revision lookup. See https://en.wikipedia.org/wiki/Wikipedia:Copyrights.
Project code does not change the source article's authorship or license.

Synthetic spam pages are created by project.experiment in output/attack/;
they are not mixed into this original corpus. Rebuild all derived artifacts
with python -m project.preprocess after changing saved text, labels or links.

Raw bytes are preserved by .gitattributes so source hashes remain stable
across Windows and Unix checkouts. No live download is needed to reproduce
the saved-corpus evaluation.
