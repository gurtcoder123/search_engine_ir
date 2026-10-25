"""Build an L2-normalized document-term CSR matrix from the saved raw counts."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix, save_npz

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import PREPROCESSING_DIR


def build_tfidf_matrix(term_counts, df, doc_ids):
    """Rows follow doc_ids; columns follow a sorted vocabulary.

    TF = 1 + log10(count), IDF = log10(N / DF). Normalize each document
    before adding its nonzero weights to the sparse matrix. Zero rows stay zero.
    """
    terms = sorted(df)
    term_to_col = {term: column for column, term in enumerate(terms)}
    df_vector = np.array([df[term] for term in terms], dtype=np.int64)
    if len(terms):
        idf = np.log10(len(doc_ids) / df_vector)
    else:
        idf = np.array([], dtype=float)

    rows, columns, values = [], [], []
    for row, doc_id in enumerate(doc_ids):
        weights = []
        norm_squared = 0.0
        for term, count in term_counts[doc_id].items():
            column = term_to_col[term]
            weight = (1 + np.log10(count)) * idf[column]
            if weight != 0:
                weights.append((column, weight))
                norm_squared += weight * weight

        norm = np.sqrt(norm_squared)
        if norm > 0:
            for column, weight in weights:
                rows.append(row)
                columns.append(column)
                values.append(weight / norm)

    matrix = csr_matrix((values, (rows, columns)),
                        shape=(len(doc_ids), len(terms)), dtype=float)
    return matrix, term_to_col, df_vector, idf


def save_tfidf_matrix(matrix, term_to_col, df_vector, idf, doc_ids, output_dir):
    """Save the sparse matrix, numerical weights and their shared ordering."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_dir / "tfidf_matrix.npz", matrix)
    np.save(output_dir / "df.npy", df_vector)
    np.save(output_dir / "idf.npy", idf)
    mappings = {
        "term_to_col.json": term_to_col,
        "doc_ids.json": doc_ids,
        "doc_to_row.json": {doc_id: row for row, doc_id in enumerate(doc_ids)},
    }
    for filename, contents in mappings.items():
        (output_dir / filename).write_text(
            json.dumps(contents, ensure_ascii=False, indent=2), encoding="utf-8",
        )


if __name__ == "__main__":
    index_dir = PREPROCESSING_DIR / "Index"
    term_counts = json.loads((index_dir / "term_counts.json").read_text(encoding="utf-8"))
    df = json.loads((index_dir / "df.json").read_text(encoding="utf-8"))
    doc_ids = json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8"))
    matrix, term_to_col, df_vector, idf = build_tfidf_matrix(term_counts, df, doc_ids)
    save_tfidf_matrix(matrix, term_to_col, df_vector, idf, doc_ids, index_dir)
    print(f"Saved L2-normalized CSR matrix: {matrix.shape}, {matrix.nnz} nonzero weights.")
    print(f"Saved vocabulary, DF/IDF and row mappings to {index_dir}")
