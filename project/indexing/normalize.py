"""Simple tokenisation and saved token lists, not indexing.

Hyphenated words/numbers stay together. Dotted initials lose their dots
(U.S.A. -> usa), but are not expanded. Plain English words use Porter stemming.
"""

import json
from pathlib import Path
import sys
import re
import unicodedata

from nltk.stem import PorterStemmer

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import PREPROCESSING_DIR

STOP_WORDS = set(Path(__file__).with_name("stopwords.txt").read_text(encoding="utf-8").split())
TOKENS_DIR = PREPROCESSING_DIR / "Tokens"
STEMMER = PorterStemmer(mode=PorterStemmer.NLTK_EXTENSIONS)


def normalize_word(word):
    """Lowercase and stem English words; leave numbers/compounds intact."""
    word = unicodedata.normalize("NFKC", word).lower()
    if word.isascii() and word.isalpha():
        return STEMMER.stem(word)
    return word


def tokenize(text):
    """Return ordered tokens with repetitions; documents and queries share this."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = re.sub(r"(?<=\w)'s\b", "", text)  # computer's -> computer
    text = re.sub(r"\b(?:[a-z]\.){2,}(?:[a-z]\b)?",
                  lambda match: match.group().replace(".", ""), text)
    text = text.replace("'", "")  # don't -> dont, teachers' -> teachers
    words = re.findall(r"[^\W_]+(?:-[^\W_]+)*", text)
    tokens = []
    for word in words:
        if word not in STOP_WORDS:
            tokens.append(normalize_word(word))
    return tokens


def load_document(documentcollection, output_dir=TOKENS_DIR):
    """Tokenize/save approved documents only. Return doc_id -> token list."""
    tokens_by_document = {}
    for document in documentcollection.get_documents():
        if document.approved is not True:
            continue
        tokens = tokenize(document.text)
        save_document(document, tokens, output_dir)
        tokens_by_document[document.doc_id] = tokens
    return tokens_by_document


def save_document(document, tokens, output_dir=TOKENS_DIR):
    """Save tokens separately, leaving original articles and review untouched."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{document.doc_id}.json"
    path.write_text(json.dumps({"doc_id": document.doc_id, "tokens": tokens},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    from project.collection.crawler import Document, DocumentCollection

    records = json.loads((PREPROCESSING_DIR / "Document_Collection.json").read_text(encoding="utf-8"))
    collection = DocumentCollection()
    for record in records:
        collection.add_document(Document(**record))
    result = load_document(collection)
    print(f"Saved tokens for {len(result)} approved documents to {TOKENS_DIR}")
