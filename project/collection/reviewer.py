"""Review articles without deleting records. approved=True selects the corpus."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import PREPROCESSING_DIR, TOPICS
from project.collection.crawler import Document, normalize_title


def review_documents(data_dir=PREPROCESSING_DIR, *, review_again=False):
    """Update the in-memory list, article file and collection after each choice.

    Incoming links are title matches from the saved collection, not global
    Wikipedia backlinks. Uncollected/unresolved redirect sources are unknown.
    """
    data_dir = Path(data_dir).resolve()
    collection_file = data_dir / "Document_Collection.json"
    records = json.loads(collection_file.read_text(encoding="utf-8"))
    documents = [Document(**record) for record in records]

    for position, document in enumerate(documents, start=1):
        if document.approved is not None and not review_again:
            continue
        incoming = [other.title for other in documents
                    if other.page_id != document.page_id
                    and normalize_title(document.title) in {
                        normalize_title(link) for link in other.outgoing_links}]
        print(f"\n[{position}/{len(documents)}] {document.title}\n{document.url}")
        print(f"\n{document.preview}\n")
        print(f"Incoming from saved pages ({len(incoming)}): {', '.join(incoming) or 'None'}")
        print(f"Outgoing ({len(document.outgoing_links)}; first 20): "
              f"{', '.join(document.outgoing_links[:20]) or 'None'}")
        print(f"Current decision: approved={document.approved}, topic={document.topic}")
        for number, topic in enumerate(TOPICS, start=1):
            print(f"  {number}. {topic.replace('_', ' ')}")
        print("  r. Reject   s. Skip   v. View saved text   l. All outgoing links   q. Quit")

        while True:
            choice = input("Choose: ").strip().lower()
            if choice == "q":
                return documents
            if choice == "s":
                break
            if choice == "v":
                print(document.text)
                continue
            if choice == "l":
                print("\n".join(document.outgoing_links) or "None")
                continue
            if choice == "r":
                approved, topic = False, None
            elif choice in {str(i) for i in range(1, len(TOPICS) + 1)}:
                approved, topic = True, TOPICS[int(choice) - 1]
            else:
                print("Choose a topic number, r, s, v, l, or q.")
                continue

            source = (data_dir / (document.file_path or f"Documents/{document.doc_id}.json")).resolve()
            destination = data_dir / "Documents"
            if approved:
                destination = destination / topic
            destination = (destination / f"{document.doc_id}.json").resolve()
            if not source.is_relative_to(data_dir) or not destination.is_relative_to(data_dir):
                raise ValueError("Article paths must stay inside the collection directory")
            if not source.is_file():
                raise FileNotFoundError(f"Article file missing: {source}")
            if source != destination and destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            document.approved, document.topic = approved, topic
            document.file_path = destination.relative_to(data_dir).as_posix()
            source.write_text(json.dumps(asdict(document), ensure_ascii=False, indent=2), encoding="utf-8")
            if source != destination:
                source.rename(destination)
            collection_file.write_text(json.dumps(
                [asdict(d) for d in documents], ensure_ascii=False, indent=2,
            ), encoding="utf-8")
            break
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PREPROCESSING_DIR)
    parser.add_argument("--review-again", action="store_true")
    args = parser.parse_args()
    try:
        documents = review_documents(args.data_dir, review_again=args.review_again)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped; completed decisions are saved. Run again to resume.")
        return
    approved = [d for d in documents if d.approved is True]
    print(f"\nApproved: {len(approved)}; rejected: {sum(d.approved is False for d in documents)}; "
          f"unreviewed: {sum(d.approved is None for d in documents)}")
    for topic in TOPICS:
        print(f"  {topic}: {sum(d.topic == topic for d in approved)}")
    if len(approved) < 100:
        print("The final project corpus needs at least 100 approved articles.")


if __name__ == "__main__":
    main()
