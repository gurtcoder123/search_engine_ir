"""Manually label/reject crawl candidates, then save a separate final corpus."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

if __package__:
    from .crawler import Document, TOPICS
else:
    from crawler import Document, TOPICS


def review_documents(documents, decisions_path, *, review_again=False):
    """Save each decision immediately. Return None when the user quits."""
    decisions_path = Path(decisions_path)
    decisions = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else {}
    decisions_path.parent.mkdir(parents=True, exist_ok=True)

    for position, document in enumerate(documents, start=1):
        previous = decisions.get(document.doc_id, {})
        if not review_again and previous.get("status") in {"accepted", "rejected"}:
            continue
        print(f"\n[{position}/{len(documents)}] {document.title} ({document.doc_id})")
        print(f"URL: {document.url}")
        print(f"Discovered from: {document.discovered_topic or 'unassigned'}")
        print(f"\n{document.preview}\n")
        if previous:
            print(f"Previous decision: {previous}")
        for number, topic in enumerate(TOPICS, start=1):
            print(f"  {number}. {topic.replace('_', ' ')}")
        print("  r. Reject   s. Skip/leave unchanged   v. View full text   q. Save and quit")

        while True:
            choice = input("Choose: ").strip().lower()
            if choice == "q":
                return None
            if choice == "v":
                print(f"\n{document.text}\n")
                continue
            if choice == "s":
                break
            if choice == "r":
                decisions[document.doc_id] = {"status": "rejected", "topic": None}
            elif choice in {str(i) for i in range(1, len(TOPICS) + 1)}:
                decisions[document.doc_id] = {"status": "accepted", "topic": TOPICS[int(choice) - 1]}
            else:
                print("Choose a topic number, r, s, v, or q.")
                continue
            decisions_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
            break
    return decisions


def finalize_documents(documents, adjacency_lists, decisions):
    """Copy accepted documents and remove rejected nodes AND incoming links."""
    selected = []
    for document in documents:
        decision = decisions.get(document.doc_id, {})
        if decision.get("status") not in {"accepted", "rejected"}:
            raise ValueError(f"Document {document.doc_id} still needs review")
        if decision["status"] == "accepted":
            if decision.get("topic") not in TOPICS:
                raise ValueError(f"Invalid topic for {document.doc_id}")
            selected.append(replace(document, topic=decision["topic"]))
    if not selected:
        raise ValueError("No documents were accepted; the final corpus would be empty")
    selected_ids = {document.doc_id for document in selected}
    filtered_lists = {
        document.doc_id: sorted({
            target for target in adjacency_lists.get(document.doc_id, [])
            if target in selected_ids and target != document.doc_id
        })
        for document in selected
    }
    return selected, filtered_lists


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent / "data" / "candidates")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data" / "reviewed")
    parser.add_argument("--review-again", action="store_true", help="Revisit previously accepted/rejected articles")
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Use a separate output directory so the raw candidates remain unchanged")
    with (args.input_dir / "documents.jsonl").open(encoding="utf-8") as stream:
        documents = [Document(**json.loads(line)) for line in stream if line.strip()]
    adjacency_lists = json.loads((args.input_dir / "adjacency_lists.json").read_text(encoding="utf-8"))
    try:
        decisions = review_documents(
            documents, args.input_dir / "review_decisions.json", review_again=args.review_again,
        )
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Previous decisions are saved; run again to resume.")
        return
    if decisions is None:
        print("Stopped. Previous decisions are saved; run again to resume.")
        return
    pending = sum(decisions.get(doc.doc_id, {}).get("status") not in {"accepted", "rejected"} for doc in documents)
    if pending:
        print(f"{pending} documents still need review. Run again to resume; final files were not changed.")
        return
    try:
        selected, filtered_lists = finalize_documents(documents, adjacency_lists, decisions)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for document in selected:
            stream.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
    (args.output_dir / "adjacency_lists.json").write_text(
        json.dumps(filtered_lists, indent=2), encoding="utf-8",
    )
    counts = {topic: sum(doc.topic == topic for doc in selected) for topic in TOPICS}
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "stage": "reviewed", "doc_ids": [doc.doc_id for doc in selected],
        "topics": list(TOPICS), "documents_per_topic": counts,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {len(selected)} accepted documents to {args.output_dir.resolve()}")
    for topic, count in counts.items():
        print(f"  {topic}: {count}")
    print("Next: run graph.py to build the final matrix. Rebuild indexes/ranks after any later review changes.")


if __name__ == "__main__":
    main()
