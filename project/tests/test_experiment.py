"""Real rebuilds on a small corpus verify the attack control and preservation."""

from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
from scipy.sparse import load_npz

from project.collection.crawler import Document
from project.config import TOPICS
from project.experiment import run_experiment


def test_attack_changes_only_control_links_and_keeps_source_untouched(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    words = ["computer chip", "atom physics", "doctor medicine", "football sport", "election vote"]
    records = [asdict(Document(i, f"Article {i}", f"https://example.org/{i}", f"doc-{i}",
                               "now", text, text=text, topic=topic, approved=True,
                               outgoing_links=[f"Article {(i + 1) % 5}"]))
               for i, (topic, text) in enumerate(zip(TOPICS, words))]
    records.append(asdict(Document(5, "Relevant computer article", "https://example.org/5", "doc-5",
                                   "now", "computer algorithm", text="computer algorithm",
                                   topic="technology", approved=True)))
    path = source / "Document_Collection.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    before = path.read_bytes()
    qrels = tmp_path / "qrels.json"
    qrels.write_text(json.dumps({"provenance": "Synthetic fixture, not benchmark results.",
                                "queries": [{"id": "q1", "query": "computer", "relevant": ["doc-5"]}]}),
                     encoding="utf-8")
    output = tmp_path / "experiment"
    result = run_experiment(source, qrels, output, "doc-0", farm_size=2, repeats=1)
    assert path.read_bytes() == before
    assert not (source / "Index").exists()
    assert result["source_collection_unchanged"]
    assert result["source_corpus_sha256"] == hashlib.sha256(before).hexdigest()
    clean, control, attacked = (result["conditions"][name] for name in ("clean", "control", "attacked"))
    assert clean["documents"] == 6 and control["documents"] == attacked["documents"] == 8
    assert attacked["internal_links"] == control["internal_links"] + 2
    assert attacked["topic_target_score"] > control["topic_target_score"]
    control_records = json.loads((output / "control/Document_Collection.json").read_text())
    attack_records = json.loads((output / "attacked/Document_Collection.json").read_text())
    assert control_records[:6] == attack_records[:6]
    for a, b in zip(control_records[6:], attack_records[6:]):
        assert a["outgoing_links"] == [] and b["outgoing_links"] == ["Article 0"]
        assert {key: value for key, value in a.items() if key != "outgoing_links"} == {
            key: value for key, value in b.items() if key != "outgoing_links"}
    for filename in ("idf.npy", "topic_log_probs.npy"):
        np.testing.assert_array_equal(np.load(output / "control/Index" / filename),
                                      np.load(output / "attacked/Index" / filename))
    assert (load_npz(output / "control/Index/tfidf_matrix.npz") !=
            load_npz(output / "attacked/Index/tfidf_matrix.npz")).nnz == 0
    assert (output / "judgments.json").read_bytes() == qrels.read_bytes()
    assert set(attacked["systems"]) == {"tfidf", "pagerank", "tspr", "hybrid"}
    assert result["changes"]["control_to_attacked"]["tfidf"]["macro_f1"] == 0
    assert len(attacked["weight_sweep"]) == 7
    assert not attacked["target_positions"]["tspr"][0]["relevant"]


def test_experiment_refuses_overlapping_source_and_output(tmp_path):
    with pytest.raises(ValueError, match="separate directory"):
        run_experiment(tmp_path, tmp_path / "unused.json", tmp_path / "attack", "a")
