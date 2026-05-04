#!/usr/bin/env python3
"""
Inspect processed AMBIFC files before training.

Run:
    python src/data/inspect_processed_ambifc.py

This checks:
1. train/dev/test row counts
2. category counts: certain vs uncertain
3. hard-label counts
4. soft-label validity
5. evidence-selection file counts
6. retrieval corpus count
7. one readable sample from each split
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROCESSED_DIR = Path("data/processed")
SPLITS = ["train", "dev", "test"]
LABEL_NAMES = ["refuting", "neutral", "supporting"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def soft_label_is_valid(values: Any) -> bool:
    if not isinstance(values, list):
        return False
    if len(values) != 3:
        return False
    if any(not isinstance(x, (int, float)) for x in values):
        return False
    if any(x < 0 or x > 1 for x in values):
        return False
    return abs(sum(values) - 1.0) < 1e-4


def print_table(title: str, counter: Counter) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in counter.most_common():
        print(f"{str(key):20s} {value}")


def inspect_split(split: str) -> None:
    path = PROCESSED_DIR / f"{split}.jsonl"
    rows = read_jsonl(path)

    category_counts = Counter(row.get("category", "missing") for row in rows)
    hard_label_counts = Counter(row.get("hard_label_name", "missing") for row in rows)
    annotation_counts = Counter(row.get("num_passage_annotations", "missing") for row in rows)
    invalid_soft = [row.get("id") for row in rows if not soft_label_is_valid(row.get("soft_label"))]
    missing_claim = sum(1 for row in rows if not str(row.get("claim", "")).strip())
    missing_passage = sum(1 for row in rows if not str(row.get("passage", "")).strip())
    sentence_counts = [len(row.get("sentences", [])) for row in rows]
    evidence_counts = [len(row.get("evidence_sentence_ids", [])) for row in rows]

    print(f"\n{'=' * 70}")
    print(f"Split: {split}")
    print(f"File:  {path}")
    print(f"Rows:  {len(rows)}")

    print_table("Category counts", category_counts)
    print_table("Hard label counts", hard_label_counts)
    print_table("Passage annotation count distribution", annotation_counts)

    print("\nQuality checks")
    print("--------------")
    print(f"Invalid soft labels:        {len(invalid_soft)}")
    print(f"Missing claims:             {missing_claim}")
    print(f"Missing passages:           {missing_passage}")
    print(f"Average sentences/passage:  {sum(sentence_counts) / max(len(sentence_counts), 1):.2f}")
    print(f"Average evidence sentences: {sum(evidence_counts) / max(len(evidence_counts), 1):.2f}")

    sample = rows[0]
    print("\nFirst sample")
    print("------------")
    print(f"id:              {sample.get('id')}")
    print(f"category:        {sample.get('category')}")
    print(f"hard_label:      {sample.get('hard_label_name')} ({sample.get('hard_label')})")
    print(f"soft_label:      {sample.get('soft_label')}")
    print(f"claim:           {sample.get('claim')}")
    print(f"entity:          {sample.get('entity')}")
    print(f"section_title:   {sample.get('section_title')}")
    passage_preview = str(sample.get("passage", ""))[:500].replace("\n", " ")
    print(f"passage preview: {passage_preview}...")


def inspect_evidence_file(split: str) -> None:
    path = PROCESSED_DIR / f"{split}_evidence.jsonl"
    rows = read_jsonl(path)
    binary_counts = Counter(row.get("binary_evidence_label", "missing") for row in rows)
    hard_counts = Counter(row.get("hard_label_name", "missing") for row in rows)
    invalid_soft = sum(1 for row in rows if not soft_label_is_valid(row.get("soft_label")))

    print(f"\nEvidence file: {path}")
    print(f"Rows:          {len(rows)}")
    print(f"Binary labels: {dict(binary_counts)}")
    print(f"Ternary labels:{dict(hard_counts)}")
    print(f"Invalid soft:  {invalid_soft}")


def main() -> None:
    print("Processed AMBIFC Inspection")
    print("=" * 70)

    for split in SPLITS:
        inspect_split(split)

    print(f"\n{'=' * 70}")
    print("Evidence-selection files")
    print("=" * 70)
    for split in SPLITS:
        inspect_evidence_file(split)

    corpus_path = PROCESSED_DIR / "retrieval_corpus.jsonl"
    corpus_rows = read_jsonl(corpus_path)
    print(f"\n{'=' * 70}")
    print("Retrieval corpus")
    print("=" * 70)
    print(f"File: {corpus_path}")
    print(f"Rows: {len(corpus_rows)}")
    if corpus_rows:
        sample = corpus_rows[0]
        print(f"First corpus id: {sample.get('corpus_id')}")
        print(f"First text preview: {str(sample.get('text', ''))[:400]}...")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
