#!/usr/bin/env python3
"""
Prepare AMBIFC data for:
1. Experiment 1: AMBIFC Base DeBERTaV3-large
2. Experiment 2: Evidence Selection
3. Experiment 3: PDC-RAG-512
4. Experiment 4: Adaptive PDC-RAG

Expected raw AMBIFC files:
    data/raw/train.certain.jsonl
    data/raw/train.uncertain.jsonl
    data/raw/dev.certain.jsonl
    data/raw/dev.uncertain.jsonl
    data/raw/test.certain.jsonl
    data/raw/test.uncertain.jsonl

The official AMBIFC repository describes the dataset files as:
    <split>.<subset>.jsonl
where split is train/dev/test and subset is certain/uncertain.

Output files:
    data/processed/train.jsonl
    data/processed/dev.jsonl
    data/processed/test.jsonl
    data/processed/train_certain.jsonl
    data/processed/train_uncertain.jsonl
    data/processed/dev_certain.jsonl
    data/processed/dev_uncertain.jsonl
    data/processed/test_certain.jsonl
    data/processed/test_uncertain.jsonl
    data/processed/train_evidence.jsonl
    data/processed/dev_evidence.jsonl
    data/processed/test_evidence.jsonl
    data/processed/retrieval_corpus.jsonl
    data/processed/prepare_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LABEL_TO_ID = {
    "refuting": 0,
    "refuted": 0,
    "refute": 0,
    "refutes": 0,
    "contradicting": 0,
    "contradiction": 0,
    "contradicts": 0,
    "r": 0,
    "neutral": 1,
    "not enough info": 1,
    "not_enough_info": 1,
    "nei": 1,
    "n": 1,
    "supporting": 2,
    "supported": 2,
    "support": 2,
    "supports": 2,
    "s": 2,
}

ID_TO_LABEL = {
    0: "refuting",
    1: "neutral",
    2: "supporting",
}

SPLITS = ["train", "dev", "test"]
SUBSETS = ["certain", "uncertain"]


def normalize_label(label: Any) -> Optional[int]:
    """Map a raw AMBIFC label string to REFUTES/NEUTRAL/SUPPORTS id."""
    if label is None:
        return None
    text = str(label).strip().lower().replace("-", "_")
    text = re.sub(r"\s+", " ", text)
    text_space = text.replace("_", " ")
    return LABEL_TO_ID.get(text) if text in LABEL_TO_ID else LABEL_TO_ID.get(text_space)


def label_distribution(labels: Iterable[Any]) -> List[float]:
    """Convert raw labels into a 3-way probability distribution."""
    counts = Counter()
    for label in labels:
        label_id = normalize_label(label)
        if label_id is not None:
            counts[label_id] += 1

    total = sum(counts.values())
    if total == 0:
        return [0.0, 1.0, 0.0]

    return [counts[i] / total for i in range(3)]


def argmax(values: List[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {error}") from error
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def find_raw_file(raw_dir: Path, split: str, subset: str) -> Optional[Path]:
    """Find AMBIFC raw jsonl files even if they are inside a nested folder."""
    candidates = [
        raw_dir / f"{split}.{subset}.jsonl",
        raw_dir / f"{split}_{subset}.jsonl",
        raw_dir / f"{split}-{subset}.jsonl",
        raw_dir / "ambifc" / f"{split}.{subset}.jsonl",
        raw_dir / "data" / f"{split}.{subset}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    patterns = [
        f"**/{split}.{subset}.jsonl",
        f"**/{split}_{subset}.jsonl",
        f"**/{split}-{subset}.jsonl",
        f"**/*{split}*{subset}*.jsonl",
    ]
    for pattern in patterns:
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def get_sentence_items(sample: Dict[str, Any]) -> List[Tuple[str, str]]:
    sentences = sample.get("sentences", {})
    if isinstance(sentences, dict):
        def key_order(key: str) -> Tuple[int, str]:
            return (int(key), key) if str(key).isdigit() else (10**9, str(key))
        return [(str(k), str(v)) for k, v in sorted(sentences.items(), key=lambda kv: key_order(str(kv[0])))]
    if isinstance(sentences, list):
        return [(str(i), str(v)) for i, v in enumerate(sentences)]
    return []


def build_passage_text(sentence_items: List[Tuple[str, str]]) -> str:
    return " ".join(sentence.strip() for _, sentence in sentence_items if sentence and sentence.strip())


def get_passage_annotation_labels(sample: Dict[str, Any]) -> List[Any]:
    annotations = sample.get("passage_annotations", [])
    labels = []
    if isinstance(annotations, list):
        for ann in annotations:
            if isinstance(ann, dict):
                labels.append(ann.get("label"))
            else:
                labels.append(ann)
    return labels


def get_sentence_annotation_labels(sample: Dict[str, Any], sentence_id: str) -> List[Any]:
    sentence_annotations = sample.get("sentence_annotations", {})
    anns = []
    if isinstance(sentence_annotations, dict):
        anns = sentence_annotations.get(sentence_id, sentence_annotations.get(int(sentence_id), []))
    if not isinstance(anns, list):
        return []

    labels = []
    for ann in anns:
        if isinstance(ann, dict):
            labels.append(ann.get("annotation", ann.get("label")))
        else:
            labels.append(ann)
    return labels


def get_aggregated_label(sample: Dict[str, Any], soft_label: List[float]) -> int:
    labels = sample.get("labels", {})
    if isinstance(labels, dict):
        label_id = normalize_label(labels.get("passage"))
        if label_id is not None:
            return label_id
    return argmax(soft_label)


def normalize_category(sample: Dict[str, Any], fallback: str) -> str:
    category = str(sample.get("category", fallback)).strip().lower()
    if "uncertain" in category:
        return "uncertain"
    if "certain" in category:
        return "certain"
    return fallback


def should_keep_sample(sample: Dict[str, Any], subset: str, include_uncertain_less_than_5: bool) -> bool:
    if subset != "uncertain":
        return True
    if include_uncertain_less_than_5:
        return True
    return len(get_passage_annotation_labels(sample)) >= 5


def make_clean_sample(sample: Dict[str, Any], split: str, subset: str, row_index: int) -> Dict[str, Any]:
    sentence_items = get_sentence_items(sample)
    passage_text = build_passage_text(sentence_items)
    passage_ann_labels = get_passage_annotation_labels(sample)
    soft_label = label_distribution(passage_ann_labels)
    hard_label = get_aggregated_label(sample, soft_label)
    category = normalize_category(sample, subset)

    wiki_passage = str(sample.get("wiki_passage", ""))
    sample_id = wiki_passage if wiki_passage else f"{split}-{subset}-{row_index}"

    sentence_records = []
    evidence_sentence_ids = []
    sentence_soft_labels: Dict[str, List[float]] = {}
    sentence_evidence_probs: Dict[str, float] = {}

    for sent_id, sent_text in sentence_items:
        sent_ann_labels = get_sentence_annotation_labels(sample, sent_id)
        sent_soft = label_distribution(sent_ann_labels)
        evidence_prob = sent_soft[0] + sent_soft[2]
        sentence_soft_labels[sent_id] = sent_soft
        sentence_evidence_probs[sent_id] = evidence_prob
        if sent_text.strip() and evidence_prob > 0.0:
            evidence_sentence_ids.append(sent_id)
        sentence_records.append({"id": sent_id, "text": sent_text})

    return {
        "id": sample_id,
        "split": split,
        "source_subset": subset,
        "category": category,
        "claim": sample.get("claim", ""),
        "claim_id": sample.get("claim_id", None),
        "wiki_page": sample.get("wiki_page", ""),
        "wiki_section": sample.get("wiki_section", ""),
        "wiki_passage": wiki_passage,
        "entity": sample.get("entity", ""),
        "section_title": sample.get("section", sample.get("section_title", "")),
        "passage": passage_text,
        "sentences": sentence_records,
        "soft_label": soft_label,
        "hard_label": hard_label,
        "hard_label_name": ID_TO_LABEL[hard_label],
        "num_passage_annotations": len(passage_ann_labels),
        "evidence_sentence_ids": evidence_sentence_ids,
        "sentence_soft_labels": sentence_soft_labels,
        "sentence_evidence_probs": sentence_evidence_probs,
    }


def make_evidence_rows(clean_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence_rows: List[Dict[str, Any]] = []
    for row in clean_rows:
        for sentence in row["sentences"]:
            sentence_id = str(sentence["id"])
            sentence_text = str(sentence["text"])
            if not sentence_text.strip():
                continue
            soft_label = row["sentence_soft_labels"].get(sentence_id, [0.0, 1.0, 0.0])
            evidence_prob = float(soft_label[0] + soft_label[2])
            evidence_rows.append({
                "id": f"{row['id']}::sent-{sentence_id}",
                "parent_id": row["id"],
                "split": row["split"],
                "category": row["category"],
                "claim": row["claim"],
                "sentence_id": sentence_id,
                "sentence": sentence_text,
                "entity": row.get("entity", ""),
                "section_title": row.get("section_title", ""),
                "soft_label": soft_label,
                "hard_label": argmax(soft_label),
                "hard_label_name": ID_TO_LABEL[argmax(soft_label)],
                "binary_evidence_label": 1 if evidence_prob > 0.0 else 0,
                "evidence_prob": evidence_prob,
            })
    return evidence_rows


def make_retrieval_corpus(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seen = set()
    corpus = []
    for split, rows in rows_by_split.items():
        for row in rows:
            text = row.get("passage", "").strip()
            if not text:
                continue
            key = row.get("wiki_passage") or text
            if key in seen:
                continue
            seen.add(key)
            corpus.append({
                "corpus_id": key,
                "text": text,
                "entity": row.get("entity", ""),
                "section_title": row.get("section_title", ""),
                "wiki_page": row.get("wiki_page", ""),
                "wiki_section": row.get("wiki_section", ""),
                "wiki_passage": row.get("wiki_passage", ""),
                "first_seen_split": split,
                "category": row.get("category", ""),
            })
    return corpus


def print_missing_data_help(raw_dir: Path) -> None:
    print("\nERROR: AMBIFC raw JSONL files were not found.")
    print("\nExpected files inside:")
    print(f"  {raw_dir.resolve()}")
    for split in SPLITS:
        for subset in SUBSETS:
            print(f"  - {split}.{subset}.jsonl")
    print("\nDownload the official AMBIFC dataset from the GitHub README and place/extract the six JSONL files into data/raw/.")
    print("Then run:")
    print("  python src/data/prepare_ambifc.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AMBIFC JSONL files for DeBERTaV3-large + PDC-RAG.")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"), help="Directory containing raw AMBIFC files.")
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"), help="Output directory.")
    parser.add_argument(
        "--include_uncertain_less_than_5",
        action="store_true",
        help="Keep uncertain samples with fewer than five passage annotations. Default excludes them to match AMBIFC experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files: Dict[Tuple[str, str], Path] = {}
    missing = []
    for split in SPLITS:
        for subset in SUBSETS:
            path = find_raw_file(raw_dir, split, subset)
            if path is None:
                missing.append(f"{split}.{subset}.jsonl")
            else:
                raw_files[(split, subset)] = path

    if missing:
        print_missing_data_help(raw_dir)
        print("\nMissing files:")
        for name in missing:
            print(f"  - {name}")
        sys.exit(1)

    rows_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_by_split_subset: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    summary: Dict[str, Any] = {
        "raw_files": {f"{split}.{subset}": str(path) for (split, subset), path in raw_files.items()},
        "splits": {},
        "label_order": {"0": "refuting", "1": "neutral", "2": "supporting"},
        "filtering": {
            "include_uncertain_less_than_5": args.include_uncertain_less_than_5,
        },
    }

    for split in SPLITS:
        for subset in SUBSETS:
            path = raw_files[(split, subset)]
            raw_rows = read_jsonl(path)
            clean_rows = []
            skipped = 0
            for i, sample in enumerate(raw_rows):
                if not should_keep_sample(sample, subset, args.include_uncertain_less_than_5):
                    skipped += 1
                    continue
                clean_rows.append(make_clean_sample(sample, split, subset, i))
            rows_by_split[split].extend(clean_rows)
            rows_by_split_subset[(split, subset)] = clean_rows
            summary["splits"][f"{split}_{subset}"] = {
                "raw": len(raw_rows),
                "kept": len(clean_rows),
                "skipped": skipped,
            }

    print("\n=== Writing processed veracity files ===")
    for split in SPLITS:
        rows = rows_by_split[split]
        count = write_jsonl(out_dir / f"{split}.jsonl", rows)
        print(f"{out_dir / f'{split}.jsonl'}: {count}")

        for subset in SUBSETS:
            subset_rows = rows_by_split_subset[(split, subset)]
            count = write_jsonl(out_dir / f"{split}_{subset}.jsonl", subset_rows)
            print(f"{out_dir / f'{split}_{subset}.jsonl'}: {count}")

    print("\n=== Writing evidence-selection files ===")
    for split in SPLITS:
        evidence_rows = make_evidence_rows(rows_by_split[split])
        count = write_jsonl(out_dir / f"{split}_evidence.jsonl", evidence_rows)
        print(f"{out_dir / f'{split}_evidence.jsonl'}: {count}")
        summary["splits"].setdefault(split, {})
        summary["splits"][split] = {
            "veracity_rows": len(rows_by_split[split]),
            "evidence_rows": count,
            "category_counts": dict(Counter(row["category"] for row in rows_by_split[split])),
            "hard_label_counts": dict(Counter(row["hard_label_name"] for row in rows_by_split[split])),
        }

    print("\n=== Writing retrieval corpus ===")
    corpus = make_retrieval_corpus(rows_by_split)
    corpus_count = write_jsonl(out_dir / "retrieval_corpus.jsonl", corpus)
    print(f"{out_dir / 'retrieval_corpus.jsonl'}: {corpus_count}")
    summary["retrieval_corpus_rows"] = corpus_count

    summary_path = out_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    print("\nDone. Next files to check:")
    print("  data/processed/train.jsonl")
    print("  data/processed/dev.jsonl")
    print("  data/processed/test.jsonl")
    print("  data/processed/train_evidence.jsonl")
    print("  data/processed/retrieval_corpus.jsonl")


if __name__ == "__main__":
    main()
