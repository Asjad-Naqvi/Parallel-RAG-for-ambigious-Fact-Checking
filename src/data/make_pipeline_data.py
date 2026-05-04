#!/usr/bin/env python3
"""
Create AMBIFC pipeline veracity data.

Pipeline idea:
    Evidence selector chooses sentence-level evidence.
    Veracity model receives only selected evidence sentences.

Train split:
    Uses gold evidence_sentence_ids from processed AMBIFC.
    If no gold evidence exists, randomly samples 1-2 sentences, following
    the original pipeline training idea.

Dev/Test splits:
    Uses evidence selector predictions.
    If no evidence is selected, passage is empty and no_evidence_selected=True.

Output:
    data/processed/pipeline_train_binary.jsonl
    data/processed/pipeline_dev_binary.jsonl
    data/processed/pipeline_test_binary.jsonl
    data/processed/pipeline_threshold_binary.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sentence_map(row: Dict[str, Any]) -> Dict[str, str]:
    return {str(s["id"]): str(s["text"]) for s in row.get("sentences", [])}


def join_sentences(sentences: List[str]) -> str:
    return " ".join(s.strip() for s in sentences if s and s.strip())


def tune_threshold(dev_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    gold = np.array([int(p["gold_label"]) for p in dev_predictions])
    scores = np.array([float(p["evidence_score"]) for p in dev_predictions])

    best = {
        "threshold": 0.50,
        "precision": 0.0,
        "recall": 0.0,
        "f1": -1.0,
    }

    for threshold in np.arange(0.00, 1.001, 0.01):
        pred = (scores >= threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            gold,
            pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )

        if f1 > best["f1"]:
            best = {
                "threshold": float(round(threshold, 2)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    return best


def group_selected_predictions(
    predictions: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)

    for pred in predictions:
        score = float(pred["evidence_score"])
        if score >= threshold:
            grouped[str(pred["parent_id"])].append(pred)

    for parent_id in grouped:
        grouped[parent_id] = sorted(
            grouped[parent_id],
            key=lambda x: float(x["evidence_score"]),
            reverse=True,
        )

    return dict(grouped)


def make_train_pipeline_row(
    row: Dict[str, Any],
    max_train_random_sentences: int,
    rng: random.Random,
) -> Dict[str, Any]:
    sent_map = sentence_map(row)
    gold_ids = [str(x) for x in row.get("evidence_sentence_ids", [])]

    selected_texts = [sent_map[sid] for sid in gold_ids if sid in sent_map]

    no_evidence = len(selected_texts) == 0

    if no_evidence:
        all_sentences = list(sent_map.values())
        if all_sentences:
            k = min(len(all_sentences), rng.randint(1, max_train_random_sentences))
            selected_texts = rng.sample(all_sentences, k=k)

    pipeline_passage = join_sentences(selected_texts)

    out = dict(row)
    out["original_passage"] = row.get("passage", "")
    out["passage"] = pipeline_passage
    out["pipeline_evidence"] = pipeline_passage
    out["selected_evidence_sentence_ids"] = gold_ids
    out["num_selected_evidence"] = len(selected_texts)
    out["no_evidence_selected"] = no_evidence
    out["pipeline_source"] = "gold_train_evidence"

    return out


def make_predicted_pipeline_row(
    row: Dict[str, Any],
    selected_by_parent: Dict[str, List[Dict[str, Any]]],
    top_k: int | None,
) -> Dict[str, Any]:
    sent_map = sentence_map(row)
    parent_id = str(row["id"])

    selected_predictions = selected_by_parent.get(parent_id, [])

    if top_k is not None:
        selected_predictions = selected_predictions[:top_k]

    selected_ids = [str(p["sentence_id"]) for p in selected_predictions]
    selected_scores = [float(p["evidence_score"]) for p in selected_predictions]

    selected_texts = [sent_map[sid] for sid in selected_ids if sid in sent_map]

    pipeline_passage = join_sentences(selected_texts)

    out = dict(row)
    out["original_passage"] = row.get("passage", "")
    out["passage"] = pipeline_passage
    out["pipeline_evidence"] = pipeline_passage
    out["selected_evidence_sentence_ids"] = selected_ids
    out["selected_evidence_scores"] = selected_scores
    out["num_selected_evidence"] = len(selected_texts)
    out["no_evidence_selected"] = len(selected_texts) == 0
    out["pipeline_source"] = "predicted_evidence"

    return out


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = [int(r.get("num_selected_evidence", 0)) for r in rows]
    no_evidence = sum(1 for r in rows if r.get("no_evidence_selected"))

    return {
        "rows": len(rows),
        "avg_selected_evidence": float(np.mean(counts)) if counts else 0.0,
        "max_selected_evidence": int(max(counts)) if counts else 0,
        "no_evidence_rows": int(no_evidence),
        "no_evidence_rate": float(no_evidence / max(len(rows), 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pipeline veracity data from evidence predictions.")

    parser.add_argument("--train_veracity", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--dev_veracity", type=Path, default=Path("data/processed/dev.jsonl"))
    parser.add_argument("--test_veracity", type=Path, default=Path("data/processed/test.jsonl"))

    parser.add_argument(
        "--dev_predictions",
        type=Path,
        default=Path("checkpoints/original_paper/exp2a_binary_evidence_deberta_base/dev_predictions.jsonl"),
    )
    parser.add_argument(
        "--test_predictions",
        type=Path,
        default=Path("checkpoints/original_paper/exp2a_binary_evidence_deberta_base/test_predictions.jsonl"),
    )

    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))

    parser.add_argument("--prefix", type=str, default="pipeline_binary")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_train_random_sentences", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print("\n=== Loading files ===")
    train_rows = read_jsonl(args.train_veracity)
    dev_rows = read_jsonl(args.dev_veracity)
    test_rows = read_jsonl(args.test_veracity)

    dev_predictions = read_jsonl(args.dev_predictions)
    test_predictions = read_jsonl(args.test_predictions)

    print(f"Train veracity rows: {len(train_rows)}")
    print(f"Dev veracity rows:   {len(dev_rows)}")
    print(f"Test veracity rows:  {len(test_rows)}")
    print(f"Dev evidence preds:  {len(dev_predictions)}")
    print(f"Test evidence preds: {len(test_predictions)}")

    print("\n=== Tuning evidence threshold on dev predictions ===")
    threshold_info = tune_threshold(dev_predictions)
    threshold = float(threshold_info["threshold"])

    print(f"Best threshold: {threshold:.2f}")
    print(f"Dev precision:  {threshold_info['precision']:.4f}")
    print(f"Dev recall:     {threshold_info['recall']:.4f}")
    print(f"Dev F1:         {threshold_info['f1']:.4f}")

    dev_selected = group_selected_predictions(dev_predictions, threshold)
    test_selected = group_selected_predictions(test_predictions, threshold)

    print("\n=== Creating pipeline rows ===")
    pipeline_train = [
        make_train_pipeline_row(
            row,
            max_train_random_sentences=args.max_train_random_sentences,
            rng=rng,
        )
        for row in train_rows
    ]

    pipeline_dev = [
        make_predicted_pipeline_row(
            row,
            selected_by_parent=dev_selected,
            top_k=args.top_k,
        )
        for row in dev_rows
    ]

    pipeline_test = [
        make_predicted_pipeline_row(
            row,
            selected_by_parent=test_selected,
            top_k=args.top_k,
        )
        for row in test_rows
    ]

    train_out = args.out_dir / f"{args.prefix}_train.jsonl"
    dev_out = args.out_dir / f"{args.prefix}_dev.jsonl"
    test_out = args.out_dir / f"{args.prefix}_test.jsonl"
    threshold_out = args.out_dir / f"{args.prefix}_threshold.json"

    write_jsonl(train_out, pipeline_train)
    write_jsonl(dev_out, pipeline_dev)
    write_jsonl(test_out, pipeline_test)

    summary = {
        "threshold": threshold_info,
        "top_k": args.top_k,
        "train": summarize(pipeline_train),
        "dev": summarize(pipeline_dev),
        "test": summarize(pipeline_test),
        "notes": {
            "train": "Gold evidence is used for training. No-evidence rows use random 1-2 sentence sampling.",
            "dev_test": "Predicted evidence is used. If no sentence passes threshold, pipeline evidence is empty.",
        },
    }

    save_json(threshold_out, summary)

    print("\n=== Saved ===")
    print(train_out)
    print(dev_out)
    print(test_out)
    print(threshold_out)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()