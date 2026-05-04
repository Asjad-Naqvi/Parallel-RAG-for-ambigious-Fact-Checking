#!/usr/bin/env python3
"""
Test AMBIFC hybrid retrieval.

This script checks:
1. BM25 retrieval
2. Dense FAISS retrieval
3. Reciprocal Rank Fusion retrieval

Input:
    data/indexes/corpus.jsonl
    data/indexes/bm25.pkl
    data/indexes/faiss.index
    data/processed/dev.jsonl

Output:
    results/pdc_rag/retrieval_test_samples.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_bm25(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def bm25_search(bm25, corpus: List[Dict[str, Any]], query: str, top_k: int, exclude_id: str | None = None):
    scores = bm25.get_scores(tokenize(query))
    ranked = np.argsort(scores)[::-1]

    results = []
    for idx in ranked:
        row = corpus[int(idx)]

        if exclude_id is not None and str(row.get("corpus_id")) == exclude_id:
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "corpus_index": int(idx),
                "corpus_id": row.get("corpus_id"),
                "score": float(scores[idx]),
                "text": row.get("text", ""),
                "entity": row.get("entity", ""),
                "section_title": row.get("section_title", ""),
            }
        )

        if len(results) >= top_k:
            break

    return results


def dense_search(
    model,
    faiss_index,
    corpus: List[Dict[str, Any]],
    query: str,
    top_k: int,
    exclude_id: str | None = None,
):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    search_k = top_k + 10
    scores, indices = faiss_index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = corpus[int(idx)]

        if exclude_id is not None and str(row.get("corpus_id")) == exclude_id:
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "corpus_index": int(idx),
                "corpus_id": row.get("corpus_id"),
                "score": float(score),
                "text": row.get("text", ""),
                "entity": row.get("entity", ""),
                "section_title": row.get("section_title", ""),
            }
        )

        if len(results) >= top_k:
            break

    return results


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 5,
):
    scores = {}
    payloads = {}

    for result_list in ranked_lists:
        for rank, item in enumerate(result_list, start=1):
            corpus_id = item["corpus_id"]
            scores[corpus_id] = scores.get(corpus_id, 0.0) + 1.0 / (k + rank)
            payloads[corpus_id] = item

    fused = []
    for corpus_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(payloads[corpus_id])
        item["rrf_score"] = float(score)
        item["rank"] = len(fused) + 1
        fused.append(item)

        if len(fused) >= top_k:
            break

    return fused


def preview(text: str, n: int = 250) -> str:
    text = " ".join(str(text).split())
    return text[:n] + ("..." if len(text) > n else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test AMBIFC hybrid retrieval.")

    parser.add_argument("--index_dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--queries", type=Path, default=Path("data/processed/dev.jsonl"))
    parser.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rrf_k", type=int, default=60)

    parser.add_argument("--output", type=Path, default=Path("results/pdc_rag/retrieval_test_samples.json"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_path = args.index_dir / "corpus.jsonl"
    bm25_path = args.index_dir / "bm25.pkl"
    faiss_path = args.index_dir / "faiss.index"

    print("\n=== Loading indexes ===")
    corpus = read_jsonl(corpus_path)
    bm25 = load_bm25(bm25_path)
    faiss_index = faiss.read_index(str(faiss_path))
    dense_model = SentenceTransformer(args.dense_model, device=args.device)

    query_rows = read_jsonl(args.queries)[: args.num_examples]

    print(f"Corpus rows: {len(corpus)}")
    print(f"Query rows:  {len(query_rows)}")
    print(f"FAISS size:  {faiss_index.ntotal}")

    outputs = []

    for i, row in enumerate(query_rows, start=1):
        claim = str(row.get("claim", ""))
        entity = str(row.get("entity", ""))
        query = claim
        if entity.strip():
            query = claim + " " + entity

        exclude_id = str(row.get("id", ""))

        bm25_results = bm25_search(
            bm25=bm25,
            corpus=corpus,
            query=query,
            top_k=args.top_k,
            exclude_id=exclude_id,
        )

        dense_results = dense_search(
            model=dense_model,
            faiss_index=faiss_index,
            corpus=corpus,
            query=query,
            top_k=args.top_k,
            exclude_id=exclude_id,
        )

        fused_results = reciprocal_rank_fusion(
            ranked_lists=[bm25_results, dense_results],
            k=args.rrf_k,
            top_k=args.top_k,
        )

        sample = {
            "example_number": i,
            "id": row.get("id"),
            "claim": claim,
            "entity": entity,
            "gold_label": row.get("hard_label_name"),
            "bm25": bm25_results,
            "dense": dense_results,
            "rrf": fused_results,
        }

        outputs.append(sample)

        print("\n" + "=" * 90)
        print(f"Example {i}")
        print(f"Claim:  {claim}")
        print(f"Entity: {entity}")
        print(f"Gold:   {row.get('hard_label_name')}")

        print("\nTop RRF results")
        for result in fused_results:
            print(
                f"[{result['rank']}] RRF={result['rrf_score']:.4f} "
                f"ID={result['corpus_id']} | Entity={result.get('entity', '')}"
            )
            print(f"    {preview(result.get('text', ''))}")

    save_json(args.output, outputs)

    print("\n=== Saved retrieval test samples ===")
    print(args.output)


if __name__ == "__main__":
    main()