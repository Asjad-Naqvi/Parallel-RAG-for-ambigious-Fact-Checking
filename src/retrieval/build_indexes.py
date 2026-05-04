#!/usr/bin/env python3
"""
Build retrieval indexes for AMBIFC PDC-RAG.

Input:
    data/processed/retrieval_corpus.jsonl

Outputs:
    data/indexes/corpus.jsonl
    data/indexes/bm25.pkl
    data/indexes/faiss.index
    data/indexes/dense_embeddings.npy
    data/indexes/build_summary.json

Retrieval design:
    1. BM25 for sparse lexical retrieval
    2. SentenceTransformer + FAISS for dense semantic retrieval
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


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


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]


def clean_corpus(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    seen = set()

    for i, row in enumerate(rows):
        text = str(row.get("text", "")).strip()

        if not text:
            continue

        corpus_id = str(row.get("corpus_id", "")).strip()
        if not corpus_id:
            corpus_id = f"corpus-{i}"

        key = corpus_id

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(
            {
                "corpus_id": corpus_id,
                "text": text,
                "entity": row.get("entity", ""),
                "section_title": row.get("section_title", ""),
                "wiki_page": row.get("wiki_page", ""),
                "wiki_section": row.get("wiki_section", ""),
                "wiki_passage": row.get("wiki_passage", ""),
                "first_seen_split": row.get("first_seen_split", ""),
                "category": row.get("category", ""),
            }
        )

    return cleaned


def build_bm25(corpus: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized_corpus = [tokenize(row["text"]) for row in tqdm(corpus, desc="Tokenizing corpus")]
    return BM25Okapi(tokenized_corpus)


def build_dense_embeddings(
    corpus: List[Dict[str, Any]],
    model_name: str,
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)

    texts = [row["text"] for row in corpus]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes for AMBIFC PDC-RAG.")

    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/retrieval_corpus.jsonl"),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/indexes"),
    )

    parser.add_argument(
        "--dense_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Use cuda or cpu. Default lets SentenceTransformer choose.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("\n=== Loading retrieval corpus ===")
    raw_rows = read_jsonl(args.corpus)
    corpus = clean_corpus(raw_rows)

    print(f"Raw corpus rows:   {len(raw_rows)}")
    print(f"Clean corpus rows: {len(corpus)}")

    corpus_out = args.output_dir / "corpus.jsonl"
    write_jsonl(corpus_out, corpus)
    print(f"Saved corpus: {corpus_out}")

    print("\n=== Building BM25 index ===")
    bm25_start = time.time()
    bm25 = build_bm25(corpus)
    bm25_time = time.time() - bm25_start

    bm25_out = args.output_dir / "bm25.pkl"
    with bm25_out.open("wb") as file:
        pickle.dump(bm25, file)

    print(f"Saved BM25 index: {bm25_out}")
    print(f"BM25 build time:  {bm25_time:.2f}s")

    print("\n=== Building dense embeddings ===")
    dense_start = time.time()
    embeddings = build_dense_embeddings(
        corpus=corpus,
        model_name=args.dense_model,
        batch_size=args.batch_size,
        device=args.device,
    )
    dense_time = time.time() - dense_start

    embeddings_out = args.output_dir / "dense_embeddings.npy"
    np.save(embeddings_out, embeddings)

    print(f"Saved embeddings: {embeddings_out}")
    print(f"Embedding shape:  {embeddings.shape}")
    print(f"Dense build time: {dense_time:.2f}s")

    print("\n=== Building FAISS index ===")
    faiss_start = time.time()
    index = build_faiss_index(embeddings)
    faiss_time = time.time() - faiss_start

    faiss_out = args.output_dir / "faiss.index"
    faiss.write_index(index, str(faiss_out))

    print(f"Saved FAISS index: {faiss_out}")
    print(f"FAISS vectors:      {index.ntotal}")
    print(f"FAISS build time:   {faiss_time:.2f}s")

    total_time = time.time() - start_time

    summary = {
        "input_corpus": str(args.corpus),
        "output_dir": str(args.output_dir),
        "dense_model": args.dense_model,
        "batch_size": args.batch_size,
        "raw_corpus_rows": len(raw_rows),
        "clean_corpus_rows": len(corpus),
        "embedding_shape": list(embeddings.shape),
        "bm25_time_seconds": bm25_time,
        "dense_embedding_time_seconds": dense_time,
        "faiss_build_time_seconds": faiss_time,
        "total_time_seconds": total_time,
        "outputs": {
            "corpus": str(corpus_out),
            "bm25": str(bm25_out),
            "dense_embeddings": str(embeddings_out),
            "faiss": str(faiss_out),
        },
    }

    summary_out = args.output_dir / "build_summary.json"
    save_json(summary_out, summary)

    print("\n=== Done ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Summary:    {summary_out}")


if __name__ == "__main__":
    main()