#!/usr/bin/env python3
"""
Generate full AMBIFC RAG data using multiprocessing BM25 retrieval.

PDC design:
    1. BM25 retrieval is CPU-heavy, so it is parallelized using ProcessPoolExecutor.
    2. Dense FAISS retrieval is done after BM25, using GPU batching.
    3. Ambiguity detection is lightweight and runs in the main process.
    4. BM25 + Dense outputs are fused with Reciprocal Rank Fusion.
    5. Final row contains original passage + top retrieved passages.

Important implementation detail:
    We DO NOT load SentenceTransformer/CUDA before starting multiprocessing.
    Loading CUDA before ProcessPoolExecutor can cause freezing/hanging in WSL/Linux.

Inputs:
    data/processed/train.jsonl
    data/processed/dev.jsonl
    data/processed/test.jsonl

Indexes:
    data/indexes/corpus.jsonl
    data/indexes/bm25.pkl
    data/indexes/faiss.index

Outputs:
    data/rag/train_rag_mp.jsonl
    data/rag/dev_rag_mp.jsonl
    data/rag/test_rag_mp.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

AMBIGUITY_PATTERNS = {
    "vagueness": [
        r"\bcan\b",
        r"\bcould\b",
        r"\bmay\b",
        r"\bmight\b",
        r"\boften\b",
        r"\busually\b",
        r"\bgenerally\b",
        r"\brare\b",
        r"\bcommon\b",
        r"\bmany\b",
        r"\bsome\b",
        r"\bmost\b",
    ],
    "underspecification": [
        r"\billegal\b",
        r"\blegal\b",
        r"\ballowed\b",
        r"\brequired\b",
        r"\bmust\b",
        r"\beverywhere\b",
        r"\balways\b",
        r"\bnever\b",
        r"\ball\b",
        r"\bany\b",
    ],
    "temporal": [
        r"\blast\b",
        r"\bcurrent\b",
        r"\bcurrently\b",
        r"\bnow\b",
        r"\brecent\b",
        r"\bformer\b",
        r"\bfirst\b",
        r"\bfinal\b",
    ],
    "coreference": [
        r"\bhe\b",
        r"\bshe\b",
        r"\bit\b",
        r"\bthey\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",
    ],
    "negation": [
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bwithout\b",
        r"\bnone\b",
        r"\bcannot\b",
        r"\bcan't\b",
    ],
}


_MP_BM25 = None
_MP_CORPUS = None


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]


def read_jsonl(path: Path, max_rows: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            rows.append(json.loads(line))

            if max_rows is not None and len(rows) >= max_rows:
                break

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_bm25(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def make_query(row: Dict[str, Any]) -> str:
    claim = str(row.get("claim", "")).strip()
    entity = str(row.get("entity", "")).strip()

    if entity:
        return f"{claim} {entity}"

    return claim


def make_query_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []

    for index, row in enumerate(rows):
        records.append(
            {
                "index": index,
                "id": str(row.get("id", "")),
                "query": make_query(row),
                "claim": str(row.get("claim", "")),
                "entity": str(row.get("entity", "")),
                "passage": str(row.get("passage", "")),
            }
        )

    return records


def bm25_search_single(
    bm25,
    corpus: List[Dict[str, Any]],
    query: str,
    top_k: int,
    exclude_id: str | None,
) -> List[Dict[str, Any]]:
    scores = bm25.get_scores(tokenize(query))
    ranked = np.argsort(scores)[::-1]

    results = []

    for idx in ranked:
        corpus_row = corpus[int(idx)]
        corpus_id = str(corpus_row.get("corpus_id", ""))

        if exclude_id and corpus_id == exclude_id:
            continue

        results.append(
            {
                "source": "bm25",
                "rank": len(results) + 1,
                "corpus_index": int(idx),
                "corpus_id": corpus_id,
                "score": float(scores[idx]),
                "text": corpus_row.get("text", ""),
                "entity": corpus_row.get("entity", ""),
                "section_title": corpus_row.get("section_title", ""),
            }
        )

        if len(results) >= top_k:
            break

    return results


def _mp_init_worker(bm25_path: str, corpus_path: str) -> None:
    """
    Initializer for each multiprocessing worker.

    Each process loads its own BM25 index and corpus.
    This avoids sending large Python objects through process pipes.
    """
    global _MP_BM25, _MP_CORPUS

    _MP_BM25 = load_bm25(Path(bm25_path))
    _MP_CORPUS = read_jsonl(Path(corpus_path))


def _mp_bm25_chunk_worker(
    chunk: List[Dict[str, Any]],
    top_k: int,
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    global _MP_BM25, _MP_CORPUS

    if _MP_BM25 is None or _MP_CORPUS is None:
        raise RuntimeError("Multiprocessing BM25 worker was not initialized.")

    outputs = []

    for record in chunk:
        results = bm25_search_single(
            bm25=_MP_BM25,
            corpus=_MP_CORPUS,
            query=record["query"],
            top_k=top_k,
            exclude_id=record["id"],
        )

        outputs.append((int(record["index"]), results))

    return outputs


def chunk_records(
    records: List[Dict[str, Any]],
    workers: int,
    chunks_per_worker: int,
) -> List[List[Dict[str, Any]]]:
    """
    Split records into many smaller chunks.

    Example:
        workers = 8
        chunks_per_worker = 16
        num_chunks = 128

    Only 8 processes run at once, but tqdm updates 128 times.
    """
    if not records:
        return []

    workers = max(1, workers)
    chunks_per_worker = max(1, chunks_per_worker)

    num_chunks = min(len(records), workers * chunks_per_worker)

    chunks = [[] for _ in range(num_chunks)]

    for i, record in enumerate(records):
        chunks[i % num_chunks].append(record)

    return [chunk for chunk in chunks if chunk]


def bm25_multiprocessing_search(
    query_records: List[Dict[str, Any]],
    bm25_path: Path,
    corpus_path: Path,
    top_k: int,
    workers: int,
    chunks_per_worker: int,
) -> List[List[Dict[str, Any]]]:
    chunks = chunk_records(
        records=query_records,
        workers=workers,
        chunks_per_worker=chunks_per_worker,
    )

    ordered_outputs: List[List[Dict[str, Any]] | None] = [None] * len(query_records)

    print(f"BM25 chunks:     {len(chunks)}")
    print(f"BM25 workers:    {workers}")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_mp_init_worker,
        initargs=(str(bm25_path), str(corpus_path)),
    ) as executor:
        futures = [
            executor.submit(_mp_bm25_chunk_worker, chunk, top_k)
            for chunk in chunks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"BM25 multiprocessing ({workers} workers)",
        ):
            chunk_outputs = future.result()

            for original_index, results in chunk_outputs:
                ordered_outputs[original_index] = results

    missing = sum(1 for output in ordered_outputs if output is None)
    if missing:
        raise RuntimeError(f"{missing} BM25 multiprocessing outputs are missing.")

    return ordered_outputs  # type: ignore


def dense_batch_search(
    dense_model,
    faiss_index,
    corpus: List[Dict[str, Any]],
    query_records: List[Dict[str, Any]],
    top_k: int,
    batch_size: int,
) -> List[List[Dict[str, Any]]]:
    queries = [record["query"] for record in query_records]
    exclude_ids = [record["id"] for record in query_records]

    all_outputs: List[List[Dict[str, Any]]] = []
    search_k = top_k + 10

    for start in tqdm(range(0, len(queries), batch_size), desc="Dense FAISS retrieval"):
        batch_queries = queries[start : start + batch_size]
        batch_exclude_ids = exclude_ids[start : start + batch_size]

        embeddings = dense_model.encode(
            batch_queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = faiss_index.search(embeddings, search_k)

        for row_scores, row_indices, exclude_id in zip(scores, indices, batch_exclude_ids):
            results = []

            for score, idx in zip(row_scores, row_indices):
                if idx < 0:
                    continue

                corpus_row = corpus[int(idx)]
                corpus_id = str(corpus_row.get("corpus_id", ""))

                if exclude_id and corpus_id == exclude_id:
                    continue

                results.append(
                    {
                        "source": "dense",
                        "rank": len(results) + 1,
                        "corpus_index": int(idx),
                        "corpus_id": corpus_id,
                        "score": float(score),
                        "text": corpus_row.get("text", ""),
                        "entity": corpus_row.get("entity", ""),
                        "section_title": corpus_row.get("section_title", ""),
                    }
                )

                if len(results) >= top_k:
                    break

            all_outputs.append(results)

    return all_outputs


def detect_ambiguity_single(record: Dict[str, Any]) -> Dict[str, Any]:
    claim = str(record.get("claim", "")).lower()
    passage = str(record.get("passage", "")).lower()
    text = claim + " " + passage[:500]

    matched_types = []
    matched_cues = []

    for ambiguity_type, patterns in AMBIGUITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matched_types.append(ambiguity_type)
                matched_cues.append(pattern.replace(r"\b", "").replace("\\", ""))
                break

    matched_types = sorted(set(matched_types))

    if not matched_types:
        primary_type = "none"
    elif "underspecification" in matched_types:
        primary_type = "underspecification"
    elif "vagueness" in matched_types:
        primary_type = "vagueness"
    elif "temporal" in matched_types:
        primary_type = "temporal"
    elif "coreference" in matched_types:
        primary_type = "coreference"
    else:
        primary_type = matched_types[0]

    return {
        "ambiguity_type": primary_type,
        "ambiguity_types": matched_types,
        "ambiguity_cues": matched_cues,
        "ambiguity_score": len(matched_types),
    }


def ambiguity_batch_detect(query_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        detect_ambiguity_single(record)
        for record in tqdm(query_records, desc="Ambiguity detection")
    ]


def reciprocal_rank_fusion_single(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    entity: str,
    ambiguity_info: Dict[str, Any],
    rrf_k: int,
    top_k_final: int,
) -> List[Dict[str, Any]]:
    scores = {}
    payloads = {}
    sources = {}

    entity_norm = str(entity).strip().lower()
    ambiguity_score = int(ambiguity_info.get("ambiguity_score", 0))

    for result_list in [bm25_results, dense_results]:
        for rank, item in enumerate(result_list, start=1):
            corpus_id = str(item["corpus_id"])
            score = 1.0 / (rrf_k + rank)

            item_entity = str(item.get("entity", "")).strip().lower()

            if entity_norm and item_entity == entity_norm:
                score += 0.01

            if ambiguity_score > 0 and entity_norm and item_entity == entity_norm:
                score += 0.005

            scores[corpus_id] = scores.get(corpus_id, 0.0) + score
            payloads[corpus_id] = item
            sources.setdefault(corpus_id, set()).add(item.get("source", "unknown"))

    fused = []

    for corpus_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(payloads[corpus_id])
        item["rank"] = len(fused) + 1
        item["rrf_score"] = float(score)
        item["source"] = "+".join(sorted(sources.get(corpus_id, {"unknown"})))

        fused.append(item)

        if len(fused) >= top_k_final:
            break

    return fused


def build_rag_context(original_passage: str, retrieved_passages: List[Dict[str, Any]]) -> str:
    parts = []

    original_passage = str(original_passage).strip()
    if original_passage:
        parts.append("[ORIGINAL] " + original_passage)

    for i, item in enumerate(retrieved_passages, start=1):
        text = str(item.get("text", "")).strip()
        if text:
            parts.append(f"[RETRIEVED_{i}] {text}")

    return " ".join(parts)


def build_rag_rows(
    input_rows: List[Dict[str, Any]],
    query_records: List[Dict[str, Any]],
    bm25_outputs: List[List[Dict[str, Any]]],
    dense_outputs: List[List[Dict[str, Any]]],
    ambiguity_outputs: List[Dict[str, Any]],
    rrf_k: int,
    top_k_final: int,
) -> List[Dict[str, Any]]:
    rag_rows = []

    for row, record, bm25_result, dense_result, ambiguity_info in tqdm(
        zip(input_rows, query_records, bm25_outputs, dense_outputs, ambiguity_outputs),
        total=len(input_rows),
        desc="Building RAG rows",
    ):
        fused = reciprocal_rank_fusion_single(
            bm25_results=bm25_result,
            dense_results=dense_result,
            entity=record.get("entity", ""),
            ambiguity_info=ambiguity_info,
            rrf_k=rrf_k,
            top_k_final=top_k_final,
        )

        original_passage = str(row.get("passage", ""))

        out = dict(row)
        out["original_passage"] = original_passage
        out["retrieved_passages"] = fused
        out["rag_context"] = build_rag_context(original_passage, fused)
        out["num_retrieved_passages"] = len(fused)
        out["retrieval_query"] = record["query"]
        out["ambiguity_info"] = ambiguity_info
        out["rag_generation_mode"] = "bm25_multiprocessing_dense_batched_rrf"

        rag_rows.append(out)

    return rag_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AMBIFC RAG data with multiprocessing BM25.")

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--index_dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--top_k_bm25", type=int, default=20)
    parser.add_argument("--top_k_dense", type=int, default=20)
    parser.add_argument("--top_k_final", type=int, default=3)
    parser.add_argument("--rrf_k", type=int, default=60)

    parser.add_argument("--dense_batch_size", type=int, default=128)
    parser.add_argument("--bm25_workers", type=int, default=4)
    parser.add_argument("--chunks_per_worker", type=int, default=16)

    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--summary", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_total = time.perf_counter()

    corpus_path = args.index_dir / "corpus.jsonl"
    bm25_path = args.index_dir / "bm25.pkl"
    faiss_path = args.index_dir / "faiss.index"

    print("\n=== Loading input and corpus ===")
    input_rows = read_jsonl(args.input, max_rows=args.max_rows)
    corpus = read_jsonl(corpus_path)
    query_records = make_query_records(input_rows)

    print(f"Input rows:        {len(input_rows)}")
    print(f"Corpus rows:       {len(corpus)}")
    print(f"BM25 workers:      {args.bm25_workers}")
    print(f"Chunks per worker: {args.chunks_per_worker}")
    print(f"Dense batch size:  {args.dense_batch_size}")

    print("\n=== Step 1: BM25 multiprocessing retrieval ===")
    bm25_start = time.perf_counter()

    bm25_outputs = bm25_multiprocessing_search(
        query_records=query_records,
        bm25_path=bm25_path,
        corpus_path=corpus_path,
        top_k=args.top_k_bm25,
        workers=args.bm25_workers,
        chunks_per_worker=args.chunks_per_worker,
    )

    bm25_time = time.perf_counter() - bm25_start
    print(f"BM25 multiprocessing time: {bm25_time:.2f}s")

    print("\n=== Step 2: Loading dense model and FAISS index ===")
    dense_load_start = time.perf_counter()

    faiss_index = faiss.read_index(str(faiss_path))
    dense_model = SentenceTransformer(args.dense_model, device=args.device)

    dense_load_time = time.perf_counter() - dense_load_start

    print(f"FAISS vectors:      {faiss_index.ntotal}")
    print(f"Dense load time:    {dense_load_time:.2f}s")

    print("\n=== Step 3: Dense FAISS batched retrieval ===")
    dense_start = time.perf_counter()

    dense_outputs = dense_batch_search(
        dense_model=dense_model,
        faiss_index=faiss_index,
        corpus=corpus,
        query_records=query_records,
        top_k=args.top_k_dense,
        batch_size=args.dense_batch_size,
    )

    dense_time = time.perf_counter() - dense_start
    print(f"Dense retrieval time: {dense_time:.2f}s")

    print("\n=== Step 4: Ambiguity detection ===")
    ambiguity_start = time.perf_counter()

    ambiguity_outputs = ambiguity_batch_detect(query_records)

    ambiguity_time = time.perf_counter() - ambiguity_start
    print(f"Ambiguity detection time: {ambiguity_time:.2f}s")

    print("\n=== Step 5: RRF fusion and RAG context construction ===")
    fusion_start = time.perf_counter()

    rag_rows = build_rag_rows(
        input_rows=input_rows,
        query_records=query_records,
        bm25_outputs=bm25_outputs,
        dense_outputs=dense_outputs,
        ambiguity_outputs=ambiguity_outputs,
        rrf_k=args.rrf_k,
        top_k_final=args.top_k_final,
    )

    fusion_time = time.perf_counter() - fusion_start
    print(f"Fusion/context time: {fusion_time:.2f}s")

    print("\n=== Step 6: Saving RAG data ===")
    write_jsonl(args.output, rag_rows)

    total_time = time.perf_counter() - start_total

    retrieved_counts = [row.get("num_retrieved_passages", 0) for row in rag_rows]

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "index_dir": str(args.index_dir),
        "dense_model": args.dense_model,
        "rows": len(rag_rows),
        "bm25_workers": args.bm25_workers,
        "chunks_per_worker": args.chunks_per_worker,
        "dense_batch_size": args.dense_batch_size,
        "top_k_bm25": args.top_k_bm25,
        "top_k_dense": args.top_k_dense,
        "top_k_final": args.top_k_final,
        "rrf_k": args.rrf_k,
        "avg_retrieved_passages": float(np.mean(retrieved_counts)) if retrieved_counts else 0.0,
        "bm25_time_seconds": bm25_time,
        "dense_load_time_seconds": dense_load_time,
        "dense_time_seconds": dense_time,
        "ambiguity_time_seconds": ambiguity_time,
        "fusion_context_time_seconds": fusion_time,
        "total_time_seconds": total_time,
        "rows_per_second": len(rag_rows) / total_time if total_time > 0 else 0.0,
    }

    summary_path = args.summary
    if summary_path is None:
        summary_path = args.output.with_suffix(".summary.json")

    save_json(summary_path, summary)

    print("\n=== Done ===")
    print(f"Saved RAG data: {args.output}")
    print(f"Saved summary:  {summary_path}")
    print(f"Total time:     {total_time:.2f}s")
    print(f"Rows/sec:       {summary['rows_per_second']:.2f}")


if __name__ == "__main__":
    main()  