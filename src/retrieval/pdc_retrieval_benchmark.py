#!/usr/bin/env python3
"""
PDC Retrieval Benchmark for AMBIFC PDC-RAG.

This script implements the benchmark style described in the project note:

1. Sequential retrieval:
   BM25 -> Dense FAISS -> Ambiguity detection -> RRF fusion

2. Async/task-level parallel retrieval:
   BM25 worker + Dense FAISS worker + Ambiguity worker run concurrently -> RRF fusion

3. Profiling:
   Measures BM25 time, dense time, ambiguity time, and fusion time separately.

4. Bottleneck-aware multiprocessing:
   Splits claims into chunks and runs BM25 retrieval across multiple CPU processes.

Outputs:
    results/benchmark/pdc_benchmark_summary.json
    results/benchmark/pdc_benchmark_table.csv
    results/benchmark/pdc_profile_breakdown.json
    results/benchmark/pdc_benchmark_samples.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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


# Globals used by multiprocessing BM25 workers.
_MP_BM25 = None
_MP_CORPUS = None


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]


def read_jsonl(path: Path, max_rows: int | None = None) -> List[Dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            rows.append(json.loads(line))

            if max_rows is not None and len(rows) >= max_rows:
                break

    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    for i, row in enumerate(rows):
        records.append(
            {
                "index": i,
                "id": str(row.get("id", "")),
                "query": make_query(row),
                "entity": str(row.get("entity", "")),
                "claim": str(row.get("claim", "")),
                "passage": str(row.get("passage", "")),
            }
        )

    return records


def bm25_search_single(
    bm25,
    corpus: List[Dict[str, Any]],
    query: str,
    top_k: int,
    exclude_id: str | None = None,
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


def bm25_batch_search(
    bm25,
    corpus: List[Dict[str, Any]],
    query_records: List[Dict[str, Any]],
    top_k: int,
) -> List[List[Dict[str, Any]]]:
    outputs = []

    for record in query_records:
        outputs.append(
            bm25_search_single(
                bm25=bm25,
                corpus=corpus,
                query=record["query"],
                top_k=top_k,
                exclude_id=record["id"],
            )
        )

    return outputs


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

    for start in range(0, len(queries), batch_size):
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
    return [detect_ambiguity_single(record) for record in query_records]


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


def fusion_batch(
    query_records: List[Dict[str, Any]],
    bm25_outputs: List[List[Dict[str, Any]]],
    dense_outputs: List[List[Dict[str, Any]]],
    ambiguity_outputs: List[Dict[str, Any]],
    rrf_k: int,
    top_k_final: int,
) -> List[Dict[str, Any]]:
    final_outputs = []

    for record, bm25_result, dense_result, ambiguity_info in zip(
        query_records,
        bm25_outputs,
        dense_outputs,
        ambiguity_outputs,
    ):
        retrieved = reciprocal_rank_fusion_single(
            bm25_results=bm25_result,
            dense_results=dense_result,
            entity=record.get("entity", ""),
            ambiguity_info=ambiguity_info,
            rrf_k=rrf_k,
            top_k_final=top_k_final,
        )

        final_outputs.append(
            {
                "id": record["id"],
                "claim": record["claim"],
                "query": record["query"],
                "ambiguity": ambiguity_info,
                "retrieved": retrieved,
            }
        )

    return final_outputs


def _mp_init_worker(bm25_path: str, corpus_path: str) -> None:
    global _MP_BM25, _MP_CORPUS

    _MP_BM25 = load_bm25(Path(bm25_path))
    _MP_CORPUS = read_jsonl(Path(corpus_path))


def _mp_bm25_chunk_worker(
    chunk: List[Dict[str, Any]],
    top_k: int,
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    global _MP_BM25, _MP_CORPUS

    if _MP_BM25 is None or _MP_CORPUS is None:
        raise RuntimeError("Multiprocessing worker was not initialized.")

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


def chunk_records(records: List[Dict[str, Any]], num_chunks: int) -> List[List[Dict[str, Any]]]:
    if num_chunks <= 1:
        return [records]

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
) -> List[List[Dict[str, Any]]]:
    chunks = chunk_records(query_records, workers)

    ordered_outputs: List[List[Dict[str, Any]] | None] = [None] * len(query_records)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_mp_init_worker,
        initargs=(str(bm25_path), str(corpus_path)),
    ) as executor:
        futures = [
            executor.submit(_mp_bm25_chunk_worker, chunk, top_k)
            for chunk in chunks
        ]

        for future in as_completed(futures):
            chunk_outputs = future.result()

            for original_index, result in chunk_outputs:
                ordered_outputs[original_index] = result

    if any(item is None for item in ordered_outputs):
        raise RuntimeError("Some multiprocessing BM25 outputs are missing.")

    return ordered_outputs  # type: ignore


def benchmark_sequential(
    query_records: List[Dict[str, Any]],
    bm25,
    dense_model,
    faiss_index,
    corpus: List[Dict[str, Any]],
    top_k_bm25: int,
    top_k_dense: int,
    top_k_final: int,
    rrf_k: int,
    dense_batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    t0 = time.perf_counter()

    bm25_start = time.perf_counter()
    bm25_outputs = bm25_batch_search(bm25, corpus, query_records, top_k_bm25)
    bm25_time = time.perf_counter() - bm25_start

    dense_start = time.perf_counter()
    dense_outputs = dense_batch_search(
        dense_model=dense_model,
        faiss_index=faiss_index,
        corpus=corpus,
        query_records=query_records,
        top_k=top_k_dense,
        batch_size=dense_batch_size,
    )
    dense_time = time.perf_counter() - dense_start

    amb_start = time.perf_counter()
    ambiguity_outputs = ambiguity_batch_detect(query_records)
    ambiguity_time = time.perf_counter() - amb_start

    fusion_start = time.perf_counter()
    final_outputs = fusion_batch(
        query_records=query_records,
        bm25_outputs=bm25_outputs,
        dense_outputs=dense_outputs,
        ambiguity_outputs=ambiguity_outputs,
        rrf_k=rrf_k,
        top_k_final=top_k_final,
    )
    fusion_time = time.perf_counter() - fusion_start

    total_time = time.perf_counter() - t0
    n = len(query_records)

    metrics = {
        "mode": "sequential",
        "workers": 1,
        "num_claims": n,
        "total_time_seconds": total_time,
        "claims_per_second": n / total_time if total_time > 0 else 0.0,
        "avg_latency_ms": (total_time / n) * 1000 if n > 0 else 0.0,
    }

    profile = {
        "bm25_time_seconds": bm25_time,
        "dense_time_seconds": dense_time,
        "ambiguity_time_seconds": ambiguity_time,
        "fusion_time_seconds": fusion_time,
        "total_time_seconds": total_time,
    }

    return metrics, final_outputs, profile


def benchmark_async_task_level(
    query_records: List[Dict[str, Any]],
    bm25,
    dense_model,
    faiss_index,
    corpus: List[Dict[str, Any]],
    top_k_bm25: int,
    top_k_dense: int,
    top_k_final: int,
    rrf_k: int,
    dense_batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=3) as executor:
        bm25_future = executor.submit(
            bm25_batch_search,
            bm25,
            corpus,
            query_records,
            top_k_bm25,
        )

        dense_future = executor.submit(
            dense_batch_search,
            dense_model,
            faiss_index,
            corpus,
            query_records,
            top_k_dense,
            dense_batch_size,
        )

        ambiguity_future = executor.submit(
            ambiguity_batch_detect,
            query_records,
        )

        bm25_outputs = bm25_future.result()
        dense_outputs = dense_future.result()
        ambiguity_outputs = ambiguity_future.result()

    final_outputs = fusion_batch(
        query_records=query_records,
        bm25_outputs=bm25_outputs,
        dense_outputs=dense_outputs,
        ambiguity_outputs=ambiguity_outputs,
        rrf_k=rrf_k,
        top_k_final=top_k_final,
    )

    total_time = time.perf_counter() - t0
    n = len(query_records)

    metrics = {
        "mode": "async_task_level",
        "workers": 3,
        "num_claims": n,
        "total_time_seconds": total_time,
        "claims_per_second": n / total_time if total_time > 0 else 0.0,
        "avg_latency_ms": (total_time / n) * 1000 if n > 0 else 0.0,
    }

    return metrics, final_outputs


def benchmark_bm25_multiprocessing(
    query_records: List[Dict[str, Any]],
    bm25_path: Path,
    corpus_path: Path,
    dense_model,
    faiss_index,
    corpus: List[Dict[str, Any]],
    top_k_bm25: int,
    top_k_dense: int,
    top_k_final: int,
    rrf_k: int,
    dense_batch_size: int,
    workers: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    t0 = time.perf_counter()

    bm25_start = time.perf_counter()
    bm25_outputs = bm25_multiprocessing_search(
        query_records=query_records,
        bm25_path=bm25_path,
        corpus_path=corpus_path,
        top_k=top_k_bm25,
        workers=workers,
    )
    bm25_time = time.perf_counter() - bm25_start

    dense_start = time.perf_counter()
    dense_outputs = dense_batch_search(
        dense_model=dense_model,
        faiss_index=faiss_index,
        corpus=corpus,
        query_records=query_records,
        top_k=top_k_dense,
        batch_size=dense_batch_size,
    )
    dense_time = time.perf_counter() - dense_start

    amb_start = time.perf_counter()
    ambiguity_outputs = ambiguity_batch_detect(query_records)
    ambiguity_time = time.perf_counter() - amb_start

    fusion_start = time.perf_counter()
    final_outputs = fusion_batch(
        query_records=query_records,
        bm25_outputs=bm25_outputs,
        dense_outputs=dense_outputs,
        ambiguity_outputs=ambiguity_outputs,
        rrf_k=rrf_k,
        top_k_final=top_k_final,
    )
    fusion_time = time.perf_counter() - fusion_start

    total_time = time.perf_counter() - t0
    n = len(query_records)

    metrics = {
        "mode": "bm25_multiprocessing",
        "workers": workers,
        "num_claims": n,
        "total_time_seconds": total_time,
        "claims_per_second": n / total_time if total_time > 0 else 0.0,
        "avg_latency_ms": (total_time / n) * 1000 if n > 0 else 0.0,
    }

    profile = {
        "bm25_time_seconds": bm25_time,
        "dense_time_seconds": dense_time,
        "ambiguity_time_seconds": ambiguity_time,
        "fusion_time_seconds": fusion_time,
        "total_time_seconds": total_time,
    }

    return metrics, final_outputs, profile


def add_speedup(metrics: Dict[str, Any], sequential_time: float) -> Dict[str, Any]:
    metrics = dict(metrics)
    metrics["speedup_vs_sequential"] = (
        sequential_time / metrics["total_time_seconds"]
        if metrics["total_time_seconds"] > 0
        else 0.0
    )
    return metrics


def profile_percentages(profile: Dict[str, float]) -> Dict[str, float]:
    total = profile.get("total_time_seconds", 0.0)

    if total <= 0:
        return {}

    return {
        "bm25_percent": 100.0 * profile.get("bm25_time_seconds", 0.0) / total,
        "dense_percent": 100.0 * profile.get("dense_time_seconds", 0.0) / total,
        "ambiguity_percent": 100.0 * profile.get("ambiguity_time_seconds", 0.0) / total,
        "fusion_percent": 100.0 * profile.get("fusion_time_seconds", 0.0) / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDC benchmark: sequential vs async vs BM25 multiprocessing.")

    parser.add_argument("--input", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--index_dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--mp_workers", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--warmup_rows", type=int, default=5)

    parser.add_argument("--top_k_bm25", type=int, default=20)
    parser.add_argument("--top_k_dense", type=int, default=20)
    parser.add_argument("--top_k_final", type=int, default=3)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--dense_batch_size", type=int, default=128)

    parser.add_argument("--output_json", type=Path, default=Path("results/benchmark/pdc_benchmark_summary.json"))
    parser.add_argument("--output_csv", type=Path, default=Path("results/benchmark/pdc_benchmark_table.csv"))
    parser.add_argument("--output_profile", type=Path, default=Path("results/benchmark/pdc_profile_breakdown.json"))
    parser.add_argument("--output_samples", type=Path, default=Path("results/benchmark/pdc_benchmark_samples.json"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_path = args.index_dir / "corpus.jsonl"
    bm25_path = args.index_dir / "bm25.pkl"
    faiss_path = args.index_dir / "faiss.index"

    max_needed = max(args.sizes) + args.warmup_rows

    print("\n=== Loading data and indexes ===")
    rows = read_jsonl(args.input, max_rows=max_needed)
    corpus = read_jsonl(corpus_path)
    bm25 = load_bm25(bm25_path)
    faiss_index = faiss.read_index(str(faiss_path))
    dense_model = SentenceTransformer(args.dense_model, device=args.device)

    print(f"Input rows loaded: {len(rows)}")
    print(f"Corpus rows:       {len(corpus)}")
    print(f"FAISS vectors:     {faiss_index.ntotal}")
    print(f"Benchmark sizes:   {args.sizes}")
    print(f"MP workers:        {args.mp_workers}")

    warmup_records = make_query_records(rows[: args.warmup_rows])

    if warmup_records:
        print("\n=== Warmup ===")
        _ = bm25_batch_search(bm25, corpus, warmup_records, args.top_k_bm25)
        _ = dense_batch_search(
            dense_model=dense_model,
            faiss_index=faiss_index,
            corpus=corpus,
            query_records=warmup_records,
            top_k=args.top_k_dense,
            batch_size=args.dense_batch_size,
        )
        _ = ambiguity_batch_detect(warmup_records)

    benchmark_source = rows[args.warmup_rows :]

    all_metrics = []
    all_profiles = {}
    all_samples = {}

    print("\n=== Benchmarking ===")

    for size in args.sizes:
        print("\n" + "=" * 90)
        print(f"Benchmark size: {size} claims")

        current_rows = benchmark_source[:size]
        query_records = make_query_records(current_rows)

        print("\n[1] Sequential retrieval")
        seq_metrics, seq_outputs, seq_profile = benchmark_sequential(
            query_records=query_records,
            bm25=bm25,
            dense_model=dense_model,
            faiss_index=faiss_index,
            corpus=corpus,
            top_k_bm25=args.top_k_bm25,
            top_k_dense=args.top_k_dense,
            top_k_final=args.top_k_final,
            rrf_k=args.rrf_k,
            dense_batch_size=args.dense_batch_size,
        )

        seq_metrics = add_speedup(seq_metrics, seq_metrics["total_time_seconds"])
        all_metrics.append(seq_metrics)

        seq_profile_full = {
            **seq_profile,
            **profile_percentages(seq_profile),
        }
        all_profiles[f"{size}_sequential"] = seq_profile_full

        print(f"Sequential time: {seq_metrics['total_time_seconds']:.2f}s")
        print(f"Claims/sec:      {seq_metrics['claims_per_second']:.2f}")

        print("\n[2] Async task-level retrieval")
        async_metrics, async_outputs = benchmark_async_task_level(
            query_records=query_records,
            bm25=bm25,
            dense_model=dense_model,
            faiss_index=faiss_index,
            corpus=corpus,
            top_k_bm25=args.top_k_bm25,
            top_k_dense=args.top_k_dense,
            top_k_final=args.top_k_final,
            rrf_k=args.rrf_k,
            dense_batch_size=args.dense_batch_size,
        )

        async_metrics = add_speedup(async_metrics, seq_metrics["total_time_seconds"])
        all_metrics.append(async_metrics)

        print(f"Async time:      {async_metrics['total_time_seconds']:.2f}s")
        print(f"Claims/sec:      {async_metrics['claims_per_second']:.2f}")
        print(f"Speedup:         {async_metrics['speedup_vs_sequential']:.2f}x")

        print("\n[3] BM25 multiprocessing retrieval")

        for workers in args.mp_workers:
            print(f"\nBM25 multiprocessing with {workers} workers")

            mp_metrics, mp_outputs, mp_profile = benchmark_bm25_multiprocessing(
                query_records=query_records,
                bm25_path=bm25_path,
                corpus_path=corpus_path,
                dense_model=dense_model,
                faiss_index=faiss_index,
                corpus=corpus,
                top_k_bm25=args.top_k_bm25,
                top_k_dense=args.top_k_dense,
                top_k_final=args.top_k_final,
                rrf_k=args.rrf_k,
                dense_batch_size=args.dense_batch_size,
                workers=workers,
            )

            mp_metrics = add_speedup(mp_metrics, seq_metrics["total_time_seconds"])
            all_metrics.append(mp_metrics)

            mp_profile_full = {
                **mp_profile,
                **profile_percentages(mp_profile),
            }
            all_profiles[f"{size}_bm25_mp_{workers}_workers"] = mp_profile_full

            print(f"MP time:         {mp_metrics['total_time_seconds']:.2f}s")
            print(f"Claims/sec:      {mp_metrics['claims_per_second']:.2f}")
            print(f"Speedup:         {mp_metrics['speedup_vs_sequential']:.2f}x")

        all_samples[str(size)] = {
            "sequential_samples": seq_outputs[:3],
            "async_samples": async_outputs[:3],
        }

    save_json(args.output_json, all_metrics)
    save_csv(args.output_csv, all_metrics)
    save_json(args.output_profile, all_profiles)
    save_json(args.output_samples, all_samples)

    print("\n=== Saved benchmark outputs ===")
    print(args.output_json)
    print(args.output_csv)
    print(args.output_profile)
    print(args.output_samples)


if __name__ == "__main__":
    main()