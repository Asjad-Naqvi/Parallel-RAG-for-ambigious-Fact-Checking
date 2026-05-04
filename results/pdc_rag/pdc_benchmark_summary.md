# PDC Retrieval Benchmark Summary

The retrieval benchmark compared three retrieval execution strategies:

1. Sequential retrieval: BM25 → Dense FAISS → Ambiguity detection → RRF.
2. Async task-level retrieval: BM25, Dense FAISS, and Ambiguity detection run concurrently.
3. BM25 multiprocessing: BM25 retrieval is split across CPU processes, followed by dense retrieval, ambiguity detection, and RRF.

## Key Result

For 1000 claims:

| Method | Runtime | Claims/sec | Speedup |
|---|---:|---:|---:|
| Sequential | 71.48s | 13.99 | 1.00× |
| Async task-level | 64.07s | 15.61 | 1.12× |
| BM25 multiprocessing, 4 workers | 30.87s | 32.40 | 2.32× |
| BM25 multiprocessing, 8 workers | 25.58s | 39.10 | 2.79× |

## Conclusion

Task-level async parallelism produced only limited speedup because the retrieval pipeline was dominated by BM25. Bottleneck-aware BM25 multiprocessing produced the strongest improvement, reducing 1000-claim retrieval time from 71.48s to 25.58s and achieving a 2.79× speedup.
