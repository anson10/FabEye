| Backend | Batch | p50 ms | p95 ms | p99 ms | Wafers/s |
|---|---|---|---|---|---|
| pytorch_cpu | 1 | 7.07 | 8.66 | 9.82 | 141 |
| onnxruntime_cpu | 1 | 3.11 | 3.57 | 6.28 | 347 |
| pytorch_cuda | 1 | 3.27 | 6.49 | 8.18 | 274 |
| pytorch_cpu | 32 | 190.14 | 207.18 | 219.05 | 169 |
| onnxruntime_cpu | 32 | 79.56 | 90.18 | 94.99 | 412 |
| pytorch_cuda | 32 | 7.17 | 8.22 | 9.54 | 4363 |
