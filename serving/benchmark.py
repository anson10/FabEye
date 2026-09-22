"""Latency and throughput benchmark: PyTorch CPU/GPU vs ONNX Runtime.

Usage: python serving/benchmark.py --ckpt ... --onnx serving/wafer_cnn.onnx
Writes results/latency_benchmark.json and results/latency_benchmark.md
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.wm_models import WaferCNN  # noqa: E402


def timeit(fn, warmup=30, iters=300):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return ts


def stats(ts, batch):
    return {
        "p50_ms": float(np.percentile(ts, 50)),
        "p95_ms": float(np.percentile(ts, 95)),
        "p99_ms": float(np.percentile(ts, 99)),
        "throughput_wafers_per_s": float(batch / (ts.mean() / 1000)),
    }


def main(ckpt, onnx_path, threads):
    torch.set_num_threads(threads)
    model = WaferCNN()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    rows = {}
    for batch in (1, 32):
        x = torch.randn(batch, 3, 64, 64)
        xn = x.numpy()
        with torch.no_grad():
            rows[f"pytorch_cpu_b{batch}"] = stats(timeit(lambda: model(x)), batch)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        rows[f"onnxruntime_cpu_b{batch}"] = stats(
            timeit(lambda: sess.run(None, {"wafer": xn})), batch
        )
        if torch.cuda.is_available():
            mg, xg = model.cuda(), x.cuda()

            def gpu():
                with torch.no_grad():
                    mg(xg)
                torch.cuda.synchronize()

            rows[f"pytorch_cuda_b{batch}"] = stats(timeit(gpu), batch)
            model.cpu()
        if "CUDAExecutionProvider" in ort.get_available_providers():
            sg = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
            rows[f"onnxruntime_cuda_b{batch}"] = stats(
                timeit(lambda: sg.run(None, {"wafer": xn})), batch
            )
    meta = {
        "threads": threads,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
        "providers": ort.get_available_providers(),
    }
    with open("results/latency_benchmark.json", "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    lines = [
        "| Backend | Batch | p50 ms | p95 ms | p99 ms | Wafers/s |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in rows.items():
        name, b = k.rsplit("_b", 1)
        lines.append(
            f"| {name} | {b} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} | "
            f"{v['p99_ms']:.2f} | {v['throughput_wafers_per_s']:.0f} |"
        )
    open("results/latency_benchmark.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/wm_lot_cnn_seed0.pt")
    ap.add_argument("--onnx", default="serving/wafer_cnn.onnx")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    main(a.ckpt, a.onnx, a.threads)
