"""Export a trained WaferCNN checkpoint to ONNX and verify it against PyTorch.

Usage: python serving/export_onnx.py --ckpt checkpoints/wm_lot_cnn_seed0.pt --out serving/wafer_cnn.onnx
The GNN is not exported. Its scatter-based message passing does not map cleanly
to ONNX, so the CNN is the deployable model.
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.wm_models import WaferCNN  # noqa: E402


def export(ckpt, out, opset=17):
    model = WaferCNN()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        out,
        opset_version=opset,
        dynamo=False,
        input_names=["wafer"],
        output_names=["logits"],
        dynamic_axes={"wafer": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(out))

    x = torch.randn(16, 3, 64, 64)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"wafer": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    return diff, os.path.getsize(out) / 1e6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/wm_lot_cnn_seed0.pt")
    ap.add_argument("--out", default="serving/wafer_cnn.onnx")
    a = ap.parse_args()
    diff, mb = export(a.ckpt, a.out)
    print(f"exported {a.out} ({mb:.2f} MB), max abs logit diff vs PyTorch {diff:.2e}")
    assert diff < 1e-3, "ONNX output disagrees with PyTorch"
