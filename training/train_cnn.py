"""
Main CNN training script with DistributedDataParallel (DDP) support.

Single GPU:
    PYTHONPATH=. python3 training/train_cnn.py

Multi-GPU (2 GPUs):
    PYTHONPATH=. torchrun --nproc_per_node=2 training/train_cnn.py
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data.image_loader import WaferImageDataset, collate_fn
from torch.utils.data import DataLoader, random_split
from models.cnn import DefectDetectionCNN, DEFECT_NAMES
from training.cnn_trainer import CNNTrainer
from training.utils import set_seed, plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN defect detector")
    p.add_argument("--image-dir",       default="data/wafer_images")
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=5e-3)
    p.add_argument("--patience",        type=int,   default=8)
    p.add_argument("--backbone-layers", type=int,   default=3)
    p.add_argument("--score-thresh",    type=float, default=0.4)
    p.add_argument("--iou-thresh",      type=float, default=0.5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--results-dir",     default="results")
    p.add_argument("--checkpoint-dir",  default="checkpoints")
    p.add_argument("--no-pretrained",   action="store_true")
    return p.parse_args()


def _plot_cnn_curves(history: dict, save_path: str):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CNN Training Curves", fontsize=14)

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Total Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["val_precision"], label="Precision")
    axes[1].plot(history["val_recall"],    label="Recall")
    axes[1].plot(history["val_f1"],        label="F1")
    axes[1].axhline(0.75, color="red", linestyle="--", label="Target 75%")
    axes[1].set_title("Detection Metrics"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(history["val_cls_acc"], label="Cls Acc", color="purple")
    axes[2].axhline(0.80, color="red", linestyle="--", label="Target 80%")
    axes[2].set_title("Classification Accuracy"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


def main():
    args = parse_args()

    # ── DDP setup ────────────────────────────────────────────────────────────
    ddp = "RANK" in os.environ  # torchrun sets this automatically
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    if is_main:
        print(f"Using device: {device}")
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}  ({'DDP torchrun' if ddp else 'single GPU'})")

    set_seed(args.seed)

    # ── Data loaders ─────────────────────────────────────────────────────────
    dataset = WaferImageDataset(args.image_dir)
    n = len(dataset)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    if is_main:
        print(f"Image dataset split — train: {n_train}, val: {n_val}, test: {n_test}")

    loader_kw = dict(collate_fn=collate_fn, num_workers=4, pin_memory=True)
    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader  = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, **loader_kw)
    else:
        train_loader  = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    # ── Model ────────────────────────────────────────────────────────────────
    model = DefectDetectionCNN(
        pretrained=not args.no_pretrained,
        trainable_backbone_layers=args.backbone_layers,
        score_threshold=args.score_thresh,
        nms_iou_threshold=args.iou_thresh,
    )
    if is_main:
        print(f"Model parameters: {model.n_parameters:,}")

    model = model.to(device)
    if ddp:
        model.model = DDP(model.model, device_ids=[local_rank])

    # ── Optimizer ────────────────────────────────────────────────────────────
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
    )

    trainer = CNNTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        iou_threshold=args.iou_thresh,
    )

    if is_main:
        print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")

    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=args.epochs,
        patience=args.patience,
        is_main=is_main,
        ddp_sampler=train_sampler if ddp else None,
    )

    if ddp:
        dist.destroy_process_group()

    # ── Evaluate + save (main process only) ──────────────────────────────────
    if is_main:
        print("\nEvaluating best model on test set...")
        trainer.load_checkpoint("best_cnn.pt")
        test_metrics = trainer.validate_epoch(test_loader)

        trainer.model.eval()
        dummy = [torch.rand(3, 512, 512).to(device)]
        with torch.no_grad():
            for _ in range(5):
                trainer.model(dummy)
            t0 = time.perf_counter()
            for _ in range(20):
                trainer.model(dummy)
        inf_ms = (time.perf_counter() - t0) / 20 * 1000

        print("\n=== Test Set Results ===")
        print(f"  Precision : {test_metrics['precision']:.4f}")
        print(f"  Recall    : {test_metrics['recall']:.4f}  (target >0.75)")
        print(f"  F1        : {test_metrics['f1']:.4f}")
        print(f"  Cls Acc   : {test_metrics['cls_acc']:.4f}  (target >0.80)")
        print(f"  Inference : {inf_ms:.1f} ms  (target <150ms)")
        print(f"  TP/FP/FN  : {test_metrics['tp']}/{test_metrics['fp']}/{test_metrics['fn']}")

        os.makedirs(args.results_dir, exist_ok=True)
        metrics_path = os.path.join(args.results_dir, "cnn_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"test": {**test_metrics, "inference_ms": inf_ms}, "history": history}, f, indent=2)
        print(f"Metrics saved → {metrics_path}")

        _plot_cnn_curves(history, save_path=os.path.join(args.results_dir, "cnn_training_curves.png"))


if __name__ == "__main__":
    main()
