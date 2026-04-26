"""
Main GNN training script.

Usage:
    python3 training/train_gnn.py
    python3 training/train_gnn.py --epochs 50 --lr 0.001 --batch-size 64
"""

import argparse
import json
import os

import torch

from data.loader import create_data_loaders
from models.gnn import DefectPredictionGNN, DefectLoss
from training.gnn_trainer import GNNTrainer
from training.utils import set_seed, get_device, plot_training_curves
from evaluation.metrics import DefectMetrics, timed_inference


def parse_args():
    p = argparse.ArgumentParser(description="Train GNN defect predictor")
    p.add_argument("--data",       default="data/raw/synthetic_wafers.json")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden",     type=int,   default=64)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Data
    train_loader, val_loader, test_loader = create_data_loaders(
        json_path=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Infer input feature dimension from first batch
    sample_batch = next(iter(train_loader))
    in_channels = sample_batch.x.shape[1]
    print(f"Input feature dim: {in_channels}")

    # Model
    model = DefectPredictionGNN(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        dropout=args.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = DefectLoss(type_weight=1.0, location_weight=0.5, severity_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )

    # Train
    trainer = GNNTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    history = trainer.fit(train_loader, val_loader, n_epochs=args.epochs, patience=args.patience)

    # Evaluate on test set using best checkpoint
    print("\nEvaluating best model on test set...")
    trainer.load_checkpoint("best_gnn.pt")
    test_metrics = DefectMetrics()

    trainer.model.eval()
    for batch in test_loader:
        outputs, inf_ms = timed_inference(trainer.model, batch, device=str(device))
        type_logits, location, severity = outputs
        y_loc = batch.y_loc.view(-1, 2).to(device)
        test_metrics.update(
            type_logits, location, severity,
            batch.y_type.to(device), y_loc, batch.y_severity.to(device),
            inference_time_ms=inf_ms,
        )

    final = test_metrics.compute()
    print("\n=== Test Set Results ===")
    print(f"  Type Accuracy : {final['type_accuracy']:.4f}  (target >0.85)")
    print(f"  Location MSE  : {final['location_mse']:.4f}  (target <0.05)")
    print(f"  Severity RMSE : {final['severity_rmse']:.4f}  (target <0.60)")
    print(f"  Inference     : {final['avg_inference_ms']:.2f} ms  (target <30ms)")
    print("\nClassification Report:")
    print(test_metrics.classification_report())

    # Save artefacts
    os.makedirs(args.results_dir, exist_ok=True)

    metrics_path = os.path.join(args.results_dir, "gnn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"test": final, "history": history}, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    plot_training_curves(history, save_path=os.path.join(args.results_dir, "gnn_training_curves.png"))


if __name__ == "__main__":
    main()
