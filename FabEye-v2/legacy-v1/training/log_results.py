"""
Hooks that connect the GNNTrainer to the DatabaseManager so every
training run is automatically logged to the database.
"""

from __future__ import annotations
import os
from database.db_utils import DatabaseManager


def log_training_run(
    db: DatabaseManager,
    run_name: str,
    hparams: dict,
    history: dict,
    test_metrics: dict,
    checkpoint_dir: str = "checkpoints",
) -> int:
    """
    Create an experiment record, log all epoch metrics, register the
    best checkpoint, and write final test metrics.

    Returns the experiment id.
    """
    exp_id = db.create_experiment(
        run_name=run_name,
        model_type="gnn",
        hidden_channels=hparams.get("hidden"),
        dropout=hparams.get("dropout"),
        learning_rate=hparams.get("lr"),
        batch_size=hparams.get("batch_size"),
        n_epochs=len(history.get("train_loss", [])),
        patience=hparams.get("patience"),
        seed=hparams.get("seed"),
    )

    # Per-epoch rows
    n_epochs = len(history["train_loss"])
    for epoch in range(n_epochs):
        db.log_epoch(
            experiment_id=exp_id,
            epoch=epoch + 1,
            train_loss=history["train_loss"][epoch],
            val_loss=history["val_loss"][epoch],
            train_type_acc=history["train_type_acc"][epoch],
            val_type_acc=history["val_type_acc"][epoch],
            train_loc_mse=history["train_loc_mse"][epoch],
            val_loc_mse=history["val_loc_mse"][epoch],
            learning_rate=history["lr"][epoch],
        )

    # Best checkpoint
    ckpt_path = os.path.join(checkpoint_dir, "best_gnn.pt")
    if os.path.exists(ckpt_path):
        db.log_checkpoint(
            experiment_id=exp_id,
            epoch=n_epochs,
            file_path=ckpt_path,
            val_loss=min(history["val_loss"]),
            is_best=True,
        )

    # Final test metrics
    db.update_final_metrics(
        experiment_id=exp_id,
        type_accuracy=test_metrics.get("type_accuracy"),
        location_mse=test_metrics.get("location_mse"),
        severity_rmse=test_metrics.get("severity_rmse"),
        avg_inference_ms=test_metrics.get("avg_inference_ms"),
    )

    print(f"Logged experiment '{run_name}' → id={exp_id}")
    return exp_id
