"""Common training utilities: seeding, device selection, curve plotting."""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def plot_training_curves(history: dict, save_path: str = "results/gnn_training_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("GNN Training Curves", fontsize=14)

    # Total loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # Type accuracy
    axes[1].plot(history["train_type_acc"], label="Train")
    axes[1].plot(history["val_type_acc"],   label="Val")
    axes[1].axhline(0.85, color="red", linestyle="--", label="Target 85%")
    axes[1].set_title("Type Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True)

    # Location MSE
    axes[2].plot(history["train_loc_mse"], label="Train")
    axes[2].plot(history["val_loc_mse"],   label="Val")
    axes[2].axhline(0.05, color="red", linestyle="--", label="Target 0.05")
    axes[2].set_title("Location MSE")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")
