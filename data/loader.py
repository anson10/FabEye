"""
PyTorch Geometric data loaders for the GNN pipeline.

Converts synthetic_wafers.json into PyG Data objects where:
  - x          : node feature matrix  [n_steps, feature_dim]
  - edge_index  : COO edge list        [2, n_edges]
  - y_type      : defect type label    scalar int
  - y_loc       : (x, y) location      [2] float
  - y_severity  : severity score       scalar float
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


class WaferGraphDataset(Dataset):
    def __init__(self, json_path: str = "data/raw/synthetic_wafers.json"):
        super().__init__()
        with open(json_path) as f:
            self.samples = json.load(f)

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        s = self.samples[idx]

        x = torch.tensor(s["node_features"], dtype=torch.float)  # [n_steps, feat_dim]

        edges = s["adjacency"]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        defect = s["defect"]
        y_type = torch.tensor(defect["defect_type"], dtype=torch.long)
        y_loc = torch.tensor([defect["location_x"], defect["location_y"]], dtype=torch.float)
        y_severity = torch.tensor(defect["severity"], dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            y_type=y_type,
            y_loc=y_loc,
            y_severity=y_severity,
            wafer_id=s["wafer_id"],
        )


def create_data_loaders(
    json_path: str = "data/raw/synthetic_wafers.json",
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split dataset into train/val/test and return DataLoaders."""
    dataset = WaferGraphDataset(json_path)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"Dataset split — train: {n_train}, val: {n_val}, test: {n_test}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_data_loaders()
    batch = next(iter(train_loader))
    print(f"Batch x shape:      {batch.x.shape}")
    print(f"Batch edge_index:   {batch.edge_index.shape}")
    print(f"Batch y_type:       {batch.y_type.shape}")
    print(f"Batch y_loc:        {batch.y_loc.shape}")
    print(f"Batch y_severity:   {batch.y_severity.shape}")
