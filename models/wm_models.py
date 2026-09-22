"""Models for WM-811K wafer map pattern classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

N_CLASSES = 9


class WaferCNN(nn.Module):
    """Four-block convnet over the 64x64 one-hot wafer map."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        chans = [3, 32, 64, 128, 256]
        blocks = []
        for a, b in zip(chans[:-1], chans[1:]):
            blocks += [
                nn.Conv2d(a, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(),
                nn.Conv2d(b, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, n_classes))

    def embed(self, x):
        """256-d wafer embedding, used by the lot-context model."""
        return self.features(x).mean((2, 3))

    def forward(self, x):
        return self.head(self.embed(x))


class WaferGNN(nn.Module):
    """GraphSAGE over the die grid with mean+max readout per wafer."""

    def __init__(self, in_dim=6, hidden=64, layers=4, n_classes=N_CLASSES):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden * layers, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        pooled = []
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, ei)))
            pooled += [global_mean_pool(x, b), global_max_pool(x, b)]
        return self.head(torch.cat(pooled, 1))
