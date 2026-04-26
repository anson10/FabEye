"""
GNN architecture for semiconductor defect prediction.

DefectPredictionGNN uses three GCN layers to propagate information
across the process-step graph, then branches into three prediction heads:
  - type_head:     6-class softmax  (defect type)
  - location_head: 2-output sigmoid (x, y location normalized [0,1])
  - severity_head: 1-output sigmoid (severity score [0,1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class DefectPredictionGNN(nn.Module):
    """
    Args:
        in_channels:   number of input features per node (process-step params)
        hidden_channels: width of GCN hidden layers
        n_defect_types: number of defect classes (default 6)
        dropout:       dropout probability applied after each GCN layer
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        n_defect_types: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Three GCN message-passing layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # Batch-norm after each conv for training stability
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # Prediction heads (operate on graph-level embedding)
        self.type_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, n_defect_types),
        )

        self.location_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2),
            nn.Sigmoid(),
        )

        self.severity_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            x:          node features  [total_nodes, in_channels]
            edge_index: edge list      [2, total_edges]
            batch:      batch vector   [total_nodes]  (PyG batching)

        Returns:
            type_logits:  [B, n_defect_types]
            location:     [B, 2]   values in [0,1]
            severity:     [B, 1]   values in [0,1]
        """
        # GCN layers with residual-style skip where dimensions match
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.relu(self.bn3(self.conv3(h, edge_index)))

        # Pool all node embeddings to a single graph-level vector
        graph_emb = global_mean_pool(h, batch)  # [B, hidden_channels]

        type_logits = self.type_head(graph_emb)
        location    = self.location_head(graph_emb)
        severity    = self.severity_head(graph_emb)

        return type_logits, location, severity


class DefectLoss(nn.Module):
    """
    Combined loss for the three prediction heads.

    type_loss:     cross-entropy over defect classes
    location_loss: MSE (only on defective wafers where has_defect=True)
    severity_loss: MSE (only on defective wafers)

    Args:
        type_weight:     weight for classification loss
        location_weight: weight for location regression loss
        severity_weight: weight for severity regression loss
    """

    def __init__(
        self,
        type_weight: float = 1.0,
        location_weight: float = 0.5,
        severity_weight: float = 0.5,
    ):
        super().__init__()
        self.type_weight     = type_weight
        self.location_weight = location_weight
        self.severity_weight = severity_weight
        self.ce_loss  = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        type_logits: torch.Tensor,
        location_pred: torch.Tensor,
        severity_pred: torch.Tensor,
        y_type: torch.Tensor,
        y_loc: torch.Tensor,
        y_severity: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:

        type_loss = self.ce_loss(type_logits, y_type)

        # Compute regression losses only on defective samples
        defect_mask = (y_type > 0)
        if defect_mask.sum() > 0:
            loc_loss = self.mse_loss(location_pred[defect_mask], y_loc[defect_mask])
            sev_loss = self.mse_loss(
                severity_pred[defect_mask].squeeze(-1),
                y_severity[defect_mask],
            )
        else:
            loc_loss = torch.tensor(0.0, device=type_logits.device)
            sev_loss = torch.tensor(0.0, device=type_logits.device)

        total = (
            self.type_weight     * type_loss
            + self.location_weight * loc_loss
            + self.severity_weight * sev_loss
        )

        return total, {
            "type_loss":     type_loss.item(),
            "location_loss": loc_loss.item(),
            "severity_loss": sev_loss.item(),
            "total_loss":    total.item(),
        }


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    # Smoke test: 2 wafers, 8 process steps each, 3 features per step
    def _make_dummy(n_steps=8, feat_dim=3):
        x = torch.randn(n_steps, feat_dim)
        edge_index = torch.tensor(
            [[i, i + 1] for i in range(n_steps - 1)]
            + [[i + 1, i] for i in range(n_steps - 1)],
            dtype=torch.long,
        ).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    batch = Batch.from_data_list([_make_dummy(), _make_dummy()])
    model = DefectPredictionGNN(in_channels=3, hidden_channels=64)
    type_logits, location, severity = model(batch.x, batch.edge_index, batch.batch)

    print(f"type_logits shape: {type_logits.shape}")   # [2, 6]
    print(f"location shape:    {location.shape}")       # [2, 2]
    print(f"severity shape:    {severity.shape}")       # [2, 1]

    criterion = DefectLoss()
    y_type = torch.tensor([0, 2])
    y_loc  = torch.tensor([[0.0, 0.0], [0.4, 0.6]])
    y_sev  = torch.tensor([0.0, 0.7])
    loss, breakdown = criterion(type_logits, location, severity, y_type, y_loc, y_sev)
    print(f"Loss: {loss.item():.4f}  breakdown: {breakdown}")
