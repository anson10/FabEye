"""Tests for GNN model architecture and loss function."""

import torch
import pytest
from torch_geometric.data import Data, Batch


def _make_batch(n_graphs: int = 4, n_steps: int = 8, feat_dim: int = 3) -> Batch:
    graphs = []
    for _ in range(n_graphs):
        x = torch.randn(n_steps, feat_dim)
        edges = (
            [[i, i + 1] for i in range(n_steps - 1)]
            + [[i + 1, i] for i in range(n_steps - 1)]
        )
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


class TestDefectPredictionGNN:
    def setup_method(self):
        from models.gnn import DefectPredictionGNN
        self.model = DefectPredictionGNN(in_channels=3, hidden_channels=32)

    def test_output_shapes(self):
        batch = _make_batch(n_graphs=4)
        types, locs, sevs = self.model(batch.x, batch.edge_index, batch.batch)
        assert types.shape == (4, 6)
        assert locs.shape  == (4, 2)
        assert sevs.shape  == (4, 1)

    def test_location_in_range(self):
        batch = _make_batch(n_graphs=8)
        _, locs, sevs = self.model(batch.x, batch.edge_index, batch.batch)
        assert locs.min() >= 0.0 and locs.max() <= 1.0
        assert sevs.min() >= 0.0 and sevs.max() <= 1.0

    def test_type_logits_not_constrained(self):
        """Type logits are raw (pre-softmax), so they can be any real value."""
        batch = _make_batch(n_graphs=4)
        types, _, _ = self.model(batch.x, batch.edge_index, batch.batch)
        assert types.dtype == torch.float32

    def test_grad_flows(self):
        batch = _make_batch(n_graphs=4)
        types, locs, sevs = self.model(batch.x, batch.edge_index, batch.batch)
        loss = types.sum() + locs.sum() + sevs.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_different_batch_sizes(self):
        for bs in [1, 2, 16]:
            batch = _make_batch(n_graphs=bs)
            types, locs, sevs = self.model(batch.x, batch.edge_index, batch.batch)
            assert types.shape[0] == bs


class TestDefectLoss:
    def setup_method(self):
        from models.gnn import DefectLoss
        self.criterion = DefectLoss()

    def test_loss_is_scalar(self):
        bs = 8
        type_logits = torch.randn(bs, 6)
        loc_pred    = torch.sigmoid(torch.randn(bs, 2))
        sev_pred    = torch.sigmoid(torch.randn(bs, 1))
        y_type      = torch.randint(0, 6, (bs,))
        y_loc       = torch.rand(bs, 2)
        y_sev       = torch.rand(bs)
        loss, breakdown = self.criterion(type_logits, loc_pred, sev_pred, y_type, y_loc, y_sev)
        assert loss.shape == ()
        assert loss.item() > 0
        assert set(breakdown.keys()) == {"type_loss", "location_loss", "severity_loss", "total_loss"}

    def test_loss_all_no_defect(self):
        """When all samples are class 0 (no defect), regression losses should be 0."""
        bs = 8
        type_logits = torch.zeros(bs, 6)
        type_logits[:, 0] = 10.0  # strongly predict class 0
        loc_pred = torch.rand(bs, 2)
        sev_pred = torch.rand(bs, 1)
        y_type   = torch.zeros(bs, dtype=torch.long)
        y_loc    = torch.zeros(bs, 2)
        y_sev    = torch.zeros(bs)
        loss, breakdown = self.criterion(type_logits, loc_pred, sev_pred, y_type, y_loc, y_sev)
        assert breakdown["location_loss"] == 0.0
        assert breakdown["severity_loss"] == 0.0

    def test_loss_decreases_on_trivial_case(self):
        """A perfect prediction should give lower loss than a random one."""
        bs = 4
        y_type = torch.tensor([1, 2, 0, 3])
        y_loc  = torch.tensor([[0.3, 0.4], [0.5, 0.6], [0.0, 0.0], [0.7, 0.2]])
        y_sev  = torch.tensor([0.5, 0.8, 0.0, 0.3])

        # Perfect type prediction
        perfect_logits = torch.zeros(bs, 6)
        for i, t in enumerate(y_type):
            perfect_logits[i, t] = 10.0

        # Random logits
        random_logits = torch.randn(bs, 6)

        loc_pred = y_loc.clone()
        sev_pred = y_sev.unsqueeze(-1).clone()

        from models.gnn import DefectLoss
        crit = DefectLoss()
        loss_perfect, _ = crit(perfect_logits, loc_pred, sev_pred, y_type, y_loc, y_sev)
        loss_random,  _ = crit(random_logits,  loc_pred, sev_pred, y_type, y_loc, y_sev)

        assert loss_perfect.item() < loss_random.item()
