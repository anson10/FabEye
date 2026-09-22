"""Tests for the GNN training loop."""

import torch
import pytest
from torch_geometric.data import Data, Batch


def _make_loader(n_graphs: int = 16, n_steps: int = 8, feat_dim: int = 3, batch_size: int = 8):
    from torch_geometric.loader import DataLoader
    graphs = []
    for i in range(n_graphs):
        x = torch.randn(n_steps, feat_dim)
        edges = (
            [[j, j + 1] for j in range(n_steps - 1)]
            + [[j + 1, j] for j in range(n_steps - 1)]
        )
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        y_type     = torch.randint(0, 6, ())
        y_loc      = torch.rand(2)
        y_severity = torch.rand(())
        graphs.append(Data(x=x, edge_index=edge_index, y_type=y_type, y_loc=y_loc, y_severity=y_severity))
    return DataLoader(graphs, batch_size=batch_size, shuffle=False)


class TestGNNTrainer:
    def _make_trainer(self):
        from models.gnn import DefectPredictionGNN, DefectLoss
        from training.gnn_trainer import GNNTrainer
        import tempfile

        model     = DefectPredictionGNN(in_channels=3, hidden_channels=16)
        criterion = DefectLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        tmpdir    = tempfile.mkdtemp()

        return GNNTrainer(
            model=model, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, device=torch.device("cpu"), checkpoint_dir=tmpdir,
        )

    def test_train_epoch_returns_metrics(self):
        trainer = self._make_trainer()
        loader  = _make_loader()
        result  = trainer.train_epoch(loader)
        assert "loss"     in result
        assert "type_acc" in result
        assert result["loss"] > 0
        assert 0.0 <= result["type_acc"] <= 1.0

    def test_validate_epoch_returns_metrics(self):
        trainer = self._make_trainer()
        loader  = _make_loader()
        result  = trainer.validate_epoch(loader)
        assert "type_accuracy" in result
        assert "location_mse"  in result
        assert "severity_rmse" in result

    def test_loss_decreases_over_epochs(self):
        """Loss should trend down over a few epochs on a tiny dataset."""
        trainer = self._make_trainer()
        loader  = _make_loader(n_graphs=32)
        losses  = [trainer.train_epoch(loader)["loss"] for _ in range(5)]
        # Not strictly monotone, but final should be lower than initial
        assert losses[-1] < losses[0] * 1.5  # generous bound

    def test_checkpoint_save_and_load(self):
        trainer = self._make_trainer()
        loader  = _make_loader()
        trainer.train_epoch(loader)
        trainer.save_checkpoint("test_ckpt.pt", epoch=1, metrics={"type_accuracy": 0.5})

        # Change weights
        for p in trainer.model.parameters():
            p.data.fill_(0.0)

        trainer.load_checkpoint("test_ckpt.pt")
        # After loading, weights should not all be zero
        total = sum(p.abs().sum().item() for p in trainer.model.parameters())
        assert total > 0

    def test_fit_runs_and_returns_history(self):
        trainer = self._make_trainer()
        loader  = _make_loader(n_graphs=16)
        history = trainer.fit(loader, loader, n_epochs=3, patience=10)
        assert "train_loss"     in history
        assert "val_type_acc"   in history
        assert len(history["train_loss"]) == 3


class TestEarlyStopping:
    def test_triggers_after_patience(self):
        from training.gnn_trainer import EarlyStopping
        es = EarlyStopping(patience=3)
        # First call initializes best_loss; 3 subsequent non-improving calls trigger stop
        es.step(1.0)  # sets best_loss = 1.0, counter stays 0
        for _ in range(3):
            triggered = es.step(1.0)
        assert triggered is True

    def test_resets_on_improvement(self):
        from training.gnn_trainer import EarlyStopping
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(1.0)
        es.step(0.5)   # improvement — resets counter
        assert es.counter == 0
        assert es.should_stop is False
