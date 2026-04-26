"""
DatabaseManager — SQLAlchemy-based interface for logging experiment results.

Supports PostgreSQL (production) and SQLite (local dev, no setup required).

PostgreSQL:
    db = DatabaseManager("postgresql://user:pass@localhost/fabeye")

SQLite (default):
    db = DatabaseManager()   # creates fabeye.db in project root
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, text,
    Column, Integer, Float, String, Boolean, Text, DateTime, ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    run_name         = Column(String(128), nullable=False)
    model_type       = Column(String(32), nullable=False, default="gnn")
    created_at       = Column(DateTime, default=datetime.utcnow)
    hidden_channels  = Column(Integer)
    dropout          = Column(Float)
    learning_rate    = Column(Float)
    batch_size       = Column(Integer)
    n_epochs         = Column(Integer)
    patience         = Column(Integer)
    seed             = Column(Integer)
    type_accuracy    = Column(Float)
    location_mse     = Column(Float)
    severity_rmse    = Column(Float)
    avg_inference_ms = Column(Float)
    notes            = Column(Text)

    epoch_metrics = relationship("EpochMetric", back_populates="experiment", cascade="all, delete")
    checkpoints   = relationship("Checkpoint",   back_populates="experiment", cascade="all, delete")


class EpochMetric(Base):
    __tablename__ = "epoch_metrics"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    epoch         = Column(Integer, nullable=False)
    train_loss    = Column(Float)
    val_loss      = Column(Float)
    train_type_acc = Column(Float)
    val_type_acc  = Column(Float)
    train_loc_mse = Column(Float)
    val_loc_mse   = Column(Float)
    learning_rate = Column(Float)
    created_at    = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="epoch_metrics")


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    epoch         = Column(Integer, nullable=False)
    file_path     = Column(String(256), nullable=False)
    val_loss      = Column(Float)
    is_best       = Column(Boolean, default=False)
    created_at    = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="checkpoints")


class DatabaseManager:
    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            db_url = os.getenv("FABEYE_DB_URL", "sqlite:///fabeye.db")
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------
    def create_experiment(self, run_name: str, model_type: str = "gnn", **hparams) -> int:
        """Insert a new experiment row and return its id."""
        with Session(self.engine) as session:
            exp = Experiment(run_name=run_name, model_type=model_type, **hparams)
            session.add(exp)
            session.commit()
            return exp.id

    def log_epoch(self, experiment_id: int, epoch: int, **metrics):
        with Session(self.engine) as session:
            row = EpochMetric(experiment_id=experiment_id, epoch=epoch, **metrics)
            session.add(row)
            session.commit()

    def log_checkpoint(self, experiment_id: int, epoch: int, file_path: str,
                       val_loss: float, is_best: bool = False):
        with Session(self.engine) as session:
            row = Checkpoint(
                experiment_id=experiment_id, epoch=epoch,
                file_path=file_path, val_loss=val_loss, is_best=is_best,
            )
            session.add(row)
            session.commit()

    def update_final_metrics(self, experiment_id: int, **metrics):
        """Write test-set metrics back to the experiment row after training."""
        with Session(self.engine) as session:
            exp = session.get(Experiment, experiment_id)
            for key, val in metrics.items():
                setattr(exp, key, val)
            session.commit()

    # ------------------------------------------------------------------
    def get_best_experiment(self, model_type: str = "gnn") -> Optional[dict]:
        """Return the experiment with the highest type_accuracy."""
        with Session(self.engine) as session:
            exp = (
                session.query(Experiment)
                .filter(Experiment.model_type == model_type,
                        Experiment.type_accuracy.isnot(None))
                .order_by(Experiment.type_accuracy.desc())
                .first()
            )
            if exp is None:
                return None
            return {c.name: getattr(exp, c.name) for c in Experiment.__table__.columns}

    def get_epoch_history(self, experiment_id: int) -> list[dict]:
        with Session(self.engine) as session:
            rows = (
                session.query(EpochMetric)
                .filter(EpochMetric.experiment_id == experiment_id)
                .order_by(EpochMetric.epoch)
                .all()
            )
            return [{c.name: getattr(r, c.name) for c in EpochMetric.__table__.columns}
                    for r in rows]

    def list_experiments(self, model_type: Optional[str] = None) -> list[dict]:
        with Session(self.engine) as session:
            q = session.query(Experiment)
            if model_type:
                q = q.filter(Experiment.model_type == model_type)
            rows = q.order_by(Experiment.created_at.desc()).all()
            return [{c.name: getattr(r, c.name) for c in Experiment.__table__.columns}
                    for r in rows]
