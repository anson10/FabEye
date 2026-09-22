-- FabEye PostgreSQL schema
-- Run once to set up the database:
--   psql -U postgres -d fabeye -f database/schema.sql

-- Stores one record per training run
CREATE TABLE IF NOT EXISTS experiments (
    id              SERIAL PRIMARY KEY,
    run_name        VARCHAR(128) NOT NULL,
    model_type      VARCHAR(32)  NOT NULL DEFAULT 'gnn',
    created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
    -- hyperparameters
    hidden_channels INTEGER,
    dropout         FLOAT,
    learning_rate   FLOAT,
    batch_size      INTEGER,
    n_epochs        INTEGER,
    patience        INTEGER,
    seed            INTEGER,
    -- final test metrics
    type_accuracy   FLOAT,
    location_mse    FLOAT,
    severity_rmse   FLOAT,
    avg_inference_ms FLOAT,
    notes           TEXT
);

-- Per-epoch metrics for each experiment
CREATE TABLE IF NOT EXISTS epoch_metrics (
    id              SERIAL PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    epoch           INTEGER NOT NULL,
    train_loss      FLOAT,
    val_loss        FLOAT,
    train_type_acc  FLOAT,
    val_type_acc    FLOAT,
    train_loc_mse   FLOAT,
    val_loc_mse     FLOAT,
    learning_rate   FLOAT,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Checkpoint metadata (actual weights stored on disk)
CREATE TABLE IF NOT EXISTS checkpoints (
    id              SERIAL PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    epoch           INTEGER NOT NULL,
    file_path       VARCHAR(256) NOT NULL,
    val_loss        FLOAT,
    is_best         BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_epoch_metrics_exp ON epoch_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_exp   ON checkpoints(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiments_model ON experiments(model_type);
