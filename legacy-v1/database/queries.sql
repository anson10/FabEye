-- Common analytical queries for FabEye results
-- Run against PostgreSQL: psql -U postgres -d fabeye -f database/queries.sql

-- 1. Best run per model type
SELECT model_type, run_name, type_accuracy, location_mse, severity_rmse, avg_inference_ms
FROM experiments
WHERE type_accuracy IS NOT NULL
ORDER BY model_type, type_accuracy DESC;

-- 2. Full epoch history for the best GNN run
SELECT e.run_name, em.epoch, em.train_loss, em.val_loss, em.val_type_acc, em.val_loc_mse
FROM epoch_metrics em
JOIN experiments e ON e.id = em.experiment_id
WHERE e.id = (
    SELECT id FROM experiments
    WHERE model_type = 'gnn' AND type_accuracy IS NOT NULL
    ORDER BY type_accuracy DESC LIMIT 1
)
ORDER BY em.epoch;

-- 3. Experiments that hit the 85% accuracy target
SELECT run_name, type_accuracy, location_mse, severity_rmse, learning_rate, hidden_channels
FROM experiments
WHERE model_type = 'gnn' AND type_accuracy >= 0.85
ORDER BY type_accuracy DESC;

-- 4. Learning rate vs final accuracy (hyperparameter analysis)
SELECT learning_rate, hidden_channels, dropout,
       ROUND(AVG(type_accuracy)::numeric, 4) AS avg_acc,
       COUNT(*) AS n_runs
FROM experiments
WHERE model_type = 'gnn' AND type_accuracy IS NOT NULL
GROUP BY learning_rate, hidden_channels, dropout
ORDER BY avg_acc DESC;

-- 5. Epoch at which best validation accuracy was reached per run
SELECT e.run_name, em.epoch, em.val_type_acc
FROM epoch_metrics em
JOIN experiments e ON e.id = em.experiment_id
WHERE em.val_type_acc = (
    SELECT MAX(val_type_acc) FROM epoch_metrics
    WHERE experiment_id = em.experiment_id
)
ORDER BY em.val_type_acc DESC;
