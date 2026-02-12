# Databricks notebook source
# MAGIC %md
# MAGIC # Demo 6: Investment Signal Prioritization with Ray
# MAGIC
# MAGIC **Scenario:** Rank investment signals to help analysts focus on highest-priority opportunities and risk flags first.

# COMMAND ----------

%pip install -q ray[default,data]==2.7.1 mlflow==2.9.2 click==8.0.4 xgboost==2.0.3
dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

import ray
from ray import data as ray_data

import sys
sys.path.append("/Workspace/Shared/acceleray/files/src")
from utils.ray_cluster import RayClusterManager, print_cluster_info
from data.data_loader import DataLoader

print("Imports loaded")

# COMMAND ----------

cluster_manager = RayClusterManager()
cluster_info = cluster_manager.initialize_cluster(
    num_worker_nodes=4,
    num_cpus_per_node=4,
    collect_log_to_path="/dbfs/ray_logs/investment_signal_prioritization"
)
print_cluster_info()
health = cluster_manager.health_check(timeout_seconds=30)
print(f"Ray health check passed ({health['latency_ms']} ms)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Build Synthetic Investment Signals

# COMMAND ----------

data_loader = DataLoader()
X, y = data_loader.load_sample_classification_data(
    n_samples=300000,
    n_features=25,
    n_classes=2,
    class_sep=0.9,
    random_state=42
)

# Simulate strongly imbalanced priority labels (~1%)
np.random.seed(42)
y = (np.random.rand(len(y)) < 0.01).astype(int)
X = X.copy()
X["trade_size"] = np.random.lognormal(mean=4.0, sigma=1.0, size=len(X))
X["sector_volatility_flag"] = np.random.binomial(1, 0.15, size=len(X))
X["macro_event_flag"] = np.random.binomial(1, 0.05, size=len(X))
X["momentum_shift_flag"] = np.random.binomial(1, 0.08, size=len(X))

labels = pd.Series(y, name="is_priority_signal")
print(f"Signals: {len(X):,}, priority rate: {labels.mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Train Prioritization Model

# COMMAND ----------

dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
dtest = xgb.DMatrix(X_test.values, label=y_test.values)

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
    "tree_method": "hist"
}

model = xgb.train(params, dtrain, num_boost_round=120, verbose_eval=False)
test_scores = model.predict(dtest)
ap = average_precision_score(y_test.values, test_scores)
print(f"Average precision: {ap:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Distributed Scoring and Priority Buckets

# COMMAND ----------

signals_df = X_test.copy()
signals_df["is_priority_signal"] = y_test.values
signals_df["signal_id"] = np.arange(len(signals_df))

ds = ray_data.from_pandas(signals_df)

class SignalScorer:
    def __init__(self):
        self.model = model

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        features = batch.drop(columns=["is_priority_signal", "signal_id"], errors="ignore")
        dmatrix = xgb.DMatrix(features.values)
        scores = self.model.predict(dmatrix)
        batch = batch.copy()
        batch["risk_score"] = scores
        batch["priority_bucket"] = pd.cut(
            batch["risk_score"],
            bins=[-0.01, 0.2, 0.5, 1.0],
            labels=["P3_low", "P2_medium", "P1_high"]
        )
        return batch

scored = ds.map_batches(SignalScorer, batch_format="pandas", compute=ray.data.ActorPoolStrategy(size=4))
scored_df = scored.to_pandas()

top_queue = scored_df.sort_values("risk_score", ascending=False).head(200)
print(top_queue[["signal_id", "risk_score", "priority_bucket"]].head(10))
print(scored_df["priority_bucket"].value_counts())

# COMMAND ----------

current_user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{current_user}/investment-signal-prioritization-ray")
with mlflow.start_run(run_name="investment_signal_prioritization"):
    mlflow.log_metric("average_precision", ap)
    mlflow.log_metric("top200_hits", int(top_queue["is_priority_signal"].sum()))
    mlflow.log_param("signals_scored", len(scored_df))
    mlflow.xgboost.log_model(model, "signal_prioritization_model")

print("Investment signal prioritization flow complete")

# COMMAND ----------

cluster_manager.shutdown_cluster()
