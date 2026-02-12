# Databricks notebook source
# MAGIC %md
# MAGIC # Demo 5: Portfolio Model Calibration with Ray on Databricks
# MAGIC
# MAGIC **Scenario:** Optimize a portfolio risk model and decision threshold for precision/recall trade-offs.

# COMMAND ----------

# Install required packages
%pip install -q ray[default,tune]==2.7.1 mlflow==2.9.2 click==8.0.4 xgboost==2.0.3
dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import xgboost as xgb

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import sys
sys.path.append("/Workspace/Shared/acceleray/files/src")
from utils.ray_cluster import RayClusterManager, print_cluster_info
from data.data_loader import DataLoader

print("Imports loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Initialize Ray Cluster

# COMMAND ----------

cluster_manager = RayClusterManager()
cluster_info = cluster_manager.initialize_cluster(
    num_worker_nodes=4,
    num_cpus_per_node=4,
    collect_log_to_path="/dbfs/ray_logs/portfolio_model_calibration"
)
print_cluster_info()
health = cluster_manager.health_check(timeout_seconds=30)
print(f"Ray health check passed ({health['latency_ms']} ms)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Generate Portfolio Risk Dataset

# COMMAND ----------

data_loader = DataLoader()
X, y = data_loader.load_sample_classification_data(
    n_samples=120000,
    n_features=30,
    n_classes=2,
    class_sep=1.1,
    random_state=42
)

# Build realistic rare-event imbalance (~2.5% positives)
high_risk_fraction = 0.025
positive_idx = np.where(y == 1)[0]
negative_idx = np.where(y == 0)[0]
np.random.seed(42)
keep_pos = np.random.choice(positive_idx, size=int(len(X) * high_risk_fraction), replace=False)
keep_neg = np.random.choice(negative_idx, size=len(X) - len(keep_pos), replace=False)
keep_idx = np.concatenate([keep_pos, keep_neg])
np.random.shuffle(keep_idx)

X = X.iloc[keep_idx].reset_index(drop=True)
y = y.iloc[keep_idx].reset_index(drop=True)
print(f"Dataset rows: {len(X):,}")
print(f"High-risk event rate: {y.mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Tune Model and Decision Threshold

# COMMAND ----------

def tune_portfolio_model(config):
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dval = xgb.DMatrix(X_val.values, label=y_val.values)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "subsample": config["subsample"],
        "colsample_bytree": config["colsample_bytree"],
        "tree_method": "hist"
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    preds = model.predict(dval)
    threshold = config["decision_threshold"]
    pred_labels = (preds >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val.values, pred_labels, average="binary", zero_division=0
    )
    aucpr = roc_auc_score(y_val.values, preds)
    business_score = 0.65 * recall + 0.35 * precision

    tune.report(
        precision=precision,
        recall=recall,
        f1=f1,
        aucpr=aucpr,
        business_score=business_score
    )


search_space = {
    "max_depth": tune.choice([3, 4, 5, 6]),
    "learning_rate": tune.choice([0.03, 0.05, 0.1]),
    "subsample": tune.choice([0.7, 0.85, 1.0]),
    "colsample_bytree": tune.choice([0.7, 0.85, 1.0]),
    "n_estimators": tune.choice([40, 80, 120]),
    "decision_threshold": tune.choice([0.25, 0.35, 0.45, 0.55])
}

scheduler = ASHAScheduler(metric="business_score", mode="max", max_t=60, grace_period=5)
analysis = tune.run(
    tune_portfolio_model,
    config=search_space,
    num_samples=8,
    scheduler=scheduler,
    resources_per_trial={"cpu": 2},
    metric="business_score",
    mode="max",
    verbose=1
)

best = analysis.best_config
print("Best model config:", best)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Final Model + MLflow Logging

# COMMAND ----------

dtrain_full = xgb.DMatrix(X_train.values, label=y_train.values)
dtest = xgb.DMatrix(X_test.values, label=y_test.values)
params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": best["max_depth"],
    "learning_rate": best["learning_rate"],
    "subsample": best["subsample"],
    "colsample_bytree": best["colsample_bytree"],
    "tree_method": "hist"
}
final_model = xgb.train(params, dtrain_full, num_boost_round=best["n_estimators"], verbose_eval=False)
test_preds = final_model.predict(dtest)
test_labels = (test_preds >= best["decision_threshold"]).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test.values, test_labels, average="binary", zero_division=0
)

current_user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{current_user}/portfolio-model-calibration-ray")
with mlflow.start_run(run_name="portfolio_threshold_optimization"):
    mlflow.log_params(best)
    mlflow.log_metrics({"precision": precision, "recall": recall, "f1": f1})
    mlflow.xgboost.log_model(final_model, "portfolio_risk_model")

print({"precision": precision, "recall": recall, "f1": f1})

# COMMAND ----------

cluster_manager.shutdown_cluster()
print("Done")
