# Databricks notebook source
# MAGIC %md
# MAGIC # Demo 7: Credit Risk Stress Testing with Ray Tasks
# MAGIC
# MAGIC **Scenario:** Run distributed stress scenarios to estimate portfolio loss distributions.

# COMMAND ----------

%pip install -q ray[default]==2.7.1 mlflow==2.9.2 click==8.0.4
dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow

import ray

import sys
sys.path.append("/Workspace/Shared/acceleray/files/src")
from utils.ray_cluster import RayClusterManager, print_cluster_info

print("Imports loaded")

# COMMAND ----------

cluster_manager = RayClusterManager()
cluster_info = cluster_manager.initialize_cluster(
    num_worker_nodes=4,
    num_cpus_per_node=4,
    collect_log_to_path="/dbfs/ray_logs/risk_stress_testing"
)
print_cluster_info()
health = cluster_manager.health_check(timeout_seconds=30)
print(f"Ray health check passed ({health['latency_ms']} ms)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Synthetic Portfolio

# COMMAND ----------

np.random.seed(42)
n_accounts = 200000
portfolio = pd.DataFrame({
    "account_id": np.arange(n_accounts),
    "ead": np.random.lognormal(mean=8.0, sigma=0.7, size=n_accounts),  # exposure at default
    "base_pd": np.clip(np.random.beta(2, 30, size=n_accounts), 0.001, 0.25),
    "lgd": np.clip(np.random.normal(0.45, 0.12, size=n_accounts), 0.1, 0.9),
    "segment": np.random.choice(["retail", "sme", "corp"], size=n_accounts, p=[0.7, 0.2, 0.1])
})

print(portfolio.head())
print(f"Portfolio accounts: {len(portfolio):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Stress Scenario Grid

# COMMAND ----------

scenarios = [
    {"name": "base", "pd_multiplier": 1.0, "lgd_shift": 0.00},
    {"name": "mild_recession", "pd_multiplier": 1.3, "lgd_shift": 0.03},
    {"name": "severe_recession", "pd_multiplier": 1.8, "lgd_shift": 0.07},
    {"name": "rate_shock", "pd_multiplier": 1.5, "lgd_shift": 0.05},
]

n_simulations = 100

@ray.remote
def run_scenario_simulation(scenario, sim_id):
    rng = np.random.default_rng(1000 + sim_id)
    pd_stress = np.clip(portfolio["base_pd"].values * scenario["pd_multiplier"], 0.001, 0.95)
    lgd_stress = np.clip(portfolio["lgd"].values + scenario["lgd_shift"], 0.05, 0.99)

    # Bernoulli default events for each account
    defaults = rng.binomial(1, pd_stress)
    losses = portfolio["ead"].values * lgd_stress * defaults

    return {
        "scenario": scenario["name"],
        "sim_id": sim_id,
        "total_loss": float(losses.sum()),
        "mean_loss_per_account": float(losses.mean()),
        "p99_account_loss": float(np.quantile(losses, 0.99))
    }

futures = []
for sc in scenarios:
    for sim_id in range(n_simulations):
        futures.append(run_scenario_simulation.remote(sc, sim_id))

results = pd.DataFrame(ray.get(futures))
print(results.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Aggregate and Persist Results

# COMMAND ----------

summary = (
    results.groupby("scenario")
    .agg(
        mean_total_loss=("total_loss", "mean"),
        p95_total_loss=("total_loss", lambda s: np.quantile(s, 0.95)),
        p99_total_loss=("total_loss", lambda s: np.quantile(s, 0.99)),
        mean_p99_account_loss=("p99_account_loss", "mean")
    )
    .reset_index()
    .sort_values("mean_total_loss", ascending=False)
)

print(summary)

spark.createDataFrame(results).write.format("delta").mode("overwrite").saveAsTable("risk_stress_simulation_runs")
spark.createDataFrame(summary).write.format("delta").mode("overwrite").saveAsTable("risk_stress_simulation_summary")
print("Saved Delta tables: risk_stress_simulation_runs, risk_stress_simulation_summary")

# COMMAND ----------

current_user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{current_user}/risk-stress-testing-ray")
with mlflow.start_run(run_name="credit_risk_stress_testing"):
    for _, row in summary.iterrows():
        prefix = row["scenario"]
        mlflow.log_metric(f"{prefix}_mean_total_loss", float(row["mean_total_loss"]))
        mlflow.log_metric(f"{prefix}_p99_total_loss", float(row["p99_total_loss"]))
    mlflow.log_param("simulations_per_scenario", n_simulations)
    mlflow.log_param("scenario_count", len(scenarios))

print("Stress testing flow complete")

# COMMAND ----------

cluster_manager.shutdown_cluster()
