# Databricks notebook source
# MAGIC %pip install -q ray[default]==2.7.1 mlflow==2.9.2 click==8.0.4

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import ray
import sys
sys.path.append("/Workspace/Shared/acceleray/files/src")
from utils.ray_cluster import RayClusterManager

print(f"Ray version: {ray.__version__}")

# COMMAND ----------

cluster_manager = RayClusterManager()
cluster_manager.initialize_cluster(
    num_worker_nodes=2,
    num_cpus_per_node=4
)

print("Ray cluster started")
health = cluster_manager.health_check(timeout_seconds=30)
print(f"Health: {health['status']} ({health['latency_ms']} ms)")
print(f"Resources: {health['available_resources']}")

# COMMAND ----------

@ray.remote
def ping(x: int) -> int:
    return x + 1

result = ray.get(ping.remote(41))
print(f"Smoke test result: {result}")
assert result == 42, "Ray remote execution failed"
print("Smoke test passed")
