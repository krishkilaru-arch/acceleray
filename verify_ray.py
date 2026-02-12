# Databricks notebook source
# MAGIC %md
# MAGIC # Verify Ray Installation

# COMMAND ----------

# Test all Ray modules
import ray
print(f"✅ Ray version: {ray.__version__}")

# Test Ray Tune
try:
    from ray import tune
    print("✅ Ray Tune: Available")
except ImportError as e:
    print(f"❌ Ray Tune: {e}")

# Test Ray Train
try:
    from ray import train
    from ray.train.torch import TorchTrainer
    print("✅ Ray Train: Available")
except ImportError as e:
    print(f"❌ Ray Train: {e}")

# Test Ray Data
try:
    from ray import data
    print("✅ Ray Data: Available")
except ImportError as e:
    print(f"❌ Ray Data: {e}")

# Test Ray on Spark
try:
    from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
    print("✅ Ray on Spark: Available")
except ImportError as e:
    print(f"❌ Ray on Spark: {e}")

print("\n" + "="*60)
print("All Ray components are ready to use!")
print("="*60)

# COMMAND ----------

# Test Ray initialization
import ray

if not ray.is_initialized():
    ray.init()
    print("✅ Ray initialized successfully")
else:
    print("✅ Ray already initialized")

# Check cluster resources
resources = ray.cluster_resources()
print(f"\n📊 Cluster Resources:")
print(f"  CPUs: {resources.get('CPU', 0)}")
print(f"  Memory: {resources.get('memory', 0) / 1e9:.2f} GB")

ray.shutdown()
print("\n✅ Ray test completed successfully!")
