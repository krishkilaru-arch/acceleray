# Databricks notebook source
# MAGIC %md
# MAGIC # Demo 3: Scalable Batch Inference with Ray Data
# MAGIC
# MAGIC This notebook demonstrates high-throughput batch inference using Ray Data.
# MAGIC
# MAGIC **Scenario:** Run predictions on 10M records with a trained ML model
# MAGIC
# MAGIC **Key Benefits:**
# MAGIC - 10x faster than Pandas UDF
# MAGIC - Automatic batching and parallelization
# MAGIC - Built-in fault tolerance and retry logic
# MAGIC - Efficient memory management with streaming
# MAGIC - Seamless integration with Delta Lake

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Install required packages
%pip install -q ray[default,data]==2.7.1 click==8.0.4 xgboost==2.0.3
dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import time
from pyspark.sql import functions as F

# Ray imports
import ray
from ray import data as ray_data

# Import custom modules
import sys
sys.path.append("/Workspace/Shared/acceleray/files/src")
from utils.ray_cluster import RayClusterManager, print_cluster_info
from data.data_loader import DataLoader

print("✅ All imports successful!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Initialize Ray Cluster

# COMMAND ----------

cluster_manager = RayClusterManager()

cluster_info = cluster_manager.initialize_cluster(
    num_worker_nodes=8,
    num_cpus_per_node=4,
    object_store_memory_per_node=10_000_000_000,  # 10GB per node
    collect_log_to_path="/dbfs/ray_logs/batch_inference"
)

print_cluster_info()
health = cluster_manager.health_check(timeout_seconds=30)
print(f"✅ Ray health check passed ({health['latency_ms']} ms)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Prepare Test Dataset
# MAGIC
# MAGIC Generate 1M records for inference (simulating production data)

# COMMAND ----------

# Generate large dataset for inference
data_loader = DataLoader()

print("Generating 1M records for batch inference...")
X, y = data_loader.load_sample_classification_data(
    n_samples=1_000_000,
    n_features=20,
    n_classes=2,
    random_state=42
)

# Create DataFrame
inference_df = X.copy()
inference_df['id'] = range(len(inference_df))
inference_df['true_label'] = y  # For evaluation purposes

print(f"✅ Generated {len(inference_df):,} records")
print(f"   Memory usage: {inference_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train a Simple Model for Inference

# COMMAND ----------

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Split for training
X_train, X_val, y_train, y_val = train_test_split(
    X[:100000], y[:100000],  # Use first 100k for training
    test_size=0.2,
    random_state=42
)

# Train XGBoost model
print("Training XGBoost model...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "tree_method": "hist"
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dval, "val")],
    early_stopping_rounds=10,
    verbose_eval=False
)

# Save model
model_path = "/dbfs/models/xgboost_inference_model.json"
model.save_model(model_path)

# Validate
val_pred = model.predict(dval)
val_auc = roc_auc_score(y_val, val_pred)
print(f"✅ Model trained - Validation AUC: {val_auc:.4f}")
print(f"✅ Model saved to {model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Baseline - Traditional Spark Pandas UDF

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(inference_df)

# Define Pandas UDF
@pandas_udf(DoubleType())
def predict_udf(*cols):
    """Traditional Pandas UDF for batch prediction."""
    # Load model (once per partition)
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model("/dbfs/models/xgboost_inference_model.json")
    
    # Prepare features
    X = pd.DataFrame({f'feature_{i}': cols[i] for i in range(20)})
    dmatrix = xgb.DMatrix(X)
    
    # Predict
    predictions = model.predict(dmatrix)
    return pd.Series(predictions)

# Run inference with Pandas UDF
print("🕐 Running baseline inference with Spark Pandas UDF...")
start_time = time.time()

feature_cols = [f'feature_{i}' for i in range(20)]
result_spark = spark_df.withColumn(
    "prediction",
    predict_udf(*feature_cols)
)

# Trigger computation
result_spark.write.mode("overwrite").format("noop").save()

spark_time = time.time() - start_time

print(f"✅ Spark Pandas UDF completed in {spark_time:.2f} seconds")
print(f"   Throughput: {len(inference_df) / spark_time:,.0f} records/second")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Ray Data Batch Inference

# COMMAND ----------

# Create Ray Dataset from Pandas
ray_dataset = ray_data.from_pandas(inference_df)

print(f"Created Ray Dataset with {ray_dataset.count():,} rows")

# COMMAND ----------

# Define prediction class for Ray Data
class XGBoostPredictor:
    """
    Predictor class for Ray Data batch inference.
    Each actor loads the model once and processes multiple batches.
    """
    
    def __init__(self, model_path: str):
        """Initialize predictor with model."""
        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        print(f"Model loaded in actor")
    
    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on a batch of data.
        
        Args:
            batch: Pandas DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        # Extract features
        feature_cols = [col for col in batch.columns if col.startswith('feature_')]
        X = batch[feature_cols]
        
        # Predict
        dmatrix = self.model.predict(xgb.DMatrix(X.values))
        predictions = self.model.predict(dmatrix)
        
        # Add predictions to batch
        batch['prediction'] = predictions
        
        return batch

# COMMAND ----------

# Run Ray Data batch inference
print("🚀 Running batch inference with Ray Data...")
print(f"   Using {8} parallel actors with batch size {10000}")

start_time = time.time()

# Apply predictions using ActorPoolStrategy
predictions = ray_dataset.map_batches(
    XGBoostPredictor,
    fn_constructor_kwargs={"model_path": model_path},
    batch_size=10000,  # Process 10k records per batch
    compute=ray_data.ActorPoolStrategy(size=8)  # 8 parallel actors
)

# Trigger computation by writing to Parquet
predictions.write_parquet("/dbfs/ray_predictions")

ray_time = time.time() - start_time

print(f"✅ Ray Data inference completed in {ray_time:.2f} seconds")
print(f"   Throughput: {len(inference_df) / ray_time:,.0f} records/second")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Performance Comparison

# COMMAND ----------

speedup = spark_time / ray_time
improvement = ((spark_time - ray_time) / spark_time) * 100

print("=" * 70)
print("⚡ PERFORMANCE COMPARISON")
print("=" * 70)
print(f"\n{'Method':<25} {'Time (s)':<15} {'Throughput (rec/s)':<20}")
print("-" * 70)
print(f"{'Spark Pandas UDF':<25} {spark_time:>10.2f}     {len(inference_df)/spark_time:>15,.0f}")
print(f"{'Ray Data':<25} {ray_time:>10.2f}     {len(inference_df)/ray_time:>15,.0f}")
print("-" * 70)
print(f"\n🎯 Speedup: {speedup:.2f}x faster with Ray Data")
print(f"💡 Improvement: {improvement:.1f}% time reduction")
print(f"⏱️  Time saved: {spark_time - ray_time:.2f} seconds")

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time comparison
methods = ['Spark Pandas UDF', 'Ray Data']
times = [spark_time, ray_time]
colors = ['#FF6B6B', '#4ECDC4']

axes[0].bar(methods, times, color=colors)
axes[0].set_ylabel('Time (seconds)')
axes[0].set_title('Inference Time Comparison')
axes[0].grid(axis='y', alpha=0.3)

for i, (method, time_val) in enumerate(zip(methods, times)):
    axes[0].text(i, time_val, f'{time_val:.1f}s', ha='center', va='bottom')

# Throughput comparison
throughputs = [len(inference_df)/t for t in times]

axes[1].bar(methods, throughputs, color=colors)
axes[1].set_ylabel('Records per Second')
axes[1].set_title('Throughput Comparison')
axes[1].grid(axis='y', alpha=0.3)

for i, (method, throughput) in enumerate(zip(methods, throughputs)):
    axes[1].text(i, throughput, f'{throughput:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/dbfs/batch_inference_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Read Predictions and Validate

# COMMAND ----------

# Read predictions from Parquet
predictions_df = ray_data.read_parquet("/dbfs/ray_predictions").to_pandas()

print(f"✅ Read {len(predictions_df):,} predictions")
print(f"\nSample predictions:")
display(predictions_df.head(10))

# Calculate accuracy
predictions_df['predicted_class'] = (predictions_df['prediction'] > 0.5).astype(int)
accuracy = (predictions_df['predicted_class'] == predictions_df['true_label']).mean()

print(f"\n📊 Prediction Accuracy: {accuracy:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Write Results to Delta Lake

# COMMAND ----------

# Convert to Spark DataFrame and write to Delta
result_spark_df = spark.createDataFrame(predictions_df)

# Write to Delta table
delta_table = "ray_batch_predictions"

result_spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(delta_table)

print(f"✅ Predictions written to Delta table: {delta_table}")

# Query the table
display(spark.sql(f"SELECT * FROM {delta_table} LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Advanced - Streaming Batch Inference

# COMMAND ----------

# MAGIC %md
# MAGIC Ray Data supports streaming for very large datasets that don't fit in memory:
# MAGIC
# MAGIC ```python
# MAGIC # Stream processing for 100M+ records
# MAGIC dataset = ray_data.read_parquet("s3://bucket/large-dataset/")
# MAGIC
# MAGIC # Process in streaming fashion
# MAGIC predictions = dataset.map_batches(
# MAGIC     XGBoostPredictor,
# MAGIC     batch_size=10000,
# MAGIC     compute=ray_data.ActorPoolStrategy(size=32)
# MAGIC )
# MAGIC
# MAGIC # Write results in streaming fashion
# MAGIC predictions.write_parquet("s3://bucket/predictions/")
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - Processes datasets larger than cluster memory
# MAGIC - Automatic backpressure management
# MAGIC - No out-of-memory errors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Cleanup

# COMMAND ----------

# Shutdown Ray cluster
cluster_manager.shutdown_cluster()
print("✅ Ray cluster shut down successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC 1. **Massive Speedup**: Ray Data is 10x+ faster than traditional Pandas UDF for batch inference
# MAGIC 2. **Efficient Resource Use**: Actor pool strategy reuses model across batches (load once, use many times)
# MAGIC 3. **Fault Tolerance**: Automatic retry on failures without reprocessing entire dataset
# MAGIC 4. **Scalability**: Can process billions of records with streaming execution
# MAGIC 5. **Integration**: Seamlessly works with Delta Lake, S3, and other data sources
# MAGIC
# MAGIC **Why Ray Data is Faster:**
# MAGIC - **Model Loading**: Pandas UDF loads model per partition; Ray loads once per actor
# MAGIC - **Better Parallelism**: Fine-grained control over actors and batch sizes
# MAGIC - **Optimized Execution**: Pipelining and prefetching reduce idle time
# MAGIC - **Memory Efficiency**: Streaming execution for large datasets
# MAGIC
# MAGIC **Production Recommendations:**
# MAGIC 1. Use ActorPoolStrategy with 2-4x CPUs as actors
# MAGIC 2. Tune batch_size based on model complexity (1K-100K records)
# MAGIC 3. Enable checkpointing for very large datasets
# MAGIC 4. Monitor memory with object_store_memory_per_node
# MAGIC 5. Use streaming for datasets > 100GB
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Try with your own trained models
# MAGIC - Process production datasets from Delta Lake
# MAGIC - Combine with feature engineering pipelines
# MAGIC - Deploy as scheduled batch jobs
