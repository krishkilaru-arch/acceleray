"""Utilities for managing Ray clusters on Databricks."""

import logging
from typing import Optional, Dict, Any
import time
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RayClusterManager:
    """Manager for Ray cluster lifecycle on Databricks."""
    
    def __init__(self):
        self.cluster_initialized = False
        
    def initialize_cluster(
        self,
        num_worker_nodes: int = 2,
        num_cpus_per_node: int = 4,
        num_gpus_per_node: int = 0,
        object_store_memory_per_node: Optional[int] = None,
        collect_log_to_path: str = "/dbfs/ray_logs",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initialize Ray cluster on Spark.
        
        Args:
            num_worker_nodes: Number of Ray worker nodes
            num_cpus_per_node: CPUs per worker node
            num_gpus_per_node: GPUs per worker node (0 for CPU-only)
            object_store_memory_per_node: Memory for object store in bytes
            collect_log_to_path: Path to collect Ray logs
            **kwargs: Additional arguments for setup_ray_cluster
            
        Returns:
            Dictionary with cluster information
        """
        if self.cluster_initialized:
            logger.warning("Ray cluster already initialized. Shutting down first...")
            self.shutdown_cluster()
        
        logger.info(f"Initializing Ray cluster with {num_worker_nodes} workers...")
        
        try:
            # Ray on Spark treats the GPU argument as "GPU mode requested" even
            # when it is set to 0. For CPU clusters, omit the argument entirely.
            setup_kwargs = {
                "num_worker_nodes": num_worker_nodes,
                "num_cpus_per_node": num_cpus_per_node,
                "collect_log_to_path": collect_log_to_path,
                **kwargs,
            }
            if object_store_memory_per_node is not None:
                setup_kwargs["object_store_memory_per_node"] = object_store_memory_per_node
            if num_gpus_per_node and num_gpus_per_node > 0:
                setup_kwargs["num_gpus_per_node"] = num_gpus_per_node

            setup_ray_cluster(**setup_kwargs)

            # Connect the current Python process to the Ray cluster started on Spark.
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
            
            self.cluster_initialized = True
            
            cluster_info = {
                "num_worker_nodes": num_worker_nodes,
                "num_cpus_per_node": num_cpus_per_node,
                "num_gpus_per_node": num_gpus_per_node,
                "total_cpus": num_worker_nodes * num_cpus_per_node,
                "total_gpus": num_worker_nodes * num_gpus_per_node,
                "dashboard_url": _get_dashboard_url(),
                "ray_version": ray.__version__
            }
            
            logger.info(f"Ray cluster initialized successfully!")
            logger.info(f"Dashboard URL: {cluster_info['dashboard_url']}")
            logger.info(f"Total CPUs: {cluster_info['total_cpus']}")
            logger.info(f"Total GPUs: {cluster_info['total_gpus']}")
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {str(e)}")
            raise

    def health_check(self, timeout_seconds: int = 30) -> Dict[str, Any]:
        """
        Validate Ray connectivity with a lightweight remote task.

        Args:
            timeout_seconds: Max seconds to wait for health check task.

        Returns:
            Health-check metadata and status.
        """
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Call initialize_cluster() first.")

        @ray.remote
        def _ping(value: int) -> int:
            return value + 1

        start = time.time()
        try:
            result = ray.get(_ping.remote(41), timeout=timeout_seconds)
        except Exception as e:
            raise RuntimeError(f"Ray health check failed: {str(e)}") from e

        elapsed_ms = int((time.time() - start) * 1000)
        if result != 42:
            raise RuntimeError(f"Unexpected health-check result: {result}")

        return {
            "status": "ok",
            "latency_ms": elapsed_ms,
            "ray_version": ray.__version__,
            "dashboard_url": _get_dashboard_url(),
            "available_resources": ray.available_resources(),
        }
    
    def shutdown_cluster(self):
        """Shutdown Ray cluster."""
        if not self.cluster_initialized:
            logger.warning("No Ray cluster to shutdown")
            return
        
        logger.info("Shutting down Ray cluster...")
        try:
            if ray.is_initialized():
                ray.shutdown()
            shutdown_ray_cluster()
            self.cluster_initialized = False
            logger.info("Ray cluster shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down Ray cluster: {str(e)}")
            raise
    
    def get_cluster_resources(self) -> Dict[str, Any]:
        """Get current cluster resources."""
        if not self.cluster_initialized:
            raise RuntimeError("Ray cluster not initialized")
        
        resources = ray.cluster_resources()
        return {
            "cpu": resources.get("CPU", 0),
            "gpu": resources.get("GPU", 0),
            "memory": resources.get("memory", 0),
            "object_store_memory": resources.get("object_store_memory", 0)
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cluster is shut down."""
        self.shutdown_cluster()


def check_ray_availability() -> bool:
    """
    Check if Ray cluster is available and initialized.
    
    Returns:
        True if Ray is initialized, False otherwise
    """
    try:
        ray.get_runtime_context()
        return True
    except Exception:
        return False


def print_cluster_info():
    """Print detailed information about the Ray cluster."""
    if not check_ray_availability():
        print("Ray cluster is not initialized")
        return
    
    print("=" * 60)
    print("Ray Cluster Information")
    print("=" * 60)
    
    resources = ray.cluster_resources()
    print(f"\n📊 Resources:")
    print(f"  - CPUs: {resources.get('CPU', 0)}")
    print(f"  - GPUs: {resources.get('GPU', 0)}")
    print(f"  - Memory: {resources.get('memory', 0) / 1e9:.2f} GB")
    print(f"  - Object Store: {resources.get('object_store_memory', 0) / 1e9:.2f} GB")
    
    print(f"\n🌐 Dashboard: {_get_dashboard_url()}")
    print(f"🔢 Ray Version: {ray.__version__}")
    
    nodes = ray.nodes()
    print(f"\n🖥️  Nodes: {len(nodes)}")
    for i, node in enumerate(nodes):
        status = "✅" if node['Alive'] else "❌"
        print(f"  {i+1}. {status} {node.get('NodeManagerAddress', 'unknown')}")
    
    print("=" * 60)


def _get_dashboard_url() -> str:
    """Best-effort Ray dashboard URL across Ray versions."""
    getter = getattr(ray, "get_dashboard_url", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return "unavailable"
    return "unavailable"
