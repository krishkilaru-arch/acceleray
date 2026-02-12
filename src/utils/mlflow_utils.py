"""MLflow integration utilities for Ray workloads."""

import mlflow
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowLogger:
    """Helper class for MLflow logging in Ray applications."""
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        """
        Initialize MLflow logger.
        
        Args:
            experiment_name: Name of MLflow experiment
            run_name: Optional name for the run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Set up MLflow experiment."""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment set to: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment: {str(e)}")
            raise
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Context manager for MLflow run.
        
        Args:
            run_name: Optional run name (overrides instance run_name)
            nested: Whether this is a nested run
        """
        name = run_name or self.run_name
        with mlflow.start_run(run_name=name, nested=nested) as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")
            yield run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.warning(f"Failed to log parameters: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {str(e)}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Log model to MLflow.
        
        Args:
            model: Model to log
            artifact_path: Path within run's artifact directory
            registered_model_name: Name for model registry
            **kwargs: Additional arguments for model logging
        """
        try:
            # Determine model flavor based on type
            model_type = type(model).__name__
            
            if "sklearn" in str(type(model).__module__):
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
            elif "xgboost" in str(type(model).__module__):
                mlflow.xgboost.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
            elif "torch" in str(type(model).__module__):
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
            else:
                # Generic model logging
                mlflow.pyfunc.log_model(
                    artifact_path,
                    python_model=model,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
            
            logger.info(f"Logged {model_type} model to {artifact_path}")
            
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
            raise
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact: {str(e)}")
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the current run."""
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.warning(f"Failed to set tag: {str(e)}")
    
    def set_tags(self, tags: Dict[str, Any]):
        """Set multiple tags."""
        for key, value in tags.items():
            self.set_tag(key, value)


def log_ray_tune_results(results, experiment_name: str):
    """
    Log Ray Tune results to MLflow.
    
    Args:
        results: Ray Tune ResultGrid
        experiment_name: MLflow experiment name
    """
    logger = MLflowLogger(experiment_name)
    
    with logger.start_run(run_name="ray_tune_summary"):
        # Log best result
        best_result = results.get_best_result()
        
        logger.log_params({
            "num_trials": len(results),
            "search_algorithm": results.experiment_analysis._searcher.__class__.__name__
            if hasattr(results, 'experiment_analysis') else "unknown"
        })
        
        logger.log_metrics({
            "best_score": best_result.metrics.get("score", 0),
            "best_loss": best_result.metrics.get("loss", float('inf'))
        })
        
        # Log individual trials as nested runs
        for i, result in enumerate(results):
            with logger.start_run(run_name=f"trial_{i}", nested=True):
                logger.log_params(result.config)
                logger.log_metrics(result.metrics)
                
                logger.set_tags({
                    "trial_id": result.log_dir.split("/")[-1] if hasattr(result, 'log_dir') else str(i),
                    "status": "completed"
                })
        
        logger.info(f"Logged {len(results)} Ray Tune trials to MLflow")


def create_mlflow_callback_for_ray():
    """
    Create an MLflow callback for Ray Train.
    
    Returns:
        MLflow callback function
    """
    from ray.train.mlflow import MLflowLoggerCallback
    
    return MLflowLoggerCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name
        if mlflow.active_run() else "default",
        save_artifact=True
    )
