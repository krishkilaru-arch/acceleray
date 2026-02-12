"""Batch inference with Ray Data."""

import logging
from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
import ray
from ray.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchPredictor:
    """Batch prediction using Ray Data."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        batch_size: int = 100,
        num_actors: int = 4
    ):
        """
        Initialize batch predictor.
        
        Args:
            model_path: Path to saved model
            model: Loaded model object
            batch_size: Batch size for predictions
            num_actors: Number of parallel actors
        """
        self.model_path = model_path
        self.model = model
        self.batch_size = batch_size
        self.num_actors = num_actors
    
    def predict_xgboost(
        self,
        dataset: Dataset,
        feature_columns: Optional[list] = None
    ) -> Dataset:
        """
        Batch prediction with XGBoost model.
        
        Args:
            dataset: Ray Dataset with features
            feature_columns: List of feature column names
            
        Returns:
            Ray Dataset with predictions
        """
        import xgboost as xgb
        
        class XGBoostPredictor:
            def __init__(self, model_path: str):
                self.model = xgb.Booster()
                self.model.load_model(model_path)
            
            def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                if feature_columns:
                    X = np.column_stack([batch[col] for col in feature_columns])
                else:
                    # Assume all columns except 'prediction' are features
                    feature_cols = [col for col in batch.keys() if col != 'prediction']
                    X = np.column_stack([batch[col] for col in feature_cols])
                
                dmatrix = xgb.DMatrix(X)
                predictions = self.model.predict(dmatrix)
                
                return {"prediction": predictions}
        
        logger.info(f"Running batch prediction with {self.num_actors} actors...")
        
        predictions = dataset.map_batches(
            XGBoostPredictor,
            fn_constructor_kwargs={"model_path": self.model_path},
            batch_size=self.batch_size,
            compute=ray.data.ActorPoolStrategy(size=self.num_actors)
        )
        
        return predictions
    
    def predict_pytorch(
        self,
        dataset: Dataset,
        model_class: type,
        model_kwargs: Dict[str, Any],
        feature_columns: Optional[list] = None
    ) -> Dataset:
        """
        Batch prediction with PyTorch model.
        
        Args:
            dataset: Ray Dataset with features
            model_class: PyTorch model class
            model_kwargs: Model initialization kwargs
            feature_columns: List of feature column names
            
        Returns:
            Ray Dataset with predictions
        """
        import torch
        
        class PyTorchPredictor:
            def __init__(self, model_path: str, model_class: type, model_kwargs: Dict):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = model_class(**model_kwargs)
                
                if model_path:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
            
            def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                if feature_columns:
                    X = np.column_stack([batch[col] for col in feature_columns])
                else:
                    feature_cols = [col for col in batch.keys() if col != 'prediction']
                    X = np.column_stack([batch[col] for col in feature_cols])
                
                X_tensor = torch.FloatTensor(X).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(X_tensor)
                    _, predictions = outputs.max(1)
                
                return {"prediction": predictions.cpu().numpy()}
        
        logger.info(f"Running batch prediction with {self.num_actors} actors...")
        
        predictions = dataset.map_batches(
            PyTorchPredictor,
            fn_constructor_kwargs={
                "model_path": self.model_path,
                "model_class": model_class,
                "model_kwargs": model_kwargs
            },
            batch_size=self.batch_size,
            compute=ray.data.ActorPoolStrategy(size=self.num_actors)
        )
        
        return predictions
    
    def predict_sklearn(
        self,
        dataset: Dataset,
        feature_columns: Optional[list] = None
    ) -> Dataset:
        """
        Batch prediction with scikit-learn model.
        
        Args:
            dataset: Ray Dataset with features
            feature_columns: List of feature column names
            
        Returns:
            Ray Dataset with predictions
        """
        import joblib
        
        class SklearnPredictor:
            def __init__(self, model_path: str):
                self.model = joblib.load(model_path)
            
            def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                if feature_columns:
                    X = np.column_stack([batch[col] for col in feature_columns])
                else:
                    feature_cols = [col for col in batch.keys() if col != 'prediction']
                    X = np.column_stack([batch[col] for col in feature_cols])
                
                predictions = self.model.predict(X)
                
                return {"prediction": predictions}
        
        logger.info(f"Running batch prediction with {self.num_actors} actors...")
        
        predictions = dataset.map_batches(
            SklearnPredictor,
            fn_constructor_kwargs={"model_path": self.model_path},
            batch_size=self.batch_size,
            compute=ray.data.ActorPoolStrategy(size=self.num_actors)
        )
        
        return predictions


def create_ray_dataset_from_spark(spark_df):
    """
    Create Ray Dataset from Spark DataFrame.
    
    Args:
        spark_df: Spark DataFrame
        
    Returns:
        Ray Dataset
    """
    # Convert to Ray Dataset
    dataset = ray.data.from_spark(spark_df)
    logger.info(f"Created Ray Dataset from Spark with {dataset.count()} rows")
    
    return dataset


def write_predictions_to_delta(predictions: Dataset, table_name: str):
    """
    Write predictions to Delta table.
    
    Args:
        predictions: Ray Dataset with predictions
        table_name: Name of Delta table
    """
    # Convert to Spark DataFrame
    spark_df = predictions.to_spark()
    
    # Write to Delta
    spark_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    
    logger.info(f"Written predictions to Delta table: {table_name}")


def batch_inference_example(
    input_data: pd.DataFrame,
    model_path: str,
    model_type: str = "xgboost"
) -> pd.DataFrame:
    """
    Example of batch inference workflow.
    
    Args:
        input_data: Input DataFrame
        model_path: Path to saved model
        model_type: Type of model ('xgboost', 'pytorch', 'sklearn')
        
    Returns:
        DataFrame with predictions
    """
    # Create Ray Dataset
    dataset = ray.data.from_pandas(input_data)
    
    # Create predictor
    predictor = BatchPredictor(
        model_path=model_path,
        batch_size=1000,
        num_actors=8
    )
    
    # Run predictions
    if model_type == "xgboost":
        predictions = predictor.predict_xgboost(dataset)
    elif model_type == "sklearn":
        predictions = predictor.predict_sklearn(dataset)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Convert to pandas
    result_df = predictions.to_pandas()
    
    logger.info(f"Batch inference completed on {len(result_df)} rows")
    
    return result_df
