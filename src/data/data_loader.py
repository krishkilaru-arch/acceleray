"""Data loading utilities for Ray on Databricks."""

import logging
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for various ML datasets."""
    
    @staticmethod
    def load_sample_classification_data(
        n_samples: int = 10000,
        n_features: int = 20,
        n_classes: int = 2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate sample classification dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            random_state: Random seed
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Generating classification dataset: {n_samples} samples, {n_features} features")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            n_classes=n_classes,
            random_state=random_state
        )
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")
        
        return X_df, y_series
    
    @staticmethod
    def load_sample_regression_data(
        n_samples: int = 10000,
        n_features: int = 20,
        noise: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate sample regression dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            noise: Noise level
            random_state: Random seed
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Generating regression dataset: {n_samples} samples, {n_features} features")
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")
        
        return X_df, y_series
    
    @staticmethod
    def load_from_delta_table(
        table_name: str,
        feature_cols: Optional[list] = None,
        target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Load data from Delta table (Databricks-specific).
        
        Args:
            table_name: Name of Delta table
            feature_cols: List of feature column names (None = all except target)
            target_col: Name of target column
            
        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        try:
            # This would be used in Databricks notebook
            spark = None
            try:
                from pyspark.sql import SparkSession
                spark = SparkSession.getActiveSession()
            except:
                logger.warning("SparkSession not available - using mock data")
                return DataLoader.load_sample_classification_data()
            
            if spark is None:
                logger.warning("No active Spark session - using mock data")
                return DataLoader.load_sample_classification_data()
            
            # Load from Delta
            df = spark.table(table_name).toPandas()
            
            if target_col:
                y = df[target_col]
                X = df[feature_cols] if feature_cols else df.drop(columns=[target_col])
                return X, y
            else:
                return df, None
                
        except Exception as e:
            logger.error(f"Error loading from Delta table: {str(e)}")
            raise
    
    @staticmethod
    def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining data)
            random_state: Random seed
            
        Returns:
            Dictionary with train, val, test splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }
    
    @staticmethod
    def create_ray_dataset(X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Create Ray Dataset from pandas DataFrame.
        
        Args:
            X: Features DataFrame
            y: Optional target Series
            
        Returns:
            Ray Dataset
        """
        import ray
        
        if y is not None:
            data = X.copy()
            data['target'] = y
        else:
            data = X
        
        dataset = ray.data.from_pandas(data)
        logger.info(f"Created Ray Dataset with {dataset.count()} rows")
        
        return dataset


class ImageDataLoader:
    """Data loader for image datasets."""
    
    @staticmethod
    def create_sample_image_dataset(
        n_samples: int = 1000,
        image_size: int = 224,
        n_classes: int = 10,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sample image dataset (random noise images for demo).
        
        Args:
            n_samples: Number of images
            image_size: Image width/height
            n_classes: Number of classes
            random_state: Random seed
            
        Returns:
            Tuple of (images array, labels array)
        """
        np.random.seed(random_state)
        
        images = np.random.rand(n_samples, 3, image_size, image_size).astype(np.float32)
        labels = np.random.randint(0, n_classes, size=n_samples)
        
        logger.info(f"Created sample image dataset: {images.shape}, {n_classes} classes")
        
        return images, labels
    
    @staticmethod
    def load_from_directory(
        data_dir: str,
        image_size: int = 224,
        batch_size: int = 32
    ):
        """
        Load images from directory structure.
        
        Args:
            data_dir: Path to data directory
            image_size: Target image size
            batch_size: Batch size for loading
            
        Returns:
            PyTorch DataLoader
        """
        try:
            from torchvision import transforms, datasets
            from torch.utils.data import DataLoader
            
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            dataset = datasets.ImageFolder(data_dir, transform=transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            logger.info(f"Loaded image dataset from {data_dir}: {len(dataset)} images")
            
            return dataloader
            
        except Exception as e:
            logger.error(f"Error loading images: {str(e)}")
            raise


def prepare_data_for_ray_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, Any]:
    """
    Prepare data dictionary for Ray Tune training.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Dictionary with data splits
    """
    return {
        "train": {
            "X": X_train.values,
            "y": y_train.values
        },
        "val": {
            "X": X_val.values,
            "y": y_val.values
        }
    }
