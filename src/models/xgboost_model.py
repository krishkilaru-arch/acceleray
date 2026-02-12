"""XGBoost model implementation with Ray Tune integration."""

import logging
from typing import Dict, Any, Optional
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """Wrapper for XGBoost model with Ray Tune support."""
    
    def __init__(self, task: str = "classification", **params):
        """
        Initialize XGBoost model.
        
        Args:
            task: Task type ('classification' or 'regression')
            **params: XGBoost parameters
        """
        self.task = task
        self.params = params
        self.model = None
        self.best_iteration = None
        
        # Set default parameters based on task
        if task == "classification":
            self.params.setdefault('objective', 'binary:logistic')
            self.params.setdefault('eval_metric', 'auc')
        else:
            self.params.setdefault('objective', 'reg:squarederror')
            self.params.setdefault('eval_metric', 'rmse')
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        num_boost_round: int = 100,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training metrics
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Train model
        evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose
        )
        
        self.best_iteration = self.model.best_iteration
        
        # Compute metrics
        metrics = {}
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            
            if self.task == "classification":
                val_pred_binary = (val_pred > 0.5).astype(int)
                metrics['val_accuracy'] = accuracy_score(y_val, val_pred_binary)
                metrics['val_auc'] = roc_auc_score(y_val, val_pred)
            else:
                metrics['val_rmse'] = mean_squared_error(y_val, val_pred, squared=False)
                metrics['val_mse'] = mean_squared_error(y_val, val_pred)
        
        logger.info(f"Training completed. Best iteration: {self.best_iteration}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features array
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.get_score(importance_type='weight')
    
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        self.model = xgb.Booster()
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")


def create_xgboost_search_space() -> Dict[str, Any]:
    """
    Create hyperparameter search space for Ray Tune.
    
    Returns:
        Dictionary with search space configuration
    """
    from ray import tune
    
    return {
        "max_depth": tune.randint(3, 10),
        "min_child_weight": tune.uniform(1, 10),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        "learning_rate": tune.loguniform(0.001, 0.3),
        "n_estimators": tune.randint(50, 500),
        "gamma": tune.uniform(0, 5),
        "reg_alpha": tune.loguniform(0.001, 100),
        "reg_lambda": tune.loguniform(0.001, 100)
    }


def train_xgboost_with_config(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, float]:
    """
    Training function for Ray Tune.
    
    Args:
        config: Hyperparameter configuration
        data: Dictionary with train and validation data
        
    Returns:
        Dictionary with evaluation metrics
    """
    from ray import train as ray_train
    
    # Extract data
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_val = data["val"]["X"]
    y_val = data["val"]["y"]
    
    # Create and train model
    model = XGBoostModel(task="classification", **config)
    metrics = model.train(
        X_train, y_train,
        X_val, y_val,
        num_boost_round=config.get("n_estimators", 100),
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Report metrics to Ray Tune
    ray_train.report(metrics)
    
    return metrics
