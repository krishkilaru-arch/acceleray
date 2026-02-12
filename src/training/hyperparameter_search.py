"""Hyperparameter search with Ray Tune."""

import logging
from typing import Dict, Any, Optional, Callable
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterSearcher:
    """Wrapper for Ray Tune hyperparameter search."""
    
    def __init__(
        self,
        search_algorithm: str = "random",
        scheduler: str = "asha",
        metric: str = "val_accuracy",
        mode: str = "max"
    ):
        """
        Initialize hyperparameter searcher.
        
        Args:
            search_algorithm: Search algorithm ('random', 'hyperopt', 'optuna')
            scheduler: Scheduler ('asha', 'pbt', 'none')
            metric: Metric to optimize
            mode: Optimization mode ('min' or 'max')
        """
        self.search_algorithm = search_algorithm
        self.scheduler_name = scheduler
        self.metric = metric
        self.mode = mode
        self.results = None
    
    def _get_search_algorithm(self, num_samples: int):
        """Get search algorithm instance."""
        if self.search_algorithm == "hyperopt":
            return HyperOptSearch(metric=self.metric, mode=self.mode)
        elif self.search_algorithm == "optuna":
            return OptunaSearch(metric=self.metric, mode=self.mode)
        else:  # random
            return None
    
    def _get_scheduler(self, max_t: int = 100):
        """Get scheduler instance."""
        if self.scheduler_name == "asha":
            return ASHAScheduler(
                time_attr='training_iteration',
                max_t=max_t,
                grace_period=10,
                reduction_factor=3,
                brackets=1
            )
        elif self.scheduler_name == "pbt":
            return PopulationBasedTraining(
                time_attr="training_iteration",
                perturbation_interval=5,
                hyperparam_mutations={}
            )
        else:
            return None
    
    def search(
        self,
        train_function: Callable,
        search_space: Dict[str, Any],
        num_samples: int = 100,
        max_concurrent_trials: int = 4,
        resources_per_trial: Optional[Dict[str, float]] = None,
        **tune_config
    ):
        """
        Run hyperparameter search.
        
        Args:
            train_function: Training function to tune
            search_space: Hyperparameter search space
            num_samples: Number of trials
            max_concurrent_trials: Maximum concurrent trials
            resources_per_trial: Resources per trial (e.g., {'cpu': 1, 'gpu': 0})
            **tune_config: Additional tune.Tuner config
            
        Returns:
            Ray Tune ResultGrid
        """
        logger.info(f"Starting hyperparameter search with {num_samples} trials...")
        
        # Default resources
        if resources_per_trial is None:
            resources_per_trial = {"cpu": 1, "gpu": 0}
        
        # Create tune config
        tune_cfg = tune.TuneConfig(
            metric=self.metric,
            mode=self.mode,
            search_alg=self._get_search_algorithm(num_samples),
            scheduler=self._get_scheduler(),
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            **tune_config
        )
        
        # Create run config
        run_cfg = None
        
        # Create tuner
        tuner = tune.Tuner(
            tune.with_resources(
                train_function,
                resources=resources_per_trial
            ),
            param_space=search_space,
            tune_config=tune_cfg,
            run_config=run_cfg
        )
        
        # Run tuning
        self.results = tuner.fit()
        
        # Log results
        best_result = self.results.get_best_result()
        logger.info(f"Best trial config: {best_result.config}")
        logger.info(f"Best trial {self.metric}: {best_result.metrics[self.metric]:.4f}")
        
        return self.results
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get best hyperparameter configuration."""
        if self.results is None:
            raise ValueError("No search results available. Run search() first.")
        
        best_result = self.results.get_best_result()
        return best_result.config
    
    def get_best_score(self) -> float:
        """Get best metric score."""
        if self.results is None:
            raise ValueError("No search results available. Run search() first.")
        
        best_result = self.results.get_best_result()
        return best_result.metrics[self.metric]
    
    def get_results_dataframe(self):
        """Get results as pandas DataFrame."""
        if self.results is None:
            raise ValueError("No search results available. Run search() first.")
        
        return self.results.get_dataframe()


def run_hyperparameter_search_xgboost(
    data: Dict[str, Any],
    num_samples: int = 100,
    max_concurrent_trials: int = 4,
    resources_per_trial: Optional[Dict[str, float]] = None
):
    """
    Run hyperparameter search for XGBoost model.
    
    Args:
        data: Dictionary with train and validation data
        num_samples: Number of trials
        max_concurrent_trials: Maximum concurrent trials
        resources_per_trial: Resources per trial
        
    Returns:
        Ray Tune ResultGrid
    """
    from ..models.xgboost_model import create_xgboost_search_space, train_xgboost_with_config
    
    # Create search space
    search_space = create_xgboost_search_space()
    
    # Create training function with data
    def train_fn(config):
        return train_xgboost_with_config(config, data)
    
    # Run search
    searcher = HyperparameterSearcher(
        search_algorithm="hyperopt",
        scheduler="asha",
        metric="val_accuracy",
        mode="max"
    )
    
    results = searcher.search(
        train_function=train_fn,
        search_space=search_space,
        num_samples=num_samples,
        max_concurrent_trials=max_concurrent_trials,
        resources_per_trial=resources_per_trial
    )
    
    return results


def run_hyperparameter_search_pytorch(
    data: Dict[str, Any],
    num_samples: int = 50,
    max_concurrent_trials: int = 4,
    resources_per_trial: Optional[Dict[str, float]] = None
):
    """
    Run hyperparameter search for PyTorch model.
    
    Args:
        data: Dictionary with train and validation data
        num_samples: Number of trials
        max_concurrent_trials: Maximum concurrent trials
        resources_per_trial: Resources per trial
        
    Returns:
        Ray Tune ResultGrid
    """
    from ..models.pytorch_model import create_pytorch_search_space, train_pytorch_with_config
    
    # Create search space
    search_space = create_pytorch_search_space()
    search_space["epochs"] = 10  # Fixed number of epochs
    
    # Create training function with data
    def train_fn(config):
        return train_pytorch_with_config(config, data)
    
    # Run search
    searcher = HyperparameterSearcher(
        search_algorithm="optuna",
        scheduler="asha",
        metric="val_accuracy",
        mode="max"
    )
    
    results = searcher.search(
        train_function=train_fn,
        search_space=search_space,
        num_samples=num_samples,
        max_concurrent_trials=max_concurrent_trials,
        resources_per_trial=resources_per_trial
    )
    
    return results
