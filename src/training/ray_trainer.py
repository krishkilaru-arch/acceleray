"""Distributed training with Ray Train."""

import logging
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedPyTorchTrainer:
    """Wrapper for distributed PyTorch training with Ray Train."""
    
    def __init__(
        self,
        num_workers: int = 2,
        use_gpu: bool = False,
        cpus_per_worker: int = 2
    ):
        """
        Initialize distributed trainer.
        
        Args:
            num_workers: Number of training workers
            use_gpu: Whether to use GPU
            cpus_per_worker: CPUs per worker
        """
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.cpus_per_worker = cpus_per_worker
        self.trainer = None
        self.results = None
    
    def create_training_function(
        self,
        model_fn: Callable,
        optimizer_fn: Callable,
        criterion: nn.Module,
        train_dataloader_fn: Callable,
        val_dataloader_fn: Optional[Callable] = None,
        epochs: int = 10
    ):
        """
        Create training function for Ray Train.
        
        Args:
            model_fn: Function that returns model
            optimizer_fn: Function that returns optimizer given model
            criterion: Loss function
            train_dataloader_fn: Function that returns training dataloader
            val_dataloader_fn: Optional function that returns validation dataloader
            epochs: Number of training epochs
            
        Returns:
            Training function compatible with Ray Train
        """
        def train_func(config: Dict[str, Any]):
            import ray.train.torch as ray_torch
            from ray import train
            
            # Get device
            device = ray_torch.get_device()
            
            # Create model
            model = model_fn()
            model = model.to(device)
            
            # Prepare model for distributed training
            model = ray_torch.prepare_model(model)
            
            # Create optimizer
            optimizer = optimizer_fn(model)
            
            # Create dataloaders
            train_loader = train_dataloader_fn()
            train_loader = ray_torch.prepare_data_loader(train_loader)
            
            if val_dataloader_fn is not None:
                val_loader = val_dataloader_fn()
                val_loader = ray_torch.prepare_data_loader(val_loader)
            else:
                val_loader = None
            
            # Training loop
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = 100. * train_correct / train_total
                
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc
                }
                
                # Validation
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = criterion(output, target)
                            
                            val_loss += loss.item()
                            _, predicted = output.max(1)
                            val_total += target.size(0)
                            val_correct += predicted.eq(target).sum().item()
                    
                    val_loss /= len(val_loader)
                    val_acc = 100. * val_correct / val_total
                    
                    metrics.update({
                        "val_loss": val_loss,
                        "val_accuracy": val_acc
                    })
                
                # Report metrics and checkpoint
                train.report(metrics)
        
        return train_func
    
    def train(
        self,
        train_func: Callable,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Run distributed training.
        
        Args:
            train_func: Training function
            config: Optional configuration dictionary
            
        Returns:
            Training results
        """
        logger.info(f"Starting distributed training with {self.num_workers} workers...")
        
        # Create scaling config
        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
            resources_per_worker={"CPU": self.cpus_per_worker}
        )
        
        # Create run config
        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_frequency=1
            )
        )
        
        # Create trainer
        self.trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            scaling_config=scaling_config,
            run_config=run_config
        )
        
        # Run training
        self.results = self.trainer.fit()
        
        logger.info("Distributed training completed!")
        
        return self.results
    
    def get_best_checkpoint(self):
        """Get best checkpoint from training."""
        if self.results is None:
            raise ValueError("No training results available. Run train() first.")
        
        return self.results.checkpoint


def create_distributed_training_example():
    """
    Example of setting up distributed training.
    
    Returns:
        Configured DistributedPyTorchTrainer
    """
    from ..models.pytorch_model import SimpleClassifier
    from torch.optim import Adam
    
    # Model factory
    def create_model():
        return SimpleClassifier(
            input_dim=20,
            hidden_dims=[128, 64],
            num_classes=2,
            dropout=0.3
        )
    
    # Optimizer factory
    def create_optimizer(model):
        return Adam(model.parameters(), lr=0.001)
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = DistributedPyTorchTrainer(
        num_workers=4,
        use_gpu=True,
        cpus_per_worker=2
    )
    
    # Note: train_dataloader_fn and val_dataloader_fn would need to be provided
    # This is just an example setup
    
    return trainer
