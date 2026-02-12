"""PyTorch model implementations for distributed training."""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleClassifier(nn.Module):
    """Simple feedforward neural network for classification."""
    
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float = 0.3):
        """
        Initialize classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ResNetClassifier(nn.Module):
    """ResNet-based image classifier."""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        Initialize ResNet classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        try:
            from torchvision import models
            
            self.resnet = models.resnet50(pretrained=pretrained)
            
            # Replace final layer
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_features, num_classes)
            
        except ImportError:
            logger.error("torchvision not installed")
            raise
    
    def forward(self, x):
        return self.resnet(x)


class TabularDataset(Dataset):
    """PyTorch dataset for tabular data."""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Initialize dataset.
        
        Args:
            X: Features array
            y: Optional labels array
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class PyTorchTrainer:
    """Trainer for PyTorch models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose:
                    logger.info(
                        f'Epoch {epoch+1}/{epochs} - '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
                    )
            else:
                if verbose:
                    logger.info(
                        f'Epoch {epoch+1}/{epochs} - '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%'
                    )
        
        return history


def create_pytorch_search_space() -> Dict[str, Any]:
    """
    Create hyperparameter search space for PyTorch models.
    
    Returns:
        Dictionary with search space configuration
    """
    from ray import tune
    
    return {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "hidden_dims": tune.choice([
            [128, 64],
            [256, 128, 64],
            [512, 256, 128],
            [256, 256],
            [512, 512]
        ]),
        "dropout": tune.uniform(0.1, 0.5),
        "weight_decay": tune.loguniform(1e-6, 1e-3)
    }


def train_pytorch_with_config(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, float]:
    """
    Training function for Ray Tune with PyTorch.
    
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
    
    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )
    
    # Create model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = SimpleClassifier(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        num_classes=num_classes,
        dropout=config["dropout"]
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Train
    trainer = PyTorchTrainer(model, optimizer, criterion)
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=config.get("epochs", 10),
        verbose=False
    )
    
    # Report final validation metrics
    metrics = {
        "val_loss": history['val_loss'][-1],
        "val_accuracy": history['val_acc'][-1]
    }
    
    ray_train.report(metrics)
    
    return metrics
