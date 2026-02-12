"""Unit tests for model implementations."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


class TestXGBoostModel:
    """Tests for XGBoost model wrapper."""
    
    def setup_method(self):
        """Set up test data."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.X_train = X[:800]
        self.y_train = y[:800]
        self.X_test = X[800:]
        self.y_test = y[800:]
    
    def test_model_training(self):
        """Test that model trains successfully."""
        from src.models.xgboost_model import XGBoostModel
        
        model = XGBoostModel(task="classification")
        metrics = model.train(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            num_boost_round=10,
            verbose=False
        )
        
        assert "val_accuracy" in metrics
        assert "val_auc" in metrics
        assert metrics["val_accuracy"] > 0.5
        assert metrics["val_auc"] > 0.5
    
    def test_model_prediction(self):
        """Test that model makes predictions."""
        from src.models.xgboost_model import XGBoostModel
        
        model = XGBoostModel(task="classification")
        model.train(self.X_train, self.y_train, num_boost_round=10, verbose=False)
        
        predictions = model.predict(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert np.all((predictions >= 0) & (predictions <= 1))


class TestPyTorchModel:
    """Tests for PyTorch model implementations."""
    
    def setup_method(self):
        """Set up test data."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.X_train = X[:800]
        self.y_train = y[:800]
        self.X_test = X[800:]
        self.y_test = y[800:]
    
    def test_simple_classifier_creation(self):
        """Test that SimpleClassifier can be created."""
        from src.models.pytorch_model import SimpleClassifier
        
        model = SimpleClassifier(
            input_dim=20,
            hidden_dims=[64, 32],
            num_classes=2,
            dropout=0.3
        )
        
        assert model is not None
    
    def test_tabular_dataset(self):
        """Test TabularDataset."""
        from src.models.pytorch_model import TabularDataset
        
        dataset = TabularDataset(self.X_train, self.y_train)
        
        assert len(dataset) == len(self.X_train)
        
        X, y = dataset[0]
        assert X.shape == (20,)
        assert isinstance(y.item(), int)


class TestDataLoader:
    """Tests for data loading utilities."""
    
    def test_load_sample_classification_data(self):
        """Test loading sample classification data."""
        from src.data.data_loader import DataLoader
        
        loader = DataLoader()
        X, y = loader.load_sample_classification_data(
            n_samples=1000,
            n_features=10,
            n_classes=2
        )
        
        assert X.shape == (1000, 10)
        assert len(y) == 1000
        assert len(np.unique(y)) == 2
    
    def test_split_data(self):
        """Test data splitting."""
        from src.data.data_loader import DataLoader
        
        loader = DataLoader()
        X, y = loader.load_sample_classification_data(n_samples=1000)
        
        splits = loader.split_data(X, y, test_size=0.2, val_size=0.1)
        
        assert "X_train" in splits
        assert "X_val" in splits
        assert "X_test" in splits
        assert len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"]) == 1000


class TestPreprocessing:
    """Tests for preprocessing utilities."""
    
    def test_data_preprocessor(self):
        """Test DataPreprocessor."""
        from src.data.preprocessing import DataPreprocessor
        
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        preprocessor = DataPreprocessor(scaling_method="standard")
        X_scaled, y_encoded = preprocessor.fit_transform(X, y)
        
        assert X_scaled.shape == X.shape
        assert len(y_encoded) == len(y)
        
        # Test mean is close to 0 and std close to 1
        assert np.abs(X_scaled.mean()) < 0.1
        assert np.abs(X_scaled.std() - 1.0) < 0.1
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        from src.data.preprocessing import handle_missing_values
        
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8]
        })
        
        result = handle_missing_values(df, strategy='mean')
        
        assert result.isnull().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
