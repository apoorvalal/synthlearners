import numpy as np
import pytest
from synthlearners.crossvalidation import PanelCrossValidator, cross_validate

@pytest.fixture
def test_horizontal_split():
    """Test that horizontal splits have the correct shape and properties."""
    cv = PanelCrossValidator(n_splits=5)
    X = np.random.randn(10, 20)  # 10 units, 20 time periods
    
    masks = cv.horizontal_split(X)
    
    # Check we get correct number of folds
    assert len(masks) == 5
    
    for train_mask, test_mask in masks:
        # Check mask shapes
        assert train_mask.shape == X.shape
        assert test_mask.shape == X.shape
        
        # Check masks are boolean
        assert train_mask.dtype == bool
        assert test_mask.dtype == bool
        
        # Check masks are mutually exclusive
        assert not np.any(train_mask & test_mask)
        
        # Check that splits are along rows (units)
        assert np.all(train_mask.sum(axis=1) == 0) | np.all(train_mask.sum(axis=1) == X.shape[1])
        assert np.all(test_mask.sum(axis=1) == 0) | np.all(test_mask.sum(axis=1) == X.shape[1])


def test_vertical_split():
    """Test that vertical splits have the correct shape and properties."""
    cv = PanelCrossValidator(n_splits=5)
    X = np.random.randn(10, 20)
    
    masks = cv.vertical_split(X)
    
    # Number of folds limited by min_train_size constraint
    assert len(masks) <= 5
    
    for train_mask, test_mask in masks:
        # Check mask shapes
        assert train_mask.shape == X.shape
        assert test_mask.shape == X.shape
        
        # Check masks are boolean
        assert train_mask.dtype == bool
        assert test_mask.dtype == bool
        
        # Check masks are mutually exclusive
        assert not np.any(train_mask & test_mask)
        
        # Check that splits are along columns (time)
        assert np.all(train_mask.sum(axis=0) == 0) | np.all(train_mask.sum(axis=0) == X.shape[0])
        assert np.all(test_mask.sum(axis=0) == 0) | np.all(test_mask.sum(axis=0) == X.shape[0])


def test_random_split():
    """Test that random splits have the correct shape and properties."""
    cv = PanelCrossValidator(n_splits=5, random_state=42)
    X = np.random.randn(10, 20)
    
    masks = cv.random_split(X)
    
    # Check we get correct number of folds
    assert len(masks) == 5
    
    for train_mask, test_mask in masks:
        # Check mask shapes  
        assert train_mask.shape == X.shape
        assert test_mask.shape == X.shape
        
        # Check masks are boolean
        assert train_mask.dtype == bool
        assert test_mask.dtype == bool
        
        # Check masks are mutually exclusive
        assert not np.any(train_mask & test_mask)
        
        # Check that expected number of elements are masked
        n_test = int(X.size / 5)
        assert abs(test_mask.sum() - n_test) <= 1


def test_cross_validation_scores(simulated_data):
    """Test that cross validation returns reasonable scores."""
    Y, config = simulated_data
    
    class DummyEstimator:
        def fit(self, X, mask):
            self.mean = X[mask].mean()
            
        def predict(self, mask):
            return np.full_like(mask, fill_value=self.mean, dtype=float)
            
        def score(self, y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)  # MSE
    
    estimator = DummyEstimator()
    
    # Test all split types
    for split_type in ['horizontal', 'vertical', 'random']:
        scores = cross_validate(estimator, Y, split_type=split_type, n_splits=5)
        
        # Check we get a score per fold
        assert len(scores) == 5
        
        # Check scores are non-negative (MSE)
        assert np.all(np.array(scores) >= 0)


def test_split_type_validation():
    """Test that invalid split types raise appropriate errors."""
    cv = PanelCrossValidator(n_splits=5)
    X = np.random.randn(10, 20)
    
    with pytest.raises(ValueError):
        cv.create_train_test_masks(X, split_type='invalid')