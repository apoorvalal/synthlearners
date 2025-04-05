import numpy as np
import pytest
from synthlearners.crossvalidation import PanelCrossValidator, cross_validate

@pytest.fixture
def test_vertical_split():
    """Test that vertical splits have the correct shape and properties."""
    cv = PanelCrossValidator(n_splits=5)
    X = np.random.randn(10, 20)  # 10 units, 20 time periods
    
    masks = cv.vertical_split(X)
    
    # Check we get correct number of folds
    assert len(masks) == 5
    
    for i, (train_mask, test_mask) in enumerate(masks):
        print(f"\n Vertical Fold {i}:")
        print("Train mask column sums:", train_mask.sum(axis=0))
        print("Test mask column sums:", test_mask.sum(axis=0))

        # Check mask shapes
        assert train_mask.shape == X.shape
        assert test_mask.shape == X.shape
        
        # Check masks are boolean
        assert train_mask.dtype == bool
        assert test_mask.dtype == bool
        
        # Check masks are mutually exclusive
        assert not np.any(train_mask & test_mask)
        
        # Check that splits are along rows (units)
        train_row_sums = train_mask.sum(axis=1)
        test_row_sums = test_mask.sum(axis=1)
        
        assert np.all((train_row_sums == 0) | (train_row_sums == X.shape[0]))
        assert np.all((test_row_sums == 0) | (test_row_sums == X.shape[0]))


def test_horizontal_split():
    """Test that horizontal splits have the correct shape and properties."""
    cv = PanelCrossValidator(n_splits=5)
    X = np.random.randn(10, 20)
    
    masks = cv.horizontal_split(X)
    
    # Number of folds limited by min_train_size constraint
    assert len(masks) <= 5
    
    for i, (train_mask, test_mask) in enumerate(masks):
        print(f"\n Horizontal Fold {i}:")
        print("Train mask column sums:", train_mask.sum(axis=0))
        print("Test mask column sums:", test_mask.sum(axis=0))
    
        # Check mask shapes
        assert train_mask.shape == X.shape
        assert test_mask.shape == X.shape
        
        # Check masks are boolean
        assert train_mask.dtype == bool
        assert test_mask.dtype == bool
        
        # Check masks are mutually exclusive
        assert not np.any(train_mask & test_mask)
        
        # Check that splits are along columns (time)
        train_column_sums = train_mask.sum(axis=0)
        test_column_sums = test_mask.sum(axis=0)
        
        assert np.all((train_column_sums == 0) | (train_column_sums == X.shape[0]))
        assert np.all((test_column_sums == 0) | (test_column_sums == X.shape[0]))


def test_random_split():
    """Test that random splits have the correct shape and properties."""
    cv = PanelCrossValidator(n_splits=5, random_state=42)
    X = np.random.randn(10, 20)
    
    masks = cv.random_split(X)
    
    # Check we get correct number of folds
    assert len(masks) == 5
    
    for i, (train_mask, test_mask) in enumerate(masks):
        print(f"\nRandom Fold {i}:")
        print("Train mask column sums:", train_mask.sum(axis=0))
        print("Test mask column sums:", test_mask.sum(axis=0))

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

def test_box_split():
    """Test that box splits have the correct shape and properties."""

    cv = PanelCrossValidator(n_splits=5, random_state=42)
    X = np.random.randn(10, 20)
    
    masks = cv.box_split(X)
    
    # Check we get correct number of folds
    assert len(masks) == 5
    
    for i, (train_mask, test_mask) in enumerate(masks):
        print(f"\nBox Fold {i}:")
        print("Train mask column sums:", train_mask.sum(axis=0))
        print("Test mask column sums:", test_mask.sum(axis=0))

        # Check mask shapes  
        assert train_mask.shape == X.shape
        assert test_mask.shape == X.shape
        
        # Check masks are boolean
        assert train_mask.dtype == bool
        assert test_mask.dtype == bool
        
        # Check masks are mutually exclusive
        assert not np.any(train_mask & test_mask)
        
        # Check that expected number of elements are masked
        expected_n = int((1-cv.cv_ratio) * X.shape[0])
        expected_t = int((1-cv.cv_ratio) * X.shape[1])
        assert abs(test_mask.sum() - expected_n * expected_t) <= 1

def test_split_type_validation():
    """Test that invalid split types raise appropriate errors."""
    cv = PanelCrossValidator(n_splits=5)
    X = np.random.randn(10, 20)
    
    with pytest.raises(ValueError):
        cv.create_train_test_masks(X, split_type='invalid')

def test_cross_validate():
    """Test that cross_validate returns expected results."""
    class MockEstimator:
        def fit(self, X, mask, **kwargs):
            return X * mask
            
        def score(self, X, X_fitted, mask):
            return np.sum(X * mask == X_fitted * mask)
    
    # Create test data
    X = np.random.randn(10, 20)
    cv = PanelCrossValidator(n_splits=3, n_jobs=1, cv_ratio=0.8, random_state=42)
    estimator = MockEstimator()
    
    # Test vertical split
    vertical_scores = cross_validate(estimator, X, cv, split_type='vertical')
    assert len(vertical_scores) == 3
    assert all(isinstance(score, (int, float, np.number)) for score in vertical_scores)
    
    # Test horizontal split
    horizontal_scores = cross_validate(estimator, X, cv, split_type='horizontal')
    assert len(horizontal_scores) <= 3  # Limited by min_train_size
    assert all(isinstance(score, (int, float, np.number)) for score in horizontal_scores)

    # Test box split
    box_scores = cross_validate(estimator, X, cv, split_type='box')
    print("Box scores:", box_scores)
    assert len(box_scores) <= 3  # Limited by min_train_size
    assert all(isinstance(score, (int, float, np.number)) for score in box_scores)
    
    # Test random split
    random_scores = cross_validate(estimator, X, cv, split_type='random')
    print("Random scores:", random_scores)
    assert len(random_scores) == 3
    assert all(isinstance(score, (int, float, np.number)) for score in random_scores)
    
    # Test with custom fit method and args
    custom_scores = cross_validate(
        estimator, 
        X, 
        cv,
        split_type='vertical',
        fit_method='fit',
        fit_args={'extra_arg': 42}
    )
    assert len(custom_scores) == 3
    
    # Test error handling
    with pytest.raises(ValueError):
        cross_validate(estimator, X, cv, split_type='invalid')
    
    with pytest.raises(AttributeError):
        cross_validate(estimator, X, cv, fit_method='invalid_method')