import numpy as np
from sklearn.model_selection import KFold
from typing import Tuple, List, Optional

class PanelCrossValidator:
    """
    A class implementing horizontal and vertical cross validation for panel data.
    Horizontal CV splits along rows (units/individuals)
    Vertical CV splits along columns (time periods) using forward-chaining validation

    Credit: Written with assistance from Claude Copilot
    """
    
    def __init__(self, n_splits: int = 5, cv_ratio: float = 0.8, random_state: Optional[int] = None):
        """
        Initialize the cross validator.
        
        Args:
            n_splits: Number of folds for cross validation
            cv_ratio: Ratio of data to use for training (between 0 and 1)
            random_state: Random seed for reproducibility
        """
        if not 0 < cv_ratio < 1:
            raise ValueError("cv_ratio must be between 0 and 1")
        self.n_splits = n_splits
        self.cv_ratio = cv_ratio
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def horizontal_split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform horizontal cross validation (splitting along rows).
        
        Args:
            X: Input matrix of shape (n_units, n_times)
            
        Returns:
            List of (train_mask, test_mask) tuples for each fold
        """
        n_units = X.shape[0]
        train_size = int(n_units * self.cv_ratio)
        masks = []
        for train_idx, test_idx in self.kf.split(np.arange(train_size)):
            train_mask = np.zeros(X.shape, dtype=bool)
            test_mask = np.zeros(X.shape, dtype=bool)
            train_mask[train_idx, :] = True
            test_mask[test_idx, :] = True
            masks.append((train_mask, test_mask))
        return masks
    
    def vertical_split(self, X: np.ndarray, min_train_size: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform vertical cross validation using forward-chaining validation.
        
        Args:
            X: Input matrix of shape (n_units, n_times)
            min_train_size: Minimum number of time periods to include in training set
            
        Returns:
            List of (train_mask, test_mask) tuples for each fold
        """
        n_times = X.shape[1]
        masks = []
        if min_train_size is None:
            min_train_size = int(n_times * self.cv_ratio)
        
        for test_size in range(1, min(self.n_splits + 1, n_times - min_train_size + 1)):
            train_mask = np.zeros(X.shape, dtype=bool)
            test_mask = np.zeros(X.shape, dtype=bool)
            train_mask[:, :n_times - test_size] = True
            test_mask[:, n_times - test_size:] = True
            masks.append((train_mask, test_mask))
        return masks
    
    def random_split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform random cross validation (splitting across all elements).

        Args:
            X: Input matrix of shape (n_units, n_times)

        Returns:
            List of (train_mask, test_mask) tuples for each fold
        """
        n_units, n_times = X.shape
        total_elements = n_units * n_times
        train_size = int(total_elements * self.cv_ratio)
        masks = []
        
        for _ in range(self.n_splits):
            train_mask = np.zeros(X.shape, dtype=bool)
            test_mask = np.zeros(X.shape, dtype=bool)
            
            # Randomly select elements for training set
            train_elements = np.random.choice(total_elements, size=train_size, replace=False)
            train_rows = train_elements // n_times
            train_cols = train_elements % n_times
            train_mask[train_rows, train_cols] = True
            test_mask = ~train_mask
            
            masks.append((train_mask, test_mask))
        return masks
    
    def create_train_test_masks(self, X: np.ndarray, split_type: str = 'horizontal', min_train_size: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create boolean masks for train and test sets.
        
        Args:
            X: Input matrix of shape (n_units, n_times)
            split_type: Type of split ('horizontal', 'vertical', or 'random')
            min_train_size: Minimum number of time periods to include in training set
        
        Returns:
            List of (train_mask, test_mask) tuples for each fold
        """
        if split_type not in ['horizontal', 'vertical', 'random']:
            raise ValueError("Invalid split_type. Expected one of ['horizontal', 'vertical', 'random']")
        
        if split_type == 'horizontal':
            return self.horizontal_split(X)
        elif split_type == 'vertical':
            return self.vertical_split(X, min_train_size=min_train_size)
        else:
            return self.random_split(X)

def cross_validate(estimator, X: np.ndarray, split_type: str = 'horizontal', 
                  n_splits: int = 5, cv_ratio: float = 0.8, random_state: Optional[int] = None):
    """
    Perform cross validation for panel data.
    
    Args:
        estimator: An estimator object implementing fit and predict
        X: Input matrix of shape (n_units, n_times)
        split_type: Type of split ('horizontal' or 'vertical')
        n_splits: Number of folds
        cv_ratio: Ratio of data to use for training (between 0 and 1)
        random_state: Random seed
        
    Returns:
        List of scores for each fold
    """
    cv = PanelCrossValidator(n_splits=n_splits, cv_ratio=cv_ratio, random_state=random_state)
    masks = cv.create_train_test_masks(X, split_type)
    scores = []
    
    for train_mask, test_mask in masks:
        estimator.fit(X, train_mask)
        predictions = estimator.predict(test_mask)
        score = estimator.score(X[test_mask], predictions[test_mask])
        scores.append(score)
    
    return scores

# TO DO:
# - Compare MC results with random, horizontal, and vertical cross-validation