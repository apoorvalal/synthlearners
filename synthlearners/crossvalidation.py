import numpy as np
from sklearn.model_selection import KFold
from typing import Tuple, List, Optional

class PanelCrossValidator:
    """
    A class implementing horizontal and vertical cross validation for panel data.
    Horizontal CV splits along rows (units/individuals)
    Vertical CV splits along columns (time periods) using forward-chaining validation
    """
    
    def __init__(self, n_splits: int = 5, random_state: Optional[int] = None):
        """
        Initialize the cross validator.
        
        Args:
            n_splits: Number of folds for cross validation
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
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
            masks = []
            for train_idx, test_idx in self.kf.split(np.arange(n_units)):
                train_mask = np.zeros(X.shape, dtype=bool)
                test_mask = np.zeros(X.shape, dtype=bool)
                train_mask[train_idx, :] = True
                test_mask[test_idx, :] = True
                masks.append((train_mask, test_mask))
            return masks
        
    def vertical_split(self, X: np.ndarray, min_train_size: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
            """
            Perform vertical cross validation using forward-chaining validation.
            Holds out the last k periods for testing, where k varies.
            
            Args:
                X: Input matrix of shape (n_units, n_times)
                min_train_size: Minimum number of time periods to include in the training set.
                                Defaults to half the total number of time periods.
                
            Returns:
                List of (train_mask, test_mask) tuples for each fold
            """
            n_times = X.shape[1]
            masks = []
            if min_train_size is None:
                min_train_size = n_times // 2  # Default to half the data for training
            
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
            flat_mask = np.ones(n_units * n_times, dtype=bool)
            for _ in range(self.n_splits):
                train_mask = np.zeros(X.shape, dtype=bool)
                test_mask = np.zeros(X.shape, dtype=bool)
                
                # Randomly select elements for test set
                n_test = int(n_units * n_times / self.n_splits)
                test_elements = np.random.choice(np.where(flat_mask)[0], size=n_test, replace=False)
                
                # Convert flat indices to 2D coordinates
                test_rows = test_elements // n_times
                test_cols = test_elements % n_times
                test_mask[test_rows, test_cols] = True
                train_mask = ~test_mask
                
                masks.append((train_mask, test_mask))
            train_mask = ~test_mask
            
            masks.append((train_mask, test_mask))
            
            return masks
        
    def create_train_test_masks(self, X: np.ndarray, split_type: str = 'horizontal', min_train_size: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
            """
            Create boolean masks for train and test sets.
            
            Args:
            X: Input matrix of shape (n_units, n_times)
            split_type: Type of split ('horizontal', 'vertical', or 'random')
            min_train_size: Minimum number of time periods to include in the training set (only for vertical split).
            
            Returns:
            List of (train_mask, test_mask) tuples for each fold
            """
            if split_type not in ['horizontal', 'vertical', 'random']:
                raise ValueError("Invalid split_type. Expected one of ['horizontal', 'vertical', 'random'], but got '{}'.".format(split_type))
            
            if split_type == 'horizontal':
                return self.horizontal_split(X)
            elif split_type == 'vertical':
                return self.vertical_split(X, min_train_size=min_train_size)
            else:
                return self.random_split(X)

def cross_validate(estimator, X: np.ndarray, split_type: str = 'horizontal', 
                  n_splits: int = 5, random_state: Optional[int] = None):
    """
    Perform cross validation for panel data.
    
    Args:
        estimator: An estimator object implementing fit and predict
        X: Input matrix of shape (n_units, n_times)
        split_type: Type of split ('horizontal' or 'vertical')
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        List of scores for each fold
    """
    cv = PanelCrossValidator(n_splits=n_splits, random_state=random_state)
    masks = cv.create_train_test_masks(X, split_type)
    scores = []
    
    for train_mask, test_mask in masks:
        # Fit on training data
        estimator.fit(X, train_mask)
        
        # Predict and calculate score
        predictions = estimator.predict(test_mask)
        score = estimator.score(X[test_mask], predictions[test_mask])
        scores.append(score)
    
    return scores

# TO DO:
# - Fix indendation errors
# - get test file to run through
# - Implement the fit and predict methods
# - Implement the score method
# - Test the cross_validate function with the MatrixCompletionEstimator class
# - Compare the results of horizontal and vertical cross validation
# - Use the cross_validate function to compare horizontal and vertical cross validation
# - Discuss the results