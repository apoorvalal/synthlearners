from typing import List, Optional, Union
import numpy as np
from sklearn.linear_model import LinearRegression
from .synth import Synth
from .crossvalidation import PanelCrossValidator

class SynthEnsemble:
    """
    Ensemble method that combines multiple synthetic control methods using cross-validation.
    """
    def __init__(self, methods: List[Synth], cv_splits: int = 5, cv_ratio: float = 0.5,
                 weighting_model=None, split_type: str = "box", **split_args):
        """
        Initialize the ensemble.
        
        Args:
            methods: List of SynthMethod instances to ensemble
            cv_splits: Number of cross-validation splits
            weighting_model: Model to learn weights (defaults to OLS)
            split_type: Type of split to use ("horizontal", "vertical", "random")
            split_args: Additional arguments to pass to the cross-validator
        """
        self.methods = methods
        self.cv_splits = cv_splits
        self.cv_ratio = cv_ratio
        self.weighting_model = weighting_model or LinearRegression()
        self.weights_ = None
        self.split_type = split_type
        self.split_args = split_args
        
    def fit(self, X: np.ndarray, treated_units: Union[int, np.ndarray], 
            T_pre: int) -> "SynthEnsemble":
        """
        Fit the ensemble by learning weights for each method using cross-validation.
        
        Args:
            X: Array of shape (n_units, n_time) containing the data
            treated_units: Index or array of indices of treated units
            T_pre: Number of pre-treatment periods
        """
        # Check if split_type is random_split, which is not supported
        if self.split_type == "random_split":
            raise ValueError("Random split type is not supported for ensemble fitting")

        # Generate cross-validation samples
        cv = PanelCrossValidator(n_splits=self.cv_splits,cv_ratio=self.cv_ratio)

        # Create clean version of X excluding treated units and post-treatment periods
        if isinstance(treated_units, int):
            treated_units = [treated_units]
        control_units = np.array([i for i in range(X.shape[0]) if i not in treated_units])
        # Define clean periods as the first T_pre columns index in X
        clean_periods = np.arange(T_pre)

        if self.split_type == "horizontal":
            # Keep clean_periods and all units
            X_clean = X[:, clean_periods]
        elif self.split_type in ("vertical","box"):
            X_clean = X[np.ix_(control_units, clean_periods)]
        
        # Get train/test masks using specified split type on clean data
        masks = cv.create_train_test_masks(X_clean, split_type=self.split_type, **self.split_args)

        # Store predictions from each method for each CV split
        all_preds = []
        all_targets = []
        
        # For each CV split
        i = 0
        for train_mask, test_mask in masks:
            i+= 1
            print(f"Processing split {i}/{self.cv_splits}")

            split_preds = []
            # Verify that train mask has some true and some false
            if not np.any(train_mask) or not np.any(~train_mask):
                raise ValueError("Train mask must contain both treated and control observations")

            # Calc number of true values by row and column in train_mask
            n_true_rows = np.sum(train_mask, axis=1)
            # Determine clean_treated as rows where n_true_cols is less than the total possible
            psuedo_treated = np.array([i for i in range(X_clean.shape[0]) if n_true_rows[i] < X_clean.shape[1]])
            # Get pre-treatment periods as the smallest number of true values in the treated units
            pseudo_T_pre = np.min(n_true_rows[psuedo_treated])
            
            # Get predictions from each method
            for method in self.methods:
                results = method.fit(X_clean, psuedo_treated, pseudo_T_pre)
                pred = results.synthetic_outcome
                split_preds.append(pred)

            print(split_preds) # List of 3 arrays of 5 values
            print(X_clean[test_mask]) # List of 4 values
            # TO DO: Figure out how to get split preds to just get the test_mask elements
            
            # Stack predictions and store targets
            all_preds.append(np.column_stack(split_preds))
            all_targets.append(X_clean[test_mask])

            # Print finished split
            print(f"Finished split {i}/{self.cv_splits}")
        
        # Combine all predictions and targets
        X_train = np.vstack(all_preds)
        y_train = np.concatenate(all_targets)
        # Verify X_train and y_train have the same number of rows
        if X_train.shape[0] != y_train.shape[0]:
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            raise ValueError("X_train and y_train must have the same number of rows")
        
        # Fit the weighting model
        self.weighting_model.fit(X_train, y_train)
        self.weights_ = self.weighting_model.coef_
        
        # Fit all methods on the full data
        for method in self.methods:
            method.fit(X, treated_units, T_pre)
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the weighted ensemble.
        
        Args:
            X: Array of shape (n_units, n_time) containing the data
            
        Returns:
            Weighted average predictions from all methods
        """
        if self.weights_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        # Get predictions from all methods
        predictions = []
        for method in self.methods:
            pred = method.predict(X)
            predictions.append(pred)
            
        # Combine predictions using learned weights
        predictions = np.column_stack(predictions)
        return predictions @ self.weights_