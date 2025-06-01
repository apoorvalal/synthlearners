from typing import List, Optional, Union
import numpy as np
from sklearn.linear_model import LinearRegression
from .synth import SynthMethod
from .crossvalidation import PanelCrossValidator

class SynthEnsemble:
    """
    Ensemble method that combines multiple synthetic control methods using cross-validation.
    """
    def __init__(self, methods: List[SynthMethod], cv_splits: int = 5, 
                 weighting_model=None, split_type: str = "horizontal", **split_args):
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
        # Generate cross-validation samples
        cv = PanelCrossValidator(n_splits=self.cv_splits)

        # Create clean version of X excluding treated units
        if isinstance(treated_units, int):
            treated_units = [treated_units]
        control_units = np.array([i for i in range(X.shape[0]) if i not in treated_units])
        X_clean = X[control_units]
        
        # Get train/test masks using specified split type on clean data
        masks = cv.create_train_test_masks(X_clean, split_type=self.split_type, **self.split_args)
        
        # Store predictions from each method for each CV split
        all_preds = []
        all_targets = []
        
        # For each CV split
        for train_mask, test_mask in masks:
            split_preds = []
            X_train = X_clean.copy()
            X_test = X_clean.copy()
            
            # Apply masks
            X_train[~train_mask] = np.nan
            X_test[~test_mask] = np.nan
            
            # Determine clean_T_pre as the longest time available in the train_mask
            
            # Determine clean_treated as any units with non-nan data in X_test

            
            # Get predictions from each method
            for method in self.methods:
                results = method.fit(X_train, clean_treated, clean_T_pre)
                pred = results.synthetic_outcome[len(treated_units):][test_mask]
                split_preds.append(pred)
            
            # Stack predictions and store targets
            all_preds.append(np.column_stack(split_preds))
            all_targets.append(X_clean[test_mask])
        
        # Combine all predictions and targets
        X_train = np.vstack(all_preds)
        y_train = np.concatenate(all_targets)
        
        # Fit the weighting model
        self.weighting_model.fit(X_train, y_train)
        self.weights_ = self.weighting_model.coef_
        
        # Fit all methods on the full data
        for method in self.methods:
            method.fit(X, treated_units, T_pre, T_post)
            
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