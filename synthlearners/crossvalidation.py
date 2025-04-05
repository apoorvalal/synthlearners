import numpy as np
from sklearn.model_selection import KFold
from typing import Tuple, List, Optional
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from .utils import tqdm_joblib


class PanelCrossValidator:
    """
    A class implementing vertical and horizontal cross validation for panel data.
    Vertical CV splits along rows (units/individuals)
    Horizontal CV splits along columns (time periods) using forward-chaining validation

    Credit: Written with assistance from Claude Copilot
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_jobs: int = 5,
        cv_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ):
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
        self.n_jobs = n_jobs
        self.cv_ratio = cv_ratio
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


    def box_split(self, X, N_prime=None, T_prime=None, specified_rows=None):
        """
        Create cross-validation masks for an N×T matrix, where the test mask is N'×T'.
        The test mask always uses the last T' columns of the matrix.
        
        The function can optionally include specific rows in each test fold.
        
        Parameters:
        -----------
        X: Input matrix of shape (n_units, n_times)
        N_prime : int
            Number of rows in the test mask, must be <= N
        T_prime : int
            Number of columns in the test mask, must be <= T
        specified_rows : array-like, default=None
            Indices of specific rows that must be included in the test set.
            If provided, N_prime should be >= len(specified_rows)
        
        Returns:
        --------
        masks: list of tuples
            Each tuple contains (train_mask, test_mask) where each mask is a boolean matrix
            of shape (N, T). True values indicate elements to be used for training/testing.
        """
        N = X.shape[0]
        T = X.shape[1]
        N_prime = int((1-self.cv_ratio) * N) if N_prime is None else N_prime
        T_prime = int((1-self.cv_ratio) * T) if T_prime is None else T_prime

        if N_prime > N or T_prime > T:
            raise ValueError("N_prime and T_prime must be less than or equal to N and T respectively")
                
        # Process specified rows if provided
        if specified_rows is not None:
            specified_rows = np.array(specified_rows)
            if len(specified_rows) > N_prime:
                raise ValueError(f"Number of specified rows ({len(specified_rows)}) exceeds N_prime ({N_prime})")
            if np.max(specified_rows) > N or np.min(specified_rows) < 0:
                raise ValueError(f"Specified row indices must be between 0 and {N}")
            # Remove duplicates and sort
            specified_rows = np.unique(specified_rows)
        
        # Create a list to store our CV splits
        masks = []

        # Always use the last T_prime columns for testing
        col_test_idx = np.arange(T - T_prime, T)
        
        # If N_prime equals N, use all rows without splitting
        if N_prime == N or (specified_rows is not None and N_prime == len(specified_rows)):
            # Use either all rows or specified rows
            row_test_idx = np.arange(N) if N_prime == N else specified_rows

            # Create empty masks
            train_mask = np.ones((N, T), dtype=bool)
            test_mask = np.zeros((N, T), dtype=bool)

            # Set test points in the masks
            for i in row_test_idx:
                for j in col_test_idx:
                    test_mask[i, j] = True
                    train_mask[i, j] = False

            masks.append((train_mask, test_mask))
        else:
            # For each fold
            for _, row_test_idx in self.kf.split(np.arange(N)):
                # Handle specified rows
                if specified_rows is not None:
                    # Calculate how many additional rows we need to select
                    additional_rows_needed = N_prime - len(specified_rows)
                    
                    # Create a set of available rows (not in specified_rows)
                    available_rows = np.setdiff1d(row_test_idx, specified_rows)
                    
                    if additional_rows_needed > 0 and len(available_rows) > 0:
                        # Randomly select the remaining rows needed
                        additional_selected = np.random.choice(
                            available_rows, 
                            size=min(additional_rows_needed, len(available_rows)), 
                            replace=False
                        )
                        # Combine specified rows with additional random rows
                        row_test_idx = np.concatenate([specified_rows, additional_selected])
                    else:
                        # Just use the specified rows
                        row_test_idx = specified_rows
                # If no specified rows but N_prime is smaller than the default test size
                elif N_prime < len(row_test_idx):
                    row_test_idx = np.random.choice(row_test_idx, size=N_prime, replace=False)
            
            
                # Create empty masks
                train_mask = np.ones((N, T), dtype=bool)
                test_mask = np.zeros((N, T), dtype=bool)
                
                # Set test points in the masks
                for i in row_test_idx:
                    for j in col_test_idx:
                        test_mask[i, j] = True
                        train_mask[i, j] = False
                
                masks.append((train_mask, test_mask))
        
        return masks

    def vertical_split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform vertical cross validation (splitting along rows).

        Args:
            X: Input matrix of shape (n_units, n_times)

        Returns:
            List of (train_mask, test_mask) tuples for each fold
        """
        n_times = X.shape[1]
        # Test is a set of test_size units for all n_times periods
        masks = self.box_split(X=X, T_prime=n_times)
        return masks

    def horizontal_split(
        self, X: np.ndarray, min_train_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform horizontal cross validation using forward-chaining validation.

        Args:
            X: Input matrix of shape (n_units, n_times)
            min_train_size: Minimum number of time periods to include in training set

        Returns:
            List of (train_mask, test_mask) tuples for each fold
        """
        n_units = X.shape[0]
        n_times = X.shape[1]
        masks = []
        if min_train_size is None:
            min_train_size = int(n_times * self.cv_ratio)
        
        for test_size in range(1, min(self.n_splits + 1, n_times - min_train_size + 1)):
            # Test is all N units for test_size time periods
            singlet_masks = self.box_split(X=X, N_prime=n_units, T_prime=test_size)
            masks.append(singlet_masks[0])
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
            train_elements = np.random.choice(
                total_elements, size=train_size, replace=False
            )
            train_rows = train_elements // n_times
            train_cols = train_elements % n_times
            train_mask[train_rows, train_cols] = True
            test_mask = ~train_mask

            masks.append((train_mask, test_mask))
        return masks

    def create_train_test_masks(
        self,
        X: np.ndarray,
        split_type: str = "horizontal",
        min_train_size: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create boolean masks for train and test sets.

        Args:
            X: Input matrix of shape (n_units, n_times)
            split_type: Type of split ('horizontal', 'vertical', or 'random')
            min_train_size: Minimum number of time periods to include in training set

        Returns:
            List of (train_mask, test_mask) tuples for each fold
        """
        if split_type not in ["horizontal", "vertical", "random","box"]:
            raise ValueError(
                "Invalid split_type. Expected one of ['horizontal', 'vertical','box', 'random']"
            )

        if split_type == "vertical":
            return self.vertical_split(X)
        elif split_type == "horizontal":
            return self.horizontal_split(X, min_train_size=min_train_size)
        elif split_type == "box":
            return self.box_split(X)
        else:
            return self.random_split(X)


def cross_validate(
    estimator,
    X: np.ndarray,
    cv: PanelCrossValidator,
    split_type: str = "vertical",
    fit_method: str = "fit",
    fit_args: dict = None,
):
    """
    Perform cross validation for panel data.

    Args:
        estimator: An estimator object implementing fit and predict
        X: Input matrix of shape (n_units, n_times)
        cv: PanelCrossValidator object
        split_type: Type of split ('horizontal' or 'vertical')
        fit_method: Name of the fit method to call on estimator
        fit_args: Dictionary of additional arguments to pass to fit_method

    Returns:
        List of scores for each fold
    """
    if fit_args is None:
        fit_args = {}

    masks = cv.create_train_test_masks(X, split_type)
    with tqdm_joblib(tqdm(total=cv.n_splits, desc="Cross validation")):
        scores = Parallel(n_jobs=cv.n_jobs)(
            delayed(
                lambda m: estimator.score(
                    X, getattr(estimator, fit_method)(X, m[0], **fit_args), m[1]
                )
            )(mask_pair)
            for mask_pair in masks
        )
        
    return scores


# TO DO:
# - Compare MC results with random, horizontal, and vertical cross-validation
