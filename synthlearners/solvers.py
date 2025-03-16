from typing import Optional

import numpy as np
from scipy.optimize import fmin_slsqp
import pyensmallen as pe
from synthlearners.crossvalidation import PanelCrossValidator

from .knn_faiss import FastNearestNeighbors

######################################################################


def _objective(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Compute objective function (squared error loss)."""
    return np.sum((np.dot(X, w) - y) ** 2)


def _gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute gradient of objective function."""
    return 2 * np.dot(X.T, (np.dot(X, w) - y))


######################################################################


def _solve_lp_norm(
    Y_control: np.ndarray,
    Y_treated: np.ndarray,
    p: float,
    max_iterations: int,
    tolerance: float,
    reg_param: Optional[float] = None,
) -> np.ndarray:
    """Solve synthetic control problem using Frank-Wolfe with Lp norm constraint."""
    N_control = Y_control.shape[0]

    def f(w, grad):
        if grad.size > 0:
            grad[:] = _gradient(w, Y_control.T, Y_treated)
        return _objective(w, Y_control.T, Y_treated)

    opt_args = {"p": p, "max_iterations": max_iterations, "tolerance": tolerance}
    if reg_param is not None:
        opt_args["lambda"] = np.ones(N_control) * reg_param

    optimizer = pe.FrankWolfe(**opt_args)
    initial_w = np.ones(N_control) / N_control
    return optimizer.optimize(f, initial_w)


def _choose_lambda(
    Y_control: np.ndarray,
    Y_treated: np.ndarray,
    p: float,
    lam_grid: np.ndarray = np.logspace(-4, 2, 20),
    n_splits: int = 2,
    val_window: int = 1,
) -> float:
    """Choose optimal lambda via time series cross-validation."""
    T_pre = Y_treated.shape[0]
    cv_ratio = (T_pre - val_window) / T_pre
    train_window = T_pre - val_window
    lambda_grid = lam_grid * np.std(Y_control)
    cv_scores = []

    for lambda_val in lambda_grid:
        # Initialize panel cross validator
        cv = PanelCrossValidator(n_splits=n_splits, cv_ratio=cv_ratio)
        
        # Get train/test splits
        X = np.vstack((Y_control, Y_treated.reshape(1, -1)))  # Combine data into single matrix
        split_errors = []
        
        # Get vertical splits with forward chaining
        masks = cv.vertical_split(X, min_train_size=train_window)
        
        for train_mask, test_mask in masks:
            # Extract training data 
            Y_control_train = Y_control[:, train_mask[0]]
            Y_treated_train = Y_treated[train_mask[0]]
            
            # Get weights using training data
            weights = _solve_lp_norm(
            Y_control_train,
            Y_treated_train,
            p=p,
            max_iterations=10000,
            tolerance=1e-8, 
            reg_param=lambda_val
            )
            
            # Compute validation error on test period
            Y_control_test = Y_control[:, test_mask[0]]
            Y_treated_test = Y_treated[test_mask[0]]
            val_pred = np.dot(Y_control_test.T, weights)
            rmse = np.sqrt(np.mean((val_pred - Y_treated_test) ** 2))
            split_errors.append(rmse)
        cv_scores.append(np.mean(split_errors))
    return lambda_grid[np.argmin(cv_scores)]


######################################################################


def _solve_linear(
    Y_control: np.ndarray, Y_treated: np.ndarray, max_iterations: int, tolerance: float
) -> np.ndarray:
    """Solve synthetic control problem using ordinary least squares."""
    return np.linalg.lstsq(Y_control.T, Y_treated, rcond=None)[0]


######################################################################


def _solve_simplex(
    Y_control: np.ndarray, Y_treated: np.ndarray, max_iterations: int, tolerance: float
) -> np.ndarray:
    def f(w, grad):
        if grad.size > 0:
            grad[:] = _gradient(w, Y_control.T, Y_treated)
        return _objective(w, Y_control.T, Y_treated)

    # Initialize optimizer
    optimizer = pe.SimplexFrankWolfe(maxIterations=max_iterations, tolerance=tolerance)
    w_init = np.repeat(1.0 / Y_control.shape[0], Y_control.shape[0])
    w_opt = optimizer.optimize(f, w_init)
    return w_opt


def _solve_simplex_slsqp(Y_control: np.ndarray, Y_treated: np.ndarray) -> np.ndarray:
    """Solve synthetic control problem with simplex constraints."""
    N_control = Y_control.shape[0]
    initial_w = np.repeat(1 / N_control, N_control)
    bounds = tuple((0, 1) for _ in range(N_control))

    weights = fmin_slsqp(
        func=lambda w: _objective(w, Y_control.T, Y_treated),
        x0=initial_w,
        f_eqcons=lambda x: np.sum(x) - 1,
        bounds=bounds,
        disp=False,
    )
    return weights


######################################################################


def _solve_matching(
    Y_control: np.ndarray,
    Y_treated: np.ndarray,
    k: int = 5,
    max_iterations: int = 1e4,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """
    Solve synthetic control problem using k-nearest neighbors matching.

    Args:
        Y_control (np.ndarray): Control unit outcomes of shape (N_control, T)
        Y_treated (np.ndarray): Treated unit outcomes of shape (T,)
        k (int): Number of nearest neighbors to match with

    Returns:
        np.ndarray: Weight vector of length N_control with k entries equal to 1/k
                   and the rest equal to 0
    """
    # Initialize the FAISS nearest neighbors finder
    # Using Mahalanobis distance to account for correlation in time series
    nn = FastNearestNeighbors(metric="mahalanobis", index_type="flatl2")

    # Fit on control units
    nn.fit(Y_control)

    # Find k nearest neighbors
    # Note: Y_treated needs to be reshaped to 2D array for FAISS
    Y_treated_2d = Y_treated.reshape(1, -1)
    _, indices = nn.kneighbors(Y_treated_2d, n_neighbors=k)

    # Create weight vector
    N_control = Y_control.shape[0]
    weights = np.zeros(N_control)

    # Assign equal weights (1/k) to the k nearest neighbors
    weights[indices[0]] = 1.0 / k

    return weights
