import logging
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
        masks = cv.create_train_test_masks(X, split_type='vertical', min_train_size=train_window)
        
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


######################################################################


def _solve_sdid_time_weights(
    Y_control_pre: np.ndarray,
    Y_control_post_avg: np.ndarray,
    regularization_strength: float,
    simplex_constraint: bool,
    num_pre_periods: int,
) -> np.ndarray:
    """
    Solve for time weights (lambda) for Synthetic Difference-in-Differences.

    Minimizes:
        sum_j (Y_control_post_avg_j - Y_control_pre_j @ lambda)^2
        + regularization_strength * ||lambda - 1/T_pre||^2_2
    Subject to simplex constraints if specified.

    Args:
        Y_control_pre: Control unit outcomes in pre-treatment (N_control x T_pre)
        Y_control_post_avg: Average post-treatment outcome for control units (N_control,)
        regularization_strength: Zeta_lambda, penalty on deviation from uniform weights.
        simplex_constraint: If True, enforce lambda_t >= 0 and sum(lambda_t) = 1.
        num_pre_periods: T_pre, number of pre-treatment periods.

    Returns:
        lambda_hat: Optimal time weights (T_pre,)
    """
    T_pre = num_pre_periods
    initial_lambda = np.ones(T_pre) / T_pre

    def objective_fn(lambda_weights: np.ndarray) -> float:
        term1 = np.sum((Y_control_post_avg - np.dot(Y_control_pre, lambda_weights)) ** 2)
        term2 = regularization_strength * np.sum((lambda_weights - (1 / T_pre)) ** 2)
        return term1 + term2

    # Gradient (optional, but can help SLSQP)
    # d_obj/d_lambda_k = -2 * sum_j (Y_j_post_avg - Y_j_pre @ lambda) * Y_jk_pre
    #                  + 2 * reg_strength * (lambda_k - 1/T_pre)
    def gradient_fn(lambda_weights: np.ndarray) -> np.ndarray:
        residuals = Y_control_post_avg - np.dot(Y_control_pre, lambda_weights)
        grad_term1 = -2 * np.dot(residuals, Y_control_pre)
        grad_term2 = 2 * regularization_strength * (lambda_weights - (1/T_pre))
        return grad_term1 + grad_term2


    if simplex_constraint:
        bounds = tuple((0, None) for _ in range(T_pre))
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        # Using SLSQP from scipy.optimize
        # For some reason, fmin_slsqp is already imported but minimize is more standard
        from scipy.optimize import minimize
        result = minimize(
            objective_fn,
            initial_lambda,
            method="SLSQP",
            jac=gradient_fn, # Providing gradient
            bounds=bounds,
            constraints=constraints,
            tol=1e-8 # A reasonable tolerance
        )
        if not result.success:
            # Fallback or warning if SLSQP fails
            logging.warning(f"SLSQP for time weights did not converge: {result.message}")
            # Could fall back to a simple ridge if simplex fails badly, or just return best effort
        return result.x
    else:
        # Standard ridge regression: (X'X + zeta*I) lambda = X'y_post + zeta * uniform_lambda_coeffs
        # where X is Y_control_pre, y_post is Y_control_post_avg
        # and I is adjusted for the (lambda - 1/T_pre)^2 penalty form.
        # Let lambda_tilde = lambda - 1/T_pre. Then lambda = lambda_tilde + 1/T_pre.
        # Objective: (y - X(lambda_tilde + 1/T_pre))^2 + zeta * lambda_tilde^2
        # ( (y - X * 1/T_pre) - X * lambda_tilde )^2 + zeta * lambda_tilde^2
        # Let y_adj = y - X * 1/T_pre.
        # Solve (X'X + zeta*I)lambda_tilde = X'y_adj for lambda_tilde
        # Then lambda = lambda_tilde + 1/T_pre
        
        # Simpler: directly solve for lambda using the objective's gradient set to zero
        # X'X lambda - X'y + zeta * (lambda - 1/T_pre) = 0
        # (X'X + zeta*I) lambda = X'y + zeta * 1/T_pre
        # where X is Y_control_pre, y is Y_control_post_avg
        XtX = np.dot(Y_control_pre.T, Y_control_pre)
        Xty = np.dot(Y_control_pre.T, Y_control_post_avg)
        
        # Add regularization term to the diagonal of XtX for the lambda part
        # and adjust the RHS for the (lambda - 1/T_pre) part of the penalty
        lhs = XtX + regularization_strength * np.eye(T_pre)
        rhs = Xty + regularization_strength * (np.ones(T_pre) / T_pre)
        
        try:
            lambda_hat = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            logging.warning("Singular matrix in time weight ridge regression. Using pseudo-inverse.")
            lambda_hat = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return lambda_hat
