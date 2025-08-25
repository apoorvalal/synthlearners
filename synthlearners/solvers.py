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
        X = np.vstack(
            (Y_control, Y_treated.reshape(1, -1))
        )  # Combine data into single matrix
        split_errors = []

        # Get vertical splits with forward chaining
        masks = cv.create_train_test_masks(
            X, split_type="horizontal", min_train_size=train_window
        )

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
                reg_param=lambda_val,
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
        # Use pyensmallen SimplexFrankWolfe optimizer for consistency with unit weights
        optimizer = pe.SimplexFrankWolfe(maxIterations=10000, tolerance=1e-8)
        lam_init = np.ones(T_pre) / T_pre
        
        def time_objective(lam, grad):
            """SDID time weights objective function for pyensmallen."""
            if grad.size > 0:
                # Compute gradient
                residuals = Y_control_post_avg - np.dot(Y_control_pre, lam)
                grad_term1 = -2 * np.dot(residuals, Y_control_pre)
                grad_term2 = 2 * regularization_strength * (lam - (1 / T_pre))
                grad[:] = grad_term1 + grad_term2
            
            # Compute objective value
            term1 = np.sum((Y_control_post_avg - np.dot(Y_control_pre, lam)) ** 2)
            term2 = regularization_strength * np.sum((lam - (1 / T_pre)) ** 2)
            return term1 + term2
        
        lam_opt = optimizer.optimize(time_objective, lam_init)
        return lam_opt
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


def _simplex_least_squares_with_intercept(
    A: np.ndarray, 
    b: np.ndarray, 
    zeta: float = 1e-6,
    max_iterations: int = 10000,
    tolerance: float = 1e-8
) -> np.ndarray:
    """
    Solve simplex-constrained least squares with intercept:
    minimize ||A @ w + w0 - b||^2 + zeta^2 * len(b) * ||w||^2
    subject to w >= 0, sum(w) = 1
    
    This follows the original SDID paper's formulation.
    """
    n_vars = A.shape[1]
    
    def objective(w, grad):
        if grad.size > 0:
            # Compute optimal intercept for given w: w0 = mean(b - A @ w)
            residual = b - A @ w
            w0_opt = np.mean(residual)
            adjusted_residual = residual - w0_opt
            
            # Gradient w.r.t. w
            grad_fit = -2 * A.T @ adjusted_residual  
            grad_reg = 2 * zeta**2 * len(b) * w
            grad[:] = grad_fit + grad_reg
        
        # Compute objective value with optimal intercept
        residual = b - A @ w
        w0_opt = np.mean(residual)
        adjusted_residual = residual - w0_opt
        
        fit_term = np.sum(adjusted_residual**2)
        reg_term = zeta**2 * len(b) * np.sum(w**2)
        return fit_term + reg_term
    
    # Use pyensmallen SimplexFrankWolfe for consistency
    optimizer = pe.SimplexFrankWolfe(maxIterations=max_iterations, tolerance=tolerance)
    w_init = np.ones(n_vars) / n_vars
    w_opt = optimizer.optimize(objective, w_init)
    return w_opt


def _solve_sdid_matrix(
    Y: np.ndarray,
    treated_units: np.ndarray, 
    T_pre: int,
    zeta_omega: float = None,
    zeta_lambda: float = 1e-6
) -> tuple:
    """
    Solve SDID using original matrix formulation from Arkhangelsky et al. (2021).
    
    Args:
        Y: Panel data matrix (N_units x T_periods)
        treated_units: Array of treated unit indices
        T_pre: Number of pre-treatment periods
        zeta_omega: Regularization for unit weights (if None, uses paper's default)
        zeta_lambda: Regularization for time weights
        
    Returns:
        tuple: (unit_weights, time_weights, sdid_estimate)
    """
    N, T = Y.shape
    control_units = np.setdiff1d(range(N), treated_units)
    N0 = len(control_units)  # Number of control units
    N1 = len(treated_units)  # Number of treated units  
    T0 = T_pre  # Number of pre-treatment periods
    T1 = T - T_pre  # Number of post-treatment periods
    
    # Rearrange data: controls first, then treated (like original R code)
    unit_order = np.concatenate([control_units, treated_units])
    Y_reordered = Y[unit_order, :]
    
    # Extract submatrices
    Y_control_pre = Y_reordered[:N0, :T0]  # Control units, pre-treatment
    Y_control_post = Y_reordered[:N0, T0:]  # Control units, post-treatment  
    Y_treated_pre = Y_reordered[N0:, :T0]   # Treated units, pre-treatment
    
    # Default regularization following original paper
    if zeta_omega is None:
        sigma_est = np.std([np.std(np.diff(Y_reordered[i, :T0])) for i in range(N0)])
        zeta_omega = ((N1 * T1) ** 0.25) * sigma_est
    
    # Solve for time weights λ
    # Predict post-treatment control outcomes from pre-treatment control outcomes
    Y_control_post_mean = np.mean(Y_control_post, axis=1)  # Average over post periods
    lambda_weights = _simplex_least_squares_with_intercept(
        Y_control_pre, Y_control_post_mean, zeta=zeta_lambda
    )
    
    # Solve for unit weights ω  
    # Predict treated pre-treatment outcomes from control pre-treatment outcomes
    Y_treated_pre_mean = np.mean(Y_treated_pre, axis=0)  # Average over treated units
    omega_weights = _simplex_least_squares_with_intercept(
        Y_control_pre.T, Y_treated_pre_mean, zeta=zeta_omega  
    )
    
    # Compute SDID estimate using matrix formulation
    # Unit weight vector: [-ω, 1/N1, 1/N1, ...]  
    unit_weight_vec = np.concatenate([-omega_weights, np.ones(N1) / N1])
    # Time weight vector: [-λ, 1/T1, 1/T1, ...]
    time_weight_vec = np.concatenate([-lambda_weights, np.ones(T1) / T1])
    
    # Final estimate: unit_weights^T @ Y @ time_weights
    sdid_estimate = unit_weight_vec.T @ Y_reordered @ time_weight_vec
    
    return omega_weights, lambda_weights, float(sdid_estimate)


######################################################################


