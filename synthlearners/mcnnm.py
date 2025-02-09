# src/estimator.py

import numpy as np

class MatrixCompletionEstimator:
    """
    A basic matrix completion estimator for treatment effect estimation.
    
    The model assumes that the full outcome matrix Y can be decomposed into a low-rank component,
    and that missing entries (e.g. counterfactual outcomes) can be imputed by solving a nuclear norm 
    regularized least-squares problem.
    
    The optimization problem is:
    
        minimize_X 0.5 * || P_Omega(X - Y) ||_F^2 + lambda * ||X||_*
    
    where P_Omega is the projection onto the observed entries.

    #### References
    Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi. <b>Matrix Completion Methods for Causal Panel Data Models</b> [<a href="http://arxiv.org/abs/1710.10251">link</a>]

    R equivalent - [MCPanel](https://github.com/susanathey/MCPanel/tree/master).

    NOTE: Built in collaboration with ChatGPT o3-mini-high

    """
    def __init__(self, lambda_param=1.0, tau=1.0, max_iter=500, tol=1e-4, verbose=False):
        """
        Parameters:
          lambda_param: regularization strength (the weight on the nuclear norm penalty)
          tau: step size for the gradient descent update
          max_iter: maximum number of iterations
          tol: relative tolerance for convergence
          verbose: if True, print progress information
        """
        self.lambda_param = lambda_param
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.completed_matrix_ = None

    def _svt(self, M, threshold):
        """
        Apply singular value thresholding (soft-thresholding of singular values).
        
        Parameters:
          M: matrix to threshold.
          threshold: threshold level (usually tau * lambda_param)
        
        Returns:
          The matrix after applying soft-thresholding to its singular values.
        """
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Soft-threshold the singular values.
        s_thresholded = np.maximum(s - threshold, 0)
        return U @ np.diag(s_thresholded) @ Vt

    def fit(self, Y, mask):
        """
        Fit the matrix completion model.

        Parameters:
          Y: 2D numpy array of observed outcomes. For missing entries, you may set Y to 0 (or any placeholder)
          mask: binary 2D numpy array of the same shape as Y where 1 indicates an observed entry and 0 a missing one.

        Returns:
          self, with the completed matrix stored in self.completed_matrix_
        """
        # Initialize the estimate X.
        X = np.zeros_like(Y)
        for it in range(self.max_iter):
            X_old = X.copy()
            # Compute the gradient on the observed entries.
            grad = mask * (X - Y)
            # Take a gradient descent step.
            X = X - self.tau * grad
            # Apply singular value thresholding.
            X = self._svt(X, self.tau * self.lambda_param)
            # Check convergence using the relative change (Frobenius norm).
            norm_old = np.linalg.norm(X_old, 'fro') + 1e-8  # avoid div-by-zero
            diff = np.linalg.norm(X - X_old, 'fro') / norm_old
            if self.verbose:
                print(f"Iteration {it + 1:3d}, relative change = {diff:.6f}")
            if diff < self.tol:
                if self.verbose:
                    print("Convergence achieved.")
                break
        self.completed_matrix_ = X
        return self

    def predict(self, mask_missing):
        """
        Predict (impute) the missing entries.

        Parameters:
          mask_missing: binary 2D numpy array of the same shape as the outcome matrix where 1 indicates
                        a missing (to-be-imputed) entry.
                        
        Returns:
          A 2D numpy array of predicted (imputed) outcomes for the missing entries.
        """
        if self.completed_matrix_ is None:
            raise ValueError("You must call fit() before predict().")
        return mask_missing * self.completed_matrix_
    
