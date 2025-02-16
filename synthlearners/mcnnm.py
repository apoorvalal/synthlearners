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
    def __init__(self, lambda_param=1e-3, max_iter=500, tol=1e-4, verbose=False):
        """
        Parameters:
          lambda_param: regularization strength (the weight on the nuclear norm penalty)
          max_iter: maximum number of iterations
          tol: relative tolerance for convergence
          verbose: if True, print progress information
        """
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.completed_matrix_ = None

    def shrink_lambda(self, A, threshold):
        """
        Apply singular value thresholding (soft-thresholding of singular values).
        
        Parameters:
          A: matrix to threshold.
          threshold: threshold level (usually tau * lambda_param)
        
        Returns:
          The matrix after applying soft-thresholding to its singular values.
        """
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Soft-threshold the singular values.
        s_thresholded = np.maximum(s - threshold, 0)
        if self.verbose:
            print(f"singular values: {s}")
            print(f"s_thresholded: {s_thresholded}")
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
        # Initialize the estimate L.

        L = np.zeros_like(mask*Y)
        shrink_treshhold = self.lambda_param*np.sum(mask)/2
        if self.verbose:
            print(f"shrink_treshhold: {shrink_treshhold}")
        for it in range(self.max_iter):
            L_old = L.copy()
            shrink_A_input = mask*Y+(1-mask)*L_old
            # Apply shrinkage per Equation 4.5 in the paper
            L = self.shrink_lambda(shrink_A_input, shrink_treshhold)
            # Check convergence using the relative change (Nuclear norm).
            norm_old = np.linalg.norm(L_old, 'fro') + 1e-8  # avoid div-by-zero
            diff = np.linalg.norm(L - L_old, 'fro') / norm_old
            if self.verbose:
                print(f"Iteration {it + 1:3d}, relative change = {diff:.6f}")
            if diff < self.tol:
                if self.verbose:
                    print("Convergence achieved.")
                break
        self.completed_matrix_ = L
        return self

# TODO: Add cross-validation per secion 4 in the paper to select lambda_param
# TODO: Add support ofr time and unit FE per paper
    
