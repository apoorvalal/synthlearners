import numpy as np
from synthlearners.crossvalidation import PanelCrossValidator, cross_validate

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

    NOTE: Built in collaboration with ChatGPT

    """

    def __init__(self, max_iter=500, tol=1e-4, verbose=False):
        """
        Parameters:
          lambda_param: regularization strength (the weight on the nuclear norm penalty)
          max_iter: maximum number of iterations
          tol: relative tolerance for convergence
          verbose: if True, print progress information
        """
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
        return U @ np.diag(s_thresholded) @ Vt, s_thresholded

    def compute_matrix(self, L, u, v):
        """
        This function computes L + u1^T + 1v^T, which is our decomposition.

        Parameters:
        L (numpy.ndarray): Input matrix L
        u (numpy.ndarray): Vector u
        v (numpy.ndarray): Vector v

        Returns:
        numpy.ndarray: Computed matrix
        """
        L, u, v = map(np.asarray, (L, u, v))  # Convert inputs to numpy arrays

        # Compute L + u * 1^T + 1 * v^T
        res = L + np.outer(u, np.ones(L.shape[1])) + np.outer(np.ones(L.shape[0]), v)

        return res  # Return computed matrix

    def update_u(self, M, mask, L, v):
        """
        This function updates u in coordinate descent algorithm.

        Parameters:
        M (numpy.ndarray): Input matrix M
        mask (numpy.ndarray): Mask matrix
        L (numpy.ndarray): Matrix L
        v (numpy.ndarray): Vector v

        Returns:
        numpy.ndarray: Updated vector u
        """
        M, mask, L, v = map(
            np.asarray, (M, mask, L, v)
        )  # Convert inputs to numpy arrays

        b = L + v - M  # Compute the difference matrix
        b_masked = b * mask  # Apply element-wise mask
        mask_counts = np.count_nonzero(mask, axis=1)  # Count non-zero elements per row

        # Compute updated vector u, handling division by zero safely
        res = np.where(mask_counts > 0, -b_masked.sum(axis=1) / mask_counts, 0)

        return res  # Return updated vector

    def update_v(self, M, mask, L, u):
        """
        This function updates v in the coordinate descent algorithm.

        Parameters:
        M (numpy.ndarray): Input matrix M
        mask (numpy.ndarray): Mask matrix
        L (numpy.ndarray): Matrix L
        u (numpy.ndarray): Vector u

        Returns:
        numpy.ndarray: Updated vector v
        """
        M, mask, L, u = map(
            np.asarray, (M, mask, L, u)
        )  # Convert inputs to numpy arrays

        b = L + u[:, np.newaxis] - M  # Compute the difference matrix
        b_masked = b * mask  # Apply element-wise mask
        mask_counts = np.count_nonzero(
            mask, axis=0
        )  # Count non-zero elements per column

        # Compute updated vector v, handling division by zero safely
        res = np.where(mask_counts > 0, -b_masked.sum(axis=0) / mask_counts, 0)

        return res  # Return updated vector

    def compute_objval(self, M, mask, L, u, v, sum_sing_vals, lambda_L):
        """
        This function computes the objective value, which is a weighted combination
        of error plus nuclear norm.

        Parameters:
        M (numpy.ndarray): Input matrix M
        mask (numpy.ndarray): Mask matrix
        L (numpy.ndarray): Matrix L
        u (numpy.ndarray): Vector u
        v (numpy.ndarray): Vector v
        sum_sing_vals (float): Sum of singular values
        lambda_L (float): Regularization parameter

        Returns:
        float: Computed objective value
        """
        M, mask, L, u, v = map(
            np.asarray, (M, mask, L, u, v)
        )  # Convert inputs to numpy arrays

        train_size = np.sum(mask)  # Count the number of training samples

        est_mat = self.compute_matrix(L, u, v)  # Compute the estimated matrix

        err_mat = (est_mat - M) * mask  # Compute the error matrix with mask applied

        # Compute the objective value
        obj_val = (1 / train_size) * np.sum(err_mat**2) + lambda_L * sum_sing_vals

        return obj_val  # Return computed objective value

    def initialize_uv(self, M, mask, to_estimate_u=True, to_estimate_v=True):
        """
        Finds optimal u and v assuming L is zero, useful for warm start on lambda_L values.
        Also computes the smallest lambda_L that nullifies L.
        """
        M, mask = map(np.asarray, (M, mask))
        num_rows, num_cols = M.shape

        u, v = np.zeros(num_rows), np.zeros(num_cols)
        L = np.zeros((num_rows, num_cols))
        obj_val = self.compute_objval(M, mask, L, u, v, 0, 0)

        for _ in range(self.max_iter):
            if to_estimate_u:
                u = self.update_u(M, mask, L, v)
            else:
                u = np.zeros(num_rows)

            if to_estimate_v:
                v = self.update_v(M, mask, L, u)
            else:
                v = np.zeros(num_cols)

            new_obj_val = self.compute_objval(M, mask, L, u, v, 0, 0)
            rel_error = (new_obj_val - obj_val) / obj_val
            if 0 <= rel_error < self.tol:
                break
            obj_val = new_obj_val

        E = self.compute_matrix(L, u, v)
        P_omega = (M - E) * mask

        # Compute SVD and find lambda_L_max
        singular_values = np.linalg.svd(P_omega, compute_uv=False)
        lambda_L_max = 2.0 * np.max(singular_values) / np.sum(mask)

        return {"u": u, "v": v, "lambda_L_max": lambda_L_max}

    def update_L(self, M, mask, L, u, v, lambda_s):
        """
        Updates L in the coordinate descent algorithm using Singular Value Thresholding (SVT).
        Saves singular values for computing the objective value efficiently.

        Parameters:
        M (numpy.ndarray): Input matrix M
        mask (numpy.ndarray): Mask matrix
        L (numpy.ndarray): Current matrix L
        u (numpy.ndarray): Vector u
        v (numpy.ndarray): Vector v
        lambda (float): Regularization parameter

        Returns:
        dict: Updated matrix L and singular values
        """
        M, mask, L, u, v = map(np.asarray, (M, mask, L, u, v))

        H = self.compute_matrix(L, u, v)  # Compute H matrix
        P_omega = (M - H) * mask  # Compute masked projection matrix
        proj = P_omega + L  # Add previous L
        # Note: Different from no FE case [M*mask+(1-mask)*L]
        # Assume u and v are 0 [(M-L)*mask + L] = M*mask + (1-mask)*L, so equivalent

        L_upd, sing = self.shrink_lambda(proj, lambda_s)
        return {"L": L_upd, "Sigma": sing}

    # Full NNM fit with fixed effects in individual and time.
    def NNM_fit(self, M, mask, lambda_L, to_estimate_u=True, to_estimate_v=True):
        """
        Performs coordinate descent updates for matrix decomposition. Added complexity over self.fit() is due to the
        addtion of the unit and time fixed effects.

        Parameters:
        M (numpy.ndarray): Input matrix M
        mask (numpy.ndarray): Mask matrix
        lambda_L (float): Regularization parameter
        to_estimate_u (bool): Whether to estimate u
        to_estimate_v (bool): Whether to estimate v

        Returns:
        dict: Updated matrices L, u, and v
        """
        shrink_treshhold = (
            lambda_L * np.sum(mask) / 2
        )  # For now keep consistent with other method
        L_init = np.zeros_like(M)
        init_uv = self.initialize_uv(M, mask, to_estimate_u, to_estimate_v)
        lambda_L_max = init_uv["lambda_L_max"]
        shrink_treshhold_max = (
            lambda_L_max * np.sum(mask) / 2
        )  # For now keep consistent with other method
        u_init = init_uv["u"]
        v_init = init_uv["v"]

        M, mask, L, u, v = map(np.asarray, (M, mask, L_init, u_init, v_init))

        sing = np.linalg.svd(L, compute_uv=False)  # Compute singular values
        sum_sigma = np.sum(sing)
        obj_val = self.compute_objval(M, mask, L, u, v, sum_sigma, lambda_L)
        term_iter = 0

        if self.verbose:
            print(f"shrink_treshhold: {shrink_treshhold}")

        for iter in range(self.max_iter):
            # Update u
            u = self.update_u(M, mask, L, v) if to_estimate_u else np.zeros(M.shape[0])

            # Update v
            v = self.update_v(M, mask, L, u) if to_estimate_v else np.zeros(M.shape[1])

            # Update L
            upd_L = self.update_L(M, mask, L, u, v, shrink_treshhold)
            L = upd_L["L"]
            sing = upd_L["Sigma"]
            sum_sigma = np.sum(sing)

            # Check if accuracy is achieved
            new_obj_val = self.compute_objval(M, mask, L, u, v, sum_sigma, lambda_L)
            rel_error = (obj_val - new_obj_val) / obj_val

            if new_obj_val < 1e-8 or (0 <= rel_error < self.tol):
                break

            term_iter = iter
            obj_val = new_obj_val

        if self.verbose:
            print(
                f"Terminated at iteration: {term_iter}, for lambda_L: {lambda_L}, with obj_val: {new_obj_val}"
            )

        self.completed_matrix_ = self.compute_matrix(L, u, v)
        self.component_matrix = {"L": L, "u": u, "v": v}
        self.singular_values = sing
        return self

    # Simple SVT no FE (self contained for explainability)
    def simple_fit(self, M, mask, lambda_L):
        """
        Fit the matrix completion model.

        Parameters:
          M: 2D numpy array of observed outcomes. For missing entries, you may set Y to 0 (or any placeholder)
          mask: binary 2D numpy array of the same shape as Y where 1 indicates an observed entry and 0 a missing one.

        Returns:
          self, with the completed matrix stored in self.completed_matrix_
        """
        # Initialize the estimate L.

        L = np.zeros_like(mask * M)
        shrink_treshhold = lambda_L * np.sum(mask) / 2
        if self.verbose:
            print(f"shrink_treshhold: {shrink_treshhold}")
        for it in range(self.max_iter):
            L_old = L.copy()
            shrink_A_input = mask * M + (1 - mask) * L_old
            # Apply shrinkage per Equation 4.5 in the paper
            L, s_thresholded = self.shrink_lambda(shrink_A_input, shrink_treshhold)
            # Check convergence using the relative change (Nuclear norm).
            norm_old = np.linalg.norm(L_old, "fro") + 1e-8  # avoid div-by-zero
            diff = np.linalg.norm(L - L_old, "fro") / norm_old
            if self.verbose:
                print(f"Iteration {it + 1:3d}, relative change = {diff:.6f}")
            if diff < self.tol:
                if self.verbose:
                    print("Convergence achieved.")
                break
        self.completed_matrix_ = L
        self.singular_values = s_thresholded
        return self
    
    def score(self, M, mask):
        """
        Compute the mean squared error between the observed and imputed entries.

        Parameters:
          M: 2D numpy array of observed outcomes
          mask: binary 2D numpy array of the same shape as Y where 1 indicates an observed entry and 0 a missing one.

        Returns:
          float: mean squared error between observed and imputed entries
        """
        if self.completed_matrix_ is None:
            raise ValueError("Model not yet fit.")

        return np.mean((self.completed_matrix_ - M) ** 2 * mask)

    def fit(self, M, mask, unit_intercept=False, time_intercept=False, cv_split_type='random'):
        """
        Fit the matrix completion model.

        Parameters:
          M: 2D numpy array of observed outcomes. For missing entries, you may set Y to 0 (or any placeholder)
          mask: binary 2D numpy array of the same shape as Y where 1 indicates an observed entry and 0 a missing one.
          unit_intercept: boolean, if True, include unit fixed effects
          time_intercept: boolean, if True, include time fixed effects

        Returns:
          self, with the completed matrix stored in self.completed_matrix_
        """

        init_uv = self.initialize_uv(M, mask, to_estimate_u=unit_intercept, to_estimate_v=time_intercept)
        lambda_L_max = init_uv["lambda_L_max"]

        # Create a list of lambda values starting at 0 up to lambda_L_max on a log scale
        lambda_values = np.concatenate(([0], np.logspace(-5, np.log10(lambda_L_max), num=10)))

        # cv_ratio is share of mask that is observed
        cv_ratio_obs = np.sum(mask) / mask.size

        # Initialize cross-validator and store best results
        cv = PanelCrossValidator(n_splits=5, cv_ratio = cv_ratio_obs)
        best_lambda = None
        best_score = float('inf')

        if self.verbose:
            print(f"Max lambda: {lambda_L_max}")
            print(f"CV ratio in observation: {cv_ratio_obs}")

        # Get cross validation splits
        cv_splits = cv.create_train_test_masks(M, split_type=cv_split_type)
        # NOTE: Paper uses random splits for lambda selection

        # Loop over lambda values to find best one using cross validation
        # NOTE: Note taking advantage of sequential nature of lambda values as suggested in the paper
        for lambda_L in lambda_values:
            if self.verbose:
                print(f"Trying lambda: {lambda_L}")
            fold_scores = []
            
            # For each fold
            for train_mask, test_mask in cv_splits:
                # Fit on training data
                self.NNM_fit(M, train_mask, lambda_L, 
                             to_estimate_u=unit_intercept,
                             to_estimate_v=time_intercept)
                
                # Score on test data
                fold_score = self.score(M, test_mask)
                fold_scores.append(fold_score)
            
            # Calculate mean score across folds    
            mean_score = np.mean(fold_scores)
            
            # Update best if this lambda gives better score
            if mean_score < best_score:
                best_score = mean_score
                best_lambda = lambda_L

        if self.verbose:
            print(f"Lambda list: {lambda_values}")
            print(f"Best lambda: {best_lambda}, Best score: {best_score}")

        # Final fit using best lambda
        return self.NNM_fit(M, mask, best_lambda,
                     to_estimate_u=unit_intercept,
                     to_estimate_v=time_intercept)

        # return self.simple_fit(M, mask, lambda_L) - KEEP FOR RECORD OF SIMPLE IMPLEMENTATION if not intercepts


