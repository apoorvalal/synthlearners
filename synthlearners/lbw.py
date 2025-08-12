import numpy as np
import pandas as pd

import cvxpy as cp
import adelie as ad
from typing import Dict, Tuple
from dataclasses import dataclass
from typing import Optional, Union, Literal
import matplotlib.pyplot as plt


def adelie_synth(
    X: np.ndarray,
    y: np.ndarray,
    l1_ratio: float = 0.0,
    intercept: bool = True,
    noisy: bool = False,
):
    """Linear Balancing Weights for Panel Data using Adelie's fast regularized regression solver.

    Args:
        X (np.ndarray): Design matrix: T X N0 where T is the number of time periods and N0 is the number of control units.
        y (np.ndarray): Target vector: T X 1 or T X N1 where N1 is the number of treated units. If N1 > 1, the function will return a matrix of weights for each treated unit, which is the 'granular' synth
        l1_ratio (float, optional): Ratio of L1 vs L2 in elastic net. Defaults to 0.0.
        intercept (bool, optional): Intercept in Balancing weights regression. Defaults to True.
        noisy (bool, optional): Print progress-bar for cross-validation. Defaults to False.
    """
    if X.flags["F_CONTIGUOUS"] is False:
        X = np.asfortranarray(X)
    if y.flags["F_CONTIGUOUS"] is False:
        y = np.asfortranarray(y)

    ff = ad.glm.multigaussian if y.ndim == 2 else ad.glm.gaussian
    # cross validate for penalty term
    state = ad.cv_grpnet(
        X=X,
        glm=ff(y=y, dtype=np.float64),
        intercept=intercept,
        alpha=l1_ratio,
        progress_bar=noisy,
    )
    # fit again for cv erro minimizing lambda
    state = state.fit(
        X=X,
        glm=ff(y=y, dtype=np.float64),
        intercept=intercept,
        alpha=l1_ratio,
        progress_bar=noisy,
    )
    # extract optima

    # Store metadata
    coef_ = state.betas[-1]
    intercept_ = np.array([state.intercepts[-1]])
    lambda_ = np.array([state.lmdas[-1]])
    return {
        "fitter": lambda X: ad.diagnostic.predict(
            X=X, betas=coef_, intercepts=intercept_
        ).squeeze(),
        "weights": coef_.toarray(),
        "intercept": intercept_,
        "lambda": lambda_,
    }


def panel_matrices(
    df: pd.DataFrame, unit_id: str, time_id: str, treat: str, outcome: str
) -> Dict:
    """
    Reshape panel data from long to wide format.

    Python translation of R's panelMatrices function.
    """
    # Pivot to wide format for treatment matrix
    W = df.pivot(index=unit_id, columns=time_id, values=treat).fillna(0).values

    # Pivot to wide format for outcome matrix
    Y = df.pivot(index=unit_id, columns=time_id, values=outcome).values

    # Move treated units to bottom of matrices (like R code)
    treat_ids = np.where(W.sum(axis=1) > 1)[0]
    N0 = W.shape[0] - len(treat_ids)
    T0 = np.where(W.sum(axis=0) > 0)[0][0]  # First treatment period - 1

    # Reorder: controls first, then treated
    control_ids = np.setdiff1d(range(W.shape[0]), treat_ids)
    unit_order = np.concatenate([control_ids, treat_ids])

    W_reordered = W[unit_order, :]
    Y_reordered = Y[unit_order, :]

    return {"W": W_reordered, "Y": Y_reordered, "N0": N0, "T0": T0}


######################################################################
# PenguinSynth: Regularized Synthetic Control using Adelie
######################################################################


@dataclass
class PenguinResults:
    """Container for PenguinSynth results."""

    weights: np.ndarray
    intercept: Optional[np.ndarray]
    lambda_reg: np.ndarray
    treated_outcome: np.ndarray
    synthetic_outcome: np.ndarray
    treatment_effect: np.ndarray
    att: float
    pre_treatment_rmse: float
    method: str
    l1_ratio: float
    fitter: callable

    def plot(
        self,
        mode: Literal["raw", "effect"] = "raw",
        figsize: tuple = (10, 6),
        ax: Optional[plt.Axes] = None,
        T_pre: Optional[int] = None,
    ) -> plt.Axes:
        """Plot synthetic control results."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if mode == "raw":
            time_idx = np.arange(len(self.treated_outcome))
            ax.plot(
                time_idx,
                self.treated_outcome,
                label="Treated",
                color="blue",
                linewidth=2,
            )
            ax.plot(
                time_idx,
                self.synthetic_outcome,
                label="Synthetic",
                color="red",
                linewidth=2,
                linestyle="--",
            )
            if T_pre is not None:
                ax.axvline(
                    T_pre, color="black", linestyle="--", alpha=0.7, label="Treatment"
                )
            ax.set_title(f"Regularized Synthetic Control (λ₁={self.l1_ratio:.2f})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Outcome")
            ax.legend()

        elif mode == "effect":
            time_idx = np.arange(len(self.treatment_effect))
            if T_pre is not None:
                time_idx = time_idx - T_pre

            ax.plot(
                time_idx,
                self.treatment_effect,
                color="blue",
                linewidth=2,
                label="Treatment Effect",
            )
            ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
            if T_pre is not None:
                ax.axvline(
                    0, color="black", linestyle="--", alpha=0.7, label="Treatment"
                )
            ax.set_title(f"Treatment Effect (ATT: {self.att:.3f})")
            ax.set_xlabel("Time Relative to Treatment" if T_pre else "Time")
            ax.set_ylabel("Effect Size")
            ax.legend()

        return ax


class PenguinSynth:
    """Regularized Synthetic Control using Adelie's fast solvers.

    This class implements L1/L2 regularized synthetic control methods that relax
    the traditional simplex constraints, allowing for more flexible and often
    more feasible solutions.
    """

    def __init__(
        self,
        l1_ratio: float = 0.0,
        method: Literal["synth", "sdid", "did"] = "synth",
        intercept: bool = True,
        noisy: bool = False,
    ):
        """Initialize PenguinSynth estimator.

        Args:
            l1_ratio: Elastic net mixing parameter (0=Ridge, 1=Lasso, 0.5=ElasticNet)
            method: Method type - 'synth' for synthetic control, 'sdid' for synthetic DID
            intercept: Whether to include intercept in the regression
            noisy: Whether to show progress bars during cross-validation
        """
        self.l1_ratio = l1_ratio
        self.method = method
        self.intercept = intercept
        self.noisy = noisy

    def fit(
        self,
        df: pd.DataFrame,
        unit_id: str,
        time_id: str,
        treat: str,
        outcome: str,
    ) -> PenguinResults:
        """Fit regularized synthetic control model.

        Args:
            df: Panel data in long format
            unit_id: Column name for unit identifier
            time_id: Column name for time identifier
            treat: Column name for treatment indicator
            outcome: Column name for outcome variable

        Returns:
            PenguinResults object containing weights, outcomes, and treatment effects
        """
        # Reshape data using existing panel_matrices function
        matrices = panel_matrices(df, unit_id, time_id, treat, outcome)
        Y = matrices["Y"]
        W = matrices["W"]
        N0 = matrices["N0"]
        T0 = matrices["T0"]

        if self.method == "synth":
            return self._fit_synthetic_control(Y, N0, T0)
        elif self.method == "sdid":
            return self._fit_synthetic_did(Y, N0, T0)
        elif self.method == "did":
            return self._fit_diff_in_diff(Y, N0, T0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_synthetic_control(self, Y: np.ndarray, N0: int, T0: int) -> PenguinResults:
        """Fit regularized synthetic control."""
        # Prepare data: control units (pre-treatment) as features, treated unit (pre-treatment) as target
        Y_control_pre = Y[:N0, :T0]  # N0 x T0
        Y_treated_pre = Y[N0:, :T0].mean(axis=0)  # T0 x 1 (average if multiple treated)

        # Use adelie_synth for the core optimization
        result = adelie_synth(
            X=Y_control_pre.T,  # T0 x N0 (adelie expects T x N format)
            y=Y_treated_pre,  # T0 x 1
            l1_ratio=self.l1_ratio,
            intercept=self.intercept,
            noisy=self.noisy,
        )

        # Extract results
        weights = result["weights"].flatten()
        intercept = result["intercept"] if self.intercept else None
        lambda_reg = result["lambda"]
        fitter = result["fitter"]

        # Generate synthetic outcomes for full time series
        if self.intercept and intercept is not None:
            synthetic_outcome = Y[:N0, :].T @ weights + intercept
        else:
            synthetic_outcome = Y[:N0, :].T @ weights

        # Get treated outcome (average if multiple treated units)
        treated_outcome = Y[N0:, :].mean(axis=0)

        # Calculate treatment effects
        treatment_effect = treated_outcome - synthetic_outcome
        att = treatment_effect[T0:].mean()  # Average post-treatment effect

        # Pre-treatment fit quality
        pre_treatment_rmse = np.sqrt(np.mean(treatment_effect[:T0] ** 2))

        return PenguinResults(
            weights=weights,
            intercept=intercept,
            lambda_reg=lambda_reg,
            treated_outcome=treated_outcome,
            synthetic_outcome=synthetic_outcome,
            treatment_effect=treatment_effect,
            att=att,
            pre_treatment_rmse=pre_treatment_rmse,
            method=self.method,
            l1_ratio=self.l1_ratio,
            fitter=fitter,
        )

    def _fit_synthetic_did(self, Y: np.ndarray, N0: int, T0: int) -> PenguinResults:
        """Fit regularized synthetic difference-in-differences.

        Implements true SDID with two separate regularized regressions:
        1. Vertical (unit weights): regress treated pre-treatment on control pre-treatment
        2. Horizontal (time weights): regress control post-treatment on control pre-treatment
        """
        # Split data
        Y_control_pre = Y[:N0, :T0]  # N0 x T0 - control units, pre-treatment
        Y_control_post = Y[:N0, T0:]  # N0 x T_post - control units, post-treatment
        Y_treated_pre = Y[N0:, :T0]  # N1 x T0 - treated units, pre-treatment

        N1 = Y.shape[0] - N0
        T_post = Y.shape[1] - T0

        # REGRESSION 1: VERTICAL - Unit weights (ω)
        # Regress treated unit pre-treatment time series on control units pre-treatment
        Y_treated_pre_mean = Y_treated_pre.mean(
            axis=0
        )  # T0 x 1 (average if multiple treated)

        unit_result = adelie_synth(
            X=Y_control_pre.T,  # T0 x N0 (time x control_units)
            y=Y_treated_pre_mean,  # T0 x 1 (treated pre-treatment trajectory)
            l1_ratio=self.l1_ratio,
            intercept=self.intercept,
            noisy=self.noisy,
        )
        unit_weights = unit_result["weights"].flatten()  # N0 x 1

        # REGRESSION 2: HORIZONTAL - Time weights (λ)
        # Regress each control unit's post-treatment average on its pre-treatment trajectory
        Y_control_post_unit_means = Y_control_post.mean(
            axis=1
        )  # N0 x 1 (each control's post-treatment mean)

        time_result = adelie_synth(
            X=Y_control_pre,  # N0 x T0 (each row is a control unit's pre-treatment trajectory)
            y=Y_control_post_unit_means,  # N0 x 1 (each control's post-treatment average)
            l1_ratio=self.l1_ratio,
            intercept=self.intercept,
            noisy=self.noisy,
        )
        time_weights = time_result["weights"].flatten()  # T0 x 1

        # Construct SDID weight vectors following the matrix formulation
        # Unit weights: [-ω, 1/N1, ..., 1/N1] (length N0 + N1)
        unit_weight_vec = np.concatenate([-unit_weights, np.ones(N1) / N1])

        # Time weights: [-λ, 1/T_post, ..., 1/T_post] (length T0 + T_post)
        time_weight_vec = np.concatenate([-time_weights, np.ones(T_post) / T_post])

        # Compute SDID estimate: τ̂ = ω_unit^T Y λ_time
        sdid_estimate = unit_weight_vec.T @ Y @ time_weight_vec

        # For visualization purposes, create synthetic outcomes using unit weights
        # This shows the synthetic control trajectory (useful for plotting)
        treated_outcome = Y[N0:, :].mean(axis=0)  # Average treated trajectory

        if self.intercept and unit_result["intercept"] is not None:
            synthetic_outcome = Y[:N0, :].T @ unit_weights + unit_result["intercept"]
        else:
            synthetic_outcome = Y[:N0, :].T @ unit_weights

        # Treatment effect trajectory (for plotting)
        treatment_effect = treated_outcome - synthetic_outcome

        # Pre-treatment fit quality (based on unit weights regression)
        pre_treatment_rmse = np.sqrt(np.mean(treatment_effect[:T0] ** 2))

        return PenguinResults(
            weights=unit_weights,
            intercept=unit_result["intercept"] if self.intercept else None,
            lambda_reg=unit_result[
                "lambda"
            ],  # Could also include time_result["lambda"]
            treated_outcome=treated_outcome,
            synthetic_outcome=synthetic_outcome,
            treatment_effect=treatment_effect,
            att=float(sdid_estimate),  # This is the key SDID estimate
            pre_treatment_rmse=pre_treatment_rmse,
            method=self.method,
            l1_ratio=self.l1_ratio,
            fitter=unit_result["fitter"],  # Could create combined fitter
        )

    def _fit_diff_in_diff(self, Y: np.ndarray, N0: int, T0: int) -> PenguinResults:
        """Fit difference-in-differences (uniform weights)."""
        # Standard DiD with uniform weights
        N, T = Y.shape
        N1, T1 = N - N0, T - T0

        # Uniform weights
        unit_weights = np.ones(N0) / N0

        # Calculate DiD estimate
        did_estimate = diff_in_diff(Y, N0, T0)

        # Create synthetic outcome for visualization
        treated_outcome = Y[N0:, :].mean(axis=0)
        synthetic_outcome = Y[:N0, :].mean(axis=0)  # Simple average of controls

        treatment_effect = treated_outcome - synthetic_outcome
        pre_treatment_rmse = np.sqrt(np.mean(treatment_effect[:T0] ** 2))

        return PenguinResults(
            weights=unit_weights,
            intercept=None,
            lambda_reg=np.array([0.0]),
            treated_outcome=treated_outcome,
            synthetic_outcome=synthetic_outcome,
            treatment_effect=treatment_effect,
            att=float(did_estimate),
            pre_treatment_rmse=pre_treatment_rmse,
            method=self.method,
            l1_ratio=0.0,  # No regularization for DiD
            fitter=lambda X: X @ unit_weights,  # Simple linear combination
        )


def diff_in_diff(Y: np.ndarray, N0: int, T0: int) -> float:
    """
    Difference-in-differences (dID function from R).
    """
    N, T = Y.shape
    N1, T1 = N - N0, T - T0

    # Simple DiD: uniform weights for both units and time
    unit_weight_vec = np.concatenate([-np.ones(N0) / N0, np.ones(N1) / N1])
    time_weight_vec = np.concatenate([-np.ones(T0) / T0, np.ones(T1) / T1])

    estimate = unit_weight_vec.T @ Y @ time_weight_vec
    return float(estimate)
