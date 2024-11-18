from dataclasses import dataclass
from typing import Optional, Union, Tuple, Literal
from enum import Enum

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .utils import tqdm_joblib
from .solvers import (
    _solve_lp_norm,
    _solve_linear,
    _solve_simplex,
    _solve_matching,
    _choose_lambda,
)


######################################################################


class SynthMethod(Enum):
    """Enumeration of available synthetic control methods."""

    LP_NORM = "lp_norm"
    LINEAR = "linear"
    SIMPLEX = "simplex"
    MATCHING = "matching"


@dataclass
class SynthResults:
    """Container for synthetic control results."""

    unit_weights: np.ndarray
    treated_outcome: np.ndarray
    synthetic_outcome: np.ndarray
    pre_treatment_rmse: float
    post_treatment_effect: float
    method: "SynthMethod"
    p: Optional[float] = None
    reg_param: Optional[float] = None
    jackknife_effects: Optional[np.ndarray] = None
    permutation_p_value: Optional[float] = None

    def treatment_effect(self) -> np.ndarray:
        """Calculate treatment effect."""
        return self.treated_outcome - self.synthetic_outcome

    def att(self) -> float:
        """Calculate average treatment effect on the treated."""
        return self.post_treatment_effect

    def jackknife_variance(self) -> Optional[np.ndarray]:
        """Calculate jackknife variance of treatment effects."""
        if self.jackknife_effects is None:
            return None
        return np.var(self.jackknife_effects, axis=0)

    def confidence_intervals(
        self, alpha: float = 0.05
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Calculate confidence intervals using jackknife variance."""
        if self.jackknife_effects is None:
            return None
        effects = self.treatment_effect()
        std_err = np.sqrt(self.jackknife_variance())
        z_score = norm.ppf(1 - alpha / 2)
        return effects - z_score * std_err, effects + z_score * std_err


class Synth:
    def __init__(
        self,
        method: Union[str, SynthMethod] = "simplex",
        p: float = 1.0,
        intercept: bool = False,
        weight_type: str = "unit",
        max_iterations: int = 10000,
        tolerance: float = 1e-8,
        n_jobs: int = 8,
        granular_weights: bool = False,
        reg_param: Optional[float] = None,
        lam_grid: Optional[np.ndarray] = None,
        k_nn: int = 5,
    ):
        """Initialize synthetic control estimator."""
        self.method = SynthMethod(method) if isinstance(method, str) else method
        self.p = p
        self.intercept = intercept
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weight_type = weight_type
        self.n_jobs = n_jobs
        self.granular_weights = granular_weights
        self.reg_param = reg_param if self.method == SynthMethod.LP_NORM else None
        self.lam_grid = np.logspace(-4, 2, 20) if lam_grid is None else lam_grid
        self.k_nn = k_nn

        # Internal state
        self.unit_weights = None
        self.time_weights = None

    ######################################################################

    #  ██    ██  ▄▄█████▄   ▄████▄    ██▄████
    #  ██    ██  ██▄▄▄▄ ▀  ██▄▄▄▄██   ██▀
    #  ██    ██   ▀▀▀▀██▄  ██▀▀▀▀▀▀   ██
    #  ██▄▄▄███  █▄▄▄▄▄██  ▀██▄▄▄▄█   ██
    #   ▀▀▀▀ ▀▀   ▀▀▀▀▀▀     ▀▀▀▀▀    ▀▀

    ######################################################################

    def fit(
        self,
        Y: np.ndarray,
        treated_units: Union[int, np.ndarray],
        T_pre: int,
        T_post: Optional[int] = None,
        compute_jackknife: bool = False,
        compute_permutation: bool = False,
        **kwargs,
    ) -> SynthResults:
        """Fit synthetic control model."""
        # Process inputs
        treated_units = (
            np.array([treated_units])
            if isinstance(treated_units, int)
            else treated_units
        )
        T_post = Y.shape[1] - T_pre if T_post is None else T_post
        self.T_pre = T_pre

        # Split data
        control_units = np.setdiff1d(range(Y.shape[0]), treated_units)
        Y_control = Y[control_units, :]
        Y_control2 = (
            np.r_[Y_control, np.ones((1, Y_control.shape[1]))]
            if self.intercept
            else Y_control
        )

        if self.weight_type != "unit":
            raise NotImplementedError("Only 'unit' weights are currently supported.")

        Y_ctrl_pre = Y_control2[:, :T_pre]

        if self.granular_weights:
            # Handle individual unit matching
            individual_weights = []
            individual_synthetic = []

            for treated_idx in treated_units:
                Y_treat_pre = Y[treated_idx, :T_pre]
                weights = self._get_weights(Y_ctrl_pre, Y_treat_pre)
                individual_weights.append(weights)
                synthetic = np.dot(Y_control2.T, weights)
                individual_synthetic.append(synthetic)

            self.unit_weights = np.mean(individual_weights, axis=0)
            synthetic_outcome = np.mean(individual_synthetic, axis=0)
            Y_treated = Y[treated_units].mean(axis=0)

        else:
            # Average treated units first
            Y_treated = Y[treated_units].reshape(-1, Y.shape[1]).mean(axis=0)
            Y_treat_pre = Y_treated[:T_pre]

            self.unit_weights = self._get_weights(Y_ctrl_pre, Y_treat_pre)
            synthetic_outcome = np.dot(Y_control2.T, self.unit_weights)

        # Calculate fit and effects
        pre_rmse = np.sqrt(
            np.mean((Y_treated[:T_pre] - synthetic_outcome[:T_pre]) ** 2)
        )

        # Compute inference if requested
        jackknife_effects = (
            self._compute_jackknife_effects(Y, treated_units, T_pre)
            if compute_jackknife
            else None
        )
        permutation_p_value = (
            self._compute_permutation_p_value(Y, treated_units, T_pre)
            if compute_permutation
            else None
        )

        return SynthResults(
            unit_weights=self.unit_weights,
            treated_outcome=Y_treated,
            synthetic_outcome=synthetic_outcome,
            post_treatment_effect=np.mean(
                Y_treated[T_pre : T_pre + T_post]
                - synthetic_outcome[T_pre : T_pre + T_post]
            ),
            pre_treatment_rmse=pre_rmse,
            method=self.method,
            p=self.p if self.method == SynthMethod.LP_NORM else None,
            reg_param=self.reg_param if self.method == SynthMethod.LP_NORM else None,
            jackknife_effects=jackknife_effects,
            permutation_p_value=permutation_p_value,
        )

    def plot(
        self,
        results: SynthResults,
        Y: np.ndarray,
        treated_units: np.ndarray,
        T_pre: int,
        mode: Literal["raw", "effect"] = "raw",
        show_ci: bool = True,
        alpha: float = 0.05,
        figsize: Tuple[int, int] = (10, 6),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot synthetic control results."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if mode == "raw":
            # Plot raw trajectories
            ctrl_units = np.setdiff1d(range(Y.shape[0]), treated_units)
            ax.plot(Y[ctrl_units].T, color="gray", alpha=0.2, linestyle="--")
            ax.plot(results.treated_outcome, label="Treated", color="blue", linewidth=2)
            ax.plot(
                results.synthetic_outcome,
                label="Synthetic Control",
                color="red",
                linewidth=2,
                linestyle="--",
            )
            ax.axvline(T_pre, color="black", linestyle="--", label="Treatment")
            ax.set_title("Raw Trajectories")
            ax.legend()

        elif mode == "effect":
            # Plot treatment effects
            effects = results.treatment_effect()
            t = np.arange(len(effects)) - T_pre

            ax.plot(t, effects, color="blue", linewidth=2, label="Treatment Effect")
            ax.axvline(0, color="black", linestyle="--", label="Treatment")
            ax.axhline(0, color="gray", linestyle=":")

            if show_ci and results.jackknife_effects is not None:
                ci = results.confidence_intervals(alpha)
                if ci is not None:
                    lower, upper = ci
                    ax.fill_between(
                        t,
                        lower,
                        upper,
                        alpha=0.2,
                        color="blue",
                        label=f"{int((1-alpha)*100)}% CI",
                    )

            title = (
                f"Treatment Effect \n ATT: {results.att():.2f}"
                f" (p={results.permutation_p_value:.3f})"
                if results.permutation_p_value is not None
                else f"Treatment Effect \n ATT: {results.att():.2f}"
            )

            ax.set_title(title)
            ax.set_xlabel("Time Relative to Treatment")
            ax.set_ylabel("Effect Size")
            ax.legend()

        return ax

    ######################################################################

    #     ██                                                                 ▄▄▄▄
    #     ▀▀                 ██                                              ▀▀██
    #   ████     ██▄████▄  ███████    ▄████▄    ██▄████  ██▄████▄   ▄█████▄    ██
    #     ██     ██▀   ██    ██      ██▄▄▄▄██   ██▀      ██▀   ██   ▀ ▄▄▄██    ██
    #     ██     ██    ██    ██      ██▀▀▀▀▀▀   ██       ██    ██  ▄██▀▀▀██    ██
    #  ▄▄▄██▄▄▄  ██    ██    ██▄▄▄   ▀██▄▄▄▄█   ██       ██    ██  ██▄▄▄███    ██▄▄▄
    #  ▀▀▀▀▀▀▀▀  ▀▀    ▀▀     ▀▀▀▀     ▀▀▀▀▀    ▀▀       ▀▀    ▀▀   ▀▀▀▀ ▀▀     ▀▀▀▀

    ######################################################################
    def _get_weights(
        self, Y_control: np.ndarray, Y_treated: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Compute weights for synthetic control.

        Centralizes weight computation for all methods and handles regularization.
        """
        if self.method == SynthMethod.LP_NORM:
            # Handle regularization parameter if needed
            if self.reg_param is None:
                print(
                    "Choosing regularization parameter using sequential cross-validation"
                )
                self.reg_param = _choose_lambda(
                    Y_control=Y_control,
                    Y_treated=Y_treated,
                    p=self.p,
                    lam_grid=self.lam_grid,
                )

            return _solve_lp_norm(
                Y_control,
                Y_treated,
                self.p,
                self.max_iterations,
                self.tolerance,
                reg_param=self.reg_param,
            )

        elif self.method == SynthMethod.LINEAR:
            return _solve_linear(
                Y_control,
                Y_treated,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )

        elif self.method == SynthMethod.MATCHING:
            return _solve_matching(
                Y_control,
                Y_treated,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                k=self.k_nn,
            )

        elif self.method == SynthMethod.SIMPLEX:
            return _solve_simplex(
                Y_control,
                Y_treated,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )

    def _jackknife_single_run(
        self, Y: np.ndarray, treated_units: np.ndarray, T_pre: int, leave_out_idx: int
    ) -> np.ndarray:
        """Run single jackknife iteration leaving out one unit."""
        # Create reduced dataset
        Y_reduced = np.delete(Y, leave_out_idx, axis=0)
        adjusted_treated = np.array(
            [
                i if i < leave_out_idx else i - 1
                for i in treated_units
                if i != leave_out_idx
            ]
        )

        # Split data
        control_units = np.setdiff1d(range(Y_reduced.shape[0]), adjusted_treated)
        Y_control = Y_reduced[control_units, :]
        Y_control2 = (
            np.r_[Y_control, np.ones((1, Y_control.shape[1]))]
            if self.intercept
            else Y_control
        )
        Y_ctrl_pre = Y_control2[:, :T_pre]

        # Get treated outcomes
        Y_treated = (
            Y_reduced[adjusted_treated].reshape(-1, Y_reduced.shape[1]).mean(axis=0)
        )
        Y_treat_pre = Y_treated[:T_pre]

        # Compute weights and synthetic outcome
        weights = self._get_weights(Y_ctrl_pre, Y_treat_pre)
        synthetic = np.dot(Y_control2.T, weights)

        return Y_treated - synthetic

    def _compute_jackknife_effects(
        self,
        Y: np.ndarray,
        treated_units: np.ndarray,
        T_pre: int,
    ) -> Optional[np.ndarray]:
        """Compute jackknife treatment effects in parallel."""
        if len(treated_units) <= 1:
            return None

        n = Y.shape[0]
        with tqdm_joblib(tqdm(total=n, desc="Computing jackknife estimates")):
            effects = Parallel(n_jobs=self.n_jobs)(
                delayed(self._jackknife_single_run)(Y, treated_units, T_pre, i)
                for i in range(n)
            )
        return np.array(effects)

    def _compute_placebo_effect(
        self, Y: np.ndarray, placebo_unit: int, original_treated: np.ndarray, T_pre: int
    ) -> float:
        """Compute effect for a single placebo treatment."""
        # Remove original treated units and prepare data
        Y_reduced = np.delete(Y, original_treated, axis=0)
        adjusted_placebo = placebo_unit - np.sum(original_treated < placebo_unit)

        # Split data
        control_units = np.setdiff1d(range(Y_reduced.shape[0]), [adjusted_placebo])
        Y_control = Y_reduced[control_units, :]
        Y_control2 = (
            np.r_[Y_control, np.ones((1, Y_control.shape[1]))]
            if self.intercept
            else Y_control
        )
        Y_ctrl_pre = Y_control2[:, :T_pre]

        # Get treated outcomes
        Y_treated = Y_reduced[adjusted_placebo].reshape(-1, Y_reduced.shape[1])
        Y_treat_pre = Y_treated[:, :T_pre].squeeze()

        # Compute weights and effect
        weights = self._get_weights(Y_ctrl_pre, Y_treat_pre)
        synthetic = np.dot(Y_control2.T, weights)

        return np.mean(Y_treated[:, T_pre:] - synthetic[T_pre:])

    def _compute_permutation_p_value(
        self, Y: np.ndarray, treated_units: np.ndarray, T_pre: int
    ) -> float:
        """Compute permutation test p-value."""
        # Get true effect
        true_results = self.fit(
            Y, treated_units, T_pre, compute_jackknife=False, compute_permutation=False
        )
        true_effect = np.abs(true_results.post_treatment_effect)

        # Warning for small sample
        n = Y.shape[0]
        if (n - 1) <= 20:
            print(
                f"You have {n} units, so the lowest possible p-value is {1/(n-1)}, which is smaller than traditional α of 0.05 \nPermutation test may be unreliable"
            )

        # Get control units and compute placebo effects
        control_units = np.setdiff1d(range(Y.shape[0]), treated_units)
        with tqdm_joblib(
            tqdm(total=len(control_units), desc="Computing permutation test")
        ):
            placebo_effects = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_placebo_effect)(
                    Y, control_unit, treated_units, T_pre
                )
                for control_unit in control_units
            )

        return np.mean(np.abs(placebo_effects) >= true_effect)
