from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .lbw import adelie_synth

try:
    from scipy.optimize import Bounds, LinearConstraint, minimize

    HAS_DYNBAL_DEPENDENCIES = True
except ImportError:
    HAS_DYNBAL_DEPENDENCIES = False


def _raise_missing_dynbal_dependency() -> None:
    raise ImportError(
        "DynamicBalance requires scipy. Install the package with full dependencies: "
        "uv add 'synthlearners[full]'"
    )


@dataclass
class DynamicHistoryEstimate:
    """Estimated mean potential outcome for one treatment history."""

    history: tuple
    mean_outcome: float
    weights_by_period: list[np.ndarray]
    effective_sample_sizes: np.ndarray
    eligible_counts: np.ndarray
    balance_gaps: list[np.ndarray]


@dataclass
class DynamicBalanceResults:
    """Results for a dynamic treatment-history contrast."""

    target: DynamicHistoryEstimate
    reference: DynamicHistoryEstimate
    contrast: float
    periods: np.ndarray
    covariates: tuple[str, ...]
    final_period: Any

    def summary_frame(self) -> pd.DataFrame:
        """Return a one-row summary of the fitted contrast."""
        return pd.DataFrame(
            {
                "target_history": [self.target.history],
                "reference_history": [self.reference.history],
                "target_mean": [self.target.mean_outcome],
                "reference_mean": [self.reference.mean_outcome],
                "contrast": [self.contrast],
            }
        )


class DynamicBalance:
    """Dynamic balancing weights for treatment-sequence contrasts.

    This estimator implements a scoped Python version of dynamic covariate
    balancing for exact treatment histories in long-format panel data.

    The implementation follows the recursive structure in Viviano and Bradic
    (2021, 2026 revision) and the DynBalancing R package:

    - estimate recursive outcome projections for a target history
    - construct balancing weights sequentially across history prefixes
    - contrast two treatment histories at a chosen final period

    The current implementation is intentionally scoped:

    - balanced panels over the selected history window
    - exact-history contrasts
    - a single final-period outcome
    - no pooled regression or clustered inference
    """

    def __init__(
        self,
        l1_ratio: float = 0.0,
        intercept: bool = True,
        noisy: bool = False,
        balance_lb: float = 1e-4,
        balance_ub: float = 2.0,
        balance_grid_size: int = 25,
        max_weight: Optional[float] = None,
        solver_max_iter: int = 1000,
        solver_tol: float = 1e-9,
    ):
        """Initialize a dynamic balancing estimator.

        Parameters
        ----------
        - `l1_ratio`: Elastic-net mixing parameter for recursive outcome
          projections.
        - `intercept`: Whether to include an intercept in the projection model.
        - `noisy`: Whether adelie should report cross-validation progress.
        - `balance_lb`: Lower end of the balancing-tuning grid.
        - `balance_ub`: Upper end of the balancing-tuning grid.
        - `balance_grid_size`: Number of grid points used when searching for a
          feasible balancing tolerance.
        - `max_weight`: Optional upper bound on each balancing weight. If not
          provided, uses the paper-inspired default `log(n) * n**(-2/3)`.
        - `solver_max_iter`: Maximum number of optimizer iterations.
        - `solver_tol`: Numerical tolerance for the balancing optimizer.
        """
        if not HAS_DYNBAL_DEPENDENCIES:
            _raise_missing_dynbal_dependency()
        if balance_lb <= 0 or balance_ub < balance_lb:
            raise ValueError("Expected 0 < balance_lb <= balance_ub.")
        if balance_grid_size < 1:
            raise ValueError("balance_grid_size must be at least 1.")

        self.l1_ratio = l1_ratio
        self.intercept = intercept
        self.noisy = noisy
        self.balance_lb = balance_lb
        self.balance_ub = balance_ub
        self.balance_grid_size = balance_grid_size
        self.max_weight = max_weight
        self.solver_max_iter = solver_max_iter
        self.solver_tol = solver_tol

    def fit(
        self,
        df: pd.DataFrame,
        unit_id: str,
        time_id: str,
        treatment: str,
        outcome: str,
        covariates: Sequence[str],
        target_history: Sequence[float],
        reference_history: Sequence[float],
        final_period: Optional[Any] = None,
    ) -> DynamicBalanceResults:
        """Estimate a contrast between two treatment histories.

        Parameters
        ----------
        - `df`: Panel data in long format.
        - `unit_id`: Column containing unit identifiers.
        - `time_id`: Column containing time identifiers.
        - `treatment`: Column containing the time-varying treatment.
        - `outcome`: Column containing the observed outcome.
        - `covariates`: Time-varying covariates used for recursive projections
          and balancing. Include lagged outcomes or lagged treatments here if
          they should be balanced on.
        - `target_history`: Treatment sequence defining the first potential
          outcome.
        - `reference_history`: Treatment sequence defining the comparison
          potential outcome.
        - `final_period`: Optional final period to anchor the history window.
          Defaults to the latest observed period.

        Returns
        -------
        - `DynamicBalanceResults`: Estimated potential outcomes and their
          contrast for the two histories.
        """
        prepared = _prepare_dynamic_panel(
            df=df,
            unit_id=unit_id,
            time_id=time_id,
            treatment=treatment,
            outcome=outcome,
            covariates=tuple(covariates),
            target_history=target_history,
            reference_history=reference_history,
            final_period=final_period,
        )

        target = self._estimate_history(
            covariate_matrices=prepared["covariate_matrices"],
            treatment_matrix=prepared["treatment_matrix"],
            final_outcome=prepared["final_outcome"],
            history=prepared["target_history"],
        )
        reference = self._estimate_history(
            covariate_matrices=prepared["covariate_matrices"],
            treatment_matrix=prepared["treatment_matrix"],
            final_outcome=prepared["final_outcome"],
            history=prepared["reference_history"],
        )

        return DynamicBalanceResults(
            target=target,
            reference=reference,
            contrast=target.mean_outcome - reference.mean_outcome,
            periods=prepared["periods"],
            covariates=prepared["covariates"],
            final_period=prepared["final_period"],
        )

    def _estimate_history(
        self,
        covariate_matrices: list[np.ndarray],
        treatment_matrix: np.ndarray,
        final_outcome: np.ndarray,
        history: tuple,
    ) -> DynamicHistoryEstimate:
        predictions = self._recursive_predictions(
            covariate_matrices=covariate_matrices,
            treatment_matrix=treatment_matrix,
            final_outcome=final_outcome,
            history=history,
        )
        weights, balance_gaps = self._sequential_weights(
            covariate_matrices=covariate_matrices,
            treatment_matrix=treatment_matrix,
            history=history,
        )

        n_units = treatment_matrix.shape[0]
        uniform_weights = np.full(n_units, 1.0 / n_units)
        correction = (weights[0] - uniform_weights) @ predictions[0]
        for idx in range(1, len(history)):
            correction += (weights[idx] - weights[idx - 1]) @ predictions[idx]

        mean_outcome = float(weights[-1] @ final_outcome - correction)
        effective_sample_sizes = np.array(
            [1.0 / np.sum(weight**2) for weight in weights], dtype=float
        )
        eligible_counts = np.array(
            [
                np.sum(
                    np.all(treatment_matrix[:, : idx + 1] == history[: idx + 1], axis=1)
                )
                for idx in range(len(history))
            ],
            dtype=int,
        )

        return DynamicHistoryEstimate(
            history=history,
            mean_outcome=mean_outcome,
            weights_by_period=weights,
            effective_sample_sizes=effective_sample_sizes,
            eligible_counts=eligible_counts,
            balance_gaps=balance_gaps,
        )

    def _recursive_predictions(
        self,
        covariate_matrices: list[np.ndarray],
        treatment_matrix: np.ndarray,
        final_outcome: np.ndarray,
        history: tuple,
    ) -> list[np.ndarray]:
        pseudo_outcome = final_outcome.astype(float).copy()
        predictions: list[np.ndarray] = [
            np.zeros_like(final_outcome, dtype=float)
        ] * len(history)

        for period_idx in range(len(history) - 1, -1, -1):
            covariates_t = covariate_matrices[period_idx]
            observed_history = treatment_matrix[:, : period_idx + 1]
            train_features = np.column_stack([covariates_t, observed_history])

            target_column = np.full((treatment_matrix.shape[0], 1), history[period_idx])
            predict_features = np.column_stack(
                [covariates_t, treatment_matrix[:, :period_idx], target_column]
            )

            predictions[period_idx] = _fit_projection(
                train_features=train_features,
                predict_features=predict_features,
                outcome=pseudo_outcome,
                l1_ratio=self.l1_ratio,
                intercept=self.intercept,
                noisy=self.noisy,
            )
            pseudo_outcome = predictions[period_idx]

        return predictions

    def _sequential_weights(
        self,
        covariate_matrices: list[np.ndarray],
        treatment_matrix: np.ndarray,
        history: tuple,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        weights: list[np.ndarray] = []
        balance_gaps: list[np.ndarray] = []

        for period_idx, covariates_t in enumerate(covariate_matrices):
            history_prefix = np.array(history[: period_idx + 1])
            eligible = np.all(
                treatment_matrix[:, : period_idx + 1] == history_prefix,
                axis=1,
            )
            if not np.any(eligible):
                raise ValueError(
                    f"No units match the treatment history prefix {tuple(history_prefix)}."
                )

            target_mean = covariates_t.mean(axis=0)
            if period_idx > 0:
                target_mean = weights[-1] @ covariates_t

            weight_t = self._fit_balancing_weights(
                covariates=covariates_t,
                eligible=eligible,
                target_mean=target_mean,
            )
            weights.append(weight_t)
            balance_gaps.append(weight_t @ covariates_t - target_mean)

        return weights, balance_gaps

    def _fit_balancing_weights(
        self,
        covariates: np.ndarray,
        eligible: np.ndarray,
        target_mean: np.ndarray,
    ) -> np.ndarray:
        eligible_indices = np.flatnonzero(eligible)
        eligible_covariates = covariates[eligible]
        n_eligible = eligible_covariates.shape[0]
        upper_bound = self.max_weight
        if upper_bound is None:
            upper_bound = np.log(max(n_eligible, 2)) * (n_eligible ** (-2.0 / 3.0))
        upper_bound = float(min(1.0, max(upper_bound, 1.0 / n_eligible + 1e-10)))

        grid = np.linspace(self.balance_lb, self.balance_ub, self.balance_grid_size)
        last_error: Optional[str] = None
        for constant in grid:
            try:
                solution = _solve_balancing_problem(
                    covariates=eligible_covariates,
                    target_mean=target_mean,
                    constant=constant,
                    max_weight=upper_bound,
                    max_iter=self.solver_max_iter,
                    tol=self.solver_tol,
                )
                full_weights = np.zeros(covariates.shape[0], dtype=float)
                full_weights[eligible_indices] = solution
                return full_weights
            except ValueError as exc:
                last_error = str(exc)

        raise ValueError(
            "Unable to find feasible dynamic balancing weights. "
            f"Last solver error: {last_error}"
        )


def _prepare_dynamic_panel(
    df: pd.DataFrame,
    unit_id: str,
    time_id: str,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
    target_history: Sequence[float],
    reference_history: Sequence[float],
    final_period: Optional[Any],
) -> dict:
    target_history = tuple(target_history)
    reference_history = tuple(reference_history)
    if len(target_history) == 0:
        raise ValueError("target_history must contain at least one period.")
    if len(target_history) != len(reference_history):
        raise ValueError(
            "target_history and reference_history must have the same length."
        )

    needed_columns = {unit_id, time_id, treatment, outcome, *covariates}
    missing_columns = sorted(needed_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working = df.loc[:, list(needed_columns)].copy()
    working = working.sort_values([unit_id, time_id])

    time_values = np.array(sorted(working[time_id].dropna().unique()))
    if len(time_values) < len(target_history):
        raise ValueError(
            "The panel does not contain enough periods for the requested history."
        )
    if final_period is None:
        final_period = time_values[-1]
    if final_period not in set(time_values):
        raise ValueError("final_period must be one of the observed time values.")

    final_index = int(np.where(time_values == final_period)[0][0])
    if final_index + 1 < len(target_history):
        raise ValueError("final_period is too early for the requested history length.")

    periods = time_values[final_index - len(target_history) + 1 : final_index + 1]
    window = working[working[time_id].isin(periods)].copy()

    unit_counts = window.groupby(unit_id)[time_id].nunique()
    complete_units = unit_counts[unit_counts == len(periods)].index
    window = window[window[unit_id].isin(complete_units)].copy()
    if window.empty:
        raise ValueError(
            "No units remain after restricting to a balanced history window."
        )

    panel = window.pivot(index=unit_id, columns=time_id)
    required_matrix_columns = [(treatment, period) for period in periods]
    required_matrix_columns += [(outcome, periods[-1])]
    required_matrix_columns += [
        (covariate, period) for covariate in covariates for period in periods
    ]
    missing_entries = panel.loc[:, required_matrix_columns].isna().any(axis=1)
    panel = panel.loc[~missing_entries]
    if panel.empty:
        raise ValueError("No complete units remain after removing missing values.")

    treatment_matrix = panel[treatment].loc[:, periods].to_numpy(dtype=float)
    final_outcome = panel[outcome].loc[:, periods[-1]].to_numpy(dtype=float)
    covariate_matrices = [
        panel.loc[:, pd.IndexSlice[covariates, period]].to_numpy(dtype=float)
        if len(covariates) > 0
        else np.zeros((panel.shape[0], 0), dtype=float)
        for period in periods
    ]

    return {
        "covariate_matrices": covariate_matrices,
        "treatment_matrix": treatment_matrix,
        "final_outcome": final_outcome,
        "target_history": target_history,
        "reference_history": reference_history,
        "periods": periods,
        "covariates": covariates,
        "final_period": final_period,
    }


def _fit_projection(
    train_features: np.ndarray,
    predict_features: np.ndarray,
    outcome: np.ndarray,
    l1_ratio: float,
    intercept: bool,
    noisy: bool,
) -> np.ndarray:
    finite_rows = np.isfinite(outcome) & np.all(np.isfinite(train_features), axis=1)
    if not np.any(finite_rows):
        raise ValueError("No finite observations available for recursive projection.")

    train_x = train_features[finite_rows]
    train_y = outcome[finite_rows]

    if train_x.shape[1] == 0:
        return np.full(predict_features.shape[0], float(np.mean(train_y)))

    non_constant = np.std(train_x, axis=0) > 1e-12
    if not np.any(non_constant):
        return np.full(predict_features.shape[0], float(np.mean(train_y)))

    train_x = train_x[:, non_constant]
    predict_x = predict_features[:, non_constant]

    fitted = adelie_synth(
        X=train_x,
        y=train_y,
        l1_ratio=l1_ratio,
        intercept=intercept,
        noisy=noisy,
    )
    return np.asarray(fitted["fitter"](predict_x), dtype=float).reshape(-1)


def _solve_balancing_problem(
    covariates: np.ndarray,
    target_mean: np.ndarray,
    constant: float,
    max_weight: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    n_obs = covariates.shape[0]
    if n_obs == 0:
        raise ValueError("Balancing problem has no eligible observations.")

    if covariates.shape[1] == 0:
        return np.full(n_obs, 1.0 / n_obs)

    p = covariates.shape[1]
    tolerance = constant * np.sqrt(np.log(max(p, 2)) / np.sqrt(n_obs))
    x0 = np.full(n_obs, 1.0 / n_obs)

    def objective(weights: np.ndarray) -> float:
        return 0.5 * float(weights @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return weights

    constraints = [
        LinearConstraint(np.ones((1, n_obs)), np.array([1.0]), np.array([1.0])),
        LinearConstraint(
            covariates.T,
            target_mean - tolerance,
            target_mean + tolerance,
        ),
    ]
    bounds = Bounds(np.zeros(n_obs), np.full(n_obs, max_weight))

    result = minimize(
        objective,
        x0=x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": max_iter, "disp": False},
    )
    if not result.success:
        raise ValueError(result.message)

    solution = np.clip(result.x, 0.0, max_weight)
    solution_sum = solution.sum()
    if solution_sum <= 0:
        raise ValueError("Balancing weights sum to zero.")
    return solution / solution_sum
