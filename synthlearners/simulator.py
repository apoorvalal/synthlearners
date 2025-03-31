import numpy as np
import pandas as pd
from scipy.special import expit
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict
from abc import ABC, abstractmethod


def generate_panel_data(
    num_units: int = 100,
    num_periods: int = 50,
    num_treated: int = 5,
    treatment_start: int = 25,
    treatment_effect_type: str = "sharkfin",
    selection_type: str = "prob",
    selection_strength: float = 1.5,
    ar_persistence: float = 0.8,
    unit_variance: float = 2.0,
    time_variance: float = 1.0,
    noise_variance: float = 0.5,
    return_raw: bool = False,
    seed: Optional[int] = None,
) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
    """Generate synthetic panel data with treatment effects.

    Args:
        num_units: Number of units
        num_periods: Number of time periods
        num_treated: Number of treated units
        treatment_start: Period where treatment begins
        treatment_effect_type: Type of treatment effect ('sharkfin', 'constant', 'linear', 'none')
        selection_type: How treated units are selected ('random', 'top', 'threshold' 'prob')
        selection_strength: Strength of selection mechanism
        ar_persistence: AR(1) coefficient for unit-specific trends
        unit_variance: Variance of unit fixed effects
        time_variance: Variance of time fixed effects
        noise_variance: Variance of idiosyncratic noise
        return_raw: If True, return raw outcome matrices Y0 and Y1
        seed: Random seed

    Returns:
        DataFrame with columns: unit, time, outcome, treatment or dictionary with raw data
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate fixed effects
    unit_effects = np.random.normal(0, np.sqrt(unit_variance), num_units)
    time_effects = np.random.normal(0, np.sqrt(time_variance), num_periods)

    # Generate AR(1) process for each unit
    noise = np.zeros((num_units, num_periods))
    eps = np.random.normal(0, np.sqrt(noise_variance), (num_units, num_periods))
    noise[:, 0] = eps[:, 0]
    for t in range(1, num_periods):
        noise[:, t] = ar_persistence * noise[:, t - 1] + eps[:, t]

    # Combine components for control potential outcome
    Y0 = unit_effects[:, np.newaxis] + time_effects[np.newaxis, :] + noise

    # Generate treatment effects
    if treatment_effect_type == "sharkfin":
        # Effect that builds up and then fades
        post_periods = num_periods - treatment_start
        effect_path = np.zeros(post_periods)
        peak_at = int(post_periods * 0.3)  # Peak at 30% of post-treatment

        # Build up phase
        effect_path[:peak_at] = 0.2 * np.log(1 + np.arange(peak_at))
        # Fade out phase
        effect_path[peak_at:] = effect_path[peak_at - 1] * np.exp(
            -0.2 * np.arange(post_periods - peak_at)
        )

    elif treatment_effect_type == "constant":
        effect_path = np.ones(num_periods - treatment_start) * 2.0

    elif treatment_effect_type == "linear":
        effect_path = 0.1 * np.arange(num_periods - treatment_start)

    elif treatment_effect_type == "none":
        effect_path = np.zeros(num_periods - treatment_start)

    # Select treated units
    if selection_type == "random":
        treated_units = np.random.choice(num_units, num_treated, replace=False)
    elif selection_type == "top":
        # Select units with highest average pre-treatment outcomes
        pre_treat_means = Y0[:, :treatment_start].mean(axis=1)
        treated_units = np.argsort(pre_treat_means)[-num_treated:]
    elif selection_type == "threshold":
        # Select units above threshold in pre-treatment period
        pre_treat_means = Y0[:, :treatment_start].mean(axis=1)
        threshold = np.percentile(
            pre_treat_means, 100 - (num_treated / num_units) * 100
        )
        treated_units = np.where(pre_treat_means >= threshold)[0]

    elif selection_type == "prob":
        # Generate selection score based on unit effects
        score = selection_strength * unit_effects

        # Convert to probabilities using logistic function
        probs = expit(score)
        probs = probs / probs.sum()  # Normalize to sum to 1

        # Sample units based on these probabilities
        treated_units = np.random.choice(
            num_units, size=num_treated, replace=False, p=probs
        )

    # Create treatment matrix and apply effects
    W = np.zeros((num_units, num_periods))
    W[treated_units, treatment_start:] = 1

    Y1 = Y0.copy()
    Y1[:, treatment_start:] += W[:, treatment_start:] * effect_path[np.newaxis, :]

    # Convert to long format DataFrame
    unit_idx = np.repeat(np.arange(num_units), num_periods)
    time_idx = np.tile(np.arange(num_periods), num_units)
    treatment = W.flatten()
    outcome = np.where(treatment == 1, Y1.flatten(), Y0.flatten())

    if return_raw:
        Y = np.where(treatment == 1, Y1, Y0)
        return {
            "Y": Y,
            "Y1": Y1,
            "Y0": Y0,
        }

    df = pd.DataFrame(
        {
            "unit": unit_idx,
            "time": time_idx,
            "outcome": outcome,
            "treatment": treatment.astype(int),
        }
    )

    return df


######################################################################


class BaseDGP(ABC):
    """Abstract base class for data generating processes."""

    @abstractmethod
    def generate(self, N: int, T: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate panel data."""
        pass


class FactorDGP(BaseDGP):
    """Factor model DGP with time trends."""

    def __init__(
        self,
        K: int = 4,
        unit_fac_lb: float = -5,
        unit_fac_ub: float = 3,
        time_fac_lb: float = -2,
        time_fac_ub: float = 2,
        sigma: float = 0.1,
        trend_sigma: float = 0.01,
    ):
        self.K = K
        self.unit_fac_lb = unit_fac_lb
        self.unit_fac_ub = unit_fac_ub
        self.time_fac_lb = time_fac_lb
        self.time_fac_ub = time_fac_ub
        self.sigma = sigma
        self.trend_sigma = trend_sigma

    def generate(self, N: int, T: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate panel data from factor model.

        Args:
            N: Number of units
            T: Number of time periods
            **kwargs: Additional arguments (ignored)

        Returns:
            Tuple of (Y, L) where Y is outcomes and L is unit factors
        """
        F = np.random.uniform(self.time_fac_lb, self.time_fac_ub, (T, self.K))
        L = np.random.uniform(self.unit_fac_lb, self.unit_fac_ub, (N, self.K))
        time_trends = np.random.normal(0, self.trend_sigma, (N, 1)) * np.arange(
            T
        ).reshape(1, T)
        epsilon = np.random.normal(0, self.sigma, (N, T))
        Y = np.dot(L, F.T) + time_trends + epsilon
        return Y, L


@dataclass
class SimulationConfig:
    """Configuration for panel data simulation."""

    N: int = 100
    T: int = 50
    T_pre: int = 40
    n_treated: int = 1
    selection_mean: float = 1.0
    dgp: BaseDGP = FactorDGP()
    random_seed: Optional[int] = 42
    treatment_effect: Union[float, np.ndarray] = 0.0


class PanelSimulator:
    """Simulator for synthetic control panel data."""

    def __init__(self, config: SimulationConfig):
        """Initialize simulator with configuration."""
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Validate and process treatment effect
        self._process_treatment_effect()

    def _process_treatment_effect(self):
        """Process and validate treatment effect specification."""
        effect = self.config.treatment_effect
        T_post = self.config.T - self.config.T_pre

        if isinstance(effect, (int, float)):
            # Constant treatment effect
            self.treatment_effect = np.full((self.config.n_treated, T_post), effect)
        elif isinstance(effect, np.ndarray):
            # Heterogeneous treatment effects
            if effect.shape != (self.config.n_treated, T_post):
                raise ValueError(
                    f"Treatment effect array must have shape ({self.config.n_treated}, {T_post}), "
                    f"got {effect.shape}"
                )
            self.treatment_effect = effect
        else:
            raise TypeError(
                f"Treatment effect must be a number or numpy array, got {type(effect)}"
            )

    def assign_treatment(self, L: np.ndarray) -> np.ndarray:
        """Assign treatment based on unit factors.

        Args:
            L: Unit factors matrix

        Returns:
            Array of treated unit indices
        """
        sel_coef = np.random.normal(self.config.selection_mean, 1, L.shape[1])
        pscore = expit(L @ sel_coef**2)
        pscore /= pscore.sum()
        return np.random.choice(
            range(L.shape[0]), self.config.n_treated, replace=False, p=pscore
        )

    def apply_treatment_effect(
        self, Y: np.ndarray, treated_units: np.ndarray
    ) -> np.ndarray:
        """Apply treatment effects to the outcome matrix.

        Args:
            Y: Original outcome matrix
            treated_units: Indices of treated units

        Returns:
            Modified outcome matrix with treatment effects
        """
        Y_modified = Y.copy()
        T_pre = self.config.T_pre

        # Apply treatment effects to each treated unit
        for i, unit_idx in enumerate(treated_units):
            Y_modified[unit_idx, T_pre:] += self.treatment_effect[i]

        return Y_modified

    def simulate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate one simulation of panel data.

        Returns:
            Tuple of (Y, Y_cf, L, treated_units) where:
                Y: Panel data array with treatment effects
                Y_cf: Counterfactual panel data array without treatment
                L: Unit factors array
                treated_units: Array of treated unit indices
        """
        Y_cf, L = self.config.dgp.generate(self.config.N, self.config.T)
        treated_units = self.assign_treatment(L)

        # Apply treatment effects
        Y = self.apply_treatment_effect(Y_cf, treated_units)

        return Y, Y_cf, L, treated_units

    def simulate_batch(
        self, n_sims: int, parallel: bool = False, n_jobs: int = -1
    ) -> list:
        """Generate multiple simulations.

        Args:
            n_sims: Number of simulations to generate
            parallel: Whether to run simulations in parallel
            n_jobs: Number of parallel jobs (if parallel=True)

        Returns:
            List of (Y, Y_cf, L, treated_units) tuples
        """
        if not parallel:
            return [self.simulate() for _ in range(n_sims)]

        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs)(delayed(self.simulate)() for _ in range(n_sims))
