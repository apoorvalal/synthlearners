import numpy as np
import pandas as pd
import pytest

from synthlearners import DynamicBalance


def _simulate_dynamic_panel(n_units: int = 800, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    unit = np.arange(n_units)
    u = rng.normal(size=n_units)
    eps_y1 = rng.normal(scale=0.5, size=n_units)
    eps_x2 = rng.normal(scale=0.5, size=n_units)
    eps_y2 = rng.normal(scale=0.5, size=n_units)

    prob_d1 = 1.0 / (1.0 + np.exp(-0.8 * u))
    d1 = rng.binomial(1, prob_d1)

    y1 = 0.5 * u + 1.0 * d1 + eps_y1
    x2 = 0.5 * u + 0.7 * d1 + eps_x2

    prob_d2 = 1.0 / (1.0 + np.exp(-(0.4 * u + 0.6 * y1 + 0.5 * x2 + 0.5 * d1)))
    d2 = rng.binomial(1, prob_d2)

    y2 = 0.4 * u + 0.3 * y1 + 0.5 * x2 + 1.0 * d1 + 1.5 * d2 + eps_y2

    period1 = pd.DataFrame(
        {
            "unit": unit,
            "time": 1,
            "treatment": d1,
            "outcome": y1,
            "x": u,
            "lag_outcome": np.zeros(n_units),
        }
    )
    period2 = pd.DataFrame(
        {
            "unit": unit,
            "time": 2,
            "treatment": d2,
            "outcome": y2,
            "x": x2,
            "lag_outcome": y1,
        }
    )
    return pd.concat([period1, period2], ignore_index=True)


def test_dynamic_balance_recovers_two_period_contrast():
    df = _simulate_dynamic_panel()
    estimator = DynamicBalance(l1_ratio=0.0, balance_grid_size=15)
    result = estimator.fit(
        df=df,
        unit_id="unit",
        time_id="time",
        treatment="treatment",
        outcome="outcome",
        covariates=["x", "lag_outcome"],
        target_history=[1, 1],
        reference_history=[0, 0],
    )

    true_contrast = 3.15
    assert np.isfinite(result.contrast)
    np.testing.assert_allclose(result.contrast, true_contrast, atol=1.0)


def test_dynamic_balance_weights_respect_history_prefixes():
    df = _simulate_dynamic_panel(n_units=400, seed=321)
    estimator = DynamicBalance(l1_ratio=0.0, balance_grid_size=10)
    result = estimator.fit(
        df=df,
        unit_id="unit",
        time_id="time",
        treatment="treatment",
        outcome="outcome",
        covariates=["x", "lag_outcome"],
        target_history=[1, 1],
        reference_history=[0, 0],
    )

    target_weights = result.target.weights_by_period
    treatment_matrix = (
        df.pivot(index="unit", columns="time", values="treatment")
        .loc[:, [1, 2]]
        .to_numpy(dtype=float)
    )

    eligible_first = treatment_matrix[:, 0] == 1
    eligible_second = np.all(treatment_matrix == np.array([1, 1]), axis=1)

    assert np.isclose(target_weights[0].sum(), 1.0)
    assert np.isclose(target_weights[1].sum(), 1.0)
    assert np.all(target_weights[0] >= -1e-10)
    assert np.all(target_weights[1] >= -1e-10)
    assert np.allclose(target_weights[0][~eligible_first], 0.0)
    assert np.allclose(target_weights[1][~eligible_second], 0.0)


def test_dynamic_balance_requires_equal_history_lengths():
    df = _simulate_dynamic_panel(n_units=100, seed=999)
    estimator = DynamicBalance()

    with pytest.raises(ValueError):
        estimator.fit(
            df=df,
            unit_id="unit",
            time_id="time",
            treatment="treatment",
            outcome="outcome",
            covariates=["x", "lag_outcome"],
            target_history=[1, 1],
            reference_history=[0],
        )
