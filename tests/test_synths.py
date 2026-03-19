import numpy as np
import pytest
from synthlearners import Synth
from synthlearners.simulator import SimulationConfig, PanelSimulator, FactorDGP


@pytest.fixture
def simulated_data():
    """Create a simple simulated dataset for testing."""
    config = SimulationConfig(
        N=100,  # Smaller for testing
        T=50,
        T_pre=40,
        n_treated=5,
        selection_mean=1.0,
        treatment_effect=0.5,  # Known treatment effect
        dgp=FactorDGP(K=3, sigma=0.4, trend_sigma=0.01),
    )
    simulator = PanelSimulator(config)
    Y, Y_0, L, treated_units = simulator.simulate()
    return Y, treated_units, config


def test_treatment_effects_consistency(simulated_data):
    """Test that post_treatment_effect matches treatment_effect() mean."""
    Y, treated_units, config = simulated_data

    synth = Synth(method="simplex")
    results = synth.fit(Y, treated_units, config.T_pre, compute_jackknife=False)

    # Check that post_treatment_effect matches mean of treatment_effect()
    effect_series = results.treatment_effect()
    post_period_effects = effect_series[config.T_pre :]

    np.testing.assert_allclose(
        results.post_treatment_effect, np.mean(post_period_effects), rtol=1e-10
    )


def test_methods_return_reasonable_effects(simulated_data):
    """Test that all methods return effects somewhat close to true effect."""
    Y, treated_units, config = simulated_data
    true_effect = config.treatment_effect

    methods = {
        "simplex": {"method": "simplex"},
        "linear": {"method": "linear"},
        "ridge": {"method": "lp_norm", "p": 2.0},
        "lasso": {"method": "lp_norm", "p": 1.0},
        "matching": {"method": "matching"},
    }

    for name, params in methods.items():
        synth = Synth(**params)
        results = synth.fit(Y, treated_units, config.T_pre, compute_jackknife=False)

        # Test if estimated effect is within reasonable range of true effect
        np.testing.assert_allclose(
            results.post_treatment_effect,
            true_effect,
            rtol=0.5,  # Allow 50% relative error
            err_msg=f"Method {name} failed to recover treatment effect",
        )


def test_pre_treatment_fit(simulated_data):
    """Test that pre-treatment fit is better than post-treatment."""
    Y, treated_units, config = simulated_data

    synth = Synth(method="simplex")
    results = synth.fit(Y, treated_units, config.T_pre, compute_jackknife=False)

    effects = results.treatment_effect()
    pre_rmse = np.sqrt(np.mean(effects[: config.T_pre] ** 2))
    post_rmse = np.sqrt(np.mean(effects[config.T_pre :] ** 2))

    # Pre-treatment fit should be better than post-treatment
    assert pre_rmse < post_rmse


def test_jackknife_shape(simulated_data):
    """Test that jackknife effects have expected shape."""
    Y, treated_units, config = simulated_data

    synth = Synth(method="lp_norm", n_jobs=1)
    results = synth.fit(Y, treated_units, config.T_pre, compute_jackknife=True)

    # Should have one jackknife iteration per treated unit
    assert results.jackknife_effects.shape == Y.shape
