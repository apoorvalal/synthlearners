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


def test_sdid_time_weights(simulated_data):
    """Test SDID time weights functionality."""
    Y, treated_units, config = simulated_data
    T_pre = config.T_pre
    true_effect = config.treatment_effect # This is a scalar in the current fixture

    # Test with SDID time weights enabled (simplex constraint for time weights)
    synth_sdid_simplex = Synth(method="simplex", enable_time_weights=True, time_weights_simplex=True)
    results_sdid_simplex = synth_sdid_simplex.fit(Y, treated_units, T_pre)

    assert results_sdid_simplex.time_weights_pre_treatment is not None
    assert len(results_sdid_simplex.time_weights_pre_treatment) == T_pre
    assert np.all(results_sdid_simplex.time_weights_pre_treatment >= -1e-9)  # Allow for small float errors
    assert np.isclose(np.sum(results_sdid_simplex.time_weights_pre_treatment), 1.0)

    # Test with high regularization -> uniform time weights
    synth_sdid_high_reg = Synth(
        method="simplex",
        enable_time_weights=True,
        time_weights_simplex=True,
        time_weight_regularization=1e6,
    )
    results_sdid_high_reg = synth_sdid_high_reg.fit(Y, treated_units, T_pre)
    expected_uniform_weights = np.ones(T_pre) / T_pre
    np.testing.assert_allclose(
        results_sdid_high_reg.time_weights_pre_treatment,
        expected_uniform_weights,
        atol=1e-3, # Looser tolerance due to optimization
        err_msg="Time weights should be uniform with high regularization"
    )

    # Test without simplex constraint for time weights (ridge)
    synth_sdid_ridge = Synth(method="simplex", enable_time_weights=True, time_weights_simplex=False, time_weight_regularization=0.01)
    results_sdid_ridge = synth_sdid_ridge.fit(Y, treated_units, T_pre)
    assert results_sdid_ridge.time_weights_pre_treatment is not None
    assert len(results_sdid_ridge.time_weights_pre_treatment) == T_pre
    # Ridge weights don't necessarily sum to 1 or are non-negative

    # Compare ATT with and without time weights
    synth_no_time_weights = Synth(method="simplex", enable_time_weights=False)
    results_no_time_weights = synth_no_time_weights.fit(Y, treated_units, T_pre)

    # ATT should generally differ, though not guaranteed for all datasets/params
    assert not np.isclose(results_sdid_simplex.post_treatment_effect, results_no_time_weights.post_treatment_effect) or \
           np.isclose(results_sdid_simplex.post_treatment_effect, true_effect, rtol=0.1), \
           "SDID ATT should typically differ from standard SC ATT, or be close to true effect if standard SC is far off"


    # Manual SDID ATT calculation for one case
    Y_control = Y[np.setdiff1d(range(Y.shape[0]), treated_units), :]
    Y_treated_avg = Y[treated_units].mean(axis=0)
    
    Y_control_pre = Y_control[:, :T_pre]
    Y_control_post_avg = Y_control[:, T_pre:].mean(axis=1)
    Y_treated_pre_vec = Y_treated_avg[:T_pre]
    Y_treated_post_avg = Y_treated_avg[T_pre:].mean()

    lambda_hat = results_sdid_simplex.time_weights_pre_treatment
    omega_hat = results_sdid_simplex.unit_weights

    term1 = Y_treated_post_avg - np.dot(Y_treated_pre_vec, lambda_hat)
    term2_control_effect = Y_control_post_avg - np.dot(Y_control_pre, lambda_hat) # This should be N_control x 1
    term2 = np.dot(term2_control_effect, omega_hat) # omega_hat is N_control x 1
    manual_att = term1 - term2
    
    np.testing.assert_allclose(
        results_sdid_simplex.post_treatment_effect,
        manual_att,
        rtol=1e-6,
        err_msg="Manually calculated SDID ATT does not match reported ATT"
    )
