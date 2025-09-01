import pytest
import numpy as np
from synthlearners.ensemble import SynthEnsemble
from synthlearners.synth import Synth

@pytest.fixture
def sample_data():
    # Create sample data: 5 units, 10 time periods
    np.random.seed(42)
    X = np.random.randn(5, 10)
    treated_units = [0]  # First unit is treated
    T_pre = 5  # First 5 periods are pre-treatment
    return X, treated_units, T_pre

@pytest.fixture
def ensemble():
    synth_methods = [
    Synth(method="lp_norm", p=1.0),
    Synth(method="simplex"),
    Synth(method="matrix_completion", unit_intercept=True, time_intercept=True),
    ]
    return SynthEnsemble(methods=synth_methods, cv_splits=2)

def test_ensemble_initialization(ensemble):
    assert len(ensemble.methods) == 3
    assert ensemble.cv_splits == 2
    assert ensemble.weights_ is None

def test_ensemble_fit(ensemble, sample_data):
    X, treated_units, T_pre = sample_data
    
    # Test fitting
    fitted_ensemble = ensemble.fit(X, treated_units, T_pre)
    assert fitted_ensemble is ensemble
    assert fitted_ensemble.weights_ is not None
    assert len(fitted_ensemble.weights_) == 3  # One weight per method

def test_ensemble_predict(ensemble, sample_data):
    X, treated_units, T_pre = sample_data
    
    # Fit first
    ensemble.fit(X, treated_units, T_pre)
    
    # Test prediction
    predictions = ensemble.predict(X)
    assert predictions.shape == X.shape

def test_ensemble_invalid_predict(ensemble):
    # Test prediction without fitting
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        ensemble.predict(np.random.randn(5, 10))

def test_ensemble_invalid_split_type(sample_data):
    X, treated_units, T_pre = sample_data
    synth_methods = [
    Synth(method="lp_norm", p=1.0),
    Synth(method="simplex"),
    Synth(method="matrix_completion", unit_intercept=True, time_intercept=True),
    ]
    ensemble = SynthEnsemble(methods=synth_methods, cv_splits=2, split_type="random_split")
    
    with pytest.raises(ValueError, match="Random split type is not supported"):
        ensemble.fit(X, treated_units, T_pre)