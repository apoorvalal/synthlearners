import pytest
import numpy as np
import mlpack
import scipy
from synthlearners.mcnnm import MatrixCompletionEstimator

@pytest.fixture(scope="module")
def svd_methods():
    #svd_methods = ["numpy", "scipy","mlpack"]
    svd_methods = ["numpy", "scipy"]
    return svd_methods

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n, t = 10, 8
    true_matrix = np.random.randn(n, t)
    mask = np.random.binomial(1, 0.8, size=(n,t))
    observed = true_matrix * mask
    return true_matrix, observed, mask

def test_svd_methods(sample_data,svd_methods):
    true_matrix, observed, mask = sample_data
    
    for method in svd_methods:
        mc = MatrixCompletionEstimator(svd_method=method)
        completed = mc.fit(observed, mask)
        
        # Check output dimensions
        assert completed.shape == true_matrix.shape
        
        # Check if observed entries are preserved (approximately)
        np.testing.assert_allclose(
            completed[mask == 1], 
            observed[mask == 1],
            rtol=1e-1
        )
        
        # Check if completion produces finite values
        assert np.all(np.isfinite(completed))
        
        # Check error metric
        mse = mc.score(true_matrix, completed, mask)
        assert mse >= 0
        assert np.isfinite(mse)
        print(f"SVD Method: {method} works, MSE: {mse}")

def test_fixed_effects_with_svd_methods(sample_data,svd_methods):
    true_matrix, observed, mask = sample_data
    
    for method in svd_methods:
        # Test with unit fixed effects
        mc = MatrixCompletionEstimator(svd_method=method)
        completed = mc.fit(observed, mask, unit_intercept=True)
        assert completed.shape == true_matrix.shape
        
        # Test with time fixed effects
        mc = MatrixCompletionEstimator(svd_method=method)
        completed = mc.fit(observed, mask, time_intercept=True)
        assert completed.shape == true_matrix.shape
        
        # Test with both fixed effects
        mc = MatrixCompletionEstimator(svd_method=method)
        completed = mc.fit(observed, mask, unit_intercept=True, time_intercept=True)
        assert completed.shape == true_matrix.shape

def test_convergence_with_svd_methods(sample_data,svd_methods):
    true_matrix, observed, mask = sample_data
    
    for method in svd_methods:
        # Test with strict tolerance
        mc = MatrixCompletionEstimator(svd_method=method, tol=1e-6, max_iter=1000)
        completed = mc.fit(observed, mask)
        assert completed.shape == true_matrix.shape
        
        # Test with loose tolerance
        mc = MatrixCompletionEstimator(svd_method=method, tol=1e-2, max_iter=50)
        completed = mc.fit(observed, mask)
        assert completed.shape == true_matrix.shape

    # Test the run time by svd method
def test_runtime_with_svd_methods(sample_data,svd_methods):
    import time
    true_matrix, observed, mask = sample_data
    runtimes = {}   
    for method in svd_methods:
        start_time = time.time()
        mc = MatrixCompletionEstimator(svd_method=method)
        completed = mc.fit(observed, mask)
        end_time = time.time()
        runtimes[method] = end_time - start_time
        assert completed.shape == true_matrix.shape
    print("Runtimes by SVD method:", runtimes)
    #assert 0==1  # Force to see runtimes