#!/usr/bin/env python3
"""
Test script for the cleaned up SDID implementation using method="sdid".
"""

import numpy as np
from synthlearners import Synth
from synthlearners.simulator import SimulationConfig, PanelSimulator, FactorDGP


def test_clean_sdid():
    """Test the cleaned up SDID implementation."""

    # Create test data
    config = SimulationConfig(
        N=50,
        T=100,
        T_pre=60,
        n_treated=20,
        selection_mean=1.0,
        treatment_effect=2.0,
        dgp=FactorDGP(
            K=2,
            sigma=0.5,
            time_fac_lb=-0.5,
            time_fac_ub=0.5,
            trend_sigma=0.5,
        ),
    )

    simulator = PanelSimulator(config)
    Y, Y_0, L, treated_units = simulator.simulate()

    print("=" * 60)
    print("TESTING CLEAN SDID API (method='sdid')")
    print("=" * 60)
    print(f"Panel dimensions: {Y.shape[0]} units × {Y.shape[1]} periods")
    print(f"Pre-treatment periods: {config.T_pre}")
    print(f"True treatment effect: {config.treatment_effect}")
    print(f"Treated units: {treated_units}")
    print()

    # Test 1: Traditional synthetic control
    print("1. Traditional Synthetic Control:")
    synth_sc = Synth(method="simplex")
    results_sc = synth_sc.fit(Y, treated_units, config.T_pre)
    print(f"   ATT estimate: {results_sc.att():.4f}")
    print(f"   Pre-treatment RMSE: {results_sc.pre_treatment_rmse:.4f}")
    print(f"   Bias: {results_sc.att() - config.treatment_effect:.4f}")
    print()

    # Test 2: SDID with default parameters
    print("2. SDID (default parameters):")
    synth_sdid = Synth(method="sdid")
    results_sdid = synth_sdid.fit(Y, treated_units, config.T_pre, compute_jackknife=True)
    print(f"   ATT estimate: {results_sdid.att():.4f}")
    print(f"   Pre-treatment RMSE: {results_sdid.pre_treatment_rmse:.4f}")
    print(f"   Bias: {results_sdid.att() - config.treatment_effect:.4f}")
    print(
        f"   Time weights range: [{results_sdid.time_weights_pre_treatment.min():.4f}, {results_sdid.time_weights_pre_treatment.max():.4f}]"
    )
    print(
        f"   Unit weights range: [{results_sdid.unit_weights.min():.4f}, {results_sdid.unit_weights.max():.4f}]"
    )
    print()

    # Test 3: SDID with custom regularization
    print("3. SDID with higher regularization:")
    synth_sdid_reg = Synth(method="sdid", zeta_omega=0.1, zeta_lambda=0.01)
    results_sdid_reg = synth_sdid_reg.fit(Y, treated_units, config.T_pre, compute_jackknife=True)
    print(f"   ATT estimate: {results_sdid_reg.att():.4f}")
    print(f"   Pre-treatment RMSE: {results_sdid_reg.pre_treatment_rmse:.4f}")
    print(f"   Bias: {results_sdid_reg.att() - config.treatment_effect:.4f}")
    print(
        f"   Time weights range: [{results_sdid_reg.time_weights_pre_treatment.min():.4f}, {results_sdid_reg.time_weights_pre_treatment.max():.4f}]"
    )
    print(
        f"   Unit weights range: [{results_sdid_reg.unit_weights.min():.4f}, {results_sdid_reg.unit_weights.max():.4f}]"
    )
    print()

    # Test 4: Compare methods
    print("4. Method Comparison:")
    methods = {
        "Simplex SC": results_sc,
        "SDID (default)": results_sdid,
        "SDID (regularized)": results_sdid_reg,
    }

    print(f"{'Method':<18} {'ATT':<8} {'Bias':<8} {'Pre-RMSE':<10}")
    print("-" * 50)
    for name, result in methods.items():
        bias = result.att() - config.treatment_effect
        print(
            f"{name:<18} {result.att():<8.4f} {bias:<8.4f} {result.pre_treatment_rmse:<10.4f}"
        )

    print("\n" + "=" * 60)
    print("CLEAN API TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)
    test_clean_sdid()
