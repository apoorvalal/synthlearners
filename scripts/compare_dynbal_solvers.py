#!/usr/bin/env bash
set -euo pipefail

uv run python - "$@" <<'PY'
import argparse
import time

import numpy as np
import pyensmallen as pe
from scipy.optimize import Bounds, LinearConstraint, minimize

from synthlearners.dynbalance import _solve_balancing_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact dynamic-balance weight optimization against "
            "simplex-only penalized approximations."
        )
    )
    parser.add_argument("--n-obs", type=int, default=150)
    parser.add_argument("--n-features", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-constant", type=float, default=0.15)
    parser.add_argument("--penalty-strength", type=float, default=1e4)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-9)
    return parser.parse_args()


def generate_problem(
    n_obs: int,
    n_features: int,
    balance_constant: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n_obs, n_features))

    # Construct a feasible target mean from a random simplex combination.
    true_weights = rng.dirichlet(np.ones(n_obs))
    target_mean = true_weights @ covariates

    tolerance = balance_constant * np.sqrt(np.log(max(n_features, 2)) / np.sqrt(n_obs))
    return covariates, target_mean, tolerance


def solve_penalized_slsqp(
    covariates: np.ndarray,
    target_mean: np.ndarray,
    tolerance: float,
    penalty_strength: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    n_obs = covariates.shape[0]
    x0 = np.full(n_obs, 1.0 / n_obs)

    def objective(weights: np.ndarray) -> float:
        residual = covariates.T @ weights - target_mean
        slack = np.maximum(np.abs(residual) - tolerance, 0.0)
        return 0.5 * float(weights @ weights) + penalty_strength * float(slack @ slack)

    def gradient(weights: np.ndarray) -> np.ndarray:
        residual = covariates.T @ weights - target_mean
        slack = np.maximum(np.abs(residual) - tolerance, 0.0)
        penalty_grad = covariates @ (2.0 * penalty_strength * np.sign(residual) * slack)
        return weights + penalty_grad

    constraints = [LinearConstraint(np.ones((1, n_obs)), np.array([1.0]), np.array([1.0]))]
    bounds = Bounds(np.zeros(n_obs), np.ones(n_obs))
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
        raise RuntimeError(f"Penalized SLSQP failed: {result.message}")
    return result.x


def solve_penalized_ensmallen(
    covariates: np.ndarray,
    target_mean: np.ndarray,
    tolerance: float,
    penalty_strength: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    n_obs = covariates.shape[0]
    initial = np.full(n_obs, 1.0 / n_obs)
    optimizer = pe.SimplexFrankWolfe(maxIterations=max_iter, tolerance=tol)

    def objective(weights: np.ndarray, grad: np.ndarray) -> float:
        residual = covariates.T @ weights - target_mean
        slack = np.maximum(np.abs(residual) - tolerance, 0.0)
        if grad.size > 0:
            penalty_grad = covariates @ (
                2.0 * penalty_strength * np.sign(residual) * slack
            )
            grad[:] = weights + penalty_grad
        return 0.5 * float(weights @ weights) + penalty_strength * float(slack @ slack)

    return optimizer.optimize(objective, initial)


def max_imbalance(
    weights: np.ndarray,
    covariates: np.ndarray,
    target_mean: np.ndarray,
) -> float:
    return float(np.max(np.abs(covariates.T @ weights - target_mean)))


def run_solver(label: str, fn, repeats: int) -> dict[str, float | str | np.ndarray]:
    timings = []
    solution = None
    for _ in range(repeats):
        start = time.perf_counter()
        solution = fn()
        timings.append(time.perf_counter() - start)
    return {
        "label": label,
        "solution": solution,
        "mean_time_ms": 1000.0 * float(np.mean(timings)),
        "min_time_ms": 1000.0 * float(np.min(timings)),
    }


def main() -> None:
    args = parse_args()
    covariates, target_mean, tolerance = generate_problem(
        n_obs=args.n_obs,
        n_features=args.n_features,
        balance_constant=args.balance_constant,
        seed=args.seed,
    )

    exact = run_solver(
        "exact_slsqp_qp",
        lambda: _solve_balancing_problem(
            covariates=covariates,
            target_mean=target_mean,
            constant=args.balance_constant,
            max_weight=1.0,
            max_iter=args.max_iter,
            tol=args.tol,
        ),
        repeats=args.repeats,
    )
    penalized_slsqp = run_solver(
        "penalized_slsqp",
        lambda: solve_penalized_slsqp(
            covariates=covariates,
            target_mean=target_mean,
            tolerance=tolerance,
            penalty_strength=args.penalty_strength,
            max_iter=args.max_iter,
            tol=args.tol,
        ),
        repeats=args.repeats,
    )
    ensmallen_fw = run_solver(
        "penalized_ensmallen_fw",
        lambda: solve_penalized_ensmallen(
            covariates=covariates,
            target_mean=target_mean,
            tolerance=tolerance,
            penalty_strength=args.penalty_strength,
            max_iter=args.max_iter,
            tol=args.tol,
        ),
        repeats=args.repeats,
    )

    results = [exact, penalized_slsqp, ensmallen_fw]
    print("Solver comparison on a toy dynamic-balance instance")
    print(
        "Only the first row solves the exact dynamic-balance constrained quadratic problem."
    )
    print(
        "The pyensmallen row solves a penalized simplex approximation because the "
        "available optimizer does not expose the additional linear balance constraints."
    )
    print()
    print(f"n_obs={args.n_obs}, n_features={args.n_features}, repeats={args.repeats}")
    print(f"balance tolerance={tolerance:.6e}")
    print()
    print(
        f"{'solver':<24} {'mean_ms':>12} {'min_ms':>12} "
        f"{'max_abs_imbalance':>20} {'within_tol':>12} {'l2_norm':>12}"
    )
    for result in results:
        weights = result["solution"]
        imbalance = max_imbalance(weights, covariates, target_mean)
        within_tol = imbalance <= (tolerance + max(args.tol, 1e-12))
        print(
            f"{result['label']:<24} "
            f"{result['mean_time_ms']:>12.3f} "
            f"{result['min_time_ms']:>12.3f} "
            f"{imbalance:>20.6e} "
            f"{str(within_tol):>12} "
            f"{np.linalg.norm(weights):>12.6f}"
        )


if __name__ == "__main__":
    main()
PY
