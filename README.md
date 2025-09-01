# `synthlearners`: Scalable Synthetic Control Methods in Python

fast, scalable synthetic control methods. Full version powered by the [`pyensmallen`](https://github.com/apoorvalal/pyensmallen) library for fast optimisation, and lean version powered by [`adelie`](https://github.com/JamesYang007/adelie) for blazing fast regression solvers.
Check out the `notebooks` directory for synthetic and real data examples.

## features

features are indicated by
- [ ] pending; good first PR; contributions welcome
- [x] done

### weights
  - [x] unit weights [`/solvers.py`]
    - [x] simplex (Abadie, Diamond, Hainmueller [2010](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.ap08746?casa_token=HHoPpXX1iigAAAAA:zCB_ZwLLTs1uWBzAVrwgCKtA_FPZXdoqLoxKgZzGAvCCgLpA5WlFm4DphUiz2U_udE5GM329XdjWoQ), [2015](https://onlinelibrary.wiley.com/doi/full/10.1111/ajps.12116?casa_token=bKtsjsYAkAIAAAAA%3AuS7vADpexw4q0BACgWtaYDal1fwCI3k3bHruSUgCJyEVs_PrUlnmcenEK58f6QoqgCPBgZGTy0mssg))
    - [x] lasso ([Hollingsworth and Wing 2024+](https://osf.io/fc9xt/))
    - [x] ridge ([Imbens and Doudchenko 2016](https://www.nber.org/papers/w22791), [Arkhangelsky et al 2021](https://www.aeaweb.org/articles?id=10.1257/aer.20190159))
    - [x] matching ([Imai, Kim, Wang 2023](https://onlinelibrary.wiley.com/doi/full/10.1111/ajps.12685?casa_token=vap307wR7DwAAAAA%3AHGX_puzkDArA-O-mTfxOedqsr1zdVH4VgwgBA8pi8LnzUg1IVVUHEeVrIcCZZ1gA7gfqsrebAgIEJg))
    - [x] support intercept term ([Ferman and Pinto 2021](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE1596), Doudchenko and Imbens)
    - [ ] entropy weights ([Hainmueller 2012](https://www.cambridge.org/core/journals/political-analysis/article/entropy-balancing-for-causal-effects-a-multivariate-reweighting-method-to-produce-balanced-samples-in-observational-studies/220E4FC838066552B53128E647E4FAA7), [Hirschberg and Arkhangelsky 2023](https://arxiv.org/abs/2311.13575), [Lal 2023](https://apoorvalal.github.io/files/papers/augbal.pdf))
  - [x] with multiple treated units, match aggregate outcomes (default) or individual outcomes ([Abadie and L'Hour 2021](https://economics.mit.edu/sites/default/files/publications/A%20Penalized%20Synthetic%20Control%20Estimator%20for%20Disagg.pdf))
  - [x] time weights
    - [x] L2 weights (Arkhangelsky et al 2021)
    - [ ] time-distance penalised weights (Imbens et al 2024)
  - [ ] augmenting weights with outcome models ([Ben-Michael et al 2021](https://arxiv.org/abs/1811.04170))
    - [x] matrix completion ([Athey et al 2021](https://arxiv.org/abs/1710.10251))
    - [ ] latent factor models ([Xu 2017](https://yiqingxu.org/papers/english/2016_Xu_gsynth/Xu_PA_2017.pdf), Lal et al 2024)
    - [ ] two-way kernel ridge weights ([Ben-Michael et al 2023](https://arxiv.org/abs/2110.07006))

### inference
- [x] jacknife confidence intervals (multiple treated units) [Arkhangelsky et al 2021)
- [x] permutation test (Abadie et al 2010)
- [ ] conformal inference ([Chernozhukov et al 2021](https://arxiv.org/abs/1712.09089))

### visualisations
  - [x] raw outcome time series with treated average and synthetic control
  - [x] event study plot (treatment effect over time)
  - [x] weight distributions


Contributions welcome!

## installation

```
pip install git+https://github.com/apoorvalal/synthlearners/
```

or git clone and run `uv pip install -e .` and make changes.


### Lean Installation (Recommended)

For fast, regularized synthetic control methods using adelie:

```bash
pip install synthlearners
```

This installs only the core dependencies:
- `numpy`, `pandas`, `matplotlib` - Basic data handling and plotting
- `adelie` - Blazing fast regularized regression solver

**Available methods:**
- `PenguinSynth` - L1/L2 regularized synthetic control and SDID
- All traditional CVX implementations (`adelie_synth`, `synthetic_control`, `synthetic_diff_in_diff`, etc.)

## Full Installation

For all methods including traditional constrained optimization and matching:

```bash
pip install synthlearners[full]
```

For development, you would want to navigate to the repo location, and then run 

```
uv pip install -e ".[full]"
```

which tells uv to install in editable mode with `full` optional dependencies.

Additional dependencies:
- `scipy`, `scikit-learn` - Optimization and cross-validation
- `pyensmallen` - Frank-Wolfe optimization
- `faiss-cpu` - Fast nearest neighbor search
- `joblib`, `tqdm` - Parallelization and progress bars
- `seaborn`, `ipywidgets` - Enhanced plotting and notebook widgets

**Additional methods:**
- `Synth` - Traditional synthetic control with various solvers
- `MatrixCompletionEstimator` - Matrix completion methods
- Advanced cross-validation and inference tools

## Quick Start

### Lean Installation
```python
from synthlearners import PenguinSynth

# Ridge regularized synthetic control
estimator = PenguinSynth(method="synth", l1_ratio=0.0)
result = estimator.fit(df, "unit", "time", "treat", "outcome")
print(f"Treatment effect: {result.att:.3f}")
```

### Full Installation
```python
from synthlearners import Synth, PenguinSynth

# Traditional simplex-constrained synthetic control
synth = Synth(method="simplex")
result = synth.fit(Y, treated_units=15, T_pre=10)

# Or use the fast regularized version
penguin = PenguinSynth(method="synth", l1_ratio=0.5)
result = penguin.fit(df, "unit", "time", "treat", "outcome")
```

## Performance Comparison

**PenguinSynth (adelie)**: ~100x faster, no convergence issues, flexible regularization
**Traditional Synth**: Full inference support, established methodology, more solvers
