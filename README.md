# `synthlearners`: Scalable Synthetic Control Methods in Python

synthetic control methods powered by the [`pyensmallen`](https://github.com/apoorvalal/pyensmallen) library for fast optimisation.
Check out the `notebooks` directory for synthetic and real data examples.


## installation

```
pip install git+https://github.com/apoorvalal/synthlearners/
```

or git clone and run `uv pip install -e .` and make changes.

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
  - [ ] time weights
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
