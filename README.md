# `synthlearners`: Scalable Synthetic Control Methods in Python

Fast synthetic control tooling with two estimator families:

- `PenguinSynth`: lean, adelie-backed regularized estimators for long-format panel data
- `Synth`: full, pyensmallen-backed estimators for matrix-format workflows and inference

Check the `notebooks/` directory for synthetic and real-data examples.

## Features

Features are indicated by:
- [ ] pending
- [x] implemented

### Weights
- [x] unit weights
- [x] simplex
- [x] lasso
- [x] ridge
- [x] matching
- [x] intercept support
- [ ] entropy weights
- [x] multiple treated units with aggregate or granular matching
- [x] time weights for SDID
- [ ] time-distance penalized weights
- [x] matrix completion augmentation
- [ ] latent factor models
- [ ] two-way kernel ridge weights

### Inference
- [x] jackknife confidence intervals
- [x] permutation tests
- [ ] conformal inference

### Visualizations
- [x] treated versus synthetic trajectories
- [x] treatment-effect event studies
- [x] weight plots

## Installation

For local development:

```bash
git clone https://github.com/apoorvalal/synthlearners.git
cd synthlearners
uv sync --extra full --extra test --extra docs
```

### Lean Installation

For the adelie-backed API:

```bash
uv add synthlearners
```

Lean installs expose:
- `PenguinSynth`
- `PenguinResults`
- `PenguinSynth(method="synth")`
- `PenguinSynth(method="sdid")`
- `PenguinSynth(method="did")`

### Full Installation

For the traditional synthetic-control stack and auxiliary utilities:

```bash
uv sync --extra full
```

If you are adding the package to another project instead of working from source:

```bash
uv add "synthlearners[full]"
```

Full installs add:
- `Synth`
- `SynthResults`
- `DynamicBalance`
- `DynamicBalanceResults`
- `Synth(method="simplex")`
- `Synth(method="linear")`
- `Synth(method="lp_norm")`
- `Synth(method="matching")`
- `Synth(method="matrix_completion")`
- `Synth(method="sdid")`
- `DynamicBalance.fit(...)` for exact treatment-history contrasts
- `PanelCrossValidator` and related cross-validation utilities

`MatrixCompletionEstimator` is available from `synthlearners.mcnnm`; it is not exported from the package root.

## Documentation

Static API docs are generated with `pdoc` into `docs/`.

```bash
uv sync --extra full --extra docs
./scripts/build_docs.sh
```

## Quick Start

### Lean API

```python
from synthlearners import PenguinSynth

estimator = PenguinSynth(method="synth", l1_ratio=0.0)
result = estimator.fit(df, "unit", "time", "treatment", "outcome")
print(f"Treatment effect: {result.att:.3f}")
```

### Full API

```python
from synthlearners import DynamicBalance, Synth, PenguinSynth

synth = Synth(method="simplex")
result = synth.fit(Y, treated_units=15, T_pre=10)

penguin = PenguinSynth(method="synth", l1_ratio=0.5)
result = penguin.fit(df, "unit", "time", "treatment", "outcome")

dynbal = DynamicBalance(l1_ratio=0.0)
res = dynbal.fit(
    df=df,
    unit_id="unit",
    time_id="time",
    treatment="treatment",
    outcome="outcome",
    covariates=["x", "lag_outcome"],
    target_history=[1, 1],
    reference_history=[0, 0],
)
```

## Performance

`PenguinSynth` is the faster default path for regularized estimation.

`Synth` carries the broader constrained-optimization toolbox, matrix completion support, and inference utilities that rely on the full dependency set.
