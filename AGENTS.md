# Repository Guide

## Purpose

This repository studies whether whitening frozen representations before fitting
a logistic prediction head reduces reliance on spurious correlations. Prefer
transparent, matched comparisons between identity, standardization, and
whitening.

## Layout

Keep the executable code flat and task-oriented:

- `evaluate.py`: editable evaluation configuration, CLI overrides, fitting
  protocol, and terminal report;
- `hyper_param_sweep.py`: ridge-only sweeps with transforms fitted once per
  embedding bundle;
- `finetune_results.py`: fixed train/validation-split sweeps for ResNet-50 and
  future fine-tuned representations;
- `aggregate.py`: validation selection and held-out aggregation across
  fine-tuning seeds;
- `evaluate_synthetic.py`: editable standard synthetic evaluation, CLI
  overrides, tables, and optional JSON;
- `data.py`: NumPy embedding bundle loading and validation;
- `model.py`: fixed scikit-learn logistic regression;
- `whitening.py`: representation transforms;
- `metric.py`: overall, class-, and group-level metrics;
- `synthetic.py`: synthetic data generation and experiments;
- `tests/`: CPU-sized tests.

Do not recreate a package-level CLI, `last_layer.py`, or fine-tuning pipeline.
The flat sweep file above is the complete hyperparameter-selection surface;
do not grow it into an experiment framework.

`writing/main.tex` is a historical overview that will be rewritten later. Do
not update it during code refactors unless the user explicitly requests a
manuscript change.

## Data boundary

Only precomputed NumPy representation bundles under `artifacts/embeddings/` are
supported at runtime. A bundle contains memory-mappable `X`, `y`, and `g`
arrays for train, validation, and test, plus `manifest.json`.

The original `.pt` files under `data/` and `models/` are user-owned archival
sources. Never edit, rename, move, overwrite, or delete them. Runtime code must
never import PyTorch or read `.pt` files.

## Evaluation protocol

The primary comparison is:

- `identity`: no preprocessing;
- `standardize`: coordinate-wise centering and scaling;
- `whiten`: empirical whitening on the retained covariance subspace;
- `whiten_ledoit_wolf`: optional Ledoit--Wolf robustness check;
- `whiten_ledoit_wolf_nonlinear`: analytical Ledoit--Wolf robustness check.

Fit transforms on train only and apply them unchanged to validation and test.
Fit the logistic head on train only. Group labels are evaluation-only and must
not enter preprocessing, fitting, or model selection.

Keep estimator settings centralized in `model.py` rather than exposing a
solver grid:

```python
LogisticRegression(
    penalty="l2",
    solver=SOLVER,
    fit_intercept=True,
    max_iter=2000,
    tol=1e-6,
)
```

The only model choice exposed by `evaluate.py` is `ridge_lambda`, defined
relative to average log loss. Report `C = 1 / ridge_lambda`, and give
scikit-learn `C = 1 / (n_train * ridge_lambda)`. This keeps effective
regularization independent of sample size. When class balancing is enabled,
use mean-one inverse-class-frequency sample weights so every class has equal
total fitting weight.
Do not add learning rates, epochs, batch sizes, momentum, schedulers,
initialization settings, or optimizer grids.

`hyper_param_sweep.py` may vary only `ridge_lambda`. Fit each transform once
per embedding bundle and reuse it across the ridge path. `aggregate.py` selects
one ridge value separately for each transform and seed using validation
`accuracy`, exact label-balanced `class_balanced_accuracy`, `log_loss`, or
label-balanced `balanced_log_loss`. Break exact ties in favor of the larger
`ridge_lambda`. Do not use a one-mistake accuracy plateau and do not expose a
Brier-score selection rule. `balanced_log_loss` first averages log loss within
each class of `y` and then averages classes equally; it does not use groups.
Group-aware metrics remain reporting-only and are never valid selection rules.

`finetune_results.py` operates only on explicit fine-tuning seeds whose
bundles share one fixed train/validation split. Select ridge independently for
each seed and average the selected test results. Never probability-weight test
options by their hyperparameter-selection frequencies.

Both `evaluate.py` and `hyper_param_sweep.py` expose
`whitening_estimator`, with canonical values `empirical` and `ledoit-wolf`.
The generic `whiten` transform must honor this setting and record it in
transform diagnostics.

The direct evaluator reports train, validation, and test:

- overall accuracy;
- equal-group and worst-group accuracy;
- accuracy for every group;
- equal-group and worst-group log loss;
- full per-group metrics in optional JSON output.

## Whitening conventions

Embedding matrices have observations in rows. Calculate covariance in float32
using denominator `n - 1`. When `n < d`, eigendecompose the `n x n` sample
covariance and lift retained eigenvectors back to feature space. When `n >= d`,
eigendecompose the ordinary `d x d` feature covariance. In both branches,
retain an eigenvalue only when it is strictly greater than
`relative_tolerance * largest_eigenvalue`; do not assign artificial variance
to sample-null directions. When no tolerance is configured, use float32
machine epsilon without multiplying it by the observation count; that larger
factor causes unintended spectral truncation on large datasets.

For a retained eigendecomposition
`covariance = V @ diag(eigenvalues) @ V.T`, use
`(X - mean) @ V / sqrt(eigenvalues)`. Preserve transformed/original
coordinate logit equivalence to float32 rounding error when mapping fitted
coefficients back.

## Implementation standards

- Runtime dependencies are NumPy, SciPy, scikit-learn, and pandas.
- Stored embeddings and numerical preprocessing use float32 throughout.
- Use `numpy.random.Generator` in synthetic experiments.
- Keep configuration defaults in `EVAL_CONFIG` in `evaluate.py`; CLI arguments
  may override them.
- Never overwrite an existing JSON result unless the user explicitly passes
  the command's `--overwrite` option. Replacement must be atomic.
- Raise actionable errors for invalid bundles, dimensions, groups, covariance
  rank, and logistic-regression non-convergence.
- Keep changes small and readable; the flat file layout is intentional.

## Verification

Before declaring a code change complete, run:

```bash
pytest
```

Also run the smallest relevant real-data smoke evaluation. Tests must cover
memory-mapped loading, checksum validation, `.pt` rejection, whitening in
low- and high-dimensional regimes, rank deficiency, binary and multiclass
logistic regression, normalized sample weights, group metrics, CLI embedding
selection, and train-only fitting.
