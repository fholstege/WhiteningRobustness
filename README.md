# Whitening frozen representations

This repository tests whether preprocessing frozen representations before a
logistic prediction head reduces reliance on spurious correlations. The main
comparison is matched across identity, coordinate standardization, empirical
whitening, and Ledoit--Wolf whitening.

## Repository layout

- `evaluate.py`: one configured train/validation/test evaluation.
- `hyper_param_sweep.py`: ridge-only sweeps for one or more embedding bundles.
- `finetune_results.py`: the fixed-split, multi-seed result grid.
- `comparison.py`: DFR, AFR, HO_ERM, and NeuroTune comparison sweeps.
- `aggregate.py`: validation selection and held-out aggregation.
- `model.py`: the fixed logistic-regression protocol.
- `whitening.py` and `whitening_core.py`: representation transforms.
- `metric.py`: overall, class-balanced, and group metrics.
- `evaluate_synthetic.py`, `synthetic.py`, and `synthetic_sweep.py`: synthetic
  experiments.
- `gather_embeddings.py`: offline conversion into NumPy embedding bundles.
- `tests/`: CPU-sized regression tests.

## Model protocol

The prediction head is L2-penalized logistic regression. `ridge_lambda` is
defined relative to average log loss, so scikit-learn receives

```text
sklearn_C = 1 / (n_train * ridge_lambda).
```

Binary fits use `liblinear`; multiclass fits use L-BFGS. If class balancing is
enabled, each class receives equal total weight through mean-one
inverse-class-frequency sample weights. Group labels never enter ordinary ERM
preprocessing, fitting, or model selection.

Transforms are fitted on train only and applied unchanged to validation and
test. The sweep varies only `ridge_lambda`, and each transform is fitted once
per embedding bundle and reused across the full ridge path.

## Running an evaluation

Edit `EVAL_CONFIG` or use command-line overrides:

```bash
python evaluate.py \
  --dataset WB \
  --embedding-type resnet50 \
  --seed 1 \
  --transform identity standardize whiten \
  --ridge-lambda 0.001 \
  --class-balanced
```

Use `--representation` for a direct manifest representation name.

## Sweeps and aggregation

Run the configured fine-tuned representation grid:

```bash
python finetune_results.py --workers 3 --blas-threads 1
```

Result files are protected unless `--overwrite` is supplied. Aggregate one or
more sweep files with a validation criterion:

```bash
python aggregate.py \
  --sweep results_seeds/WB_resnet50_finetune.json \
  --criterion class_balanced_accuracy
```

The available ordinary selection criteria are exact accuracy,
class-balanced accuracy, log loss, balanced log loss, and the documented
two-stage rule. Exact criterion ties favor the larger ridge penalty. Group
metrics remain reporting-only except for comparison methods that explicitly
declare oracle validation-group selection.

## Comparison methods

`comparison.py` runs one or more of:

- DFR: an oracle group-balanced validation refit.
- AFR: confidence- and class-weighted validation-head adaptation.
- HO_ERM: class-balanced ERM fitted on one label-stratified validation half,
  with ridge selected on the other half.
- NeuroTune†: split validation label-stratified into equal halves; identify
  coordinates from absolute activation magnitudes on the first half, fit the
  masked head on the disjoint second half, and select candidates by SFit on
  the first half. Group labels are unused and the selected head is not refit.

SFit is the default NeuroTune selection rule. To use another group-free
criterion, pass for example `--criterion class_balanced_accuracy` to
`aggregate.py`. This applies the criterion to ordinary methods and NeuroTune;
DFR and AFR retain their method-specific selection protocols. For the rarer
case of a transform-specific override, `--selection-rule
neurotune_identity=accuracy` remains available.

Example:

```bash
python comparison.py --methods DFR AFR HO_ERM --workers 3 --blas-threads 1
```

Comparison result files can be aggregated alone or together with the matching
ordinary sweep file.

## Whitening

Embedding matrices store observations in rows. Empirical covariance is
computed in float32 with denominator `n - 1`. When `n < d`, whitening uses the
sample-space eigendecomposition and lifts retained eigenvectors back to
feature space. Otherwise it uses the feature covariance directly.

Directions are retained only when their raw eigenvalue is strictly greater
than `relative_tolerance * largest_eigenvalue`. The default tolerance is
float32 machine epsilon. Retained coordinates are

```text
(X - train_mean) @ eigenvectors / sqrt(eigenvalues).
```

## Data boundary

Runtime code reads only NumPy embedding bundles under
`artifacts/embeddings/`. Each bundle contains memory-mappable `X`, `y`, and
`g` arrays for train, validation, and test plus `manifest.json`.

The original `.pt` files under `data/` and `models/` are archival inputs and
must not be changed. PyTorch is used only by the offline embedding-gathering
script, never by runtime evaluation code.

## Synthetic experiments

Run a small configured synthetic evaluation with:

```bash
python evaluate_synthetic.py
```

Run the larger editable sweep with:

```bash
python synthetic_sweep.py
```

Both use NumPy random generators and the same logistic and whitening
implementations as the real-data experiments.

The default synthetic sweep writes `results/synthetic_gamma_sweep.json`.
Edit `gamma_values` in `SYNTHETIC_SWEEP_CONFIG` to change the appendix values
(default: `[0.2, 1.0, 5.0]`); `gamma` selects the original Figure 2 value and
must occur in that list. Theorem/proposition checks are disabled by default.
All gamma values use matched random seeds, dimensions, and sample sizes.

```bash
python synthetic_sweep.py --workers 4 --blas-threads 1
python viz_synthetic.py results/synthetic_gamma_sweep.json --output-dir plots/synthetic_gamma --overwrite
```

The plotting command creates `performance_vs_q_over_n.png` (Figure 2, including
standardization) and `performance_by_gamma.png` (the gamma appendix), as well
as the decision-boundary and empirical-whitening performance plots. It does
not rerun simulations or theory checks.

Regenerate the empirical 2x2 figures, with worst-group and equal-group accuracy
above their corresponding paired differences:

```bash
# Figures 3 and 10: main and appendix backbones.
python viz.py --figure both --group-accuracy-panels --output plots/group_accuracy --overwrite
# Figure 9: empirical versus nonlinear whitening, main backbones.
python viz.py --figure main --compare-whitening-estimators --group-accuracy-panels --output plots/whitening_estimators_group_accuracy --overwrite
```

These write `group_accuracy_main.png`, `group_accuracy_appendix.png`, and
`whitening_estimators_group_accuracy.png` under `plots/`. Existing result JSON
files are never replaced unless the sweep itself receives `--overwrite`.

## Setup and verification

### Frozen pretrained backbones

Extract DINOv3 ViT-B/16 CLS features and OpenAI CLIP ViT-B/16 normalized image
projections once, without task fine-tuning:

```bash
python gather_embeddings.py --dataset waterbirds --backbone dinov3-vitb16 --dtype float32 --batch-size 16
python gather_embeddings.py --dataset waterbirds --backbone clip-vitb16 --dtype float32 --batch-size 16
python gather_embeddings.py --dataset celeba --backbone dinov3-vitb16 --dtype float32 --batch-size 16
python gather_embeddings.py --dataset celeba --backbone clip-vitb16 --dtype float32 --batch-size 16
```

DINOv3 requires access to Meta's gated Hugging Face checkpoint. The extraction
dependencies are optional (`pip install -e '.[embeddings]'`); evaluation remains
NumPy-only. Existing complete bundles are reused, and interrupted extraction
resumes from its staging files. These pretrained bundles have no representation
seed and declare `task_finetuned: false` in their provenance.

Then edit `FROZEN_RESULTS_CONFIG` in `frozen_results.py` and run:

```bash
python frozen_results.py --workers 2 --blas-threads 1
```

The default four experiments cover both backbones on WB and CelebA, with ten
editable `split_seeds`. Each seed repartitions the original train+validation
pool using target-label stratification and the original split sizes. Group
labels do not determine the partition, and the original test split is fixed.
All backbones and transforms use matched assignments. Transforms and heads
are fitted on each new training split only, with each transform fitted once
per split and reused across the ridge path.

The script uses the fine-tuned ridge grid plus four lower values
(`1e-7`, `3e-7`, `1e-6`, and `3e-6`) and the same whitening variants. It
selects ridge independently for each split/transform using class-balanced
validation accuracy, and averages selected test metrics. One JSON per
dataset/backbone is written to `results_frozen/`, containing both the complete
sweep and `selected_results`. `--dataset WB`, `--embedding dinov3_vitb16`, and
`--split-seeds 1 2` can restrict a run. Outputs require explicit `--overwrite`
to replace. Split seeds are recorded separately from representation seeds;
intervals describe split variability conditional on this checkpoint and test
set, not variability across independent pretrained models or test datasets.
The files also work with the existing aggregate and plot commands:

```bash
python viz.py --sweep results_frozen/WB_dinov3_vitb16_frozen.json results_frozen/CelebA_dinov3_vitb16_frozen.json --group-accuracy-panels --output plots/frozen_dinov3 --overwrite
```

### Tests

Install the locked environment used by the repository, then run:

```bash
pytest
```

For code changes, also run the smallest relevant real-data smoke evaluation.
