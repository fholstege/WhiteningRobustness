# Whitening frozen representations

This repository studies whether preprocessing frozen representations before
fitting a logistic prediction head reduces reliance on spurious correlations.
We compare no preprocessing, coordinate-wise standardization, and whitening
under a matched fitting and model-selection protocol.

## Setup

Install the evaluation, plotting, and test dependencies with

```bash
pip install -e '.[test,plot]'
```

Runtime experiments use memory-mappable NumPy bundles under
`artifacts/embeddings/`. Each bundle contains train, validation, and test
arrays for representations, labels, and groups, together with a manifest.
The original `.pt` files are archival inputs and are not read at runtime.

## Experiments

The main experiment configurations are defined near the top of their
corresponding scripts. Typical commands are

```bash
# One train/validation/test evaluation.
python evaluate.py --dataset WB --embedding-type resnet50 --seed 1

# Ridge sweeps across fine-tuned representations.
python finetune_results.py --workers 3 --blas-threads 1

# Ridge sweeps across frozen DINOv3 and CLIP representations.
python frozen_results.py --workers 2 --blas-threads 1

# DFR, AFR, HO_ERM, and NeuroTune comparisons.
python comparison.py --methods DFR AFR HO_ERM --workers 3 --blas-threads 1
```

Transforms and logistic heads are fitted on training observations only. Ridge
parameters are selected on validation data, while group labels remain
reporting-only unless a comparison method explicitly defines a group-aware
oracle. Existing result files are protected unless `--overwrite` is supplied.

## Results and figures

Aggregate saved sweeps with a label-only validation criterion:

```bash
python aggregate.py --sweep results_paper/WB_resnet50_finetune.json \
  --criterion class_balanced_accuracy
```

The principal empirical and frozen-representation figures are generated with

```bash
python viz.py --figure both --group-accuracy-panels \
  --output plots/group_accuracy --overwrite

python viz_experiments.py --plot-type frozen --overwrite
```

The frozen simplicity plot selects each ridge parameter by validation balanced
log loss and caches the corresponding fitted coefficients. Synthetic
experiments are run with `evaluate_synthetic.py` or `synthetic_sweep.py` and
plotted with `viz_synthetic.py`.

## Verification

Run the test suite and the smallest relevant real-data smoke evaluation:

```bash
pytest
```
