# Whitening representations

This repository evaluates whether fitting a linear prediction head after
standardization or whitening makes fine-tuned representations less reliant on
spurious correlations. The manuscript compares identity preprocessing,
coordinate-wise standardization, and whitening with matched ridge selection.

## Reproducing the paper results

Use the following order after obtaining the data and checkpoints described in
[Data and checkpoints](#data-and-checkpoints). Each step writes the inputs for
the next one. Run the commands from the repository root.

### 1. Create the fine-tuned representations

Run the project's fine-tuning script first. It is intentionally separate from
the NumPy/scikit-learn evaluation code in this repository. Its job is to train
the requested representation seeds and save their checkpoints and final linear
heads.

```bash
python finetune.py
```

`finetune.py` is not included in this checkout, so obtain it with the training
code and use its documented configuration for the paper seeds. Before moving
on, use `gather_embeddings.py` (or the training script's corresponding export
step) to create the required NumPy embedding bundles under
`artifacts/embeddings/`. The paper workflow expects fixed train/validation
splits and the saved heads under `artifacts/last_layers/`.

### 2. Regenerate all paper-result JSON files

Install the evaluation, plotting, and embedding dependencies. A virtual
environment keeps the commands below isolated.

```bash
python -m venv .env
.env/bin/python -m pip install -r requirements.txt
```

Then generate the complete paper result set:

```bash
.env/bin/python regenerate_results_paper.py --overwrite --workers 4
```

This runs the fixed paper ridge sweeps for the six representation families,
the DFR/AFR/NeuroTune comparison sweeps for the three main datasets, their
paper-specific validation aggregation, and the synthetic gamma sweep. It
writes 25 JSON reports under `results_paper/`. Existing reports are only
replaced when `--overwrite` is supplied.

To also regenerate the synthetic theory-validation report required by the two
optional theory figures, run:

```bash
.env/bin/python regenerate_results_paper.py --overwrite --workers 4 \
  --include-theory-checks
```

You can inspect the exact output plan without running experiments:

```bash
.env/bin/python regenerate_results_paper.py --dry-run
```

To test a full regeneration without replacing the manuscript results, choose
another directory explicitly, for example `--results-dir results_paper_test`.

### 3. Regenerate the manuscript figures

Once the JSON files exist, render every standard PNG used by the manuscript:

```bash
.env/bin/python regenerate_paper_plots.py
```

For the optional theory figures, use the theory report generated in step 2:

```bash
.env/bin/python regenerate_paper_plots.py --add-theory-plots
```

The plotting script checks that the required result reports are present and
that the comparison aggregation uses the paper's restricted AFR gamma grid.

## Data and checkpoints

Large data, checkpoints, embedding bundles, results, and figures are ignored
by Git. They must be downloaded or transferred separately before reproduction.
Place them at the following locations (or pass the equivalent explicit paths
to the extraction/training tools):

| Item | Expected location | Used for |
| --- | --- | --- |
| Custom Waterbirds-95 source pickle | `data/WB/data_WB_95.pkl` | Waterbirds fine-tuning and embedding export |
| CelebA images and metadata | `data/CelebA/` | CelebA fine-tuning and embedding export |
| Cached MultiNLI features and metadata | `data/multiNLI/` | MultiNLI fine-tuning and embedding export |
| Fine-tuned model checkpoints | `models/` | Recreating representation seeds |
| Saved final linear heads | `artifacts/last_layers/` | Warm starts and paper-result provenance |
| NumPy representation bundles | `artifacts/embeddings/` | `regenerate_results_paper.py` |

An embedding bundle contains memory-mappable `X`, `y`, and `g` arrays for the
train, validation, and test splits plus `manifest.json`. The evaluator reads
only these NumPy bundles; it never reads PyTorch checkpoint files at runtime.
Each paper bundle must contain the correct dataset, representation, and seed
named in `regenerate_results_paper.py`.

### Waterbirds protocol

Use the project's custom Waterbirds-95 construction rather than treating the
downloaded benchmark split as final. We create a version in which the target
and background are spuriously correlated in the validation set as well as in
the training set. The resulting `data/WB/data_WB_95.pkl` is the required input
to fine-tuning and embedding export; do not replace it with an unmodified
Waterbirds validation split when reproducing the paper.

The repository does not redistribute datasets or fine-tuned checkpoints.
Download the original [Waterbirds data](https://github.com/kohpangwei/group_DRO/tree/master/dataset_scripts),
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and
[MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) releases, or obtain the
corresponding project-data archive. Follow each source's terms of use.
`gather_embeddings.py --help` lists the supported data-path and checkpoint
overrides for exporting bundles.

## Verification

Run the test suite after changing code or before a full reproduction:

```bash
.env/bin/python -m pytest
```

For a quick configuration check that does not fit models, use the result
regenerator's `--dry-run` option shown above.
