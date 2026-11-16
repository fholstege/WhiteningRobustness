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

Use the paper’s fine-tuned checkpoints and export fixed train, validation,
and test embedding bundles with `gather_embeddings.py` or the original
training code. The result regenerator takes these bundles as inputs and
re-fits the linear probes. See [the control experiments](#control-experiments)
for their additional representation bundles.

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

This reproduces the experiments in `writing/iclr2027/main_v2.tex`: the
six benchmark representation families, the DFR/AFR/NeuronTune comparisons,
the two controls, and the main and appendix synthetic sweeps. It writes
30 JSON reports under `results_paper/`. Pass `--overwrite` to replace an
existing report.

To rerun only the controls or only the synthetic experiments:

```bash
.env/bin/python regenerate_results_paper.py --section controls --workers 4
.env/bin/python regenerate_results_paper.py --section synthetic --workers 4
```

The synthetic reports use `gamma=5` for the main experiment and
`gamma=0.5, 1, 3` for the appendix, with 100 matched simulations per
condition. Both use the same sample size and random seeds.

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

## Control experiments

Use ImageNet-pretrained ResNet-50 backbones fine-tuned on the corresponding
task. The regenerator expects these NumPy bundles:

| Control | Bundle directory | Representation seeds | Validation selection |
| --- | --- | --- | --- |
| Waterbirds without background correlation | `artifacts/embeddings/WB_no_spurious/resnet50/` | 1–5 | Class-balanced accuracy |
| CIFAR-10 | `artifacts/embeddings/CIFAR10/resnet50/` | 0–2 | Balanced log loss |

Use the original decorrelated Waterbirds split, where background is
independent of class in training. For CIFAR-10, use the fixed seed-0,
stratified 45,000/5,000 training/validation split of the official training
set and the official 10,000-image test set. The split seed is independent
of the backbone seeds. Set the CIFAR-10 evaluation group IDs equal to the
class labels.

If you have the original trusted PyTorch embedding exports, convert them
before running the regenerator:

```bash
.env/bin/python prepare_wb_no_spurious.py
.env/bin/python prepare_cifar10_finetuned.py
```

The Waterbirds converter reads archives under
`artifacts/embeddings/WB/resnet50_no_spurious/`. The CIFAR-10 converter
expects the legacy seed directories at its bundle directory above and
replaces them in place after validating all seeds. It uses the split indices
from `artifacts/embeddings/CIFAR10/resnet50_imagenet_v2/default/`; create that
reference bundle with `gather_embeddings.py --dataset cifar10` if needed.
The reference bundle supplies only the split and label order.

`--section controls` writes one sweep and one selected result per control.
The plot uses paired differences within representation seed, with 95%
confidence intervals. To render it on its own:

```bash
.env/bin/python viz.py --plot-type no-spurious --overwrite
```

If you regenerated results in a separate directory, give that directory to
`viz.py --control-results-dir`.

## Verification

Run the test suite after changing code or before a full reproduction:

```bash
.env/bin/python -m pytest
```

For a quick configuration check that does not fit models, use the result
regenerator's `--dry-run` option shown above.
