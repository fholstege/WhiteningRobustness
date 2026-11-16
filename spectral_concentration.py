#!/usr/bin/env python3
"""Measure core-orthogonalized spurious concentration along representation PCs.

The analysis is fit on the complete, original (unwhitened) training
representation without resampling. A balanced core direction (or MultiNLI
subspace) is removed before spurious conditional means are formed. Groups
receive equal formal weight regardless of sample size.

Examples
--------
Analyze one of this repository's embedding bundles::

    python spectral_concentration.py \
        --dataset waterbirds \
        --bundle artifacts/embeddings/WB/resnet50/seed_001 \
        --output-dir results/spectral_concentration/waterbirds

Analyze standalone NumPy arrays::

    python spectral_concentration.py --dataset multinli \
        --x X_train.npy --y y_train.npy --a a_train.npy \
        --output-dir results/spectral_concentration/multinli
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASET_ALIASES = {
    "waterbirds": "waterbirds",
    "waterbird": "waterbirds",
    "wb": "waterbirds",
    "celeba": "celeba",
    "celeb_a": "celeba",
    "multinli": "multinli",
    "multi_nli": "multinli",
    "mnli": "multinli",
}

DIRECTION_REPRESENTATIONS = ("spurious",)
PLOT_TOP_COMPONENTS = 500
DEFAULT_BATCH_SIZE = 2048


def _normalize_dataset(value: str) -> str:
    try:
        return DATASET_ALIASES[value.strip().lower()]
    except KeyError as exc:
        choices = "waterbirds, celeba, multinli"
        raise ValueError(f"Unknown dataset {value!r}; choose from {choices}.") from exc


def _safe_manifest_array(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest array filenames must be non-empty strings.")
    path = (root / value).resolve()
    if path.parent != root:
        raise ValueError(f"Manifest array path escapes its bundle: {value!r}.")
    return path


def load_bundle_split(
    bundle: Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, and recover a from one repository embedding-bundle split."""
    root = bundle.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_spec = manifest.get("splits", {}).get(split)
    if not isinstance(split_spec, dict):
        raise ValueError(f"Manifest has no valid {split!r} split.")
    paths = {}
    for name in ("X", "y", "g"):
        spec = split_spec.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"Manifest {split!r} split has no {name!r} array.")
        paths[name] = _safe_manifest_array(root, spec.get("file"))
    X = np.load(paths["X"], mmap_mode="r", allow_pickle=False)
    y = np.load(paths["y"], mmap_mode="r", allow_pickle=False)
    groups = np.load(paths["g"], mmap_mode="r", allow_pickle=False)
    if not np.array_equal(np.asarray(groups), 2 * np.asarray(y) + groups % 2):
        raise ValueError("Bundle groups do not follow g = 2*y + a.")
    return X, y, np.asarray(groups % 2)


def load_training_data(
    *,
    bundle: Path | None = None,
    x_path: Path | None = None,
    y_path: Path | None = None,
    a_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the full training split from a bundle or three standalone arrays."""
    if bundle is not None:
        if any(path is not None for path in (x_path, y_path, a_path)):
            raise ValueError("Use either --bundle or --x/--y/--a, not both.")
        root = bundle.expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        X, y, a = load_bundle_split(root, "train")
        source = {
            "kind": "embedding_bundle",
            "path": str(root),
            "split": "train",
            "manifest_dataset": manifest.get("dataset"),
            "representation": manifest.get("representation"),
        }
        return X, y, a, source

    if any(path is None for path in (x_path, y_path, a_path)):
        raise ValueError("Provide --bundle or all of --x, --y, and --a.")
    assert x_path is not None and y_path is not None and a_path is not None
    X = np.load(x_path.expanduser().resolve(), mmap_mode="r", allow_pickle=False)
    y = np.load(y_path.expanduser().resolve(), mmap_mode="r", allow_pickle=False)
    a = np.load(a_path.expanduser().resolve(), mmap_mode="r", allow_pickle=False)
    source = {
        "kind": "npy_arrays",
        "X": str(x_path.expanduser().resolve()),
        "y": str(y_path.expanduser().resolve()),
        "a": str(a_path.expanduser().resolve()),
        "split": "train",
    }
    return X, y, a, source


def _validate_inputs(
    X: np.ndarray, y: np.ndarray, a: np.ndarray, dataset: str
) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must have shape (n, d) with n >= 2.")
    y = np.asarray(y).reshape(-1)
    a = np.asarray(a).reshape(-1)
    if len(X) != y.size or y.size != a.size:
        raise ValueError("X, y, and a have incompatible numbers of observations.")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")
    expected_y = np.arange(3 if dataset == "multinli" else 2)
    if not np.array_equal(np.unique(y), expected_y):
        raise ValueError(f"{dataset} target labels must be {expected_y.tolist()}.")
    if not np.array_equal(np.unique(a), np.array([0, 1])):
        raise ValueError("Spurious attribute labels must be binary 0/1.")
    for target in expected_y:
        for attribute in (0, 1):
            if not np.any((y == target) & (a == attribute)):
                raise ValueError(f"Training cell (y={target}, a={attribute}) is empty.")
    return y.astype(np.int64, copy=False), a.astype(np.int64, copy=False)


def _orthonormal_basis(vectors: np.ndarray) -> np.ndarray:
    """Return a numerical orthonormal basis for column vectors."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    left, singular_values, _ = np.linalg.svd(vectors, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        raise ValueError("A requested conditional-mean direction is zero.")
    tolerance = max(vectors.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == 0:
        raise ValueError("A requested conditional-mean direction has numerical rank zero.")
    return left[:, :rank]


def _cell_feature_means(
    X: np.ndarray, y: np.ndarray, a: np.ndarray, classes: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate group means without selecting a potentially huge group matrix."""
    sums = np.zeros((classes.size, 2, X.shape[1]), dtype=np.float64)
    counts = np.zeros((classes.size, 2), dtype=np.int64)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        values = np.asarray(X[start:end], dtype=np.float64)
        batch_y, batch_a = y[start:end], a[start:end]
        for target in classes:
            for attribute in (0, 1):
                mask = (batch_y == target) & (batch_a == attribute)
                if np.any(mask):
                    sums[target, attribute] += values[mask].sum(axis=0)
                    counts[target, attribute] += int(mask.sum())
    return sums / counts[:, :, None], counts


def _project_in_batches(
    X: np.ndarray, center: np.ndarray, basis: np.ndarray, batch_size: int
) -> np.ndarray:
    result = np.empty((len(X), basis.shape[1]), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        values = np.asarray(X[start:end], dtype=np.float64)
        result[start:end] = (values - center) @ basis
    return result


def _core_basis_from_cell_means(
    cell_means: np.ndarray, dataset: str
) -> np.ndarray:
    """Balanced binary core direction or MultiNLI class-centroid subspace."""
    if dataset == "multinli":
        class_centroids = cell_means.mean(axis=1)
        vectors = (class_centroids - class_centroids.mean(axis=0)).T
    else:
        vectors = (cell_means[1] - cell_means[0]).mean(axis=0)
    return _orthonormal_basis(vectors)


def _remove_core_from_contrasts(
    contrasts: np.ndarray, core_basis: np.ndarray
) -> np.ndarray:
    return contrasts - (contrasts @ core_basis) @ core_basis.T


def _classification_metrics(
    features: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    *,
    evaluation_features: np.ndarray | None = None,
    evaluation_y: np.ndarray | None = None,
    evaluation_a: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit unweighted heads and report y/a accuracy and y WGA."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            "Direction diagnostics require scikit-learn from requirements.txt."
        ) from exc

    features = np.asarray(features, dtype=np.float64)
    feature_mean = features.mean(axis=0)
    scale = features.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    standardized = (features - feature_mean) / scale
    y_head = LogisticRegression(max_iter=2000, solver="lbfgs")
    a_head = LogisticRegression(max_iter=2000, solver="lbfgs")
    y_head.fit(standardized, y)
    a_head.fit(standardized, a)
    if evaluation_features is None:
        evaluation_features, evaluation_y, evaluation_a = features, y, a
    assert evaluation_y is not None and evaluation_a is not None
    evaluation_standardized = (np.asarray(evaluation_features) - feature_mean) / scale
    y_prediction = y_head.predict(evaluation_standardized)
    a_prediction = a_head.predict(evaluation_standardized)
    group_accuracies = []
    for target in np.unique(evaluation_y):
        for attribute in np.unique(evaluation_a):
            mask = (evaluation_y == target) & (evaluation_a == attribute)
            group_accuracies.append(float(np.mean(y_prediction[mask] == evaluation_y[mask])))
    y_balanced = np.mean([
        np.mean(y_prediction[evaluation_y == value] == value)
        for value in np.unique(evaluation_y)
    ])
    a_balanced = np.mean([
        np.mean(a_prediction[evaluation_a == value] == value)
        for value in np.unique(evaluation_a)
    ])
    return {
        "y_accuracy": float(np.mean(y_prediction == evaluation_y)),
        "y_balanced_accuracy": float(y_balanced),
        "a_accuracy": float(np.mean(a_prediction == evaluation_a)),
        "a_balanced_accuracy": float(a_balanced),
        "y_worst_group_accuracy": min(group_accuracies),
    }


def conditional_mean_direction_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    *,
    dataset: str,
    evaluation_X: np.ndarray | None = None,
    evaluation_y: np.ndarray | None = None,
    evaluation_a: np.ndarray | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Evaluate spurious means after removing the balanced core subspace."""
    classes = np.arange(3 if dataset == "multinli" else 2)
    cell_means, _ = _cell_feature_means(X, y, a, classes, batch_size)
    core_basis = _core_basis_from_cell_means(cell_means, dataset)
    contrasts = cell_means[:, 1] - cell_means[:, 0]
    residual_contrasts = _remove_core_from_contrasts(contrasts, core_basis)
    spurious_vector = residual_contrasts.mean(axis=0)
    spurious_basis = _orthonormal_basis(spurious_vector)
    center = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, len(X), batch_size):
        center += np.asarray(X[start:start + batch_size], dtype=np.float64).sum(axis=0)
    center /= len(X)
    train_features = _project_in_batches(X, center, spurious_basis, batch_size)
    evaluation_features = None
    if evaluation_X is not None:
        evaluation_features = _project_in_batches(
            evaluation_X, center, spurious_basis, batch_size
        )
    return {
        "spurious": {
            "dimension": int(spurious_basis.shape[1]),
            **_classification_metrics(
                train_features, y, a, evaluation_features=evaluation_features,
                evaluation_y=evaluation_y, evaluation_a=evaluation_a,
            ),
        },
    }


def compute_spectral_concentration(
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    *,
    dataset: str,
    evaluation_X: np.ndarray | None = None,
    evaluation_y: np.ndarray | None = None,
    evaluation_a: np.ndarray | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, np.ndarray | float | int]:
    """Compute raw spurious concentration with bounded working memory."""
    dataset = _normalize_dataset(dataset)
    y, a = _validate_inputs(X, y, a, dataset)
    classes = np.arange(3 if dataset == "multinli" else 2)
    cell_means, counts = _cell_feature_means(X, y, a, classes, batch_size)

    center = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, len(X), batch_size):
        center += np.asarray(X[start:start + batch_size], dtype=np.float64).sum(axis=0)
    center /= len(X)
    covariance = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        values = np.asarray(X[start:start + batch_size], dtype=np.float64) - center
        covariance += values.T @ values
    covariance /= len(X) - 1

    try:
        from scipy.linalg import eigh
    except ImportError as exc:
        raise RuntimeError("Top-500 PCA requires scipy from requirements.txt.") from exc
    components = min(PLOT_TOP_COMPONENTS, X.shape[1], len(X) - 1)
    eigenvalues, eigenvectors = eigh(
        covariance,
        subset_by_index=(X.shape[1] - components, X.shape[1] - 1),
        check_finite=False,
        overwrite_a=True,
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    contrasts = cell_means[:, 1] - cell_means[:, 0]
    core_basis = _core_basis_from_cell_means(cell_means, dataset)
    residual_contrasts = _remove_core_from_contrasts(contrasts, core_basis)
    projected_contrasts = residual_contrasts @ eigenvectors
    if dataset == "multinli":
        weighted_spurious = 0.5 * np.mean(projected_contrasts ** 2, axis=0)
        total_spurious = 0.5 * float(np.mean(np.sum(residual_contrasts ** 2, axis=1)))
    else:
        weighted_spurious = 0.5 * np.sum(projected_contrasts ** 2, axis=0)
        total_spurious = 0.5 * float(np.sum(residual_contrasts ** 2))
    if total_spurious <= 0.0:
        raise ValueError("Total spurious signal must be positive.")
    cumulative_spurious = np.cumsum(weighted_spurious) / total_spurious
    spurious = weighted_spurious / eigenvalues

    direction_diagnostics = conditional_mean_direction_diagnostics(
        X, y, a, dataset=dataset, evaluation_X=evaluation_X,
        evaluation_y=evaluation_y, evaluation_a=evaluation_a, batch_size=batch_size,
    )

    return {
        "center": center,
        "eigenvalues": eigenvalues,
        "spurious_scores": spurious,
        "weighted_spurious_signal": weighted_spurious,
        "cumulative_spurious": cumulative_spurious,
        "fraction_components": np.arange(1, components + 1, dtype=np.float64) / components,
        "conditional_feature_means": cell_means,
        "group_counts": counts,
        "rank": components,
        "total_spurious_signal": total_spurious,
        "direction_diagnostics": direction_diagnostics,
        "direction_diagnostics_evaluation": (
            "held-out split" if evaluation_X is not None else "training split (in-sample)"
        ),
    }


def _plot_results(results: dict[str, Any], dataset: str, output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = min(PLOT_TOP_COMPONENTS, int(results["rank"]))
    x = np.arange(1, limit + 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(x, results["cumulative_spurious"][:limit], label="Spurious", linewidth=2.4)
    ax.set(xlim=(1, limit), ylim=(0, 1), xlabel="Number of top eigenvalues included",
           ylabel="Cumulative signal concentration",
           title=f"Spectral signal concentration — {dataset}")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_results(
    results: dict[str, Any], *, dataset: str, source: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in results.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(output_dir / "spectral_concentration.npz", **arrays)

    rank = int(results["rank"])
    with (output_dir / "spectral_concentration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "component", "fraction_components", "eigenvalue", "spurious_score",
            "weighted_spurious_signal", "cumulative_spurious",
        ])
        for j in range(rank):
            writer.writerow([
                j + 1, results["fraction_components"][j], results["eigenvalues"][j],
                results["spurious_scores"][j], results["weighted_spurious_signal"][j],
                results["cumulative_spurious"][j],
            ])

    metadata = {
        "dataset": dataset,
        "source": source,
        "split": "train",
        "observations": int(np.asarray(results["group_counts"]).sum()),
        "computed_leading_components": rank,
        "group_counts": np.asarray(results["group_counts"]).tolist(),
        "resampling": "none",
        "group_weighting": "equal in score definitions",
        "spurious_definition": (
            "equal-weight within-y conditional-mean contrasts after projecting out "
            "the balanced core direction/subspace"
        ),
        "core_direction_computed_only_for_residualization": True,
        "memory_strategy": "memory-mapped X; two-pass batched covariance; top-500 eigensystem",
        "direction_diagnostics": results["direction_diagnostics"],
        "direction_diagnostics_protocol": (
            "Unweighted logistic heads fit on the full training set and evaluated on "
            f"the {results['direction_diagnostics_evaluation']}; WGA is y accuracy "
            "minimized over joint (y,a) groups."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_results(results, dataset, output_dir / "spectral_concentration")
    with (output_dir / "direction_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "representation", "dimension", "y_accuracy", "y_balanced_accuracy",
            "a_accuracy", "a_balanced_accuracy", "y_worst_group_accuracy",
        ])
        for name in DIRECTION_REPRESENTATIONS:
            values = results["direction_diagnostics"][name]
            writer.writerow([
                name, values["dimension"], values["y_accuracy"],
                values["y_balanced_accuracy"], values["a_accuracy"],
                values["a_balanced_accuracy"], values["y_worst_group_accuracy"],
            ])


def print_checkpoints(results: dict[str, Any]) -> None:
    rank = int(results["rank"])
    print(f"Leading covariance components computed: {rank}")
    k = min(PLOT_TOP_COMPONENTS, rank)
    fs = float(results["cumulative_spurious"][k - 1])
    print(f"Top-{k} spurious concentration: F_s={fs:.6f}")
    diagnostics = results.get("direction_diagnostics")
    if diagnostics is not None:
        evaluation = results.get("direction_diagnostics_evaluation", "seed evaluation split")
        print(f"Conditional-mean direction diagnostics ({evaluation}):")
        for name in DIRECTION_REPRESENTATIONS:
            values = diagnostics[name]
            print(
                f"  {name:8s} (dim={values['dimension']}): "
                f"y acc={values['y_accuracy']:.6f}, "
                f"y bal acc={values['y_balanced_accuracy']:.6f}, "
                f"a acc={values['a_accuracy']:.6f}, "
                f"a bal acc={values['a_balanced_accuracy']:.6f}, "
                f"y WGA={values['y_worst_group_accuracy']:.6f}"
            )


def discover_seed_bundles(bundle: Path) -> list[Path]:
    """Find valid seed_* bundles beside the supplied bundle."""
    resolved = bundle.expanduser().resolve()
    parent = resolved.parent if resolved.name.startswith("seed_") else resolved
    candidates = sorted(
        path for path in parent.glob("seed_*") if (path / "manifest.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No seed_* embedding bundles found in {parent}.")
    return candidates


def aggregate_seed_results(
    seed_results: list[tuple[str, dict[str, Any]]], output_dir: Path, dataset: str
) -> dict[str, Any]:
    """Interpolate seed curves to a common fraction grid and save mean/std."""
    if not seed_results:
        raise ValueError("At least one seed result is required.")
    grid_size = max(int(result["rank"]) for _, result in seed_results)
    grid = np.arange(1, grid_size + 1, dtype=np.float64) / grid_size
    spurious_curves = []
    for _, result in seed_results:
        x = np.asarray(result["fraction_components"])
        # Include the mathematical origin so interpolation below the first PC
        # does not incorrectly hold the first observed concentration constant.
        source_x = np.concatenate(([0.0], x))
        spurious_curves.append(np.interp(
            grid, source_x, np.concatenate(([0.0], result["cumulative_spurious"]))
        ))
    spurious_curves = np.asarray(spurious_curves)
    aggregate = {
        "fraction_components": grid,
        "cumulative_spurious": spurious_curves.mean(axis=0),
        "cumulative_spurious_std": spurious_curves.std(axis=0, ddof=0),
        "per_seed_cumulative_spurious": spurious_curves,
        "rank": grid_size,
    }
    diagnostic_summary: dict[str, dict[str, float | int]] = {}
    for representation in DIRECTION_REPRESENTATIONS:
        values = [result["direction_diagnostics"][representation] for _, result in seed_results]
        diagnostic_summary[representation] = {
            "dimension": int(values[0]["dimension"]),
            "y_accuracy": float(np.mean([value["y_accuracy"] for value in values])),
            "y_accuracy_std": float(np.std([value["y_accuracy"] for value in values])),
            "y_balanced_accuracy": float(np.mean([
                value["y_balanced_accuracy"] for value in values
            ])),
            "y_balanced_accuracy_std": float(np.std([
                value["y_balanced_accuracy"] for value in values
            ])),
            "a_accuracy": float(np.mean([value["a_accuracy"] for value in values])),
            "a_accuracy_std": float(np.std([value["a_accuracy"] for value in values])),
            "a_balanced_accuracy": float(np.mean([
                value["a_balanced_accuracy"] for value in values
            ])),
            "a_balanced_accuracy_std": float(np.std([
                value["a_balanced_accuracy"] for value in values
            ])),
            "y_worst_group_accuracy": float(np.mean([
                value["y_worst_group_accuracy"] for value in values
            ])),
            "y_worst_group_accuracy_std": float(np.std([
                value["y_worst_group_accuracy"] for value in values
            ])),
        }
        if "retained_spurious_contrast_norm_fraction" in values[0]:
            retained = [
                value["retained_spurious_contrast_norm_fraction"] for value in values
            ]
            diagnostic_summary[representation][
                "retained_spurious_contrast_norm_fraction"
            ] = float(np.mean(retained))
            diagnostic_summary[representation][
                "retained_spurious_contrast_norm_fraction_std"
            ] = float(np.std(retained))
    aggregate["direction_diagnostics"] = diagnostic_summary
    aggregate["direction_diagnostics_evaluation"] = "test split (held out)"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "spectral_concentration_aggregate.npz", **{
        key: value for key, value in aggregate.items() if isinstance(value, np.ndarray)
    })
    with (output_dir / "spectral_concentration_aggregate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "component", "mean_cumulative_spurious", "std_cumulative_spurious",
        ])
        for index in range(grid_size):
            writer.writerow([
                index + 1, aggregate["cumulative_spurious"][index],
                aggregate["cumulative_spurious_std"][index],
            ])
    metadata = {
        "dataset": dataset,
        "aggregation": "pointwise arithmetic mean across seed curves",
        "variability": "population standard deviation across seeds",
        "common_grid": "absolute leading-component index 1..500",
        "seeds": [name for name, _ in seed_results],
        "seed_ranks": {name: int(result["rank"]) for name, result in seed_results},
        "direction_diagnostics": diagnostic_summary,
    }
    (output_dir / "metadata_aggregate.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_aggregate(aggregate, dataset, output_dir / "spectral_concentration_aggregate")
    with (output_dir / "direction_diagnostics_aggregate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "representation", "dimension", "mean_y_accuracy", "std_y_accuracy",
            "mean_y_balanced_accuracy", "std_y_balanced_accuracy",
            "mean_a_accuracy", "std_a_accuracy", "mean_a_balanced_accuracy",
            "std_a_balanced_accuracy", "mean_y_worst_group_accuracy",
            "std_y_worst_group_accuracy",
        ])
        for name in DIRECTION_REPRESENTATIONS:
            values = diagnostic_summary[name]
            writer.writerow([
                name, values["dimension"], values["y_accuracy"], values["y_accuracy_std"],
                values["y_balanced_accuracy"], values["y_balanced_accuracy_std"],
                values["a_accuracy"], values["a_accuracy_std"],
                values["a_balanced_accuracy"], values["a_balanced_accuracy_std"],
                values["y_worst_group_accuracy"], values["y_worst_group_accuracy_std"],
            ])
    return aggregate


def _plot_aggregate(results: dict[str, Any], dataset: str, output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = min(PLOT_TOP_COMPONENTS, int(results["rank"]))
    x = np.arange(1, limit + 1)
    fs = results["cumulative_spurious"][:limit]
    ss = results["cumulative_spurious_std"][:limit]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    line_s, = ax.plot(x, fs, label="Spurious (seed mean)", linewidth=2.2)
    ax.fill_between(x, np.maximum(0, fs - ss), np.minimum(1, fs + ss),
                    color=line_s.get_color(), alpha=0.18, linewidth=0)
    ax.set(xlim=(1, limit), ylim=(0, 1),
           xlabel="Number of top eigenvalues included",
           ylabel="Cumulative signal concentration",
           title=f"Spectral signal concentration — {dataset} (across seeds)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spurious_dataset_comparison(
    aggregate_files: dict[str, Path], output_stem: Path
) -> dict[str, float]:
    """Plot top-500 core-orthogonalized spurious concentration by dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Waterbirds": "#39c77a", "CelebA": "#9163c2", "MultiNLI": "#3db7c5"}
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    top_values: dict[str, float] = {}
    for label, path in aggregate_files.items():
        with np.load(path, allow_pickle=False) as stored:
            curve = np.asarray(stored["cumulative_spurious"])
        limit = min(PLOT_TOP_COMPONENTS, curve.size)
        x = np.arange(1, limit + 1)
        ax.plot(x, curve[:limit], color=colors.get(label), linewidth=2.4, label=label)
        top_values[label] = float(curve[limit - 1])
    ax.set(
        xlim=(1, PLOT_TOP_COMPONENTS), ylim=(0, 1),
        xlabel="Number of top eigenvalues included",
        ylabel="Cumulative spurious signal concentration",
    )
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return top_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="waterbirds, celeba, or multinli")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="Embedding bundle; uses its train split")
    source.add_argument("--x", type=Path, help="Training feature matrix in .npy format")
    parser.add_argument("--y", type=Path, help="Training target labels in .npy format")
    parser.add_argument("--a", type=Path, help="Training spurious labels in .npy format")
    parser.add_argument(
        "--all-seeds", action="store_true",
        help="Analyze every seed_* bundle beside --bundle and aggregate the curves",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = _normalize_dataset(args.dataset)
    output_dir = args.output_dir.expanduser().resolve()
    if args.all_seeds:
        if args.bundle is None:
            raise ValueError("--all-seeds requires --bundle.")
        seed_results = []
        bundles = discover_seed_bundles(args.bundle)
        print(f"Discovered {len(bundles)} seed bundles in {bundles[0].parent}")
        for bundle in bundles:
            print(f"\nAnalyzing {bundle.name} ...", flush=True)
            X, y, a, source = load_training_data(bundle=bundle)
            test_X, test_y, test_a = load_bundle_split(bundle, "test")
            result = compute_spectral_concentration(
                X, y, a, dataset=dataset, evaluation_X=test_X,
                evaluation_y=test_y, evaluation_a=test_a,
            )
            save_results(
                result, dataset=dataset, source=source, output_dir=output_dir / bundle.name
            )
            print_checkpoints(result)
            seed_results.append((bundle.name, result))
        aggregate = aggregate_seed_results(seed_results, output_dir, dataset)
        print("\nAcross-seed mean:")
        print_checkpoints(aggregate)
        print(f"Saved per-seed and aggregate results to {output_dir}")
        return
    X, y, a, source = load_training_data(
        bundle=args.bundle, x_path=args.x, y_path=args.y, a_path=args.a
    )
    evaluation = (
        load_bundle_split(args.bundle, "test") if args.bundle is not None else (None, None, None)
    )
    results = compute_spectral_concentration(
        X, y, a, dataset=dataset, evaluation_X=evaluation[0],
        evaluation_y=evaluation[1], evaluation_a=evaluation[2],
    )
    save_results(
        results, dataset=dataset, source=source, output_dir=output_dir
    )
    print_checkpoints(results)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
