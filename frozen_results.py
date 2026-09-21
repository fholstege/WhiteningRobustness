#!/usr/bin/env python3
"""Ridge sweeps on repeated train/validation splits of pretrained features.

Edit FROZEN_RESULTS_CONFIG, extract the four bundles with gather_embeddings.py,
then run ``python frozen_results.py --workers 2 --blas-threads 1``.
Like finetune_results.py, this reuses the existing transform/ridge sweep and
writes one aggregate-compatible JSON per dataset/backbone. Here replicates
are split seeds, not independently trained backbones. The original test split
is fixed; only original train+validation observations are repartitioned.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from data import EmbeddingBundle, SplitArrays
from evaluate import find_bundle
from finetune_results import (
    FINETUNE_RESULTS_CONFIG, _filename_component, _resolved_root,
    _sweep_variants, _validated_parallelism, _validated_progress_every,
    _validated_whitening_variants,
)
from hyper_param_sweep import _validated_ridge_lambdas, save_sweep


FROZEN_RESULTS_CONFIG: dict[str, Any] = {
    "artifact_root": "artifacts/embeddings",
    "results_dir": "results_frozen",
    "verify_hashes": True,
    "overwrite": True,
    "workers": 1,
    "blas_threads": 1,
    "progress_every": 10,
    "split_seeds": list(range(1, 11)),
    # Retain the fine-tuned ridge path and extend its low end for unit-norm
    # frozen CLIP features. This does not mutate the fine-tuned configuration.
    "sweep_defaults": {
        **deepcopy(FINETUNE_RESULTS_CONFIG["sweep_defaults"]),
        "ridge_lambdas": [
            1e-7, 5e-7, 1e-6, 5e-6,
            *FINETUNE_RESULTS_CONFIG["sweep_defaults"]["ridge_lambdas"],
        ],
    },
    "selection_rule": "balanced_log_loss",
    "experiments": [
        {"dataset": dataset, "embedding": embedding, "class_balanced": True}
        for dataset in ("WB", "CelebA")
        for embedding in ("dinov3_vitb16", "clip_openai_vitb16")
    ],
}


def _validate_pretrained_bundle(bundle: EmbeddingBundle) -> None:
    metadata = bundle.manifest.get("metadata", {})
    if (metadata.get("frozen_backbone") is not True
            or metadata.get("task_finetuned") is not False
            or bundle.manifest.get("representation_seed") is not None
            or metadata.get("randomized_train_val_split", False)):
        raise ValueError(
            f"{bundle.root} must be an original, pretrained-only embedding bundle. "
            "Extract it with gather_embeddings.py using a registered pretrained "
            "checkpoint; fine-tuned or already repartitioned bundles are invalid."
        )


def split_indices(bundle: EmbeddingBundle, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Label-stratified split with the original train/validation sizes; no g."""
    if isinstance(split_seed, bool) or int(split_seed) != split_seed or not 0 <= split_seed < 2**32:
        raise ValueError("split_seed must be an integer in [0, 2**32).")
    y = np.concatenate((bundle.split("train").y, bundle.split("val").y))
    indices = np.arange(y.size, dtype=np.int64)
    try:
        train, val = train_test_split(
            indices, test_size=len(bundle.split("val").y),
            stratify=y, random_state=int(split_seed),
        )
    except ValueError as error:
        raise ValueError(f"Cannot form a label-stratified train/validation split: {error}") from error
    return np.sort(train), np.sort(val)


def resplit_bundle(bundle: EmbeddingBundle, split_seed: int) -> EmbeddingBundle:
    """Construct one reproducible in-memory view; retain the test memory map."""
    _validate_pretrained_bundle(bundle)
    train_indices, val_indices = split_indices(bundle, split_seed)
    index_hash = sha256(train_indices.astype("<i8").tobytes()
                        + val_indices.astype("<i8").tobytes()).hexdigest()
    protocol = {
        "version": 1,
        "split_seed": int(split_seed),
        "source_fingerprint": bundle.fingerprint,
        "indices_sha256": index_hash,
        "pool": "original_train_plus_val",
        "stratification": "target_label_only",
        "test_split": "original_unchanged",
    }
    fingerprint = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    # Materialize one field at a time, rather than retaining a second full
    # embedding pool for the complete ridge sweep. No derived bundles on disk.
    arrays: dict[str, dict[str, np.ndarray]] = {"train": {}, "val": {}}
    for field in ("X", "y", "g"):
        pool = np.concatenate((getattr(bundle.split("train"), field),
                               getattr(bundle.split("val"), field)), axis=0)
        arrays["train"][field] = pool[train_indices]
        arrays["val"][field] = pool[val_indices]
        del pool
    splits = {name: SplitArrays(**values) for name, values in arrays.items()}
    splits["test"] = bundle.split("test")
    manifest = deepcopy(dict(bundle.manifest))
    manifest["metadata"].update(randomized_train_val_split=True, frozen_split=protocol)
    # This manifest is view metadata, not an on-disk array manifest.
    manifest["representation_seed"] = None
    return EmbeddingBundle(bundle.root, manifest, splits, fingerprint)


def _run_split_job(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    with threadpool_limits(limits=int(job["blas_threads"])):
        original = EmbeddingBundle.load(job["bundle_path"], verify_hashes=job["verify_hashes"])
        bundle = resplit_bundle(original, job["split_seed"])
        total = len(job["ridge_lambdas"]) * (
            len(job["baselines"]) + (len(job["variants"]) if job["include_whitening"] else 0)
        )
        completed = 0

        def progress(run):
            nonlocal completed
            completed += 1
            if completed % job["progress_every"] == 0 or completed == total:
                print(f"  split_seed={job['split_seed']} {completed}/{total} "
                      f"method={run['method']} ridge={run['ridge_lambda']:g}", flush=True)

        runs = _sweep_variants(bundle, job, progress)
        protocol = bundle.manifest["metadata"]["frozen_split"]
        for run in runs:
            run["bundle"]["frozen_split"] = protocol
            run["bundle"]["split_seed"] = job["split_seed"]
            run.setdefault("protocol", {}).update(
                replicate_unit="train_validation_split", test_split_fixed=True,
            )
        return runs


def run_frozen_sweeps(config: Mapping[str, Any]) -> list[Path]:
    """Validate all sources, sweep split seeds, and save selected-test summaries."""
    from aggregate import aggregate_runs, format_aggregate

    if config.get("selection_rule") not in {
        "accuracy", "class_balanced_accuracy", "log_loss", "balanced_log_loss",
    }:
        raise ValueError("selection_rule must be a label-only validation metric.")
    workers, blas_threads = _validated_parallelism(config)
    progress_every = _validated_progress_every(config)
    seeds = list(config["split_seeds"])
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError("split_seeds must contain distinct integers in [0, 2**32).")
    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("No frozen experiments selected.")
    prepared = []
    signatures = {}
    destinations = set()
    for experiment in experiments:
        dataset, embedding = experiment["dataset"], experiment["embedding"]
        path = find_bundle(_resolved_root(config["artifact_root"]),
                           dataset=dataset, representation=embedding, seed=None)
        bundle = EmbeddingBundle.load(path, verify_hashes=bool(config["verify_hashes"]))
        _validate_pretrained_bundle(bundle)
        # Matching source identity/order is required for paired comparisons
        # between backbones. Groups are checked here, never used to partition.
        signature = (
            bundle.manifest["metadata"].get("source_fingerprint"),
            tuple(bundle.manifest["splits"][s][f]["sha256"]
                  for s in ("train", "val", "test") for f in ("y", "g")),
        )
        if not signature[0]:
            raise ValueError(f"Missing source_fingerprint in {path}.")
        if dataset in signatures and signatures[dataset] != signature:
            raise ValueError(f"Frozen backbones for {dataset} must share original source/order/splits.")
        signatures[dataset] = signature
        output = _resolved_root(config["results_dir"]) / (
            f"{_filename_component(dataset)}_{_filename_component(embedding)}_frozen.json"
        )
        if output in destinations:
            raise ValueError(f"Duplicate frozen experiment: {dataset}/{embedding}.")
        destinations.add(output)
        if output.exists() and not config.get("overwrite", False):
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite to replace it.")
        settings = {**config["sweep_defaults"], **experiment}
        transforms = list(settings["transforms"])
        if not transforms or set(transforms) - {"identity", "standardize", "whiten"}:
            raise ValueError("transforms must use identity, standardize, and/or whiten.")
        if len(transforms) != len(set(transforms)):
            raise ValueError("transforms must not contain duplicates.")
        base_job = {
            "bundle_path": str(path), "verify_hashes": bool(config["verify_hashes"]),
            "blas_threads": blas_threads, "progress_every": progress_every,
            "ridge_lambdas": _validated_ridge_lambdas(settings["ridge_lambdas"]),
            "baselines": [name for name in transforms if name != "whiten"],
            "include_whitening": "whiten" in transforms,
            "variants": _validated_whitening_variants(settings["whitening_variants"]),
            "class_balanced": bool(settings["class_balanced"]),
            "whitening_relative_tolerance": settings["whitening_relative_tolerance"],
            "warm_start": False,
        }
        prepared.append((experiment, output, base_job))
        del bundle

    outputs = []
    for experiment, output, base_job in prepared:
        jobs = [dict(base_job, split_seed=seed) for seed in seeds]
        print(f"{experiment['dataset']}/{experiment['embedding']}: "
              f"{len(seeds)} matched train/validation splits", flush=True)
        if workers == 1:
            results = [_run_split_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
                results = list(executor.map(_run_split_job, jobs))
        runs = [run for rows in results for run in rows]
        failed = sum(run["status"] != "complete" for run in runs)
        summary = aggregate_runs(
            runs, default_selection_rule=config["selection_rule"],
            refit_train_val=False, confidence=0.95,
        )
        summary["replicate_unit"] = "train_validation_split"
        summary["uncertainty_scope"] = "split variability conditional on a fixed pretrained checkpoint and test set"
        report = {
            "schema_version": 1, "kind": "ridge_lambda_sweep",
            "status": "partial" if failed else "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {**base_job, **experiment, "split_seeds": seeds,
                       "workers": workers, "selection_rule": config["selection_rule"],
                       "replicate_unit": "train_validation_split", "model_choices": ["ridge_lambda"]},
            "run_count": len(runs), "failed_run_count": failed,
            "runs": runs, "selected_results": summary,
        }
        outputs.append(save_sweep(report, output=output, results_dir=config["results_dir"],
                                  overwrite=bool(config.get("overwrite", False))))
        formatted = format_aggregate(summary).replace(
            "For repeated representations,", "For repeated train/validation splits,"
        ).replace("matched representation seeds.", "matched split seeds.")
        print(formatted, flush=True)
        print("Replicates above are split seeds; the pretrained checkpoint and test set are fixed.", flush=True)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", help="Filter configured dataset: WB or CelebA.")
    parser.add_argument("--embedding", help="Filter configured representation.")
    parser.add_argument("--split-seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--blas-threads", type=int)
    parser.add_argument("--progress-every", type=int)
    parser.add_argument("--artifact-root")
    parser.add_argument("--results-dir")
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = deepcopy(FROZEN_RESULTS_CONFIG)
    for key in ("split_seeds", "workers", "blas_threads", "progress_every",
                "artifact_root", "results_dir", "overwrite"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config["experiments"] = [entry for entry in config["experiments"]
                             if (args.dataset is None or entry["dataset"].lower() == args.dataset.lower())
                             and (args.embedding is None or entry["embedding"] == args.embedding)]
    for path in run_frozen_sweeps(config):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
