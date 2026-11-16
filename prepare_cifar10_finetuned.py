#!/usr/bin/env python3
"""Convert trusted fine-tuned CIFAR-10 ResNet-50 archives in place.

The incoming archives contain embeddings for all 50,000 official training
images and the unchanged 10,000-image test set. Conversion applies the
project's canonical train/validation membership and replaces each legacy
``.pt`` directory only after every seed has been staged and validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from data import EmbeddingBundle, SplitArrays, sha256_file, write_embedding_bundle


ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = ROOT / "artifacts" / "embeddings" / "CIFAR10" / "resnet50"
DEFAULT_SOURCE_BUNDLE = (
    ROOT / "artifacts" / "embeddings" / "CIFAR10"
    / "resnet50_imagenet_v2" / "default"
)
REQUIRED_KEYS = {"z_train", "y_train", "z_test", "y_test"}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Conversion requires PyTorch in the project environment.") from exc
    return torch


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"{name} may not be an object array.")
    return array


def _load_legacy(directory: Path) -> tuple[int, Path, dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Legacy manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = int(directory.name.removeprefix("seed_"))
    if int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch in {manifest_path}.")
    archive_path = directory / str(manifest.get("embeddings_file", ""))
    if not archive_path.is_file():
        raise FileNotFoundError(f"Legacy archive not found: {archive_path}")
    try:
        archive = _require_torch().load(
            archive_path, map_location="cpu", weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {archive_path}; the copy may be incomplete or corrupt."
        ) from exc
    if not isinstance(archive, dict):
        raise TypeError(f"Expected a dictionary in {archive_path}.")
    missing = sorted(REQUIRED_KEYS.difference(archive))
    if missing:
        raise KeyError(f"{archive_path} is missing arrays: {', '.join(missing)}")
    metadata = archive.get("metadata", {})
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(f"metadata in {archive_path} must be a dictionary.")
    return seed, archive_path, manifest, archive


def _canonical_splits(
    archive: Mapping[str, Any], source_bundle: EmbeddingBundle,
) -> dict[str, SplitArrays]:
    source = source_bundle.manifest["source"]
    train_indices = np.asarray(source["indices"]["train"], dtype=np.int64)
    val_indices = np.asarray(source["indices"]["val"], dtype=np.int64)
    test_indices = np.asarray(source["indices"]["test"], dtype=np.int64)
    z_train = np.asarray(_to_numpy(archive["z_train"], name="z_train"), dtype=np.float32)
    y_train = np.asarray(_to_numpy(archive["y_train"], name="y_train"), dtype=np.int64).reshape(-1)
    z_test = np.asarray(_to_numpy(archive["z_test"], name="z_test"), dtype=np.float32)
    y_test = np.asarray(_to_numpy(archive["y_test"], name="y_test"), dtype=np.int64).reshape(-1)
    if z_train.shape[0] != y_train.size or z_test.shape[0] != y_test.size:
        raise ValueError("Embedding and label row counts differ.")
    expected_train = np.empty_like(y_train)
    expected_train[train_indices] = source_bundle.split("train").y
    expected_train[val_indices] = source_bundle.split("val").y
    np.testing.assert_array_equal(y_train, expected_train)
    np.testing.assert_array_equal(y_test[test_indices], source_bundle.split("test").y)
    return {
        "train": SplitArrays(z_train[train_indices], y_train[train_indices], y_train[train_indices].copy()),
        "val": SplitArrays(z_train[val_indices], y_train[val_indices], y_train[val_indices].copy()),
        "test": SplitArrays(z_test[test_indices], y_test[test_indices], y_test[test_indices].copy()),
    }


def convert_in_place(
    root: str | Path = DEFAULT_ROOT,
    source_bundle_path: str | Path = DEFAULT_SOURCE_BUNDLE,
) -> list[Path]:
    """Validate all legacy seeds, stage standard bundles, then replace them."""
    root = Path(root).expanduser().resolve()
    source_bundle = EmbeddingBundle.load(source_bundle_path, verify_hashes=True)
    if source_bundle.manifest.get("dataset") != "CIFAR10":
        raise ValueError("The canonical source bundle must be CIFAR10.")
    directories = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not directories:
        raise FileNotFoundError(f"No seed directories found under {root}.")

    loaded = [_load_legacy(directory) for directory in directories]
    staged: list[tuple[Path, Path]] = []
    try:
        for directory, (seed, archive_path, legacy_manifest, archive) in zip(directories, loaded):
            stage = root / f".{directory.name}.converted"
            if stage.exists():
                raise FileExistsError(f"Staging directory already exists: {stage}")
            archive_metadata = dict(archive.get("metadata") or {})
            bundle = write_embedding_bundle(
                stage,
                splits=_canonical_splits(archive, source_bundle),
                dataset="CIFAR10",
                representation="resnet50",
                representation_seed=seed,
                source={
                    "format": "trusted_pt",
                    "path": _display_path(archive_path),
                    "sha256": sha256_file(archive_path),
                    "canonical_dataset_source": source_bundle.manifest["source"],
                    "checkpoint": legacy_manifest.get("checkpoint"),
                },
                metadata={
                    **archive_metadata,
                    "task_finetuned": True,
                    "frozen_backbone": False,
                    "randomized_train_val_split": False,
                    "group_definition": "g = y (class reporting only; no spurious attribute)",
                    "source_fingerprint": source_bundle.manifest["metadata"]["source_fingerprint"],
                    "legacy_manifest": legacy_manifest,
                },
            )
            EmbeddingBundle.load(bundle.root, verify_hashes=True)
            staged.append((directory, stage))

        backups: list[Path] = []
        try:
            for directory, stage in staged:
                backup = root / f".{directory.name}.legacy"
                if backup.exists():
                    raise FileExistsError(f"Backup directory already exists: {backup}")
                directory.rename(backup)
                backups.append(backup)
                stage.rename(directory)
                EmbeddingBundle.load(directory, verify_hashes=True)
        except Exception:
            for directory, _ in reversed(staged):
                backup = root / f".{directory.name}.legacy"
                if backup.exists():
                    if directory.exists():
                        shutil.rmtree(directory)
                    backup.rename(directory)
            raise
        for backup in backups:
            shutil.rmtree(backup)
    finally:
        for _, stage in staged:
            if stage.exists():
                shutil.rmtree(stage)
    return directories


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--source-bundle", default=str(DEFAULT_SOURCE_BUNDLE))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for path in convert_in_place(args.root, args.source_bundle):
        print(f"Converted and removed legacy files: {path}")


if __name__ == "__main__":
    main()
