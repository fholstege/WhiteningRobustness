#!/usr/bin/env python3
"""Convert trusted no-spurious Waterbirds ``.pt`` archives to NumPy bundles.

The source archives are retained unchanged. Converted bundles use the schema
consumed by ``evaluate.py``, ``hyper_param_sweep.py``, and
``finetune_results.py``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data import (
    SPLITS,
    EmbeddingBundle,
    SplitArrays,
    sha256_file,
    write_embedding_bundle,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = (
    ROOT / "artifacts" / "embeddings" / "WB" / "resnet50_no_spurious"
)
DEFAULT_DESTINATION_ROOT = (
    ROOT / "artifacts" / "embeddings" / "WB_no_spurious" / "resnet50"
)
REQUIRED_ARRAY_KEYS = {
    f"{name}_{split}"
    for split in SPLITS
    for name in ("z", "y", "g")
}


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Converting trusted .pt archives requires PyTorch. Run this script "
            "with the project's embedding environment."
        ) from exc
    return torch


def _seed_directories(source_root: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for directory in sorted(source_root.glob("seed_*")):
        if not directory.is_dir():
            continue
        try:
            seed = int(directory.name.removeprefix("seed_"))
        except ValueError as exc:
            raise ValueError(f"Invalid seed directory name: {directory}") from exc
        found.append((seed, directory))
    if not found:
        raise FileNotFoundError(f"No seed directories found under {source_root}.")
    return found


def _load_archive(
    seed: int, directory: Path
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")
    server_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(server_manifest.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch in {manifest_path}.")
    if server_manifest.get("no_spurious") is not True:
        raise ValueError(f"Expected no_spurious=true in {manifest_path}.")
    archive_path = directory / str(server_manifest["embeddings_file"])
    if not archive_path.is_file():
        raise FileNotFoundError(f"Embedding archive not found: {archive_path}")

    torch = _require_torch()
    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    if not isinstance(archive, dict):
        raise TypeError(f"Expected a dictionary in {archive_path}.")
    missing = sorted(REQUIRED_ARRAY_KEYS.difference(archive))
    if missing:
        raise KeyError(f"{archive_path} is missing arrays: {', '.join(missing)}")
    return archive_path, server_manifest, archive


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"{name} may not be an object array.")
    return array


def _normalized_groups(value: Any, *, name: str) -> tuple[np.ndarray, int]:
    groups = np.asarray(_to_numpy(value, name=name), dtype=np.int64).reshape(-1)
    unique = np.unique(groups)
    if unique.size == 0 or unique[0] not in (0, 1):
        raise ValueError(f"{name} group IDs must start at zero or one.")
    expected = np.arange(unique[0], unique[-1] + 1)
    if not np.array_equal(unique, expected):
        raise ValueError(f"{name} group IDs must be contiguous.")
    offset = int(unique[0])
    return groups - offset, offset


def _split_arrays(
    archive: Mapping[str, Any]
) -> tuple[dict[str, SplitArrays], int]:
    splits: dict[str, SplitArrays] = {}
    offsets: set[int] = set()
    for split in SPLITS:
        groups, offset = _normalized_groups(
            archive[f"g_{split}"], name=f"g_{split}"
        )
        offsets.add(offset)
        splits[split] = SplitArrays(
            X=np.asarray(
                _to_numpy(archive[f"z_{split}"], name=f"z_{split}"),
                dtype=np.float32,
            ),
            y=np.asarray(
                _to_numpy(archive[f"y_{split}"], name=f"y_{split}"),
                dtype=np.int64,
            ).reshape(-1),
            g=groups,
        )
    if len(offsets) != 1:
        raise ValueError("Group-ID offsets differ across splits.")
    return splits, offsets.pop()


def _fixed_split_signature(splits: Mapping[str, SplitArrays]) -> tuple[str, ...]:
    return tuple(
        sha256(np.ascontiguousarray(getattr(splits[split], field)).tobytes()).hexdigest()
        for split in ("train", "val")
        for field in ("y", "g")
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def convert_archives(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    destination_root: str | Path = DEFAULT_DESTINATION_ROOT,
) -> list[Path]:
    """Validate all seeds, then convert them without changing the sources."""
    source_root = Path(source_root).expanduser().resolve()
    destination_root = Path(destination_root).expanduser().resolve()
    seeds = _seed_directories(source_root)

    expected_signature: tuple[str, ...] | None = None
    for seed, directory in seeds:
        _, _, archive = _load_archive(seed, directory)
        splits, _ = _split_arrays(archive)
        signature = _fixed_split_signature(splits)
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ValueError(
                "No-spurious representation seeds do not share the fixed "
                f"train/validation split; mismatch at {directory}."
            )

    outputs: list[Path] = []
    for seed, directory in seeds:
        archive_path, server_manifest, archive = _load_archive(seed, directory)
        splits, group_offset = _split_arrays(archive)
        archive_metadata = archive.get("metadata", {})
        if archive_metadata is not None and not isinstance(
            archive_metadata, Mapping
        ):
            raise TypeError(f"metadata in {archive_path} must be a dictionary.")
        destination = destination_root / f"seed_{seed:03d}"
        source_sha256 = sha256_file(archive_path)
        if destination.exists():
            bundle = EmbeddingBundle.load(destination, verify_hashes=True)
            identity = (
                bundle.manifest.get("dataset"),
                bundle.manifest.get("representation"),
                bundle.manifest.get("representation_seed"),
                bundle.manifest.get("source", {}).get("sha256"),
            )
            expected_identity = (
                "WB_no_spurious",
                "resnet50",
                seed,
                source_sha256,
            )
            if identity != expected_identity:
                raise FileExistsError(
                    "Refusing to reuse an incompatible destination bundle: "
                    f"{destination}"
                )
            outputs.append(bundle.root)
            print(f"Reused seed {seed}: {bundle.root}", flush=True)
            continue
        bundle = write_embedding_bundle(
            destination,
            splits=splits,
            dataset="WB_no_spurious",
            representation="resnet50",
            representation_seed=seed,
            source={
                "format": "trusted_pt",
                "path": _display_path(archive_path),
                "sha256": source_sha256,
                "group_id_offset_removed": group_offset,
            },
            metadata={
                **dict(archive_metadata or {}),
                "no_spurious": True,
                "randomized_train_val_split": False,
                "server_manifest": server_manifest,
            },
        )
        outputs.append(bundle.root)
        print(f"Converted seed {seed}: {bundle.root}", flush=True)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert no-spurious Waterbirds .pt embeddings to NumPy bundles."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--destination-root", default=str(DEFAULT_DESTINATION_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    convert_archives(args.source_root, args.destination_root)


if __name__ == "__main__":
    main()
