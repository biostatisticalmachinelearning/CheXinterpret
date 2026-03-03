#!/usr/bin/env python3
"""Clear cached pipeline artifacts.

By default clears everything except model weights (--model-weights must be
passed explicitly, as re-downloading ~33 GB takes a long time).

Examples
--------
# Clear all pipeline outputs (manifest, features, SAE, logs, runs):
python scripts/clear_cache.py --config configs/default.yaml

# Clear only the manifest and features so they are recomputed:
python scripts/clear_cache.py --config configs/default.yaml --manifest --features

# Also wipe the downloaded CheXagent model weights from disk:
python scripts/clear_cache.py --config configs/default.yaml --model-weights
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear cached pipeline artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")

    group = parser.add_argument_group("what to clear (default: all pipeline outputs)")
    group.add_argument("--manifest", action="store_true", help="Clear manifest.csv")
    group.add_argument("--features", action="store_true", help="Clear features.npz")
    group.add_argument("--sae", action="store_true", help="Clear sae.pt and study outputs")
    group.add_argument("--logs", action="store_true", help="Clear logs/ directory")
    group.add_argument("--runs", action="store_true", help="Clear runs/ and publication/ directories")
    group.add_argument(
        "--model-weights",
        action="store_true",
        help="Clear the HuggingFace model weight cache (re-download required, ~33 GB).",
    )

    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt."
    )
    return parser.parse_args()


def _remove(path: Path, label: str, dry_run: bool = False) -> None:
    if not path.exists():
        print(f"  skip  {label}  (not found: {path})")
        return
    size = _dir_size_mb(path) if path.is_dir() else path.stat().st_size / 1e6
    print(f"  rm    {label}  ({size:.1f} MB)  {path}")
    if not dry_run:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    args = parse_args()

    # Import here so the script is importable even before the package is installed.
    try:
        from chex_sae_fairness.config import ExperimentConfig
    except ImportError:
        sys.exit("Package not installed. Run: pip install -e .")

    cfg = ExperimentConfig.from_yaml(args.config)
    output_root = cfg.output_root

    # If no specific flags were given, clear all pipeline outputs (but not model weights).
    clear_all_pipeline = not any([
        args.manifest, args.features, args.sae, args.logs, args.runs, args.model_weights
    ])

    targets: list[tuple[Path, str]] = []

    if args.manifest or clear_all_pipeline:
        targets.append((output_root / "manifest.csv", "manifest"))

    if args.features or clear_all_pipeline:
        targets.append((output_root / "features.npz", "features"))

    if args.sae or clear_all_pipeline:
        targets.append((output_root / "sae.pt", "SAE checkpoint"))
        targets.append((output_root / "study_metrics.json", "study metrics"))
        targets.append((output_root / "study_predictions.npz", "study predictions"))
        targets.append((output_root / "workspace", "workspace dir"))
        targets.append((output_root / "sae_sweep", "SAE sweep dir"))
        targets.append((output_root / "figures", "figures dir"))

    if args.logs or clear_all_pipeline:
        targets.append((output_root / "logs", "logs dir"))

    if args.runs or clear_all_pipeline:
        targets.append((output_root / "runs", "runs dir"))
        targets.append((output_root / "publication", "publication dir"))

    if args.model_weights:
        cache_dir = cfg.features.cache_dir
        if cache_dir:
            targets.append((Path(cache_dir).expanduser().resolve(), "HuggingFace model cache"))

    existing = [(p, label) for p, label in targets if p.exists()]

    if not existing:
        print("Nothing to clear — all targets already absent.")
        return

    print("Will remove:")
    for path, label in existing:
        size = _dir_size_mb(path) if path.is_dir() else path.stat().st_size / 1e6
        print(f"  {label:30s}  {size:8.1f} MB  {path}")

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print()
    for path, label in existing:
        _remove(path, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
