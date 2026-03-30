"""CLI entry points: cnn-train and cnn-sweep."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Train CLI ─────────────────────────────────────────────────────────────────

def _parse_train_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a single CNN on CheXpert Plus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to cnn_default.yaml")
    p.add_argument("--run-name", default=None, help="Output subdirectory name (default: timestamp)")
    p.add_argument("--output-dir", default=None, help="Override output root directory")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    p.add_argument("--arch", default=None,
                   help="Override architecture (densenet121|resnet50|efficientnet_b4|convnext_small)")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    p.add_argument("--image-size", type=int, default=None, help="Override image size")
    p.add_argument("--augmentation", default=None, help="Override augmentation level (light|medium|heavy)")
    return p.parse_args()


def train_cli() -> None:
    args = _parse_train_args()
    from .config import CNNConfig
    import dataclasses

    cfg = CNNConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.arch:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, architecture=args.arch))
    if args.lr is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, lr=args.lr))
    if args.batch_size is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, batch_size=args.batch_size))
    if args.epochs is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, epochs=args.epochs))
    if args.image_size is not None:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, image_size=args.image_size))
    if args.augmentation is not None:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, augmentation_level=args.augmentation))

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (Path(args.output_dir) if args.output_dir else cfg.output_root) / run_name
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, output_root=str(output_root)))

    device = _resolve_device(args.device)
    _set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("CNN TRAINING RUN: %s", run_name)
    logger.info("  Config     : %s", args.config)
    logger.info("  Arch       : %s", cfg.train.architecture)
    logger.info("  LR         : %.2e", cfg.train.lr)
    logger.info("  Batch size : %d", cfg.train.batch_size)
    logger.info("  Image size : %d", cfg.data.image_size)
    logger.info("  Augment    : %s", cfg.data.augmentation_level)
    logger.info("  Device     : %s", device)
    logger.info("  Output     : %s", output_root)
    logger.info("=" * 60)

    from .dataset import build_dataloaders, compute_pos_weight
    from .evaluate import evaluate_model
    from .models import build_model, count_parameters
    from .train import Trainer

    manifest = pd.read_csv(cfg.paths.manifest_csv)
    logger.info("Loaded manifest: %d rows", len(manifest))

    train_loader, valid_loader = build_dataloaders(manifest, cfg, device)

    model = build_model(cfg.train.architecture, num_classes=len(cfg.data.pathology_cols),
                        pretrained=cfg.train.pretrained)
    total_params, trainable_params = count_parameters(model)
    logger.info("Model: %s | params: %dM total, %dM trainable",
                cfg.train.architecture, total_params // 1_000_000, trainable_params // 1_000_000)

    train_df = manifest[manifest[cfg.data.split_col] == "train"]
    pos_weight = compute_pos_weight(train_df, cfg.data.pathology_cols, device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        pathology_cols=cfg.data.pathology_cols,
        cfg=cfg.train,
        device=device,
        output_root=output_root,
        pos_weight=pos_weight,
    )
    state = trainer.fit()

    logger.info("Training complete. Best val_auroc=%.4f (epoch stopped_early=%s)",
                state.best_val_auroc, state.stopped_early)

    # Final evaluation
    logger.info("Running final evaluation on validation split...")
    evaluate_model(
        model=model,
        loader=valid_loader,
        pathology_cols=cfg.data.pathology_cols,
        device=device,
        output_dir=output_root,
        amp=cfg.train.amp,
        split_name="valid",
    )
    logger.info("All outputs saved to %s", output_root)


# ── Sweep CLI ─────────────────────────────────────────────────────────────────

def _parse_sweep_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter sweep over CNN architectures (Optuna TPE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to YAML config (sweep section used)")
    p.add_argument("--run-name", default=None, help="Sweep output subdirectory (default: timestamp)")
    p.add_argument("--n-trials", type=int, default=None, help="Override sweep.n_trials")
    p.add_argument("--device", default=None, help="torch device")
    p.add_argument("--retrain-best", action="store_true",
                   help="After sweep, retrain best configuration from scratch and evaluate.")
    return p.parse_args()


def sweep_cli() -> None:
    args = _parse_sweep_args()
    from .config import CNNConfig
    import dataclasses

    cfg = CNNConfig.from_yaml(args.config)

    if args.n_trials is not None:
        cfg = dataclasses.replace(cfg, sweep=dataclasses.replace(cfg.sweep, n_trials=args.n_trials))

    run_name = args.run_name or ("sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_root = cfg.output_root / run_name
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, output_root=str(output_root)))

    device = _resolve_device(args.device)
    _set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("CNN HYPERPARAMETER SWEEP: %s", run_name)
    logger.info("  Config    : %s", args.config)
    logger.info("  n_trials  : %d", cfg.sweep.n_trials)
    logger.info("  Archs     : %s", cfg.sweep.architectures)
    logger.info("  Device    : %s", device)
    logger.info("  Output    : %s", output_root)
    logger.info("=" * 60)

    from .sweep import run_sweep

    study = run_sweep(cfg, device)

    if args.retrain_best:
        logger.info("Retraining best configuration from scratch...")
        best_params = study.best_trial.params
        import dataclasses

        best_train_cfg = dataclasses.replace(
            cfg.train,
            architecture=best_params.get("architecture", cfg.train.architecture),
            lr=best_params.get("lr", cfg.train.lr),
            batch_size=best_params.get("batch_size", cfg.train.batch_size),
            weight_decay=best_params.get("weight_decay", cfg.train.weight_decay),
        )
        best_data_cfg = dataclasses.replace(
            cfg.data,
            image_size=best_params.get("image_size", cfg.data.image_size),
            augmentation_level=best_params.get("augmentation_level", cfg.data.augmentation_level),
        )
        best_cfg = dataclasses.replace(
            cfg,
            train=best_train_cfg,
            data=best_data_cfg,
            paths=dataclasses.replace(cfg.paths, output_root=str(output_root / "best_retrained")),
        )

        # Invoke train_cli logic inline
        from .dataset import build_dataloaders, compute_pos_weight
        from .evaluate import evaluate_model
        from .models import build_model
        from .train import Trainer

        manifest = pd.read_csv(best_cfg.paths.manifest_csv)
        train_loader, valid_loader = build_dataloaders(manifest, best_cfg, device)
        model = build_model(best_train_cfg.architecture, num_classes=len(best_cfg.data.pathology_cols))
        train_df = manifest[manifest[best_cfg.data.split_col] == "train"]
        pos_weight = compute_pos_weight(train_df, best_cfg.data.pathology_cols, device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            pathology_cols=best_cfg.data.pathology_cols,
            cfg=best_train_cfg,
            device=device,
            output_root=best_cfg.output_root,
            pos_weight=pos_weight,
        )
        state = trainer.fit()
        evaluate_model(model, valid_loader, best_cfg.data.pathology_cols, device,
                       best_cfg.output_root, best_train_cfg.amp)
        logger.info("Best model retrained. val_auroc=%.4f", state.best_val_auroc)
        logger.info("Outputs saved to %s", best_cfg.output_root)
