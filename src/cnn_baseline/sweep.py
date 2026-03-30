"""Optuna-based hyperparameter sweep for CNN architectures."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import yaml

from .config import CNNConfig, CNNSweepConfig, CNNTrainConfig
from .dataset import build_dataloaders, compute_pos_weight
from .models import build_model
from .train import Trainer

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed. Install with: pip install optuna")


# ── Objective factory ─────────────────────────────────────────────────────────

def make_objective(
    base_cfg: CNNConfig,
    manifest: pd.DataFrame,
    device: torch.device,
) -> "Callable[[optuna.Trial], float]":
    """Returns an Optuna objective closure."""
    if not HAS_OPTUNA:
        raise ImportError("Optuna is required for sweep. Install with: pip install optuna")

    def objective(trial: "optuna.Trial") -> float:
        # Sample hyperparameters
        trial_train_cfg, data_overrides = _suggest_hyperparams(trial, base_cfg.sweep, base_cfg.train)

        # Build trial-specific config
        trial_cfg = CNNConfig(
            seed=base_cfg.seed,
            paths=base_cfg.paths,
            data=dataclasses.replace(base_cfg.data, **data_overrides),
            train=trial_train_cfg,
            sweep=base_cfg.sweep,
        )
        # Each trial writes to its own subdirectory
        trial_output = base_cfg.output_root / "trials" / f"trial_{trial.number:04d}"

        logger.info(
            "Trial %d: arch=%s lr=%.2e bs=%d wd=%.0e aug=%s img=%d",
            trial.number,
            trial_train_cfg.architecture,
            trial_train_cfg.lr,
            trial_train_cfg.batch_size,
            trial_train_cfg.weight_decay,
            data_overrides["augmentation_level"],
            data_overrides["image_size"],
        )

        torch.manual_seed(base_cfg.seed + trial.number)
        np.random.seed(base_cfg.seed + trial.number)

        train_loader, valid_loader = build_dataloaders(manifest, trial_cfg, device)
        model = build_model(
            trial_train_cfg.architecture,
            num_classes=len(base_cfg.data.pathology_cols),
            pretrained=trial_train_cfg.pretrained,
        )

        train_df = manifest[manifest[base_cfg.data.split_col] == "train"]
        pos_weight = compute_pos_weight(train_df, base_cfg.data.pathology_cols, device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            pathology_cols=base_cfg.data.pathology_cols,
            cfg=trial_train_cfg,
            device=device,
            output_root=trial_output,
            pos_weight=pos_weight,
        )
        state = trainer.fit()
        return state.best_val_auroc

    return objective


def _suggest_hyperparams(
    trial: "optuna.Trial",
    sweep_cfg: CNNSweepConfig,
    base_train_cfg: CNNTrainConfig,
) -> tuple[CNNTrainConfig, dict]:
    """Samples hyperparameters. Returns (CNNTrainConfig, data_overrides dict)."""
    architecture = trial.suggest_categorical("architecture", sweep_cfg.architectures)
    lr = trial.suggest_float("lr", sweep_cfg.lr_low, sweep_cfg.lr_high, log=True)
    batch_size = trial.suggest_categorical("batch_size", sweep_cfg.batch_sizes)
    weight_decay = trial.suggest_categorical("weight_decay", sweep_cfg.weight_decays)
    augmentation_level = trial.suggest_categorical("augmentation_level", sweep_cfg.augmentation_levels)
    image_size = trial.suggest_categorical("image_size", sweep_cfg.image_sizes)

    train_cfg = dataclasses.replace(
        base_train_cfg,
        architecture=architecture,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )
    data_overrides = {"augmentation_level": augmentation_level, "image_size": image_size}
    return train_cfg, data_overrides


# ── Sweep runner ──────────────────────────────────────────────────────────────

def run_sweep(cfg: CNNConfig, device: torch.device) -> "optuna.Study":
    """
    Creates/loads an Optuna study (SQLite), runs n_trials, saves results.
    Resumable: re-running with same config continues from last completed trial.
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna is required. Install with: pip install optuna")

    cfg.output_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(cfg.paths.manifest_csv)
    logger.info("Loaded manifest: %d rows", len(manifest))

    db_path = cfg.output_root / "sweep.db"
    storage = f"sqlite:///{db_path}"

    sampler = (
        optuna.samplers.TPESampler(seed=cfg.seed)
        if cfg.sweep.sampler == "tpe"
        else optuna.samplers.RandomSampler(seed=cfg.seed)
    )

    study = optuna.create_study(
        study_name="cnn_baseline_sweep",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    n_remaining = cfg.sweep.n_trials - len(study.trials)
    if n_remaining <= 0:
        logger.info("Study already has %d trials (≥ n_trials=%d). Nothing to do.",
                    len(study.trials), cfg.sweep.n_trials)
    else:
        logger.info("Running %d trials (already completed: %d).", n_remaining, len(study.trials))
        objective = make_objective(cfg, manifest, device)
        study.optimize(
            objective,
            n_trials=n_remaining,
            timeout=cfg.sweep.timeout_seconds,
            show_progress_bar=True,
        )

    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv(cfg.output_root / "sweep_results.csv", index=False)

    best = study.best_trial
    best_params = {
        "value": best.value,
        "params": best.params,
    }
    with open(cfg.output_root / "best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    logger.info(
        "Best trial %d: val_auroc=%.4f | params=%s",
        best.number, best.value, best.params,
    )

    return study
