"""Training loop with AMP, cosine+warmup LR schedule, and early stopping."""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import CNNTrainConfig

logger = logging.getLogger(__name__)


# ── LR scheduler ─────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: CNNTrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup for warmup_epochs, then cosine decay to lr * 0.01.
    Operates per-epoch (step called once per epoch).
    """
    warmup = cfg.warmup_epochs
    total = cfg.epochs
    min_lr_ratio = 0.01

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ── Train state ───────────────────────────────────────────────────────────────

@dataclass
class TrainState:
    epoch: int = 0
    best_val_auroc: float = 0.0
    patience_counter: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_aurocs: list[float] = field(default_factory=list)
    stopped_early: bool = False


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        pathology_cols: list[str],
        cfg: CNNTrainConfig,
        device: torch.device,
        output_root: Path,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.pathology_cols = pathology_cols
        self.cfg = cfg
        self.device = device
        self.output_root = output_root
        output_root.mkdir(parents=True, exist_ok=True)

        pw = pos_weight.to(device) if pos_weight is not None and cfg.pos_weight_mode == "auto" else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = build_scheduler(self.optimizer, cfg, len(train_loader))
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
        self.state = TrainState()

        # CSV log header
        self._log_path = output_root / "train_log.csv"
        with open(self._log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_macro_auroc", "lr"])

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self) -> TrainState:
        for epoch in range(self.cfg.epochs):
            self.state.epoch = epoch
            train_loss = self._train_epoch()
            val_auroc, _, _ = self._eval_epoch()

            current_lr = self.scheduler.get_last_lr()[0]
            self.state.train_losses.append(train_loss)
            self.state.val_aurocs.append(val_auroc)

            self.scheduler.step()

            logger.info(
                "Epoch %3d/%d | loss=%.4f | val_auroc=%.4f | lr=%.2e",
                epoch + 1, self.cfg.epochs, train_loss, val_auroc, current_lr,
            )

            with open(self._log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, f"{train_loss:.6f}", f"{val_auroc:.6f}", f"{current_lr:.2e}"])

            if val_auroc > self.state.best_val_auroc:
                self.state.best_val_auroc = val_auroc
                self.state.patience_counter = 0
                self._save_checkpoint(epoch)
                logger.info("  New best val_auroc=%.4f — checkpoint saved.", val_auroc)
            else:
                self.state.patience_counter += 1
                if self.state.patience_counter >= self.cfg.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d).",
                        epoch + 1, self.cfg.patience,
                    )
                    self.state.stopped_early = True
                    break

        self._load_best_checkpoint()
        return self.state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.amp and self.device.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            # Unscale before clipping to operate on true gradient magnitudes
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _eval_epoch(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Returns (macro_auroc, labels[N,C], probs[N,C])."""
        self.model.eval()
        all_labels, all_probs = [], []

        with torch.no_grad():
            for batch in self.valid_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["labels"].cpu().numpy()

                with torch.amp.autocast("cuda", enabled=self.cfg.amp and self.device.type == "cuda"):
                    logits = self.model(images)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_labels.append(labels)
                all_probs.append(probs)

        labels_arr = np.concatenate(all_labels, axis=0)
        probs_arr = np.concatenate(all_probs, axis=0)

        macro_auroc = _macro_auroc(labels_arr, probs_arr)
        return macro_auroc, labels_arr, probs_arr

    def _save_checkpoint(self, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_state": self.state,
            },
            self.output_root / "best_model.pt",
        )

    def _load_best_checkpoint(self) -> None:
        ckpt_path = self.output_root / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded best checkpoint from epoch %d.", ckpt["epoch"] + 1)


# ── Metric helper ─────────────────────────────────────────────────────────────

def _macro_auroc(labels: np.ndarray, probs: np.ndarray, min_positives: int = 5) -> float:
    """Macro-averaged AUROC, skipping columns with < min_positives positives."""
    scores = []
    for i in range(labels.shape[1]):
        y = labels[:, i]
        if y.sum() < min_positives or (1 - y).sum() < min_positives:
            continue
        try:
            scores.append(roc_auc_score(y, probs[:, i]))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.5
