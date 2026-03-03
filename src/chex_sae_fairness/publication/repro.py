from __future__ import annotations

from datetime import datetime, timezone
import platform
import sys
from typing import Any

import torch

from chex_sae_fairness.config import ExperimentConfig


def build_reproducibility_appendix(
    cfg: ExperimentConfig,
    report: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = report.get("counts", {}) if isinstance(report, dict) else {}
    appendix = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_seed": int(cfg.seed),
        "config_output_root": str(cfg.output_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None
        ),
        "data_filtering_counts": counts if isinstance(counts, dict) else {},
    }
    if extra:
        appendix.update(extra)
    return appendix
