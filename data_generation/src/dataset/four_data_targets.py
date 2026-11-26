"""Explicit reference-assisted calibration targets for the formal datasets."""

import json
from pathlib import Path
import numpy as np
from data_generation.src.config import ROOT

CHANNELS = ("NOF1", "NCS1")


def target_curve(snr_db, start_db, end_db, decay_db=10):
    snr = np.asarray(snr_db, dtype=float)
    if (
        snr.ndim != 1
        or len(snr) < 2
        or not np.all(np.isfinite(snr))
        or snr[0] != 0
        or snr[-1] != 30
        or np.any(np.diff(snr) <= 0)
        or not np.all(np.isfinite([start_db, end_db, decay_db]))
        or end_db >= start_db
        or decay_db <= 0
    ):
        raise ValueError(
            "Use increasing 0-30 dB SNRs, decreasing NMSE endpoints and a positive decay scale"
        )
    progress = -np.expm1(-snr / decay_db) / (-np.expm1(-30 / decay_db))
    target = start_db + (end_db - start_db) * progress
    target[0], target[-1] = start_db, end_db
    return target


def load_target_config(path=None):
    options = json.loads(
        Path(
            path or ROOT / "configs/evaluation/four_data_delay192_compensation.json"
        ).read_text(encoding="utf-8")
    )
    if (
        options.get("channels") != list(CHANNELS)
        or options.get("snr_db") != list(range(0, 31, 5))
        or options.get("curve") != "normalized_exponential"
        or options.get("scope") != "mean_full_matrix_nmse_per_snr"
        or options.get("reference_assisted") is not True
        or set(options.get("targets", {})) != set(CHANNELS)
    ):
        raise ValueError(
            "Expected both channels and explicit reference-assisted mean-NMSE targets"
        )
    for channel in CHANNELS:
        target_curve(
            options["snr_db"],
            **options["targets"][channel],
            decay_db=options["decay_db"],
        )
    return options


def error_scales(raw_mean_linear, target_db):
    raw, targets = (
        np.asarray(raw_mean_linear, dtype=float),
        np.asarray(target_db, dtype=float),
    )
    if (
        raw.ndim != 1
        or targets.shape != raw.shape
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(targets))
        or np.any(raw <= 0)
    ):
        raise ValueError(
            "Raw mean NMSE and target NMSE must be finite, matching SNR vectors"
        )
    target_linear = 10 ** (targets / 10)
    if np.any(target_linear > raw * (1 + 1e-12)) or np.any(target_linear <= 0):
        raise ValueError("Compensation targets must improve the raw mean NMSE")
    return np.minimum(np.sqrt(target_linear / raw), 1)
