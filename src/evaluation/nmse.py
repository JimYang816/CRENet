"""Full-support four-symbol CSI NMSE, averaged in the linear domain."""

import json
import numpy as np
from data_generation.src.dataset.four_data_math import coefficient_nmse
from data_generation.src.dataset.four_data_protocol import SCHEMA


def manifest_values(dataset, files):
    try:
        report = json.loads(
            (dataset.path.parent / "manifest.json").read_text(encoding="utf-8")
        )
        value = report["splits"]["test"]
        measured = np.asarray(value["mean_raw_nmse_db"], float)
        if (
            report.get("complete")
            and report["schema"] == SCHEMA
            and dataset.input_view == "raw"
            and report["channel"] == dataset.channel
            and report["config_hash"] == dataset._h.attrs["config_hash"]
            and value["checked_all_frames"]
            and value["frames_per_snr"] == dataset.frame_count
            and np.array_equal(report["snr_db"], dataset.snr_db)
            and measured.shape == dataset.snr_db.shape
            and np.isfinite(measured).all()
            and all(
                report["files"][name]["sha256"] == files[name]["sha256"]
                for name in ("test.h5", "model.h5")
            )
        ):
            return measured
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def mean_nmse_db(dataset, si):
    h = dataset._h
    windows = h["samples/window_index"][:]
    total = 0.0
    for wi in np.unique(windows):
        frames = np.flatnonzero(windows == wi)
        values = coefficient_nmse(
            h["samples/raw_coefficients"][si, frames],
            dataset._model["sampled_grams"][:],
            h["windows/reference_cross"][int(wi)],
            h["windows/reference_energy"][int(wi)],
        )
        total += float(values.sum())
    return float(
        10 * np.log10(max(total / (dataset.frame_count * 4), np.finfo(float).tiny))
    )


def evaluate_nmse(dataset, entry, files):
    accepted = manifest_values(dataset, files)
    for si in range(len(dataset.snr_db)):
        if not entry.complete[si]:
            value = (
                float(accepted[si])
                if accepted is not None
                else mean_nmse_db(dataset, si)
            )
            entry.commit((si,), mean_nmse_db=value)
