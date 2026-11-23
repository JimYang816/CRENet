"""Record-level paired uncertainty for the isolated DPSS estimator comparison."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.common.experiment import file_sha256, write_json
from src.evaluation.fingerprints import DataIdentity, digest


def paired(records, errors, nmse, ndata, *, draws=4000, seed=20261123):
    """Bootstrap physical record IDs, preserving all windows and noise repeats."""
    records = np.asarray(records)
    errors, nmse = np.asarray(errors), np.asarray(nmse)
    if errors.shape != nmse.shape or errors.ndim != 4 or errors.shape[0] != 2 or errors.shape[2] != len(records):
        raise ValueError("Expected method x SNR x frame x D arrays")
    if np.any(errors < 0) or np.any(nmse <= 0) or not np.all(np.isfinite(nmse)) or ndata < 1 or draws < 1:
        raise ValueError("Invalid paired comparison input")
    unique = np.unique(records)
    if len(unique) < 2:
        raise ValueError("At least two physical records are needed")
    error_sum, nmse_sum, counts = [], [], []
    for record in unique:
        idx = np.flatnonzero(records == record)
        error_sum.append(errors[:, :, idx, :].sum(axis=(2, 3)))
        nmse_sum.append(nmse[:, :, idx, :].sum(axis=(2, 3)))
        counts.append(len(idx)*2)
    errors_by_record = np.stack(error_sum, axis=-1)
    nmse_by_record = np.stack(nmse_sum, axis=-1)
    counts = np.asarray(counts)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(unique), size=(draws, len(unique)))
    denom = counts[sample].sum(axis=-1)
    ber_boot = errors_by_record[:, :, sample].sum(axis=-1)/denom[None, None, :]/(2*ndata)
    nmse_boot = nmse_by_record[:, :, sample].sum(axis=-1)/denom[None, None, :]
    ber_delta = ber_boot[1]-ber_boot[0]
    nmse_delta_db = 10*np.log10(nmse_boot[1]/nmse_boot[0])
    total = counts.sum()
    ber = errors_by_record.sum(axis=-1)/total/(2*ndata)
    linear = nmse_by_record.sum(axis=-1)/total
    interval = lambda x: np.quantile(x, [.025, .975], axis=-1).T.tolist()
    return dict(record_ids=unique.tolist(), blocks_per_record=counts.tolist(), draws=draws, seed=seed,
                ber_difference_omp_minus_lmmse=(ber[1]-ber[0]).tolist(),
                ber_difference_95pct_record_bootstrap=interval(ber_delta),
                nmse_db_difference_omp_minus_lmmse=(10*np.log10(linear[1]/linear[0])).tolist(),
                nmse_db_difference_95pct_record_bootstrap=interval(nmse_delta_db))


def run(datasets, source, destination, *, draws=4000, seed=20261123):
    source = Path(source)
    report = json.loads((source/"metrics.json").read_text(encoding="utf-8"))
    if not report.get("complete"):
        raise ValueError("Primary test must be complete")
    channels = {}
    for channel, supplied in datasets.items():
        folder = Path(supplied)
        identity = DataIdentity({"test.h5": folder/"test.h5"})
        if identity.files["test.h5"] != report["channels"][channel]["source"]["files"]["test.h5"]:
            raise ValueError("Test dataset does not match the completed comparison")
        with h5py.File(folder/"test.h5", "r") as data, h5py.File(source/f"{channel.lower()}_test.h5", "r") as result:
            if (not result.attrs.get("complete") or not np.all(result["window_complete"][:])
                    or result.attrs.get("identity") != digest(report["channels"][channel]["source"])):
                raise ValueError("Incomplete test state")
            frame_windows = data["samples/window_index"][:]
            records = data["windows/record"][:][frame_windows]
            errors = result["bit_errors"][:]
            nmse = np.stack((result["lmmse_nmse"][:], result["omp_nmse"][:]))
            ndata = data["samples/bits"].shape[-2]
            channels[channel] = paired(records, errors, nmse, ndata, draws=draws, seed=seed)
            channels[channel]["snr_db"] = report["channels"][channel]["snr_db"]
            channels[channel]["result_state_sha256"] = file_sha256(source/f"{channel.lower()}_test.h5")
        identity.assert_unchanged()
    result = dict(complete=True, method="paired percentile bootstrap of physical record IDs",
                  interpretation="Intervals quantify record-level variability for the fixed test configuration, not hyperparameter or training-seed uncertainty.",
                  source_metrics_sha256=file_sha256(source/"metrics.json"), channels=channels)
    write_json(destination, result)
    return result
