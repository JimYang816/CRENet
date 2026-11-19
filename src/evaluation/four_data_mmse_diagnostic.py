"""Paired 30 dB CSI-quality controls for the completed four-data experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy import linalg

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT
from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.four_data import BlockedFourDataReceiver, fixed_four_data_model
from data_generation.src.transmitter.ofdm import carrier_masks, decisions
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.data.contract import target_nmse_db
from src.evaluation.four_data_mmse import ALGORITHMS, experiment_config


MODES = ("raw", "paper_nmse_reference_assisted", "oracle_L176", "oracle_Q24_L176", "oracle_full")


def decode(matrices, observations, known, bits, variance, indices, noise, backend, whitener):
    """Use the unchanged two equalizers on the same four received blocks."""
    cancelled = observations-(matrices @ known[..., None])[..., 0]
    batch = EqualizerBatch(backend.array(matrices), backend.array(observations), backend.array(cancelled),
                           np.full(4, variance), indices, noise)
    errors = {}
    for name, equalizer in ALGORITHMS.items():
        parameters = {"whitener": whitener} if name == "full_matrix_mmse" else {}
        symbols = equalizer(batch, backend, **parameters).symbols
        errors[name] = np.count_nonzero(decisions(symbols) != bits, axis=(1, 2))
    return errors


def structure(reference, raw, known, indices, variance, noise):
    """Separate diagonal/coupling accuracy and expected CSI mismatch power."""
    error = raw-reference
    diagonal = np.diagonal(reference, axis1=-2, axis2=-1)
    error_diagonal = np.diagonal(error, axis1=-2, axis2=-1)
    raw_diagonal = np.diagonal(raw, axis1=-2, axis2=-1)
    energy = np.sum(abs(reference)**2, axis=(1, 2))
    diagonal_energy = np.sum(abs(diagonal)**2, axis=1)
    off_energy = energy-diagonal_energy
    error_energy = np.sum(abs(error)**2, axis=(1, 2))
    diagonal_error = np.sum(abs(error_diagonal)**2, axis=1)
    raw_energy = np.sum(abs(raw)**2, axis=(1, 2))
    # Unknown independent unit-energy data plus deterministic public pilots.
    mismatch = np.sum(abs(error[..., indices])**2, axis=(1, 2))
    mismatch += np.sum(abs((error @ known[..., None])[..., 0])**2, axis=1)
    thermal = variance*float(np.trace(noise).real)
    return dict(reference_offdiagonal_energy_fraction=off_energy/energy,
                raw_offdiagonal_energy_fraction=(raw_energy-np.sum(abs(raw_diagonal)**2, axis=1))/raw_energy,
                diagonal_relative_squared_error=diagonal_error/diagonal_energy,
                offdiagonal_relative_squared_error=(error_energy-diagonal_error)/off_energy,
                expected_csi_mismatch_to_thermal_noise=mismatch/thermal)


def audit_channel(channel, run, archive, backend, frame_limit=None):
    path = run/channel.lower()
    source = json.loads((path/"metrics.json").read_text(encoding="utf-8"))
    if not source["complete"]:
        raise ValueError("Audit needs a completed four-data experiment")
    config = experiment_config(channel)
    model = fixed_four_data_model(config)
    with h5py.File(path/"frozen_prior.h5", "r") as saved:
        np.testing.assert_array_equal(model.basis, saved["basis"][:])
    receiver = BlockedFourDataReceiver(config)
    dataset = WatermarkArchive(archive, channel)
    indices = np.flatnonzero(carrier_masks(config)["data"])
    sampled = model.sampled_basis
    projection = linalg.pinv(sampled, check_finite=False)
    variance_index = config["snr_db"].index(30)
    target_db = float(target_nmse_db(channel, 30))
    raw_nmse = source["summary"]["mean_nmse_linear"][variance_index]
    alpha = float(np.sqrt(10**(target_db/10)/raw_nmse))
    totals = {mode: {name: 0 for name in ALGORITHMS} for mode in MODES}
    nmse = {mode: [] for mode in MODES}
    diagnostics, frames = {}, []
    with h5py.File(path/"receiver_estimates.h5", "r") as saved:
        known, noise = saved["known"][:], saved["unit_noise_covariance"][:]
        whitener = make_whitener(noise, backend)
        for index, original in enumerate(source["frames"][:frame_limit]):
            record, start = original["record"], original["start_seconds"]
            group = saved[f"frames/{index:04d}"]
            assert group.attrs["record"] == record and group.attrs["start_seconds"] == start
            variance = float(group["rf_variances"][variance_index])
            y = group["clean_observations"][:]+np.sqrt(variance)*group["standard_noise"][:]
            bits = group["bits"][:]
            coefficients = group["coefficients"][variance_index]
            raw = np.stack([taps_to_matrix(model.reconstruct(coefficients, symbol)) for symbol in range(4)])
            sounding = dataset.load(record)
            taps = np.stack([receiver.reference_taps(sounding, start, symbol)[0] for symbol in range(4)])
            reference = np.stack([taps_to_matrix(tap) for tap in taps])
            energy = np.sum(abs(reference)**2, axis=(1, 2))
            projected = projection @ taps[..., :model.length].reshape(-1, model.length)
            matrices = dict(raw=raw, paper_nmse_reference_assisted=reference+alpha*(raw-reference),
                            oracle_L176=np.stack([taps_to_matrix(tap[:, :model.length]) for tap in taps]),
                            oracle_Q24_L176=np.stack([taps_to_matrix(model.reconstruct(projected, symbol)) for symbol in range(4)]),
                            oracle_full=reference)
            row = dict(record=record, start_seconds=start, errors={})
            for mode, matrix in matrices.items():
                errors = decode(matrix, y, known, bits, variance, indices, noise, backend, whitener)
                if mode == "raw":
                    for name, error in errors.items():
                        np.testing.assert_array_equal(error, original["bit_errors"][name][variance_index])
                row["errors"][mode] = {name: value.tolist() for name, value in errors.items()}
                for name, error in errors.items():
                    totals[mode][name] += int(error.sum())
                nmse[mode].extend((np.sum(abs(matrix-reference)**2, axis=(1, 2))/energy).tolist())
            for name, values in structure(reference, raw, known, indices, variance, noise).items():
                diagnostics.setdefault(name, []).extend(values.tolist())
            frames.append(row)
            print(f"{channel} paired CSI audit {index+1}/{min(len(source['frames']), frame_limit or len(source['frames']))}: "
                  f"{record:03d} @ {start}s", flush=True)
    count = len(frames)*4*len(indices)*2
    summary = {}
    for mode in MODES:
        mean = float(np.mean(nmse[mode]))
        summary[mode] = dict(mean_nmse_linear=mean, mean_nmse_db=float(10*np.log10(mean)) if mean > 0 else None,
                             bit_errors=totals[mode], ber={name: error/count for name, error in totals[mode].items()})
    if frame_limit is None:
        np.testing.assert_allclose(summary["paper_nmse_reference_assisted"]["mean_nmse_db"], target_db,
                                    rtol=0, atol=1e-10)
    return dict(channel=channel, snr_db=30, frame_count=len(frames), bits=count, alpha=alpha,
                paper_target_nmse_db=target_db, source_identity=source["identity"],
                mean_structure={name: float(np.mean(values)) for name, values in diagnostics.items()},
                modes=summary, frames=frames)


def run_audit(run, output, archive, *, device, frame_limit=None):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    backend = DenseBackend(device)
    results = {}
    for channel in ("NOF1", "NCS1"):
        result = audit_channel(channel, run, archive, backend, frame_limit)
        write_json(output/f"{channel.lower()}_audit.json", result)
        results[channel] = result
        print(channel, json.dumps({mode: value["ber"] for mode, value in result["modes"].items()}), flush=True)
    source = dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  code_sha256=file_sha256(Path(__file__)), device=device,
                  raw_source=json.loads((run/"source_manifest.json").read_text(encoding="utf-8")))
    report = dict(schema="four-data-mmse-paired-csi-audit-v1", complete=True, source=source,
                  reference_usage="Oracle diagnostic only; same saved observations, payload, noise and equalizers",
                  channels={channel: {key: value for key, value in result.items() if key != "frames"}
                            for channel, result in results.items()})
    write_json(output/"diagnostic.json", report)
    with (output/"diagnostic.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "mode", "snr_db", "bits", "single_tap_ber", "full_matrix_ber", "mean_nmse_db"])
        for channel, result in results.items():
            for mode, values in result["modes"].items():
                writer.writerow([channel, mode, 30, result["bits"], values["ber"]["single_tap_mmse"],
                                 values["ber"]["full_matrix_mmse"], values["mean_nmse_db"]])
    return report
