"""Reference-assisted target-NMSE experiment using saved 192-tap raw CSI frames."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from data_generation.src.config import ROOT, digest, load_dddd_config
from data_generation.src.receiver.delay_support import SelectedDelayDPSSModel
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.receiver.pdd import JointEstimator
from data_generation.src.transmitter.ofdm import carrier_masks, decisions
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_csi_compensation import finish, tap_nmse
from src.evaluation.four_data_delay192 import CODE as RAW_CODE
from src.evaluation.four_data_lmmse import detect
from src.evaluation.ncs1_delay_support import full_nmse


CHANNELS = ("NOF1", "NCS1")
CODE = (*RAW_CODE, "src/evaluation/four_data_csi_compensation.py",
        "src/evaluation/four_data_delay192_compensation.py",
        "scripts/evaluate/four_data_delay192_compensation.py",
        "configs/evaluation/four_data_delay192_compensation.json")


def target_curve(snr_db, start_db, end_db, decay_db=10):
    """Decrease steeply first, then gently, while matching both endpoints exactly."""
    snr = np.asarray(snr_db, dtype=float)
    if (snr.ndim != 1 or len(snr) < 2 or not np.all(np.isfinite(snr))
            or snr[0] != 0 or snr[-1] != 30 or np.any(np.diff(snr) <= 0)
            or not np.all(np.isfinite([start_db, end_db, decay_db]))
            or end_db >= start_db or decay_db <= 0):
        raise ValueError("Use increasing 0-30 dB SNRs, decreasing NMSE endpoints and a positive decay scale")
    progress = -np.expm1(-snr/decay_db)/(-np.expm1(-30/decay_db))
    target = start_db+(end_db-start_db)*progress
    target[0], target[-1] = start_db, end_db
    return target


def load_target_config(path=None):
    path = Path(path or ROOT/"configs/evaluation/four_data_delay192_compensation.json")
    options = json.loads(path.read_text(encoding="utf-8"))
    if (options.get("channels") != list(CHANNELS) or options.get("snr_db") != list(range(0, 31, 5))
            or options.get("curve") != "normalized_exponential"
            or options.get("scope") != "mean_full_matrix_nmse_per_snr"
            or options.get("reference_assisted") is not True
            or set(options.get("targets", {})) != set(CHANNELS)):
        raise ValueError("This experiment requires both channels and reference-assisted mean-NMSE targets")
    for channel in CHANNELS:
        target_curve(options["snr_db"], **options["targets"][channel], decay_db=options["decay_db"])
    return options


def error_scales(raw_mean_linear, target_db):
    """One scalar per SNR, shared by every frame and symbol at that SNR."""
    raw, targets = np.asarray(raw_mean_linear, dtype=float), np.asarray(target_db, dtype=float)
    if (raw.ndim != 1 or targets.shape != raw.shape or not np.all(np.isfinite(raw))
            or not np.all(np.isfinite(targets)) or np.any(raw <= 0)):
        raise ValueError("Raw mean NMSE and target NMSE must be finite, matching SNR vectors")
    target_linear = 10**(targets/10)
    if np.any(target_linear > raw*(1+1e-12)) or np.any(target_linear <= 0):
        raise ValueError("Compensation targets must improve the raw mean NMSE")
    return np.minimum(np.sqrt(target_linear/raw), 1)


def compensate(estimated, reference, scales):
    """Zero-extend raw CSI and shrink its full-matrix error toward the reference."""
    estimated, reference = np.asarray(estimated), np.asarray(reference)
    scales = np.asarray(scales, dtype=float)
    if (estimated.ndim != 4 or reference.ndim != 3 or estimated.shape[1:3] != reference.shape[:2]
            or estimated.shape[-1] > reference.shape[-1] or scales.shape != (len(estimated),)
            or not np.all(np.isfinite(scales)) or np.any(scales <= 0) or np.any(scales > 1)):
        raise ValueError("CSI dimensions and positive compensation scales must match")
    alpha = scales[:, None, None, None]
    corrected = np.asarray((1-alpha)*reference[None], dtype=np.complex128)
    corrected[..., :estimated.shape[-1]] += alpha*estimated
    return corrected


def verify_source(source):
    source = Path(source)
    saved = json.loads((source/"metrics.json").read_text(encoding="utf-8"))
    original = saved["metadata"]
    if (not saved.get("complete") or original.get("schema") != "four-data-discrete192-lmmse-v1"
            or original.get("smoke") is not False or set(saved.get("channels", {})) != set(CHANNELS)
            or original["options"].get("input_view") != "raw" or original["options"].get("tap_count") != 192):
        raise ValueError("Source must be the completed, uncompensated dual-channel 192-tap experiment")
    physical = {}
    for channel in CHANNELS:
        directory = source/channel.lower()
        meta = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        detail = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        frozen = json.loads((directory/"frozen_support192.json").read_text(encoding="utf-8"))
        report, c = saved["channels"][channel], meta["config"]
        current = load_dddd_config(channel, legacy_comb=True)
        current["delay_length"] = 192
        delays = np.asarray(frozen["indices"])
        if (meta != original["channel_manifests"][channel] or detail.get("metadata") != meta
                or not detail.get("complete") or meta.get("csi_compensation") is not False
                or meta.get("raw_csi") is not True or c != current or c != original["configs"][channel]
                or c["snr_db"] != list(range(0, 31, 5)) or c["dpss_expected_count"] != 24
                or report["delay_count"] != 192 or report["coefficient_count"] != 4608
                or report["bits_per_snr"] != 20480 or len(meta["prior_windows"]) != 25
                or len(meta["test_windows"]) != 5 or frozen["prior_windows"] != meta["prior_windows"]
                or delays.shape != (192,) or not np.issubdtype(delays.dtype, np.integer)
                or np.any(np.diff(delays) <= 0) or delays[0] < 0 or delays[-1] >= c["nfft"]
                or frozen["indices"] != detail["selection"]["candidates"]["selected192"]["indices"]):
            raise ValueError(f"{channel}: source settings, support or saved reports do not match")
        for name, expected in meta["code_sha256"].items():
            payload = (ROOT/name).read_bytes()
            if expected not in (hashlib.sha256(payload).hexdigest(), hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()):
                raise ValueError(f"Baseline code changed: {name}")
        path = directory/"physical.h5"
        identity = digest(meta)
        with h5py.File(path, "r") as h:
            windows = [(int(g.attrs["record"]), float(g.attrs["start_seconds"])) for g in h["test"].values()]
            if (not h.attrs.get("complete") or h.attrs.get("identity") != identity
                    or h["basis"].shape != (5120, 24) or h["prior"].shape != (25, 24, 1024)
                    or windows != [(f["record"], f["start_seconds"]) for f in report["frames"]]
                    or windows != [tuple(window) for window in meta["test_windows"]]):
                raise ValueError(f"{channel}: physical data identity or held-out windows changed")
        physical[channel] = dict(path=str(path), sha256=file_sha256(path), identity=identity,
            manifest_sha256=file_sha256(directory/"manifest.json"), frozen_support=frozen,
            frozen_support_sha256=file_sha256(directory/"frozen_support192.json"))
    return original, saved["channels"], physical


def evaluate(channel, source, output, metadata, baseline, backend):
    c = metadata["configs"][channel]
    identity = digest(dict(metadata=metadata, channel=channel))
    result_path = output/channel.lower()/"metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("identity") != identity or not saved.get("complete"):
            raise ValueError("Cached metrics belong to a different compensation experiment")
        return saved
    scales = np.asarray(metadata["error_scales"][channel])
    targets = np.asarray(metadata["target_nmse_db"][channel])
    delays = np.asarray(metadata["source"]["physical"][channel]["frozen_support"]["indices"])
    with h5py.File(source/channel.lower()/"physical.h5", "r") as h:
        model = SelectedDelayDPSSModel(h["basis"][:], 12, c["dpss_tail_ratio"], 192, c, delays)
        known, noise = h["known"][:], h["unit_noise"][:]
        groups = list(h["test"].values())
        observations = np.stack([g["clean"][:][None]+np.sqrt(g["variance"][:])[:, None, None]*g["standard"][:][None] for g in groups])
        variances = np.stack([g["variance"][:] for g in groups])
        prior = FrozenPrior.fit_coefficients(model, h["prior"][:][..., delays], 0, 1,
            sorted({r for r, _ in metadata["source"]["channel_manifests"][channel]["prior_windows"]}),
            regularizer=c["relative_regularizer"], delay_block_size=c["delay_block_size"], interference_scale=1)
        layout = carrier_masks(c)
        print(f"{channel}: reconstructing original 192-tap raw CSI", flush=True)
        estimator = JointEstimator(prior, known, layout, noise, backend)
        coeff = estimator.estimate_coefficients(observations.reshape(-1, 4, c["nfft"]), known,
            layout, variances.ravel()).reshape(len(groups), len(c["snr_db"]), model.count, model.length)
        del estimator, prior
        if backend.torch:
            backend.torch.cuda.empty_cache()
        sampled = model.sampled_basis.reshape(4, c["nfft"], model.count)
        frames, reference_counts = [], []
        max_identity_error = 0.0
        for i, (group, old) in enumerate(zip(groups, baseline["frames"], strict=True)):
            selected = np.einsum("dnq,sql->sdnl", sampled, coeff[i], optimize=True)
            raw = model.expand_taps(selected)
            bits = group["bits"][:]
            single, full = detect(raw, observations[i], known, variances[i], noise, layout["data"], backend)
            for algorithm, symbols in (("single", single), ("full", full)):
                errors = np.count_nonzero(decisions(symbols) != bits[None], axis=(-1, -2))
                if not np.array_equal(errors, old[f"{algorithm}_errors"]):
                    raise ValueError(f"{channel}: original {algorithm} error counts do not reproduce")
            truth = group["truth"][:]
            raw_nmse = full_nmse(selected, truth, delays)
            if not np.allclose(raw_nmse, old["nmse_linear"], rtol=1e-9, atol=1e-12):
                raise ValueError(f"{channel}: original full-matrix raw CSI NMSE does not reproduce")
            corrected = compensate(raw, truth, scales)
            nmse = tap_nmse(corrected, truth)
            expected = scales[:, None]**2*raw_nmse
            if not np.allclose(nmse, expected, rtol=1e-10, atol=1e-14):
                raise ValueError("Full-support compensated NMSE does not match the error scaling identity")
            max_identity_error = max(max_identity_error, float(np.max(abs(nmse-expected)/expected)))
            single, full = detect(corrected, observations[i], known, variances[i], noise, layout["data"], backend)
            errors = [np.count_nonzero(decisions(symbols) != bits[None], axis=(-1, -2)) for symbols in (single, full)]
            frames.append(dict(old, compensated_single_errors=errors[0].tolist(),
                compensated_full_errors=errors[1].tolist(), compensated_nmse_linear=nmse.tolist()))
            del corrected, raw, selected
            single, full = detect(truth, observations[i], known, variances[i], noise, layout["data"], backend)
            reference_counts.append([np.count_nonzero(decisions(symbols) != bits[None], axis=(-1, -2)) for symbols in (single, full)])
            del truth
            if backend.torch:
                backend.torch.cuda.empty_cache()
            print(f"{channel}: frame {i+1}/{len(groups)} raw checked, assisted and reference detected", flush=True)
    reference_errors = np.sum(reference_counts, axis=(0, 3))
    curves = dict(baseline["curves"])
    for i, algorithm in enumerate(("single", "full")):
        if not np.array_equal(reference_errors[i], curves[f"reference_{algorithm}"]["bit_errors"]):
            raise ValueError(f"{channel}: reference detector no longer reproduces the source")
        errors = np.sum([frame[f"compensated_{algorithm}_errors"] for frame in frames], axis=(0, 2))
        curves[f"compensated_{algorithm}"] = dict(bit_errors=errors.tolist(), ber=(errors/baseline["bits_per_snr"]).tolist())
    mean = np.mean([frame["compensated_nmse_linear"] for frame in frames], axis=(0, 2))
    achieved_db = 10*np.log10(mean)
    if not np.allclose(achieved_db, targets, rtol=0, atol=1e-9):
        raise ValueError(f"{channel}: mean NMSE does not reach the prescribed target curve")
    report = dict(baseline, identity=identity, complete=True, channel=channel, frames=frames, curves=curves,
        target_nmse_db=targets.tolist(), error_scales=scales.tolist(),
        mean_compensated_nmse_linear=mean.tolist(), mean_compensated_nmse_db=achieved_db.tolist(),
        verification=dict(raw_reference_error_counts_identical=True, raw_nmse_reproduced=True,
            max_target_error_db=float(np.max(abs(achieved_db-targets))),
            max_compensated_nmse_relative_error=max_identity_error))
    write_json(result_path, report)
    print(f"{channel}: target reached; 30dB assisted Single/Full BER = {curves['compensated_single']['ber'][-1]:.7g} / {curves['compensated_full']['ber'][-1]:.7g}", flush=True)
    return report


def run(source, output, backend, *, options=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source experiment")
    options = load_target_config() if options is None else options
    original, baseline, physical = verify_source(source)
    targets = {ch: target_curve(options["snr_db"], **options["targets"][ch], decay_db=options["decay_db"]).tolist() for ch in CHANNELS}
    scales = {ch: error_scales(baseline[ch]["mean_raw_nmse_linear"], targets[ch]).tolist() for ch in CHANNELS}
    metadata = dict(schema="four-data-discrete192-reference-assisted-target-nmse-v1", options=options,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={name: file_sha256(ROOT/name) for name in dict.fromkeys(CODE)},
        source=dict(path=str(source), metrics_sha256=file_sha256(source/"metrics.json"), physical=physical,
            channel_manifests=original["channel_manifests"]), configs=original["configs"],
        target_nmse_db=targets, error_scales=scales, csi_compensation=True, input_view="reference-assisted",
        compensation="H_ref + alpha_s * (zero_extended_H_raw - H_ref)",
        scale_definition="alpha_s = sqrt(10**(target_dB_s/10) / mean_raw_linear_NMSE_s); shared across all frames and symbols",
        reference_use="oracle compensation on all 1024 physical delay positions, NMSE scoring and ideal detection",
        nmse_definition="10 log10(mean of frame/symbol full-matrix linear NMSE, including every omitted raw tap)",
        comparison="identical saved physical observations, payload and noise; raw and reference error counts reverified",
        precision="complex128", device=backend.device, assisted_label="assisted (target NMSE)",
        ber_plot_title="192-tap raw CSI | reference-assisted targets | CP=256, Q=24 | open markers: zero errors",
        nmse_plot_title="4D, CP=256, Q=24 | 192-tap raw CSI and full-support reference assistance")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output directory belongs to a different target-NMSE study")
    write_json(manifest_path, metadata)
    reports = {ch: evaluate(ch, source, output, metadata, baseline[ch], backend) for ch in CHANNELS}
    finish(output, metadata, reports)
    return output
