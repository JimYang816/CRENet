"""Resumable DDDD export with selected-delay raw CSI and explicit oracle views."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import logging
import multiprocessing
from pathlib import Path
import time

import h5py
import numpy as np

from data_generation.src.config import carrier_independent_config, digest
from data_generation.src.dataset.four_data_math import (
    array_sha256, coefficient_nmse, reference_statistics, sampled_grams,
)
from data_generation.src.dataset.four_data_physical import initialize_worker, window_results
from data_generation.src.dataset.four_data_protocol import MODEL_SCHEMA, SCHEMA, SPLITS, counts, windows
from data_generation.src.receiver.delay_support import SelectedDelayDPSSModel, profile_full_frame
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.receiver.pdd import JointEstimator, symbol_layout
from data_generation.src.transmitter.ofdm import carrier_masks, make_frame
from src.common.experiment import file_sha256, write_json
from data_generation.src.dataset.four_data_targets import channel_targets, error_scales
from data_generation.src.receiver.full_frame import model_basis

LOG = logging.getLogger(__name__)


def load_prior(c, source, archive, receiver, smoke=False):
    """Reuse frozen physical positions; only independent prior records fit raw CSI."""
    source = Path(source)/c["channel"].lower()
    support = json.loads((source/"frozen_support192.json").read_text(encoding="utf-8"))
    delays = np.asarray(support["indices"])
    if (delays.shape != (192,) or not np.issubdtype(delays.dtype, np.integer)
            or np.any(np.diff(delays) <= 0) or delays[0] < 0 or delays[-1] >= c["nfft"]):
        raise ValueError("Expected the frozen 192 physical delay positions")
    if smoke:
        basis, _ = model_basis(c)
        coefficients, _ = profile_full_frame(receiver, archive.load(51), 2, basis)
        coefficients, records = coefficients[None, ..., delays], [51]
        info = dict(mode="CPU smoke: one independent prior window", prior_windows=[[51, 2]])
    else:
        meta = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
        with h5py.File(source/"physical.h5", "r") as h:
            if (carrier_independent_config(meta["config"]) != carrier_independent_config(c)
                    or not h.attrs.get("complete") or h.attrs.get("identity") != digest(meta)
                    or h["basis"].shape != (4*(c["nfft"]+c["cp"]), c["dpss_expected_count"])):
                raise ValueError("Independent prior must match the current CP80 physical frame and time basis")
            basis, coefficients = h["basis"][:], h["prior"][:][..., delays]
        records = sorted({r for r, _ in meta["prior_windows"]})
        info = dict(mode="frozen independent full-frame prior from accepted raw study",
            prior_windows=meta["prior_windows"], source_git_head=meta["git_head"],
            source_manifest_sha256=file_sha256(source/"manifest.json"),
            source_physical_sha256=file_sha256(source/"physical.h5"))
    model = SelectedDelayDPSSModel(basis, 12, c["dpss_tail_ratio"], 192, c, delays)
    prior = FrozenPrior.fit_coefficients(model, coefficients, 0, 1, records,
        regularizer=c["relative_regularizer"], delay_block_size=c["delay_block_size"], interference_scale=1)
    info.update(frozen_support=support, support_sha256=file_sha256(source/"frozen_support192.json"))
    return prior, info


def prepare_model(path, c, prior, noise, backend, identity, provenance):
    """Persist the estimator's observation spectrum, not a dense QK covariance."""
    path = Path(path)
    known, masks = make_frame(c, 1, 2).known, carrier_masks(c)
    if path.exists():
        with h5py.File(path, "r") as h:
            if h.attrs.get("schema") != MODEL_SCHEMA or h.attrs.get("identity") != identity or not h.attrs.get("prepared"):
                raise ValueError("Shared model identity mismatch; use a new run ID")
            prepared = JointEstimator.__new__(JointEstimator)
            prepared.prior, prepared.backend = prior, backend
            prepared.data_symbols, prepared.training_symbols = symbol_layout(c)
            prepared.known, prepared.masks = known, masks
            prepared.model_error_ratio = float(c["csi_model_error_ratio"])
            prepared.noise_gain = float(h["estimator"].attrs["noise_gain"])
            prepared.ridge = float(h["estimator"].attrs["ridge"])
            prepared.rows = [h[f"estimator/rows/{s}"][:] for s in range(4)]
            prepared.offsets = h["estimator/offsets"][:]
            for name in ("values", "projected", "gain"):
                setattr(prepared, name, backend.array(h["estimator/"+name][:]))
            prepared.diagnostics = json.loads(h.attrs["estimator_diagnostics"])
        return prepared
    LOG.info("Building %s independent 24 x 192 joint receiver", c["channel"])
    prepared = JointEstimator(prior, known, masks, noise, backend)
    partial = path.with_suffix(".partial.h5")
    with h5py.File(partial, "w") as h:
        h.attrs.update(schema=MODEL_SCHEMA, identity=identity, prepared=False, complete=False,
            config_json=json.dumps(c), provenance_json=json.dumps(provenance),
            reference_assisted_prior=True, raw_csi_unmodified=True,
            estimator_diagnostics=json.dumps(prepared.diagnostics))
        sampled = prior.model.sampled_basis.reshape(4, c["nfft"], 24)
        for name, value in dict(basis=prior.model.basis, sampled_basis=sampled,
                delay_indices=prior.model.delays, known=known, unit_rf_noise_covariance=noise,
                snr_db=c["snr_db"], sampled_grams=sampled_grams(sampled),
                prior_coefficients=prior.coefficients, prior_source_records=prior.source_records).items():
            h.create_dataset(name, data=value)
        for name, mask in masks.items():
            h.create_dataset("masks/"+name, data=mask)
        for i, (select, temporal, delay) in enumerate(prior.separable_blocks()):
            group = h.create_group(f"prior/blocks/{i:03d}")
            group.attrs.update(start=select.start, stop=select.stop)
            group.create_dataset("temporal", data=temporal)
            group.create_dataset("delay", data=delay)
        for name in ("values", "projected", "gain"):
            value = backend.numpy(getattr(prepared, name))
            h.create_dataset("estimator/"+name, data=value,
                **(dict(compression="lzf", chunks=(min(1024, value.shape[0]), min(128, value.shape[1]))) if value.ndim == 2 else {}))
        h.create_dataset("estimator/offsets", data=prepared.offsets)
        for s, rows in enumerate(prepared.rows):
            h.create_dataset(f"estimator/rows/{s}", data=rows)
        h["estimator"].attrs.update(ridge=prepared.ridge, noise_gain=prepared.noise_gain)
        h.attrs["prepared"] = True
    partial.replace(path)
    return prepared


def initialize_split(path, c, split, identity):
    physical = windows(c, split)
    repeats, ns, n = c["learning_dataset"]["repeats"][split], len(c["snr_db"]), c["nfft"]
    nw, nf = len(physical), len(physical)*repeats
    h = h5py.File(path, "w")
    h.attrs.update(schema=SCHEMA, model_file="model.h5", model_identity=identity, channel=c["channel"],
        split=split, config_hash=digest(c), complete=False, raw_complete=False,
        raw_csi_unmodified=True, reference_assisted_csi_adjustment=False,
        precision="complex128 physical construction; complex64 persisted reference and raw coefficients",
        snr_definition="four-symbol useful-window received power / expected filtered real-RF thermal noise")
    for name, value in {"snr_db": c["snr_db"], "windows/record": [r for r, _ in physical],
            "windows/start_seconds": [s for _, s in physical],
            "samples/window_index": np.repeat(np.arange(nw), repeats),
            "samples/repeat": np.tile(np.arange(repeats), nw)}.items():
        h.create_dataset(name, data=value)
    specs = {
        "windows/complete": ((nw,), bool), "windows/reference_sha256": ((nw,), "S64"),
        "windows/reference_energy": ((nw, 4), "float64"),
        "windows/reference_cross": ((nw, 4, 24, 192), "complex128"),
        "samples/complete": ((nf,), bool),
        "samples/clean_observations": ((nf, 4, n), "complex128"),
        "samples/standard_noise": ((nf, 4, n), "complex128"),
        "samples/model_residual": ((nf, 4, n), "complex128"),
        "samples/bits": ((nf, 4, c["data_count"], 2), "uint8"),
        "samples/rf_noise_variance": ((ns, nf), "float64"),
        "samples/effective_rf_variance": ((ns, nf), "float64"),
        "samples/raw_nmse_linear": ((ns, nf, 4), "float64"),
        "samples/actual_snr_db": ((ns, nf, 4), "float64"),
        "samples/physical_sha256": ((nf,), "S64"),
        "samples/payload_seed": ((nf, 6), "uint64"), "samples/noise_seed": ((nf, 6), "uint64"),
    }
    for name, (shape, dtype) in specs.items():
        h.create_dataset(name, shape, dtype=dtype)
    h.create_dataset("windows/reference_taps", (nw, 4, n, n), dtype="complex64",
        chunks=(1, 1, n, min(n, 128)), compression="lzf", shuffle=True)
    h.create_dataset("samples/raw_coefficients", (ns, nf, 24, 192), dtype="complex64",
        chunks=(1, 1, 24, 192), compression="lzf", shuffle=True)
    h.create_dataset("windows/diagnostics_json", (nw,), dtype=h5py.string_dtype("utf-8"))
    return h


def generate_split(path, run, c, split, archive, receiver, prepared, identity, *, pool=None, prefetch=1, stop_after=None):
    """Flush data before completion markers; resume only unfinished transactions."""
    path, run = Path(path), Path(run)
    with (h5py.File(path, "r+") if path.exists() else initialize_split(path, c, split, identity)) as h:
        if (h.attrs.get("schema") != SCHEMA or h.attrs.get("model_identity") != identity
                or h.attrs.get("config_hash") != digest(c) or h.attrs.get("split") != split):
            raise ValueError("Split identity mismatch; never overwrite another dataset")
        if h.attrs.get("raw_complete"):
            LOG.info("%s %s verified existing completed raw data", c["channel"], split)
            return
        repeats, ns = c["learning_dataset"]["repeats"][split], len(c["snr_db"])
        physical = windows(c, split)
        jobs = [(wi, split, record, start, repeats) for wi, (record, start) in enumerate(physical)
                if not np.all(h["samples/complete"][wi*repeats:(wi+1)*repeats])]
        model, masks = prepared.prior.model, carrier_masks(c)
        sampled = model.sampled_basis.reshape(4, c["nfft"], 24)
        grams = sampled_grams(sampled)
        began, generated = time.perf_counter(), 0
        for wi, packet in window_results(jobs, c, archive, receiver, pool=pool, prefetch=prefetch):
            truth = packet["reference"]
            cross, energy = reference_statistics(sampled, truth, model.delays)
            if h["windows/complete"][wi]:
                if h["windows/reference_sha256"][wi].decode() != packet["reference_sha256"]:
                    raise ValueError("Resumed physical reference differs from the saved window")
            else:
                h["windows/reference_taps"][wi] = truth
                h["windows/reference_sha256"][wi] = packet["reference_sha256"].encode()
                h["windows/reference_cross"][wi], h["windows/reference_energy"][wi] = cross, energy
                h["windows/diagnostics_json"][wi] = json.dumps(packet["diagnostics"])
                h.flush()
                h["windows/complete"][wi] = True
                h.flush()
            missing = [sample for sample in packet["samples"] if not h["samples/complete"][wi*repeats+sample["repeat"]]]
            observations = np.stack([sample["clean"][None]+np.sqrt(sample["variance"])[:, None, None]*sample["standard"][None] for sample in missing])
            variances = np.stack([sample["variance"] for sample in missing])
            coefficients = prepared.estimate_coefficients(observations.reshape(-1, 4, c["nfft"]),
                prepared.known, masks, variances.ravel()).reshape(len(missing), ns, 24, 192).astype("complex64")
            effective = prepared.effective_noise_variances(observations.reshape(-1, 4, c["nfft"]), variances.ravel()).reshape(len(missing), ns)
            for i, sample in enumerate(missing):
                fi = wi*repeats+sample["repeat"]
                s = h["samples"]
                for name, value in {"clean_observations": sample["clean"], "standard_noise": sample["standard"],
                        "model_residual": sample["model_residual"], "bits": sample["bits"],
                        "payload_seed": sample["payload_seed"], "noise_seed": sample["noise_seed"]}.items():
                    s[name][fi] = value
                s["rf_noise_variance"][:, fi] = sample["variance"]
                s["effective_rf_variance"][:, fi] = effective[i]
                s["raw_coefficients"][:, fi] = coefficients[i]
                s["raw_nmse_linear"][:, fi] = coefficient_nmse(coefficients[i], grams, cross, energy)
                realized = sample["variance"][:, None]*np.mean(abs(sample["standard"])**2, axis=1)[None]
                s["actual_snr_db"][:, fi] = 10*np.log10(np.mean(abs(sample["clean"])**2, axis=1)[None]/realized)
                s["physical_sha256"][fi] = sample["physical_sha256"].encode()
                h.flush()
                s["complete"][fi] = True
                h.flush()
            generated += 1
            completed = int(np.count_nonzero(h["samples/complete"][:]))
            progress = dict(channel=c["channel"], split=split, physical_window=wi+1, total_windows=len(physical),
                completed_frames_per_snr=completed, expected_frames_per_snr=len(physical)*repeats,
                elapsed_seconds=time.perf_counter()-began, window_seconds=packet["diagnostics"]["seconds"])
            write_json(run/"progress.json", progress)
            if generated == 1 or generated % 10 == 0 or wi+1 == len(physical):
                LOG.info("%s %s %d/%d physical windows, %d frames/SNR, %.1fs", c["channel"], split,
                    wi+1, len(physical), completed, progress["elapsed_seconds"])
            if stop_after is not None and generated >= stop_after:
                return
        h.attrs["raw_complete"] = bool(np.all(h["samples/complete"][:]) and np.all(h["windows/complete"][:]))
        h.flush()


def calibrate(directory, c, options, targets):
    """The selected scope alone determines the common per-SNR error scales."""
    directory = Path(directory)
    means = {}
    for split in SPLITS:
        with h5py.File(directory/f"{split}.h5", "r") as h:
            if not h.attrs.get("raw_complete"):
                raise ValueError("Cannot calibrate an incomplete raw split")
            means[split] = h["samples/raw_nmse_linear"][:].mean(axis=(1, 2))
    target = channel_targets(dict(targets, snr_db=c["snr_db"]), c["channel"])
    mode = options["compensation_calibration"]
    for split in SPLITS:
        scale = (np.ones(len(target)) if mode == "none" else
            error_scales(means["train"] if mode == "train_only" else means[split], target))
        with h5py.File(directory/f"{split}.h5", "r") as h:
            if (h.attrs.get("complete") and h.attrs.get("default_input_view") == options["default_input_view"]
                    and "compensation" in h and h["compensation"].attrs.get("calibration_scope") == mode
                    and np.array_equal(h["compensation/error_scales"][:], scale)
                    and np.array_equal(h["compensation/target_nmse_db"][:], target)):
                LOG.info("%s %s verified existing target calibration", c["channel"], split)
                continue
        with h5py.File(directory/f"{split}.h5", "r+") as h:
            h.attrs["complete"] = False
            h.flush()
            if "compensation" in h:
                del h["compensation"]
            group = h.create_group("compensation")
            group.attrs.update(calibration_scope=mode, source_split="train" if mode == "train_only" else split,
                reference_assisted=(mode != "none"),
                definition="H_ref + alpha_s*(zero_extended_H_raw-H_ref); common alpha across every frame and symbol")
            group.create_dataset("target_nmse_db", data=target)
            group.create_dataset("error_scales", data=scale)
            group.create_dataset("raw_mean_nmse_linear", data=means[split])
            group.create_dataset("assisted_mean_nmse_linear", data=scale**2*means[split])
            h.attrs.update(default_input_view=options["default_input_view"],
                reference_assisted_csi_adjustment=(mode != "none"), complete=True)
            h.flush()
        LOG.info("%s %s target calibration: raw=%s; assisted=%s", c["channel"], split,
            np.round(10*np.log10(means[split]), 3).tolist(), np.round(10*np.log10(means[split]*scale**2), 3).tolist())


def generate(directory, run, c, options, targets, archive, receiver, prior, backend, identity, provenance,
             *, workers=1, worker_threads=2):
    directory, run = Path(directory), Path(run)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = prepare_model(directory/"model.h5", c, prior, receiver.chain.noise_covariance, backend, identity, provenance)
    context = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker, initargs=(c, str(archive.path), worker_threads)) if workers > 1 else nullcontext(None))
    with context as pool:
        for split in SPLITS:
            generate_split(directory/f"{split}.h5", run, c, split, archive, receiver, prepared, identity,
                pool=pool, prefetch=max(1, workers*2))
    calibrate(directory, c, options, targets)
    with h5py.File(directory/"model.h5", "r+") as h:
        if not h.attrs.get("complete"):
            h.attrs["complete"] = True
    del prepared
    if backend.torch:
        backend.torch.cuda.empty_cache()
    return counts(c)
