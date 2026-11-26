"""Validation-selected frequency-domain pilot clusters with guarded ends."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from functools import partial
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT
from data_generation.src.dataset.four_data_math import array_sha256, reference_statistics
from data_generation.src.dataset.four_data_physical import initialize_worker, replay_window, window_results
from data_generation.src.dataset.four_data_protocol import channel_config, windows
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.receiver.single_symbol import SymbolPilotDesign, symbol_basis
from data_generation.src.transmitter.ofdm import carrier_masks, make_frame
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_delay192 import select_prior_delays
from src.evaluation.four_data_edge_comb import allocation, compare_source, previous_test, replay_observations, split_result
from src.evaluation.four_data_q_sweep import BlockedReceiver
from src.evaluation.four_data_single_symbol import (
    dataset_manifest, db, estimate_frames, fit_independent_prior, fit_model, project_references, projection_nmse,
    verify_file, verify_split, write_csv,
)


CONFIG = "configs/evaluation/four_data_protected_pilots.json"
LAYOUT_FIELDS = ("clusters", "pilots_per_cluster", "guard_nulls_per_side", "null_per_edge")
PARAMETERS = ("bandwidth_hz", "q", "delay_count", "model_error_ratio", "ber_frames_per_channel")


def cluster_config(base, spec):
    fields = {key: spec[key] for key in LAYOUT_FIELDS}
    if (any(type(value) is not int for value in fields.values()) or min(fields[k] for k in LAYOUT_FIELDS[:3]) < 1
            or not 0 <= fields["null_per_edge"] < base["nfft"]//2):
        raise ValueError("Cluster sizes, guards and edge counts must be valid integers")
    return dict(base, null_per_edge=fields["null_per_edge"], pilot_count=fields["clusters"]*fields["pilots_per_cluster"],
                pilot_cluster={key: fields[key] for key in LAYOUT_FIELDS[1:3]})


def default_layout(channel):
    c = channel_config(channel)
    spec = c.get("pilot_cluster")
    if spec is None:
        raise ValueError(f"{channel} default frame must declare a protected pilot cluster")
    carrier_masks(c)
    layout = dict(clusters=c["pilot_count"]//spec["pilots_per_cluster"], **spec, null_per_edge=c["null_per_edge"])
    if cluster_config(channel_config(channel, legacy_comb=True), layout) != c:
        raise ValueError(f"{channel} default physical frame differs from the accepted CP80 replay source")
    return layout


def load_options(path=None, *, smoke=False):
    options = json.loads(Path(path or ROOT/CONFIG).read_text(encoding="utf-8"))
    specs = options["candidates"]
    if (options["channels"] != ["NOF1", "NCS1"] or not specs or type(options["delay_count"]) is not int
            or not 1 <= options["delay_count"] <= 1024 or not 0 <= options["model_error_ratio"] <= 1
            or type(options["ber_frames_per_channel"]) is not int or options["ber_frames_per_channel"] < 1
            or len({tuple(item[k] for k in LAYOUT_FIELDS) for item in specs}) != len(specs)):
        raise ValueError("Invalid protected-pilot candidates")
    base = channel_config("NOF1")
    for spec in specs:
        carrier_masks(cluster_config(base, spec))
    if path is None:
        layouts = [default_layout(channel) for channel in options["channels"]]
        if layouts[0] != layouts[1]:
            raise ValueError("NOF1 and NCS1 must use the same default protected pilot layout")
        options["candidates"] = layouts[:1]
    elif smoke:
        options["candidates"] = specs[:3]
    return options


@dataclass(frozen=True)
class ChannelStudy:
    config: dict
    source: Path
    archive: WatermarkArchive
    receiver: BlockedReceiver
    prior: FrozenPrior
    prior_info: dict
    accepted: dict | None


def prepare(channel, data, archive_path, archive_sha, options, smoke):
    # The source contains comb-era taps and pairing streams. All observations are replayed.
    c = channel_config(channel, smoke, legacy_comb=True)
    source = data/(channel.lower()+"_dddd")
    accepted, source_sha = (None, None) if smoke else dataset_manifest(source, c, archive_sha)
    archive, receiver = WatermarkArchive(archive_path, channel), BlockedReceiver(c)
    if accepted is not None:
        with h5py.File(source/"model.h5", "r") as h:
            np.testing.assert_allclose(receiver.chain.noise_covariance, h["unit_rf_noise_covariance"][:], rtol=1e-10, atol=1e-12)
    band = options["bandwidth_hz"]
    basis = symbol_basis(c["nfft"], c["fs_baseband"], band, options["q"])
    coeff, powers, records, identities = fit_independent_prior(c, archive, {band: basis}, smoke)
    delays = select_prior_delays(powers, options["delay_count"])
    prior = fit_model(c, basis, delays, coeff[band][..., delays], records)
    info = dict(basis_sha256=array_sha256(basis), prior_windows=identities, delay_indices=delays.tolist(),
        prior_delay_coverage=float(powers[:, delays].sum()/powers.sum()), physical_source_manifest_sha256=source_sha)
    return ChannelStudy(c, source, archive, receiver, prior, info, accepted)


@contextmanager
def accepted_split(study, split):
    if study.accepted is None:
        yield None
        return
    path = study.source/f"{split}.h5"
    verify_file(path, study.accepted)
    with h5py.File(path, "r") as h:
        verify_split(h, study.config, split, study.accepted)
        yield h


def replay_pool(study, workers):
    if workers == 1:
        return nullcontext()
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker, initargs=(study.config, study.archive.path, 1))


def replay_cluster_window(c, archive, receiver, job, *, spec):
    return replay_window(cluster_config(c, spec), archive, receiver, job)


def replay_split(study, split, spec, pool, workers):
    c = study.config
    repeats = c["learning_dataset"]["repeats"][split]
    jobs = [(wi, split, record, start, repeats) for wi, (record, start) in enumerate(windows(c, split))]
    return window_results(jobs, c, study.archive, study.receiver, pool=pool, prefetch=2*workers,
                          compute=partial(replay_cluster_window, spec=spec))


def prepare_estimators(study, c, backend):
    prior = replace(study.prior, model=replace(study.prior.model, c=c))
    model = prior.model
    known = make_frame(c, 1, 2).known
    noise = study.receiver.chain.noise_covariance
    design = SymbolPilotDesign(model.basis, model.delays, known, carrier_masks(c), noise, backend)
    observations = len(design.rows)
    matrices = backend.numpy(design.responses).reshape(len(known), observations, -1)
    ranks = np.linalg.matrix_rank(matrices).tolist()
    return design.build(prior), known, ranks


def validation_references(study, old, plan):
    basis, delays = study.prior.model.basis, study.prior.model.delays
    if old is not None:
        return project_references(old, range(len(plan)), basis, delays)
    sampled = np.broadcast_to(basis, (4, *basis.shape))
    projections, energies = [], []
    for record, start in plan:
        sounding = study.archive.load(record)
        truth = np.stack([study.receiver.reference(sounding, start, s)[0] for s in range(4)]).astype(np.complex64)
        cross, energy = reference_statistics(sampled, truth, delays)
        projections.append(cross)
        energies.append(energy)
    return np.stack(projections), np.stack(energies)


def validate(study, options, backend, workers):
    plan = windows(study.config, "validation")
    repeats = study.config["learning_dataset"]["repeats"]["validation"]
    nframes = len(plan)*repeats
    n_snr = len(study.config["snr_db"])
    rows = []
    with accepted_split(study, "validation") as old, replay_pool(study, workers) as pool:
        projection, energy = validation_references(study, old, plan)
        for identity, spec in enumerate(options["candidates"], start=1):
            c = cluster_config(study.config, spec)
            estimators, _, ranks = prepare_estimators(study, c, backend)
            scores = np.full((n_snr, nframes, 4), np.nan)
            changed = 0
            for wi, packet in replay_split(study, "validation", spec, pool, workers):
                frame = slice(wi*repeats, (wi+1)*repeats)
                y, variance, _, count = replay_observations(packet, old, frame)
                coeff = estimate_frames(estimators, y, variance, options["model_error_ratio"])
                scores[:, frame] = projection_nmse(coeff, projection[wi], energy[wi])
                changed += count
            if not np.all(np.isfinite(scores)) or (old is not None and changed != nframes):
                raise ValueError("Validation must use every independently replayed carrier frame")
            mean = float(scores.mean())
            rows.append(dict(layout_id=identity, **spec, counts=allocation(c)["counts"], ranks=ranks,
                frames_per_snr=nframes, mean_nmse_linear=mean, all_snr_nmse_db=float(db(mean)),
                mean_nmse_db=db(scores.mean(axis=(1, 2))).tolist(), paired_noise_and_payload=old is not None))
            print(f"{c['channel']} validation {identity}/{len(options['candidates'])}: "
                  f"{spec['clusters']} clusters, {c['pilot_count']} pilots, NMSE {float(db(mean)):.3f} dB", flush=True)
            del estimators
    return rows


def choose(validation, options):
    channels = options["channels"]
    n = len(options["candidates"])
    rows = []
    for identity, trials in enumerate(zip(*(validation[channel] for channel in channels), strict=True), start=1):
        if any(trial["layout_id"] != identity for trial in trials):
            raise ValueError("Validation results must be paired by candidate order")
        score = float(np.mean([row["mean_nmse_linear"] for row in trials]))
        rows.append(dict(layout_id=identity, **options["candidates"][identity-1], **trials[0]["counts"],
            **{f"{channel}_nmse_db": trial["all_snr_nmse_db"] for channel, trial in zip(channels, trials, strict=True)},
            joint_nmse_linear=score, joint_nmse_db=float(db(score))))
    if len(rows) != n:
        raise ValueError("The selection requires every configured validation candidate")
    ranked = sorted(rows, key=lambda row: (row["joint_nmse_linear"], row["pilot"], row["layout_id"]))
    best = {channel: min(validation[channel], key=lambda row: (row["mean_nmse_linear"], row["counts"]["pilot"], row["layout_id"]))["layout_id"]
            for channel in channels}
    return ranked, dict(selected=ranked[0], best_layout_id_by_channel=best,
        rule="Equal-channel mean linear full-matrix NMSE across validation symbols and SNR; ties prefer fewer pilots")


def edge_source(path, options, archive_sha):
    manifest_path = path/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads((path/"metrics.json").read_text(encoding="utf-8"))
    if (not metrics["complete"] or manifest["archive_sha256"] != archive_sha or manifest["smoke"]
            or any(manifest["options"][key] != options[key] for key in PARAMETERS)):
        raise ValueError("The edge-only comparison must share the fixed single-symbol settings")
    return metrics["channels"], dict(path=str(path), manifest_sha256=file_sha256(manifest_path))


def edge_test(path, report, study, test, previous_nmse, identities):
    prior = report["prior"]
    if (report["physical_source_manifest_sha256"] != study.prior_info["physical_source_manifest_sha256"]
            or prior["delay_indices"] != study.prior_info["delay_indices"]
            or prior["basis_sha256"] != study.prior_info["basis_sha256"]):
        raise ValueError("The edge-only comparison must share the physical source and prior")
    old = report["test"]
    if any(test[key] != old[key] for key in ("diagnostic_bits_per_snr", "diagnostic_ber_frames", "frames_per_snr", "snr_db")):
        raise ValueError("The edge-only test rows differ from the selected cluster test")
    path = path/"test_nmse.npz"
    with np.load(path) as pairs:
        for key, values in identities.items():
            np.testing.assert_array_equal(pairs[key], values)
        np.testing.assert_array_equal(pairs["inside_null_nmse_linear"], previous_nmse)
        edge_nmse = pairs["edge_comb_nmse_linear"]
    np.testing.assert_allclose(db(edge_nmse.mean(axis=(1, 2))), old["mean_nmse_db"], rtol=0, atol=1e-12)
    return {key: old[key] for key in ("mean_nmse_db", "all_snr_nmse_db", "ber", "bit_errors")}, edge_nmse


def test_selected(study, output, spec, options, backend, workers, smoke, baseline, edge_dir, edge_report):
    c = cluster_config(study.config, spec)
    estimators, known, ranks = prepare_estimators(study, c, backend)
    with accepted_split(study, "test") as old, replay_pool(study, workers) as pool:
        replay = replay_split(study, "test", spec, pool, workers)
        test, nmse = split_result("test", c, old, replay, study.archive, study.receiver, options,
                                  estimators, known, backend, smoke)
    records, starts = zip(*windows(c, "test"), strict=True)
    repeats = c["learning_dataset"]["repeats"]["test"]
    identities = dict(snr_db=c["snr_db"], record=np.repeat(records, repeats), start_seconds=np.repeat(starts, repeats),
        repeat=np.tile(np.arange(repeats), len(records)), ber_diagnostic_frames=test["diagnostic_ber_frames"])
    comparisons = {}
    pairs = dict(**identities, protected_nmse_linear=nmse)
    if not smoke:
        comparisons["inband_comb"], previous_nmse = previous_test(baseline, c, study.accepted, study.prior, test)
        comparisons["edge_comb"], edge_nmse = edge_test(edge_dir, edge_report, study, test, previous_nmse, identities)
        pairs.update(inband_comb_nmse_linear=previous_nmse, edge_comb_nmse_linear=edge_nmse)
    np.savez_compressed(output/"paired_test_nmse.npz", **pairs)
    result = dict(prior=study.prior_info, test=test, ranks=ranks, comparisons=comparisons)
    write_json(output/"test.json", result)
    return result


def finish(output, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = dict(protected=("Protected clusters", "#146C7C", "o"),
                  inband_comb=("Comb, in-band nulls", "#A45534", "s"),
                  edge_comb=("Comb, edge nulls", "#7467A6", "^"))
    rows = []
    for metric, ylabel in (("nmse", "Mean full-matrix CSI NMSE (dB)"), ("ber", "Bit error rate")):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), layout="constrained", sharey=True)
        for ax, (channel, report) in zip(axes, reports.items(), strict=True):
            test = report["test"]
            methods = dict(protected=test, **report["comparisons"])
            snr = test["snr_db"]
            ax.set(xlabel="SNR (dB)", ylabel=ylabel, xticks=snr)
            if metric == "nmse":
                ax.set_title(f"{channel} · {test['symbols_per_snr']:,} symbols/SNR")
                for key, summary in methods.items():
                    label, color, marker = styles[key]
                    ax.plot(snr, summary["mean_nmse_db"], color=color, marker=marker, label=label,
                            linestyle="-" if key == "protected" else "--")
                for i, s in enumerate(snr):
                    rows.append(dict(channel=channel, snr_db=s, symbols_per_snr=test["symbols_per_snr"],
                        diagnostic_bits_per_snr=test["diagnostic_bits_per_snr"],
                        **{f"{key}_nmse_db": summary["mean_nmse_db"][i] for key, summary in methods.items()},
                        **{f"{key}_{detector}_ber": summary["ber"][d][i] for key, summary in methods.items()
                           for d, detector in enumerate(("single_tap", "full_matrix"))}))
            else:
                ax.set_title(f"{channel} · {test['diagnostic_bits_per_snr']:,} bits/SNR")
                lower = .5/test["diagnostic_bits_per_snr"]
                for key, summary in methods.items():
                    label, color, marker = styles[key]
                    for d, (algorithm, line) in enumerate((("Single-tap", ":"), ("Full-matrix", "-"))):
                        ax.semilogy(snr, np.maximum(summary["ber"][d], lower), color=color, marker=marker, linestyle=line,
                                    label=f"{algorithm} / {label}")
            ax.grid(alpha=.23, which="both")
            ax.legend(fontsize=7.5)
        fig.suptitle("Protected pilots · single-symbol raw CSI · CP 80 ms" if metric == "nmse"
                     else "Paired LMMSE detection · diagnostic subset")
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{metric}_snr.{ext}", dpi=170)
        plt.close(fig)
    write_csv(output/"summary.csv", rows)


def run(output, data, archive_path, options, backend, *, workers=1, smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    archive_sha = file_sha256(archive_path)
    baseline, edge_dir = (ROOT/options[key] for key in ("baseline", "edge_comb"))
    baseline_meta = None if smoke else compare_source(baseline, options, archive_sha)
    edge_reports, edge_meta = ({}, None) if smoke else edge_source(edge_dir, options, archive_sha)
    manifest = dict(schema="cp80-single-symbol-protected-pilots-v1", options=options, smoke=smoke,
        device=backend.device, archive_sha256=archive_sha, input_data=None if smoke else str(data),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        baseline=baseline_meta, edge_comb=edge_meta, pairing="Identical physical windows, payload bits and standard RF noise",
        snr_definition="Allocation-specific mean useful received power divided by expected filtered thermal noise power",
        observations="Pilot tones and the in-band guard nulls of the current symbol")
    path = output/"manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output belongs to a different protected pilot experiment")
    write_json(path, manifest)
    studies, validation = {}, {}
    for channel in options["channels"]:
        study = prepare(channel, data, archive_path, archive_sha, options, smoke)
        studies[channel] = study
        validation[channel] = validate(study, options, backend, workers)
        directory = output/channel.lower()
        directory.mkdir(exist_ok=True)
        write_json(directory/"validation.json", dict(snr_db=study.config["snr_db"], trials=validation[channel]))
    ranked, selection = choose(validation, options)
    write_csv(output/"validation_search.csv", ranked)
    write_json(output/"selection.json", selection)
    winning_id = selection["selected"]["layout_id"]
    spec = options["candidates"][winning_id-1]
    write_json(output/"carrier_allocation.json", allocation(cluster_config(studies["NOF1"].config, spec)))
    print(f"Shared layout {winning_id} selected from validation: {spec}", flush=True)
    reports = {}
    for channel, study in studies.items():
        reports[channel] = test_selected(study, output/channel.lower(), spec, options, backend, workers, smoke,
                                         baseline, edge_dir/channel.lower(), edge_reports.get(channel))
        if reports[channel]["ranks"] != validation[channel][winning_id-1]["ranks"]:
            raise ValueError("Test pilot matrix differs from validation")
    if len({report["prior"]["basis_sha256"] for report in reports.values()}) != 1:
        raise ValueError("Both channels must use the same single-symbol time basis")
    finish(output, reports)
    write_json(output/"metrics.json", dict(complete=True, selection=selection, channels=reports,
                                          wall_seconds=time.perf_counter()-began))
    return reports
