"""Paired NCS1 study of 64 physical taps selected from independent priors."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT, digest, load_config
from data_generation.src.receiver.delay_support import (
    SelectedDelayDPSSModel, delay_candidates, profile_full_frame,
)
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.pdd import JointEstimator
from data_generation.src.transmitter.ofdm import carrier_masks, decisions, fft, receive_symbols
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_lmmse import CODE_FILES, detect, model_basis
from src.evaluation.four_data_q_sweep import BlockedReceiver, make_frame


CODE = (*CODE_FILES, "data_generation/src/receiver/delay_support.py",
        "src/evaluation/ncs1_delay_support.py", "scripts/evaluate/ncs1_delay_support.py")
LABELS = dict(dense176="Continuous 0:175 (176 taps)", first64="Continuous 0:63 (64 taps)",
              window64="Best prior window (64 taps)", selected64="Prior power selection (64 taps)")


def metadata(archive_path, backend, smoke):
    c = load_config(ROOT/"configs/data/ncs1/dddd.json")
    if (c["frame_structure"], c["delay_length"], c["dpss_expected_count"], c["dpss_half_bandwidth_hz"]) != ("DDDD", 176, 24, 12):
        raise ValueError("This study requires the current DDDD / L176 / Q24 / 12 Hz baseline")
    prior_records = c["prior_records"][:2] if smoke else c["prior_records"]
    starts = [2] if smoke else c["starts_seconds"]
    tests = c["learning_dataset"]["records"]["test"][:1 if smoke else 5]
    if set(prior_records) & set(tests):
        raise ValueError("Prior and test records must be disjoint")
    if smoke:
        c["snr_db"] = [0, 15, 30]
    return dict(schema="ncs1-independent-delay-support-v1", config=c, device=backend.device,
                smoke=smoke, prior_windows=[[r, s] for r in prior_records for s in starts],
                test_windows=[[r, 2] for r in tests],
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                code_sha256={name: file_sha256(ROOT/name) for name in CODE},
                archive_sha256=file_sha256(archive_path), precision="complex128",
                selection="64 largest mean physical tap powers over independent prior windows; no test CSI",
                prior="full-frame coefficients from the same independent windows for every candidate",
                raw_csi=True, csi_compensation=False,
                reference_use="test scoring and ideal detection only, after support is frozen",
                nmse="mean frame/symbol full-matrix linear NMSE, including every omitted tap; then 10 log10",
                delay_convention="zero-based receiver delay; physical excess delay = (index - fft_advance) / fs_baseband")


def prepare(output, archive_path, meta):
    path = output/"physical.h5"
    identity = digest(meta)
    if path.exists():
        with h5py.File(path, "r") as h:
            if h.attrs.get("complete") and h.attrs.get("identity") == identity:
                return path
        raise ValueError("Existing physical data belong to a different study")
    c = meta["config"]
    archive, receiver = WatermarkArchive(archive_path, "NCS1"), BlockedReceiver(c)
    basis, concentration = model_basis(c)
    partial = output/"physical.partial.h5"
    with h5py.File(partial, "w") as h:
        h.attrs.update(identity=identity, complete=False)
        h.create_dataset("basis", data=basis)
        h.create_dataset("concentration", data=concentration)
        h.create_dataset("unit_noise", data=receiver.chain.noise_covariance)
        known = make_frame(c, *meta["test_windows"][0], 2).known
        h.create_dataset("known", data=known)
        coefficients, powers, native_powers = [], [], []
        for i, (record, start) in enumerate(meta["prior_windows"]):
            begin = time.perf_counter()
            sounding = archive.load(record)
            coeff, power = profile_full_frame(receiver, sounding, start, basis)
            coefficients.append(coeff)
            powers.append(power)
            if start == meta["prior_windows"][0][1]:
                native_powers.append(np.mean(abs(sounding.h)**2, axis=0))
            print(f"prior {i+1}/{len(meta['prior_windows'])}: {record}@{start}s, {time.perf_counter()-begin:.2f}s", flush=True)
        h.create_dataset("prior", data=coefficients)
        h.create_dataset("prior_power", data=powers)
        h.create_dataset("native_power", data=np.mean(native_powers, axis=0))
        for i, (record, start) in enumerate(meta["test_windows"]):
            begin = time.perf_counter()
            sounding, frame = archive.load(record), make_frame(c, record, start, 2)
            np.testing.assert_array_equal(frame.known, known)
            origin = start-c["guard_samples"]/c["fs_baseband"]
            clean_rf = receiver.replay(sounding, origin, frame)
            clean = receive_symbols(receiver.chain.receive(clean_rf), c)
            noise_seed = [c["seed"], 2, record, round(start*1000), 1, 0]
            rng = np.random.default_rng(np.random.SeedSequence(noise_seed))
            standard = receive_symbols(receiver.chain.receive(rng.standard_normal(len(clean_rf))), c)
            variance = np.mean(abs(clean)**2)/receiver.chain.noise_gain/10**(np.asarray(c["snr_db"])/10)
            refs = [receiver.reference(sounding, start, d) for d in range(4)]
            truth = np.stack([item[0] for item in refs])
            predicted = fft(np.sum(truth*input_windows(frame.x, c["nfft"]), axis=-1))
            g = h.create_group(f"test/{i:04d}")
            g.attrs.update(record=record, start_seconds=start, sounding_sha256=sounding.sha256,
                           noise_seed=json.dumps(noise_seed),
                           model_error=json.dumps((np.sum(abs(predicted-clean)**2, axis=1)/np.sum(abs(clean)**2, axis=1)).tolist()),
                           intersymbol_leakage=json.dumps([item[1] for item in refs]))
            for name, value in dict(truth=truth, clean=clean, standard=standard, variance=variance, bits=frame.bits).items():
                g.create_dataset(name, data=value, compression="lzf" if name == "truth" else None)
            h.flush()
            print(f"test {i+1}/{len(meta['test_windows'])}: {record}@{start}s, {time.perf_counter()-begin:.2f}s", flush=True)
        h.attrs["complete"] = True
    partial.replace(path)
    return path


def freeze_supports(path, output):
    # This function reads independent priors only, never the test group.
    with h5py.File(path, "r") as h:
        powers = h["prior_power"][:]
    power = powers.mean(axis=0)
    supports = delay_candidates(power)
    report = dict(selection_source="independent prior windows only", mean_power=power.tolist(),
                  candidates={name: dict(indices=idx.tolist(), count=len(idx), span=int(idx[-1])+1,
                    prior_mean_coverage=float(np.mean(powers[:, idx].sum(axis=1)/powers.sum(axis=1))),
                    prior_aggregate_coverage=float(power[idx].sum()/power.sum())) for name, idx in supports.items()})
    write_json(output/"selected_support.json", report)
    return supports, report


def full_nmse(selected_taps, truth, delays):
    """Frobenius error under the unitary time/frequency transform, without cropping truth."""
    energy = np.sum(abs(truth)**2, axis=(1, 2))
    retained = truth[..., delays]
    omitted = np.maximum(energy-np.sum(abs(retained)**2, axis=(1, 2)), 0)
    return (np.sum(abs(selected_taps-retained[None])**2, axis=(2, 3))+omitted[None])/energy[None]


def evaluate(path, output, meta, supports, backend):
    c = meta["config"]
    layout, results = carrier_masks(c), {}
    with h5py.File(path, "r") as h:
        known, noise, basis = h["known"][:], h["unit_noise"][:], h["basis"][:]
        groups = list(h["test"].values())
        observations = np.stack([g["clean"][:][None]+np.sqrt(g["variance"][:])[:, None, None]*g["standard"][:][None] for g in groups])
        variances = np.stack([g["variance"][:] for g in groups])
        prior_coefficients = h["prior"][:]
        for name, delays in supports.items():
            result_path = output/f"{name}_metrics.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved["identity"] != digest(dict(metadata=meta, name=name, delays=delays.tolist())):
                    raise ValueError("Saved metrics do not match this support study")
                results[name] = saved
                continue
            model = SelectedDelayDPSSModel(basis, 12, c["dpss_tail_ratio"], len(delays), c, delays)
            prior = FrozenPrior.fit_coefficients(model, prior_coefficients[..., delays], 0, 1,
                sorted({r for r, _ in meta["prior_windows"]}), regularizer=c["relative_regularizer"],
                delay_block_size=c["delay_block_size"], interference_scale=1)
            if backend.torch:
                backend.torch.cuda.synchronize()
            begin = time.perf_counter()
            estimator = JointEstimator(prior, known, layout, noise, backend)
            if backend.torch:
                backend.torch.cuda.synchronize()
            build_seconds = time.perf_counter()-begin
            begin = time.perf_counter()
            coeff = estimator.estimate_coefficients(observations.reshape(-1, 4, c["nfft"]), known,
                layout, variances.ravel()).reshape(len(groups), len(c["snr_db"]), model.count, model.length)
            if backend.torch:
                backend.torch.cuda.synchronize()
            solve_seconds = time.perf_counter()-begin
            diagnostics = estimator.diagnostics
            cross_bytes = model.count*model.length*int(estimator.offsets[-1])*16
            del estimator, prior
            if backend.torch:
                backend.torch.cuda.empty_cache()
            frames = []
            sampled = model.sampled_basis.reshape(4, c["nfft"], model.count)
            for i, group in enumerate(groups):
                selected = np.einsum("dnq,sql->sdnl", sampled, coeff[i], optimize=True)
                taps = model.expand_taps(selected)
                single, full = detect(taps, observations[i], known, variances[i], noise, layout["data"], backend)
                bits = group["bits"][:]
                errors = [np.count_nonzero(decisions(symbols) != bits[None], axis=(-1, -2)).tolist() for symbols in (single, full)]
                # Held-out CSI is read after raw detection and cannot change it.
                truth = group["truth"][:]
                energy = np.sum(abs(truth)**2, axis=(1, 2))
                coverage = np.sum(abs(truth[..., delays])**2, axis=(1, 2))/energy
                frames.append(dict(record=int(group.attrs["record"]), start_seconds=float(group.attrs["start_seconds"]),
                    nmse_linear=full_nmse(selected, truth, delays).tolist(), support_coverage=coverage.tolist(),
                    single_errors=errors[0], full_errors=errors[1],
                    model_error=json.loads(group.attrs["model_error"]),
                    intersymbol_leakage=json.loads(group.attrs["intersymbol_leakage"])))
                print(f"{name}: detected {i+1}/{len(groups)}", flush=True)
            bits_per_snr = len(groups)*4*c["data_count"]*2
            nmse = np.mean([frame["nmse_linear"] for frame in frames], axis=(0, 2))
            curves = {}
            for algorithm in ("single", "full"):
                errors = np.sum([frame[f"{algorithm}_errors"] for frame in frames], axis=(0, 2))
                curves[algorithm] = dict(errors=errors.tolist(), ber=(errors/bits_per_snr).tolist())
            coverage = float(np.mean([frame["support_coverage"] for frame in frames]))
            report = dict(identity=digest(dict(metadata=meta, name=name, delays=delays.tolist())),
                frames=frames, snr_db=c["snr_db"], bits_per_snr=bits_per_snr, curves=curves,
                mean_nmse_linear=nmse.tolist(), mean_nmse_db=(10*np.log10(nmse)).tolist(),
                test_mean_coverage=coverage, support_only_nmse_floor_db=float(10*np.log10(max(1-coverage, 1e-300))),
                coefficient_count=model.count*model.length, delay_count=model.length, support_span=int(delays[-1])+1,
                cross_covariance_bytes=cross_bytes, build_seconds=build_seconds, estimate_seconds=solve_seconds,
                estimator_diagnostics=diagnostics)
            write_json(result_path, report)
            results[name] = report
            print(f"{name}: coverage={coverage:.4%}, 30dB NMSE={nmse[-1]:.5g}, build={build_seconds:.2f}s", flush=True)
        # One common ideal reference for all candidates, with identical observations.
        frames, oracle_coverages, oracle_counts = [], [], []
        for i, group in enumerate(groups):
            truth = group["truth"][:]
            single, full = detect(truth, observations[i], known, variances[i], noise, layout["data"], backend)
            bits = group["bits"][:]
            frames.append([np.count_nonzero(decisions(symbols) != bits[None], axis=(-1, -2)) for symbols in (single, full)])
            power = np.sum(abs(truth)**2, axis=1)
            ordered = np.sort(power, axis=-1)[:, ::-1]
            cumulative = np.cumsum(ordered, axis=-1)/power.sum(axis=-1)[:, None]
            oracle_coverages.extend(cumulative[:, 63].tolist())
            oracle_counts.extend([[int(np.searchsorted(row, threshold)+1) for threshold in (.90, .95, .99)] for row in cumulative])
        ideal = np.sum(frames, axis=(0, 3))
        bits_per_snr = len(groups)*4*c["data_count"]*2
        reference = {name: dict(errors=ideal[i].tolist(), ber=(ideal[i]/bits_per_snr).tolist())
                     for i, name in enumerate(("single", "full"))}
        reference["delay_diagnostics"] = dict(
            use="scoring-only oracle bounds; never used for support selection or raw estimation",
            best64_mean_coverage=float(np.mean(oracle_coverages)),
            best64_min_coverage=float(np.min(oracle_coverages)),
            best64_max_coverage=float(np.max(oracle_coverages)),
            coverage_thresholds=[.90, .95, .99],
            minimum_taps_mean=np.mean(oracle_counts, axis=0).tolist(),
            minimum_taps_max=np.max(oracle_counts, axis=0).tolist())
    return results, reference


def finish(output, meta, selection, results, reference, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_json(output/"metrics.json", dict(complete=True, metadata=meta, selection=selection,
                                         results=results, reference=reference))
    with (output/"summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate", "snr_db", "tap_count", "coefficient_count", "prior_coverage", "test_coverage",
                         "nmse_db", "single_ber", "full_ber", "bits_per_snr"])
        for name, r in results.items():
            for i, snr in enumerate(r["snr_db"]):
                writer.writerow([name, snr, r["delay_count"], r["coefficient_count"],
                    selection["candidates"][name]["prior_mean_coverage"], r["test_mean_coverage"],
                    r["mean_nmse_db"][i], r["curves"]["single"]["ber"][i], r["curves"]["full"]["ber"][i], r["bits_per_snr"]])
    colors = dict(zip(results, ("#222222", "#D55E00", "#009E73", "#0072B2"), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, algorithm in zip(axes, ("single", "full"), strict=True):
        for name, r in results.items():
            values = np.asarray(r["curves"][algorithm]["ber"])
            floor = .5/r["bits_per_snr"]
            ax.semilogy(r["snr_db"], np.maximum(values, floor), "o-", color=colors[name], label=LABELS[name])
            zero = values == 0
            ax.scatter(np.asarray(r["snr_db"])[zero], np.full(zero.sum(), floor), facecolors="white", edgecolors=colors[name], zorder=5)
        ax.semilogy(meta["config"]["snr_db"], np.maximum(reference[algorithm]["ber"], floor), "k--", label="Ideal reference CSI")
        ax.set(title="Single-tap LMMSE" if algorithm == "single" else "Full-matrix LMMSE", xlabel="SNR (dB)", ylabel="BER")
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"NCS1, DDDD, Q=24 | {r['bits_per_snr']:,} bits/SNR | open markers: zero errors")
    for ext in ("png", "pdf"):
        fig.savefig(output/f"ber_snr.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
    for name, r in results.items():
        ax.plot(r["snr_db"], r["mean_nmse_db"], "o-", color=colors[name], label=LABELS[name])
    ax.set(title="NCS1: mean full-matrix CSI NMSE, including omitted taps", xlabel="SNR (dB)", ylabel="Mean NMSE (dB)")
    ax.text(.03, .04, "Ideal reference: NMSE = 0 (-infinity dB)", transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"nmse_snr.{ext}", dpi=180)
    plt.close(fig)
    with h5py.File(path, "r") as h:
        native, powers = h["native_power"][:], h["prior_power"][:]
    power = powers.mean(axis=0)
    power /= power.sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    axes[0].plot(np.arange(len(native))/16, 10*np.log10(np.maximum(native/native.max(), 1e-12)))
    axes[0].set(title="Independent NCS1 recordings: native PDP", xlabel="Measured excess delay (ms)", ylabel="Relative power (dB)", ylim=(-60, 3))
    indices = np.asarray(selection["candidates"]["selected64"]["indices"])
    axes[1].plot(np.arange(len(power)), 10*np.log10(np.maximum(power/power.max(), 1e-12)), label="Physical receiver PDP")
    axes[1].scatter(indices, 10*np.log10(power[indices]/power.max()), s=17, color=colors["selected64"], label="Selected 64 taps", zorder=5)
    axes[1].axvline(63.5, color=colors["first64"], linestyle="--", label="End of first 64 taps")
    axes[1].axvline(175.5, color=".3", linestyle=":", label="End of baseline 176 taps")
    axes[1].set(title="Full-frame receiver PDP after actual RF/FIR chain", xlabel="Receiver delay index (zero based)", ylabel="Relative power (dB)",
                xlim=(0, max(256, indices[-1]+10)), ylim=(-60, 3))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"delay_profile.{ext}", dpi=180)
    plt.close(fig)


def run(output, archive_path, backend, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = metadata(archive_path, backend, smoke)
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != meta:
        raise ValueError("Output directory belongs to a different support study")
    write_json(manifest_path, meta)
    path = prepare(output, archive_path, meta)
    supports, selection = freeze_supports(path, output)
    results, reference = evaluate(path, output, meta, supports, backend)
    finish(output, meta, selection, results, reference, path)
    return output
