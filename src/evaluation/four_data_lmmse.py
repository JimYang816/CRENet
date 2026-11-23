"""Small, paired WATERMARK experiment with raw and ideal-reference CSI."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
from scipy.signal.windows import dpss

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT, digest, load_config
from data_generation.src.receiver.chain import relative_energy, taps_to_matrix
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.pdd import JointEstimator
from data_generation.src.transmitter.ofdm import carrier_masks, decisions, fft, receive_symbols
from src.algorithms.full_matrix_lmmse.equalizer import equalize_batch as full_equalize
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.algorithms.single_tap_mmse.equalizer import equalize_batch as single_equalize, equalize_diagonal
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_q_sweep import BlockedReceiver, make_frame, masks, raw_nmse, time_noise_covariance


CODE_FILES = (
    "src/evaluation/four_data_lmmse.py", "scripts/evaluate/four_data_lmmse.py",
    "src/evaluation/four_data_q_sweep.py", "data_generation/src/receiver/pdd.py",
    "data_generation/src/receiver/estimation.py", "data_generation/src/receiver/chain.py",
    "data_generation/src/receiver/dpss.py", "data_generation/src/transmitter/ofdm.py",
    "data_generation/src/channel/watermark/replay.py", "src/algorithms/single_tap_mmse/equalizer.py",
    "src/algorithms/full_matrix_lmmse/equalizer.py", "src/algorithms/likelihood.py", "src/algorithms/interface.py",
    "src/common/numerics.py", "src/common/experiment.py", "data_generation/src/config.py", "data_generation/src/pdd.py",
)


def load_run_config(path):
    options = json.loads(Path(path).read_text(encoding="utf-8"))
    if (options.get("channels") != ["NOF1", "NCS1"] or options.get("input_view") != "raw"
            or options.get("algorithms") != ["single_tap_mmse", "full_matrix_lmmse"]):
        raise ValueError("This experiment requires both channels, raw CSI and the two fixed LMMSE baselines")
    prior, test = options["prior_records"], options["test_records"]
    if not prior or not test or len(set(prior)) != len(prior) or len(set(test)) != len(test) or set(prior) & set(test):
        raise ValueError("Prior and test records must be nonempty, unique and disjoint")
    return options


def channel_config(channel):
    c = load_config(ROOT/f"configs/data/{channel.lower()}/dddd.json")
    if (c["channel"] != channel or c["frame_structure"] != "DDDD"
            or (c["data_count"], c["pilot_count"], c["dpss_expected_count"], c["delay_length"]) != (512, 256, 24, 176)
            or c["dpss_half_bandwidth_hz"] != 12 or c["snr_db"] != list(range(0, 31, 5))):
        raise ValueError("The data configuration does not match the fixed four-data experiment")
    return c


def model_basis(c):
    total = 4*(c["nfft"]+c["cp"])
    nw = total*c["dpss_half_bandwidth_hz"]/c["fs_baseband"]
    vectors, concentration = dpss(total, nw, Kmax=c["dpss_expected_count"], sym=True,
                                  norm=2, return_ratios=True)
    return vectors.T.copy(), concentration


def basis_diagnostics(c, concentration):
    total = 4*(c["nfft"]+c["cp"])
    nw = total*c["dpss_half_bandwidth_hz"]/c["fs_baseband"]
    if len(concentration) != c["dpss_expected_count"]:
        raise ValueError("DPSS concentration count does not match the fixed order")
    return dict(order=c["dpss_expected_count"], time_bandwidth_product=float(nw),
                tail_ratio=float(max(0., 1-np.sum(concentration)/(2*nw))),
                tail_threshold=float(c["dpss_tail_ratio"]), tail_threshold_enforced=False,
                order_selection="explicit fixed order")


def manifest(options, archive, backend):
    configs = {ch: channel_config(ch) for ch in options["channels"]}
    for ch, c in configs.items():
        if (not set(options["prior_records"]).issubset(c["prior_records"])
                or not set(options["test_records"]).issubset(c["learning_dataset"]["records"]["test"])
                or options["start_seconds"] not in c["learning_dataset"]["starts_seconds"]):
            raise ValueError(f"{ch}: selected windows are not a subset of the existing prior/test split")
    return dict(schema="four-data-raw-lmmse-v1", git_head=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
                code_sha256={name: file_sha256(ROOT/name) for name in CODE_FILES},
                archive_sha256=file_sha256(archive), options=options, configs=configs,
                dpss_diagnostics={ch: basis_diagnostics(c, model_basis(c)[1]) for ch, c in configs.items()},
                input_view="raw", csi_compensation=False, precision="complex128", device=backend.device,
                prior="full-frame reference kernels from independent prior records only",
                reference_use="ideal-CSI detector and NMSE scoring only; never used to modify estimated CSI",
                snr_definition="mean received useful-window power over four D symbols / expected filtered RF-noise power",
                equalizers="single-tap uses raw data-bin observations; full-matrix cancels known pilots and uses all receive bins and colored thermal noise",
                nmse_definition="10*log10(mean over frames and four D symbols of ||H_est-H_ref||_F^2/||H_ref||_F^2)")


def prepare(channel, output, archive_path, metadata):
    c, options = metadata["configs"][channel], metadata["options"]
    directory = Path(output)/channel.lower()
    directory.mkdir(parents=True, exist_ok=True)
    path, partial = directory/"physical.h5", directory/"physical.partial.h5"
    identity = digest(dict(schema=metadata["schema"], config=c, options=options,
                           archive=metadata["archive_sha256"], code=metadata["code_sha256"]))
    if path.exists():
        with h5py.File(path, "r") as h:
            if h.attrs.get("complete") and h.attrs.get("identity") == identity:
                print(f"{channel}: verified existing physical data", flush=True)
                return path
        raise ValueError(f"Physical data belongs to a different run: {path}")
    archive, receiver = WatermarkArchive(archive_path, channel), BlockedReceiver(c)
    basis, concentration = model_basis(c)
    start = options["start_seconds"]
    with h5py.File(partial, "a") as h:
        if "identity" in h.attrs and h.attrs["identity"] != identity:
            raise ValueError("Partial physical data belongs to a different run")
        h.attrs.update(identity=identity, channel=channel, complete=False)
        if "basis" not in h:
            h.create_dataset("basis", data=basis)
            h.create_dataset("concentration", data=concentration)
            h.create_dataset("unit_noise", data=receiver.chain.noise_covariance)
            h.create_dataset("known", data=make_frame(c, options["test_records"][0], start, 2).known)
            h.create_dataset("prior", (len(options["prior_records"]), c["dpss_expected_count"], c["delay_length"]), dtype=np.complex128)
            h.create_dataset("prior_complete", data=np.zeros(len(options["prior_records"]), dtype=bool))
        np.testing.assert_allclose(h["basis"][:], basis, rtol=0, atol=1e-13)
        for i, record in enumerate(options["prior_records"]):
            if h["prior_complete"][i]:
                continue
            begin = time.perf_counter()
            h["prior"][i] = receiver.prior(archive.load(record), start, basis)
            h.flush()
            h["prior_complete"][i] = True
            h.flush()
            print(f"{channel}: prior {i+1}/{len(options['prior_records'])}, record {record}, {time.perf_counter()-begin:.1f}s", flush=True)
        for i, record in enumerate(options["test_records"]):
            key = f"test/{i:04d}"
            if key in h and h[key].attrs.get("complete"):
                continue
            if key in h:
                del h[key]
            begin = time.perf_counter()
            sounding = archive.load(record)
            frame = make_frame(c, record, start, 2)
            if not np.array_equal(frame.known, h["known"][:]):
                raise ValueError("Public pilot symbols changed between frames")
            origin = start-c["guard_samples"]/c["fs_baseband"]
            clean = receiver.replay(sounding, origin, frame)
            y = receive_symbols(receiver.chain.receive(clean), c)
            noise_seed = [c["seed"], 2, record, round(start*1000), 1, 0]
            rng = np.random.default_rng(np.random.SeedSequence(noise_seed))
            standard = receive_symbols(receiver.chain.receive(rng.standard_normal(len(clean))), c)
            power = float(np.mean(abs(y)**2))
            variance = power/receiver.chain.noise_gain/10**(np.asarray(c["snr_db"])/10)
            refs = [receiver.reference(sounding, start, d) for d in range(4)]
            truth = np.stack([item[0] for item in refs])
            predicted = fft(np.sum(truth*input_windows(frame.x, c["nfft"]), axis=-1))
            model_error = [relative_energy(a-b, a) for a, b in zip(y, predicted, strict=True)]
            support = np.sum(abs(truth[..., c["delay_length"]:])**2, axis=(1, 2))/np.sum(abs(truth)**2, axis=(1, 2))
            g = h.create_group(key)
            g.attrs.update(record=record, start_seconds=start, sounding_sha256=sounding.sha256,
                           payload_seed=json.dumps(frame.seed), noise_seed=json.dumps(noise_seed),
                           intersymbol_leakage=json.dumps([v for _, v in refs]), model_error=json.dumps(model_error),
                           support_residual=json.dumps(support.tolist()))
            for name, value in dict(truth=truth, clean=y, standard=standard, variance=variance, bits=frame.bits).items():
                g.create_dataset(name, data=value, compression="lzf" if name == "truth" else None)
            h.flush()
            g.attrs["complete"] = True
            h.flush()
            print(f"{channel}: test {i+1}/{len(options['test_records'])}, record {record}, {time.perf_counter()-begin:.1f}s", flush=True)
        h.attrs["complete"] = True
    partial.replace(path)
    return path


def detect(taps, observations, known, variances, noise, data_mask, backend):
    """Run both LMMSE baselines on the supplied CSI; no labels enter detection.

    CUDA uses the unitary time-coordinate form of the same full-matrix solve.
    The taps may contain one CSI per SNR or a single ideal CSI shared by all SNRs.
    """
    taps, observations = np.asarray(taps), np.asarray(observations)
    if taps.ndim == 3:
        taps = taps[None]
    count, n = len(variances), known.shape[1]
    if (taps.ndim != 4 or taps.shape[0] not in (1, count) or taps.shape[1:3] != (4, n)
            or observations.shape != (count, 4, n) or known.shape != (4, n)):
        raise ValueError("CSI and observations do not describe the same four-symbol frame")
    indices = np.flatnonzero(data_mask)
    if backend.torch is None:
        whitener = make_whitener(noise, backend)
        single = np.empty((count, 4, len(indices)), complex)
        full = np.empty_like(single)
        for d in range(4):
            for si in range(count):
                matrix = taps_to_matrix(taps[0 if len(taps) == 1 else si, d])
                y = observations[si, d]
                batch = EqualizerBatch(matrix[None], y[None], (y-matrix@known[d])[None],
                                       np.asarray(variances[si:si+1]), indices, noise)
                single[si, d] = single_equalize(batch, backend).symbols[0]
                full[si, d] = full_equalize(batch, backend, whitener=whitener).symbols[0]
        return single, full
    torch = backend.torch
    t = torch.as_tensor(taps, dtype=torch.complex128, device="cuda")
    y = torch.as_tensor(observations, dtype=torch.complex128, device="cuda")
    var = torch.as_tensor(variances, dtype=torch.float64, device="cuda")
    data = torch.as_tensor(indices, device="cuda")
    diagonal = torch.fft.fftshift(torch.fft.fft(t.mean(dim=-2), n=n, dim=-1), dim=-1).index_select(-1, data)
    noise_diag = torch.as_tensor(np.diag(noise).real[indices].copy(), device="cuda")
    single = equalize_diagonal(diagonal, y.index_select(-1, data), var[:, None, None]*noise_diag, backend)
    carriers = np.fft.fftshift(np.arange(n))[data_mask]
    response = torch.fft.fft(t, n=n, dim=-1).index_select(-1, torch.as_tensor(carriers, device="cuda"))
    phase = torch.as_tensor(np.exp(2j*np.pi/n*np.arange(n)[:, None]*carriers)/np.sqrt(n), device="cuda")
    matrix = response*phase
    known_windows = torch.as_tensor(input_windows(known, taps.shape[-1]), device="cuda")
    known_time = (t*known_windows[None]).sum(-1)
    residual = torch.fft.ifft(torch.fft.ifftshift(y, dim=-1), dim=-1, norm="ortho")-known_time
    matrix = matrix.expand(count, -1, -1, -1)
    factor = torch.linalg.cholesky(torch.as_tensor(time_noise_covariance(noise), device="cuda"))
    whitened = torch.linalg.solve_triangular(factor, torch.cat((matrix, residual[..., None]), dim=-1), upper=False)
    h, v = whitened[..., :-1], whitened[..., -1:]
    gram = h.mH @ h
    gram.diagonal(dim1=-2, dim2=-1).add_(var[:, None, None])
    full = torch.linalg.solve(gram, h.mH @ v)[..., 0]
    return single.detach().cpu().numpy(), full.detach().cpu().numpy()


def evaluate(channel, output, metadata, backend, *, physical_root=None, order=None):
    c, options = metadata["configs"][channel].copy(), metadata["options"]
    order = c["dpss_expected_count"] if order is None else order
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("DPSS order must be a positive integer")
    c["dpss_expected_count"] = order
    path = Path(physical_root or output)/channel.lower()/"physical.h5"
    result_path = Path(output)/channel.lower()/"metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(metadata=metadata, channel=channel, order=order))
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("identity") != identity:
            raise ValueError("Existing metrics do not match this experiment")
        return saved
    with h5py.File(path, "r") as h:
        if not h.attrs.get("complete"):
            raise ValueError("Physical data are incomplete")
        if h["basis"].shape[1] < order or h["prior"].shape[1] < order:
            raise ValueError("Physical source does not contain the requested DPSS order")
        model = DPSSModel(h["basis"][:, :order], c["dpss_half_bandwidth_hz"], c["dpss_tail_ratio"], c["delay_length"], c)
        known, noise = h["known"][:], h["unit_noise"][:]
        groups = list(h["test"].values())
        observations = np.stack([g["clean"][:][None]+np.sqrt(g["variance"][:])[:, None, None]*g["standard"][:][None] for g in groups])
        variances = np.stack([g["variance"][:] for g in groups])
        prior = FrozenPrior.fit_coefficients(model, h["prior"][:, :order], c["empirical_weight"], c["prior_scale"],
                  options["prior_records"], regularizer=c["relative_regularizer"], delay_block_size=c["delay_block_size"],
                  interference_scale=c["interference_scale"])
        layout = carrier_masks(c)
        print(f"{channel}: building raw joint DPSS estimator, Q={order}", flush=True)
        estimator = JointEstimator(prior, known, layout, noise, backend)
        coefficients = estimator.estimate_coefficients(observations.reshape(-1, 4, c["nfft"]), known,
            layout, variances.ravel()).reshape(len(groups), len(c["snr_db"]), model.count, model.length)
        estimator_diagnostics = estimator.diagnostics
        del estimator, prior
        if backend.torch:
            backend.torch.cuda.empty_cache()
        frames = []
        sampled = model.sampled_basis.reshape(4, c["nfft"], model.count)
        for i, group in enumerate(groups):
            begin = time.perf_counter()
            estimated = np.einsum("dnq,sql->sdnl", sampled, coefficients[i], optimize=True)
            single, full = detect(estimated, observations[i], known, variances[i], noise, layout["data"], backend)
            bits = group["bits"][:]
            raw_errors = [np.count_nonzero(decisions(x) != bits[None], axis=(-1, -2)) for x in (single, full)]
            # Reference CSI is first read after raw detection and is never used to change estimated taps.
            truth = group["truth"][:]
            nmse = raw_nmse(model, coefficients[i], truth)
            ref_single, ref_full = detect(truth, observations[i], known, variances[i], noise, layout["data"], backend)
            ref_errors = [np.count_nonzero(decisions(x) != bits[None], axis=(-1, -2)) for x in (ref_single, ref_full)]
            frames.append(dict(record=int(group.attrs["record"]), start_seconds=float(group.attrs["start_seconds"]),
                raw_single_errors=raw_errors[0].tolist(), raw_full_errors=raw_errors[1].tolist(),
                reference_single_errors=ref_errors[0].tolist(), reference_full_errors=ref_errors[1].tolist(),
                raw_nmse_linear=nmse.tolist(), model_error=json.loads(group.attrs["model_error"]),
                intersymbol_leakage=json.loads(group.attrs["intersymbol_leakage"]),
                support_residual=json.loads(group.attrs["support_residual"])))
            print(f"{channel}: evaluated {i+1}/{len(groups)}, {time.perf_counter()-begin:.1f}s", flush=True)
    bits_per_snr = len(frames)*4*c["data_count"]*2
    curves = {}
    for view in ("raw", "reference"):
        for algorithm in ("single", "full"):
            errors = np.asarray([f[f"{view}_{algorithm}_errors"] for f in frames]).sum(axis=(0, 2))
            curves[f"{view}_{algorithm}"] = dict(bit_errors=errors.tolist(), ber=(errors/bits_per_snr).tolist())
    nmse = np.asarray([f["raw_nmse_linear"] for f in frames]).mean(axis=(0, 2))
    report = dict(identity=identity, channel=channel, complete=True, q=order, frames=frames, bits_per_snr=bits_per_snr,
                  snr_db=c["snr_db"], curves=curves, mean_raw_nmse_linear=nmse.tolist(),
                  mean_raw_nmse_db=(10*np.log10(nmse)).tolist(), reference_nmse_linear=0,
                  reference_nmse_db="-inf", estimator_diagnostics=estimator_diagnostics)
    write_json(result_path, report)
    return report


def finish(output, metadata, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    orders = {report["q"] for report in reports.values()}
    if len(orders) != 1:
        raise ValueError("Channel reports must use the same fixed DPSS order")
    order = orders.pop()
    cps = {metadata["configs"][channel]["cp"] for channel in reports}
    if len(cps) != 1:
        raise ValueError("Channel reports must use the same cyclic-prefix length")
    cp = cps.pop()
    write_json(output/"metrics.json", dict(complete=True, metadata=metadata, channels=reports))
    with (output/"summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "snr_db", "bits", "raw_single_errors", "raw_single_ber", "raw_full_errors",
                         "raw_full_ber", "reference_single_errors", "reference_single_ber", "reference_full_errors",
                         "reference_full_ber", "raw_nmse_db", "reference_nmse_linear", "reference_nmse_db"])
        for channel, r in reports.items():
            for i, snr in enumerate(r["snr_db"]):
                row = [channel, snr, r["bits_per_snr"]]
                for key in ("raw_single", "raw_full", "reference_single", "reference_full"):
                    row += [r["curves"][key]["bit_errors"][i], r["curves"][key]["ber"][i]]
                writer.writerow(row+[r["mean_raw_nmse_db"][i], 0, "-inf"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), layout="constrained")
    styles = (("raw_single", "Single-tap / estimated CSI", "#C8752B", "o", "-"),
              ("raw_full", "Full-matrix / estimated CSI", "#2166AC", "s", "-"),
              ("reference_single", "Single-tap / reference CSI", "#C8752B", "o", "--"),
              ("reference_full", "Full-matrix / reference CSI", "#2166AC", "s", "--"))
    for ax, (channel, r) in zip(axes, reports.items(), strict=True):
        for key, label, color, marker, line in styles:
            values = np.asarray(r["curves"][key]["ber"])
            # Open markers at the counting floor represent zero observed errors.
            ax.plot(r["snr_db"], np.maximum(values, .5/r["bits_per_snr"]), linestyle=line,
                    marker=marker, color=color, label=label, linewidth=1.7)
            zero = values == 0
            if np.any(zero):
                ax.scatter(np.asarray(r["snr_db"])[zero], np.full(zero.sum(), .5/r["bits_per_snr"]),
                           facecolors="white", edgecolors=color, zorder=4)
        ax.set(title=f"{channel} ({r['bits_per_snr']:,} bits/SNR)", xlabel="SNR (dB)", ylabel="BER", xticks=r["snr_db"])
        ax.set_yscale("log")
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"4D, CP={cp}, Q={order} | raw vs reference CSI | zero errors at half-count floor")
    fig.savefig(output/"ber_snr.png", dpi=180)
    fig.savefig(output/"ber_snr.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), layout="constrained")
    for ax, (channel, r) in zip(axes, reports.items(), strict=True):
        ax.plot(r["snr_db"], r["mean_raw_nmse_db"], "o-", color="#2166AC", label="Estimated CSI (raw)")
        ax.text(.03, .05, "Reference CSI: NMSE = 0 (−∞ dB)", transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=.95), fontsize=9)
        ax.set(title=channel, xlabel="SNR (dB)", ylabel="Mean full-matrix CSI NMSE (dB)", xticks=r["snr_db"])
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle(f"4D, CP={cp}, 512 data + 256 pilots | 12 Hz / Q={order} / L=176")
    fig.savefig(output/"nmse_snr.png", dpi=180)
    fig.savefig(output/"nmse_snr.pdf")
    plt.close(fig)


def run(output, archive_path, options, backend, stage="all"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = manifest(options, archive_path, backend)
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output directory belongs to a different experiment")
    write_json(manifest_path, metadata)
    if stage in ("prepare", "all"):
        for channel in options["channels"]:
            prepare(channel, output, archive_path, metadata)
    if stage in ("solve", "all"):
        reports = {ch: evaluate(ch, output, metadata, backend) for ch in options["channels"]}
        finish(output, metadata, reports)
    return output
