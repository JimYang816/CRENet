"""Repeat the 4D Q sweep using only newly estimated, uncompensated CSI."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.config import ROOT, digest
from data_generation.src.receiver import pdd
from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.transmitter.ofdm import decisions
from src.algorithms.full_matrix_lmmse.equalizer import equalize_batch
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.evaluation.four_data_q_sweep import (
    BANDWIDTH_HZ, CODE as PHYSICAL_CODE, LENGTH, ORDERS, SNR_DB, choose, config,
    four_data_estimator_layout, identities, masks, output_identity, raw_nmse,
    time_noise_covariance,
)


CODE = (*PHYSICAL_CODE, "src/evaluation/four_data_raw_q_sweep.py",
        "scripts/evaluate/four_data_raw_q_sweep.py")


def detect_raw(model, coefficients, observations, known, variances, noise, data_mask, backend):
    """Full-matrix LMMSE using raw coefficient CSI; no reference or labels enter."""
    n, count, length = model.c["nfft"], model.count, model.length
    if (np.shape(coefficients) != (len(variances), count, length)
            or np.shape(observations) != (len(variances), 4, n)
            or np.shape(known) != (4, n)):
        raise ValueError("Raw channel, observations, and frame do not agree")
    if backend.torch is None:
        whitener = make_whitener(noise, backend)
        indices = np.flatnonzero(data_mask)
        symbols = np.empty((len(variances), 4, len(indices)), complex)
        for d in range(4):
            for si in range(len(variances)):
                matrix = taps_to_matrix(model.reconstruct(coefficients[si], d))
                residual = observations[si, d]-matrix@known[d]
                batch = EqualizerBatch(matrix[None], observations[si, d][None], residual[None],
                                       np.asarray(variances[si:si+1]), indices, noise)
                symbols[si, d] = equalize_batch(batch, backend, whitener=whitener).symbols[0]
        return symbols
    torch = backend.torch
    carriers = np.fft.fftshift(np.arange(n))[data_mask]
    sampled = torch.as_tensor(model.sampled_basis.reshape(4, n, count), dtype=torch.complex128, device="cuda")
    coeff = torch.as_tensor(coefficients, device="cuda")
    taps = torch.einsum("dnq,sql->sdnl", sampled, coeff)
    response = torch.fft.fft(taps, n=n, dim=-1).index_select(-1, torch.as_tensor(carriers, device="cuda"))
    phase = torch.as_tensor(np.exp(2j*np.pi/n*np.arange(n)[:, None]*carriers)/np.sqrt(n), device="cuda")
    matrix = response*phase
    known_windows = torch.as_tensor(input_windows(known, length), device="cuda")
    known_time = (taps*known_windows[None]).sum(-1)
    obs = torch.as_tensor(observations, device="cuda")
    residual = torch.fft.ifft(torch.fft.ifftshift(obs, dim=-1), dim=-1, norm="ortho")-known_time
    factor = torch.linalg.cholesky(torch.as_tensor(time_noise_covariance(noise), device="cuda"))
    whitened = torch.linalg.solve_triangular(factor, torch.cat((matrix, residual[..., None]), dim=-1), upper=False)
    h, y = whitened[..., :-1], whitened[..., -1:]
    gram = h.mH@h
    gram.diagonal(dim1=-2, dim2=-1).add_(torch.as_tensor(variances, device="cuda")[:, None, None])
    return torch.linalg.solve(gram, h.mH@y)[..., 0].detach().cpu().numpy()


def source_manifest(source, archive_path, smoke, backend):
    """Verify unchanged physical inputs; no compensated or per-Q cache is read."""
    original = json.loads((source/"source_manifest.json").read_text(encoding="utf-8"))
    if (original["smoke"] != smoke or original["orders"] != list(ORDERS)
            or original["snr_db"] != list(SNR_DB) or original["bandwidth_hz"] != BANDWIDTH_HZ
            or original["length"] != LENGTH):
        raise ValueError("Physical source has a different frame or Q grid")
    physical = {}
    for channel in ("NOF1", "NCS1"):
        path = source/channel.lower()/"physical.h5"
        with h5py.File(path, "r") as h:
            if (not h.attrs.get("complete") or h.attrs["channel"] != channel
                    or h.attrs["identity"] != output_identity(channel, archive_path, smoke)):
                raise ValueError(f"Physical source identity mismatch: {path}")
            if set(h.keys()) != {"basis", "concentration", "known", "prior", "prior_complete",
                                 "unit_noise", "validation", "test"}:
                raise ValueError("Unexpected CSI or other fields in the physical source")
            for split, ids in zip(("validation", "test"), identities(smoke)[1:]):
                actual = [(int(g.attrs["record"]), int(g.attrs["start_seconds"])) for g in h[split].values()]
                if actual != ids:
                    raise ValueError("Physical windows differ from the original experiment")
        physical[channel] = dict(path=str(path.resolve()), sha256=file_sha256(path))
    return dict(schema="four-data-256p-raw-q-v1", git_head=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                code_sha256={name: file_sha256(ROOT/name) for name in CODE},
                source_manifest_sha256=file_sha256(source/"source_manifest.json"),
                physical=physical, archive_sha256=file_sha256(archive_path), smoke=smoke,
                frame=original["frame"], carrier_pattern=original["carrier_pattern"],
                observations=original["observations"], source_splits=original["source_splits"],
                orders=list(ORDERS), snr_db=list(SNR_DB), bandwidth_hz=BANDWIDTH_HZ, length=LENGTH,
                input_view="raw", compensation=False,
                estimator="fresh joint prior LMMSE for each Q; no decision feedback",
                input_separation="physical prior and waveforms only; no old estimated CSI or evaluation cache",
                reference_use="full reference read only after detection, for NMSE reporting",
                equalizer="all 1024 receive rows, 512 data columns, full colored noise covariance",
                precision="complex128", device=backend.device)


def evaluate_candidate(channel, source, output, metadata, order, split, backend):
    run = output/channel.lower()
    run.mkdir(parents=True, exist_ok=True)
    path = run/f"{split}_q{order:02d}.json"
    identity = digest(dict(metadata=metadata, channel=channel, order=order, split=split))
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved["identity"] != identity or saved["input_view"] != "raw"
                or saved["q"] != order or saved["split"] != split):
            raise ValueError("Refusing to reuse a different view or candidate")
        return saved
    c = config(channel)
    with h5py.File(source/channel.lower()/"physical.h5", "r") as h:
        model = DPSSModel(h["basis"][:, :order], BANDWIDTH_HZ, 1e-5, LENGTH, c)
        known, noise = h["known"][:], h["unit_noise"][:]
        groups = list(h[split].values())
        obs = np.stack([g["clean"][:][None]+np.sqrt(g["variance"][:])[:, None, None]*g["standard"][:][None]
                        for g in groups])
        variance = np.stack([g["variance"][:] for g in groups])
        prior = FrozenPrior.fit_coefficients(model, h["prior"][:, :order], 0, 1, range(51, 56),
                                             regularizer=c["relative_regularizer"], delay_block_size=32)
        mask = masks(c)
        with four_data_estimator_layout():
            estimator = pdd.JointEstimator(prior, known, mask, noise, backend)
            coefficients = estimator.estimate_coefficients(obs.reshape(-1, 4, c["nfft"]), known, mask,
                variance.ravel()).reshape(len(groups), len(SNR_DB), order, LENGTH)
        del estimator, prior
        if backend.torch:
            backend.torch.cuda.empty_cache()
        frames = []
        for i, group in enumerate(groups):
            begin = time.perf_counter()
            detected = detect_raw(model, coefficients[i], obs[i], known, variance[i], noise, mask["data"], backend)
            errors = np.count_nonzero(decisions(detected) != group["bits"][:][None], axis=(-1, -2))
            # The reference is read for this metric only; detect_raw has no reference input.
            nmse = raw_nmse(model, coefficients[i], group["truth"][:])
            frames.append(dict(record=int(group.attrs["record"]), start_seconds=float(group.attrs["start_seconds"]),
                               errors=errors.tolist(), nmse_linear=nmse.tolist()))
            print(f"{channel} {split} raw Q={order} {i+1}/{len(groups)} {time.perf_counter()-begin:.1f}s", flush=True)
    errors = np.asarray([f["errors"] for f in frames]).sum(axis=(0, 2))
    nmse = np.asarray([f["nmse_linear"] for f in frames]).mean(axis=(0, 2))
    bits = len(groups)*4*512*2
    row = dict(identity=identity, input_view="raw", compensation=False, q=order, split=split,
               frames=frames, bit_errors=errors.tolist(), bits_per_snr=bits, ber=(errors/bits).tolist(),
               mean_ber=float(np.mean(errors/bits)), mean_nmse_linear=nmse.tolist(),
               mean_nmse_db=(10*np.log10(nmse)).tolist())
    write_json(path, row)
    print(f"{channel} {split} raw Q={order}: mean BER {100*row['mean_ber']:.5f}%", flush=True)
    return row


def finish(output, metadata, selection, validation, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = dict(complete=True, metadata=metadata, selection=selection, validation=validation, test=test)
    write_json(output/"metrics.json", report)
    with (output/"test_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "q", "snr_db", "bit_errors", "bits", "ber", "mean_nmse_db"])
        for channel, rows in test.items():
            for row in rows:
                for s, errors, ber, nmse in zip(SNR_DB, row["bit_errors"], row["ber"], row["mean_nmse_db"]):
                    writer.writerow([channel, row["q"], s, errors, row["bits_per_snr"], ber, nmse])
    for kind in ("ber", "nmse"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
        for ax, channel in zip(axes, ("NOF1", "NCS1")):
            winner = selection["by_channel"][channel]
            for row in test[channel]:
                values = row["ber"] if kind == "ber" else row["mean_nmse_db"]
                ax.plot(SNR_DB, values, "o-", linewidth=2.1 if row["q"] == winner else 1,
                        label=f"Q={row['q']}" + (" selected" if row["q"] == winner else ""))
            if kind == "ber":
                ax.set_yscale("log")
            ax.set(title=channel, xlabel="SNR (dB)", ylabel="BER" if kind == "ber" else "Raw CSI NMSE (dB)", xticks=SNR_DB)
            ax.grid(alpha=.25, which="both")
            ax.legend(fontsize=8)
        fig.suptitle("4D, 256 comb pilots, 12 Hz, L=176 | raw CSI | held-out test")
        fig.savefig(output/f"{kind}_snr.png", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for channel, rows in validation.items():
        axes[0].plot([r["q"] for r in rows], [100*r["mean_ber"] for r in rows], "o-", label=channel)
        axes[1].plot([r["q"] for r in rows], [10*np.log10(np.mean(r["mean_nmse_linear"])) for r in rows], "o-", label=channel)
    for ax, ylabel in zip(axes, ("Validation mean BER (%)", "Validation mean raw CSI NMSE (dB)")):
        ax.set(xlabel="Q", ylabel=ylabel, xticks=ORDERS)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(output/"validation_q_search.png", dpi=170)
    plt.close(fig)


def run(source, output, archive_path, smoke, backend):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Raw outputs must use a separate experiment directory")
    metadata = source_manifest(source, archive_path, smoke, backend)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"source_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output directory belongs to a different source, code or CSI view")
    write_json(manifest, metadata)
    orders = (16, 17) if smoke else ORDERS
    validation = {channel: [evaluate_candidate(channel, source, output, metadata, q, "validation", backend)
                            for q in orders] for channel in ("NOF1", "NCS1")}
    if smoke:
        print("Raw CPU smoke completed; no research selection", flush=True)
        return
    selected = choose(validation)
    selection_path = output/"selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selected:
        raise ValueError("Frozen validation selection changed")
    write_json(selection_path, selected)
    print(f"Frozen raw validation selection: {selected['by_channel']}; shared Q={selected['shared']}", flush=True)
    test = {}
    for channel in ("NOF1", "NCS1"):
        ranked = sorted(validation[channel], key=lambda row: (row["mean_ber"], row["q"]))
        # Q16 and Q24 are fixed reference orders.
        test_orders = sorted({selected["by_channel"][channel], selected["shared"], ranked[1]["q"], 16, 24})
        test[channel] = [evaluate_candidate(channel, source, output, metadata, q, "test", backend)
                         for q in test_orders]
    finish(output, metadata, selected, validation, test)
