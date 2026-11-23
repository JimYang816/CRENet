"""Offline delay-group energy analysis of accepted compact reference channels.

The physical reference and its full-frame DPSS projection are supervision, not
receiver inputs. Each physical window is counted once, independently of SNR,
payload and noise repeats. All calculations use the D1/D2 FFT windows.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
import subprocess

import numpy as np

from data_generation.src.dataset.compact_math import matrix_to_taps
from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset

LOG = logging.getLogger(__name__)
THRESHOLDS = (90, 95, 99)
SCOPES = ("d1", "d2", "joint")


def energy_counts(energy):
    """Minimum number of largest, nonnegative groups capturing each target."""
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1 or not np.all(np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("Energy must be a finite, nonnegative vector")
    total = float(energy.sum())
    if total <= 0:
        raise ValueError("Zero-energy reference or projection")
    order = np.argsort(-energy, kind="stable")
    curve = np.cumsum(energy[order], dtype=np.float64) / total
    curve[-1] = 1.0
    counts = {f"k{v}": int(np.searchsorted(curve, v / 100, side="left") + 1)
              for v in THRESHOLDS}
    return counts, curve, order


class TapEnergy:
    """Reference/projection energies in the circular time-delay representation."""

    def __init__(self, basis, device="cpu"):
        self.basis = np.asarray(basis, dtype=np.float64)
        if self.basis.ndim != 3 or self.basis.shape[0] != 2:
            raise ValueError("Expected two D-block basis matrices")
        self.device = device
        self.torch = None
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            self.torch = torch
            self.basis_gpu = torch.as_tensor(self.basis, device="cuda", dtype=torch.complex64)
            n = self.basis.shape[1]
            row = torch.arange(n, device="cuda")[:, None]
            self.index_gpu = ((row - torch.arange(n, device="cuda")[None, :]) % n)
        elif device != "cpu":
            raise ValueError("Unknown device")

    def calculate(self, reference, coefficients):
        """Return group powers, cross terms and direct full projection error."""
        reference = np.asarray(reference)
        coefficients = np.asarray(coefficients)
        n, q = self.basis.shape[1:]
        if reference.shape != (2, n, n) or coefficients.ndim != 2 or coefficients.shape[0] != q:
            raise ValueError("Reference, basis and coefficient dimensions do not match")
        length = coefficients.shape[1]
        if length > n or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Invalid reference or coefficient values")
        if self.torch is None:
            truth = matrix_to_taps(reference.astype(np.complex128))
            projected = np.einsum("dnq,ql->dnl", self.basis, coefficients, optimize=True)
            return (np.sum(np.abs(truth) ** 2, axis=1),
                    np.sum(np.abs(projected) ** 2, axis=1),
                    np.sum((truth[:, :, :length].conj() * projected).real, axis=1),
                    np.sum(np.abs(truth[:, :, :length] - projected) ** 2, axis=(1, 2))
                    + np.sum(np.abs(truth[:, :, length:]) ** 2, axis=(1, 2)))
        t = self.torch
        with t.no_grad():
            h = t.as_tensor(reference, device="cuda", dtype=t.complex64)
            h = t.fft.ifftshift(h, dim=(-2, -1))
            circular = t.fft.fft(t.fft.ifft(h, dim=-2, norm="ortho"), dim=-1, norm="ortho")
            index = self.index_gpu.expand(2, -1, -1)
            truth = t.gather(circular, -1, index)
            coefficients = t.as_tensor(coefficients, device="cuda", dtype=t.complex64)
            projected = t.matmul(self.basis_gpu, coefficients)
            power = lambda x, dim: t.sum(t.abs(x).square(), dim=dim, dtype=t.float64)
            p_true = power(truth, 1)
            p_proj = power(projected, 1)
            cross = t.sum((truth[:, :, :length].conj() * projected).real, dim=1, dtype=t.float64)
            error = (power(truth[:, :, :length] - projected, (1, 2))
                     + power(truth[:, :, length:], (1, 2)))
            return tuple(x.cpu().numpy() for x in (p_true, p_proj, cross, error))


def scope_metrics(true_power, proj_power, cross, direct_error, length, block_size):
    """One D block or the energy-pooled pair; all ratios are linear."""
    p = np.asarray(true_power, dtype=np.float64)
    pp = np.asarray(proj_power, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    if p.ndim != 1 or pp.shape != (length,) or cross.shape != pp.shape or length % block_size:
        raise ValueError("Invalid delay powers or delay block size")
    ref_counts, ref_curve, _ = energy_counts(p[:length])
    proj_counts, proj_curve, proj_order = energy_counts(pp)
    blocks, _, _ = energy_counts(p[:length].reshape(-1, block_size).sum(axis=1))
    projected_blocks, _, _ = energy_counts(pp.reshape(-1, block_size).sum(axis=1))
    total = float(p.sum())
    residual = float(direct_error) / total
    # Calculate true-reference error after retaining the projected-energy top K.
    increments = np.cumsum((pp - 2 * cross)[proj_order], dtype=np.float64)
    truncated = {}
    for target in THRESHOLDS:
        k = proj_counts[f"k{target}"]
        error = (total + increments[k - 1]) / total
        if error < -1e-5:
            raise ValueError("Negative truncated projection error")
        truncated[f"projected_top_k{target}_nmse"] = float(max(error, 0.0))
    result = {f"reference_{key}": value for key, value in ref_counts.items()}
    result.update({f"projected_{key}": value for key, value in proj_counts.items()})
    result.update({f"reference_block{block_size}_{key}": value for key, value in blocks.items()})
    result.update({f"projected_block{block_size}_{key}": value for key, value in projected_blocks.items()})
    result.update(outside_support_fraction=float(p[length:].sum() / total),
                  projection_nmse=residual, **truncated)
    return result, ref_curve, proj_curve


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Empty or nonfinite results")
    q = np.quantile(values, [.1, .25, .5, .75, .9, .95])
    return dict(n=len(values), mean=float(values.mean()), median=float(q[2]),
                p10=float(q[0]), p25=float(q[1]), p75=float(q[3]),
                p90=float(q[4]), p95=float(q[5]), min=float(values.min()), max=float(values.max()))


def _identity_check(dataset, split, channel):
    h, c, model = dataset._h, dataset._config, dataset._model
    if dataset.channel != channel or dataset.split != split or not np.all(h["windows/complete"][:]):
        raise ValueError("Dataset split identity or completion mismatch")
    expected = {(int(r), float(s)) for r in c["learning_dataset"]["records"][split]
                for s in c["learning_dataset"]["starts_seconds"]}
    actual = list(zip(h["windows/record"][:].astype(int), h["windows/start_seconds"][:].astype(float)))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Physical window identities do not match the fixed split")
    if h["windows/reference_csi"].shape != (len(actual), 2, c["nfft"], c["nfft"]):
        raise ValueError("Invalid physical reference dimensions")
    if h["windows/reference_projection"].shape != (len(actual), model.count, model.length):
        raise ValueError("Invalid reference projection dimensions")
    return actual


def _verify_cpu(reference, coefficients, basis, result):
    cpu = TapEnergy(basis, "cpu").calculate(reference, coefficients)
    # Projection error may be nearly zero; normalize by physical signal power,
    # not by the tiny error itself.
    scale = max(float(np.sum(cpu[0])), 1e-30)
    for expected, actual in zip(cpu, result):
        if float(np.sum(np.abs(expected - actual))) / scale > 3e-5:
            raise RuntimeError("CUDA result disagrees with complex128 CPU verification")


def run(datasets, output, device, verify_windows=1):
    """Read accepted data read-only and write per-window and aggregate reports."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, summaries, curves, metadata = [], {}, {}, {}
    common_basis = None
    for channel, folder in datasets.items():
        channel_summaries, metadata[channel] = {}, {}
        summaries[channel] = channel_summaries
        for split in ("train", "validation", "test"):
            path = Path(folder) / f"{split}.h5"
            with CompactDataset(path) as ds:
                identities = _identity_check(ds, split, channel)
                c, model, h = ds._config, ds._model, ds._h
                if common_basis is None:
                    common_basis = model.basis.copy()
                elif common_basis.shape != model.basis.shape or not np.array_equal(common_basis, model.basis):
                    raise ValueError("The channels do not share the same actual DPSS basis")
                n, length, q = c["nfft"], model.length, model.count
                basis = model.sampled_basis.reshape(4, n, q)[[1, 2]]
                block_size = int(c["delay_block_size"])
                calc = TapEnergy(basis, device)
                obs_rows = [len(ds._shared[f"estimator/rows/{i}"]) for i in range(4)]
                metadata[channel][split] = dict(path=str(path), model_identity=str(h.attrs["model_identity"]),
                    window_count=len(identities), nfft=n, delay_length=length, dpss_count=q,
                    delay_block_size=block_size, estimator_rows=obs_rows,
                    manifest_sha256=file_sha256(Path(folder) / "manifest.json"))
                window_hashes = hashlib.sha256()
                local = {scope: [] for scope in SCOPES}
                curve_acc = {scope: [np.zeros(length), np.zeros(length)] for scope in SCOPES}
                for i, (record, start) in enumerate(identities):
                    reference = h["windows/reference_csi"][i]
                    stored_hash = h["windows/reference_sha256"][i].decode("ascii")
                    actual_hash = hashlib.sha256(reference.tobytes()).hexdigest()
                    if actual_hash != stored_hash:
                        raise RuntimeError(f"Reference checksum mismatch in {channel}/{split}/{i}")
                    window_hashes.update(f"{record},{start},{actual_hash}\n".encode())
                    coeff = h["windows/reference_projection"][i]
                    result = calc.calculate(reference, coeff)
                    if device == "cuda" and i < verify_windows:
                        _verify_cpu(reference, coeff, basis, result)
                    true, proj, cross, direct_error = result
                    selections = ((0,), (1,), (0, 1))
                    for scope, indices in zip(SCOPES, selections):
                        selection = np.asarray(indices)
                        metrics, ref_curve, proj_curve = scope_metrics(
                            true[selection].sum(axis=0), proj[selection].sum(axis=0),
                            cross[selection].sum(axis=0), direct_error[selection].sum(), length, block_size)
                        local[scope].append(metrics)
                        curve_acc[scope][0] += ref_curve
                        curve_acc[scope][1] += proj_curve
                        rows.append(dict(channel=channel, split=split, record=int(record),
                                         start_seconds=float(start), scope=scope, **metrics))
                    if (i + 1) % 50 == 0 or i + 1 == len(identities):
                        LOG.info("%s %s: %d/%d physical windows", channel, split, i + 1, len(identities))
                metadata[channel][split]["reference_identity_sha256"] = window_hashes.hexdigest()
                channel_summaries[split] = {}
                curves[channel, split] = {}
                for scope in SCOPES:
                    channel_summaries[split][scope] = {key: summarize([x[key] for x in local[scope]])
                                                       for key in local[scope][0]}
                    curves[channel, split][scope] = [x / len(identities) for x in curve_acc[scope]]
    fields = list(rows[0])
    with (output / "per_window.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "energy_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "split", "scope", "k", "reference_mean_fraction", "projected_mean_fraction"])
        for (channel, split), scopes in curves.items():
            for scope, (actual, projected) in scopes.items():
                writer.writerows((channel, split, scope, i + 1, a, b)
                                 for i, (a, b) in enumerate(zip(actual, projected)))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = dict(method="one group per baseband delay, D1/D2 pooled by energy",
                  unit="one unique physical record/start window; no SNR or payload repeats",
                  scope="reference matrix converted exactly to circular taps; plus full-frame DPSS reference projection",
                  support_denominator="K values retain the specified fraction of energy inside L; outside-support fraction reported separately",
                  block_grouping="nonoverlapping consecutive prior delay blocks; one count per entire block",
                  thresholds=list(THRESHOLDS), device=device, cpu_verified_windows_per_split=verify_windows if device == "cuda" else 0,
                  commit=commit, datasets=metadata, statistics=summaries)
    write_json(output / "summary.json", report)
    _plots(output, rows, curves)
    return report


def _plots(output, rows, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for channel in ("NOF1", "NCS1"):
        values = sorted(r["reference_k95"] for r in rows
                        if r["channel"] == channel and r["split"] == "test" and r["scope"] == "joint")
        axes[0].step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=channel)
        actual, projected = curves[channel, "test"]["joint"]
        axes[1].plot(np.arange(1, len(actual) + 1), actual, label=f"{channel}: reference")
        axes[1].plot(np.arange(1, len(projected) + 1), projected, "--", label=f"{channel}: DPSS projection")
    axes[0].set(xlabel="K95 delay groups per frame", ylabel="Empirical CDF", ylim=(0, 1.01))
    axes[1].set(xlabel="Largest delay groups K", ylabel="Mean retained energy fraction", ylim=(0, 1.01))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(output / "test_delay_sparsity.png", dpi=180)
    plt.close(fig)
