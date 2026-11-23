"""NCS1 raw-CSI extension to Q33..40 using the same measured 4D inputs."""
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
from data_generation.src.config import ROOT, digest
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_q_sweep import (
    BANDWIDTH_HZ, LENGTH, SNR_DB, BlockedReceiver, config, identities, output_identity,
)
from src.evaluation.four_data_raw_q_sweep import CODE as RAW_CODE, evaluate_candidate


CHANNEL = "NCS1"
REFERENCE_ORDER = 32
ORDERS = tuple(range(32, 41))
CODE = (*RAW_CODE, "src/evaluation/four_data_raw_q_extension.py",
        "scripts/evaluate/four_data_raw_q_extension.py")


def extended_basis(c, previous_basis):
    """Preserve all original 32 columns exactly and append the next eight DPSS vectors."""
    total = 4*(c["nfft"]+c["cp"])
    nw = total*BANDWIDTH_HZ/c["fs_baseband"]
    vectors, concentration = dpss(total, nw, Kmax=max(ORDERS), sym=True, norm=2, return_ratios=True)
    generated = vectors.T
    old = np.asarray(previous_basis)
    if old.shape != (total, REFERENCE_ORDER):
        raise ValueError("Expected the original full-frame Q32 basis")
    discrepancy = float(np.max(np.abs(old-generated[:, :REFERENCE_ORDER])))
    if discrepancy > 1e-9:
        raise ValueError("The first 32 DPSS vectors do not reproduce the physical source")
    combined = np.column_stack((old, generated[:, REFERENCE_ORDER:]))
    gram_error = float(np.max(np.abs(combined.conj().T@combined-np.eye(max(ORDERS)))))
    if gram_error > 5e-10:
        raise ValueError("Extended DPSS basis is not numerically orthonormal")
    return combined, concentration, dict(first_32_max_absolute_discrepancy=discrepancy,
                                         extended_gram_max_absolute_error=gram_error)


def build_manifest(source, baseline, archive_path, smoke, backend):
    original = json.loads((source/"source_manifest.json").read_text(encoding="utf-8"))
    if (original["smoke"] != smoke or original["bandwidth_hz"] != BANDWIDTH_HZ
            or original["length"] != LENGTH or original["snr_db"] != list(SNR_DB)):
        raise ValueError("The original physical experiment does not match")
    path = source/CHANNEL.lower()/"physical.h5"
    with h5py.File(path, "r") as h:
        if (not h.attrs.get("complete") or h.attrs["channel"] != CHANNEL
                or h.attrs["identity"] != output_identity(CHANNEL, archive_path, smoke)):
            raise ValueError("Original physical input identity mismatch")
        if set(h.keys()) != {"basis", "concentration", "known", "prior", "prior_complete",
                             "unit_noise", "validation", "test"}:
            raise ValueError("Unexpected fields in original physical input")
        for split, ids in zip(("validation", "test"), identities(smoke)[1:]):
            actual = [(int(g.attrs["record"]), int(g.attrs["start_seconds"])) for g in h[split].values()]
            if actual != ids:
                raise ValueError("Original validation/test windows changed")
    physical_hash = file_sha256(path)
    prior_run = None
    if not smoke:
        last = json.loads((baseline/"source_manifest.json").read_text(encoding="utf-8"))
        if (last["input_view"] != "raw" or last["compensation"] is not False
                or last["physical"][CHANNEL]["sha256"] != physical_hash
                or last["snr_db"] != list(SNR_DB) or last["length"] != LENGTH):
            raise ValueError("Previous Q32 raw run does not use this physical source")
        prior_run = dict(manifest_sha256=file_sha256(baseline/"source_manifest.json"),
                         validation_q32_sha256=file_sha256(baseline/"ncs1/validation_q32.json"))
    return dict(schema="ncs1-four-data-raw-q33-40-v1", git_head=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                code_sha256={name: file_sha256(ROOT/name) for name in CODE},
                channel=CHANNEL, input_view="raw", compensation=False, smoke=smoke,
                bandwidth_hz=BANDWIDTH_HZ, length=LENGTH, q_reference=REFERENCE_ORDER,
                q_search=list(ORDERS[1:]), snr_db=list(SNR_DB),
                frame=original["frame"], carrier_pattern=original["carrier_pattern"],
                observations=original["observations"], source_splits=original["source_splits"],
                physical_path=str(path.resolve()), physical_sha256=physical_hash,
                physical_manifest_sha256=file_sha256(source/"source_manifest.json"),
                archive_sha256=file_sha256(archive_path), prior_raw_comparison=prior_run,
                basis="first 32 exact original DPSS columns plus newly generated 33..40",
                prior="first 32 physical-prior coefficients preserved; new 33..40 from same independent prior records",
                estimator="fresh joint prior LMMSE for every Q; no decision feedback or old estimated-CSI cache",
                reference_use="validation/test reference is read only after detection to report NMSE",
                equalizer="all 1024 receive rows, 512 data columns, full colored noise covariance",
                precision="complex128", device=backend.device)


def prepare_physical_wrapper(source, output, archive_path, metadata, smoke):
    """Store new prior modes only; waveform/labels/noise remain external physical links."""
    run = output/CHANNEL.lower()
    run.mkdir(parents=True, exist_ok=True)
    path = run/"physical.h5"
    identity = digest(metadata)
    if path.exists():
        with h5py.File(path, "r") as h:
            if h.attrs.get("complete") and h.attrs["identity"] == identity:
                return
        raise ValueError("Existing extended physical wrapper has another identity")
    partial = path.with_suffix(".partial.h5")
    old_path = source/CHANNEL.lower()/"physical.h5"
    c = config(CHANNEL)
    prior_ids = identities(smoke)[0]
    archive, receiver = WatermarkArchive(archive_path, CHANNEL), BlockedReceiver(c)
    with h5py.File(old_path, "r") as old, h5py.File(partial, "a") as h:
        if "identity" in h.attrs and h.attrs["identity"] != identity:
            raise ValueError("Partial extended prior belongs to another run")
        h.attrs.update(identity=identity, complete=False, input_view="physical_prior_only")
        basis, concentration, diagnostics = extended_basis(c, old["basis"][:])
        if "basis" not in h:
            h.create_dataset("basis", data=basis)
            h.create_dataset("concentration", data=concentration)
            prior = h.create_dataset("prior", (len(prior_ids), max(ORDERS), LENGTH), dtype=np.complex128)
            prior[:, :REFERENCE_ORDER] = old["prior"][:]
            h.create_dataset("extra_prior_complete", data=np.zeros(len(prior_ids), dtype=bool))
            h.attrs["basis_diagnostics_json"] = json.dumps(diagnostics)
            for name in ("known", "unit_noise", "validation", "test"):
                h[name] = h5py.ExternalLink(str(old_path.resolve()), f"/{name}")
            h.flush()
        else:
            np.testing.assert_array_equal(h["basis"][:], basis)
            np.testing.assert_array_equal(h["prior"][:, :REFERENCE_ORDER], old["prior"][:])
        for i, (record, start) in enumerate(prior_ids):
            if h["extra_prior_complete"][i]:
                continue
            begin = time.perf_counter()
            h["prior"][i, REFERENCE_ORDER:] = receiver.prior(archive.load(record), start, basis[:, REFERENCE_ORDER:])
            h.flush()
            h["extra_prior_complete"][i] = True
            h.flush()
            print(f"NCS1 additional physical prior {i+1}/{len(prior_ids)} {record}@{start}s "
                  f"{time.perf_counter()-begin:.1f}s", flush=True)
        h.attrs["complete"] = True
        h.flush()
    partial.replace(path)


def verify_baseline(recomputed, previous):
    if (recomputed["input_view"] != "raw" or previous["input_view"] != "raw"
            or recomputed["q"] != REFERENCE_ORDER or previous["q"] != REFERENCE_ORDER
            or recomputed["split"] != previous["split"]):
        raise ValueError("Baseline is not the same raw Q32 split")
    if len(recomputed["frames"]) != len(previous["frames"]):
        raise ValueError("Baseline frame count changed")
    max_nmse_error = 0.0
    for a, b in zip(recomputed["frames"], previous["frames"]):
        if (a["record"], a["start_seconds"]) != (b["record"], b["start_seconds"]):
            raise ValueError("Baseline physical frame changed")
        np.testing.assert_array_equal(a["errors"], b["errors"])
        np.testing.assert_allclose(a["nmse_linear"], b["nmse_linear"], rtol=1e-9, atol=1e-12)
        max_nmse_error = max(max_nmse_error, float(np.max(np.abs(
            np.asarray(a["nmse_linear"])-np.asarray(b["nmse_linear"])))))
    return dict(frames=len(recomputed["frames"]), bit_errors_identical=True,
                max_absolute_linear_nmse_difference=max_nmse_error)


def select(validation):
    if len(validation) != len(ORDERS) or {row["q"] for row in validation} != set(ORDERS):
        raise ValueError("The Q32..40 validation sweep is incomplete")
    if any(row["input_view"] != "raw" or row["compensation"] or row["split"] != "validation"
           for row in validation):
        raise ValueError("Selection requires raw validation-only results")
    ranking = sorted(validation, key=lambda row: (row["mean_ber"], row["q"]))
    extra = min((row for row in validation if row["q"] > REFERENCE_ORDER),
                key=lambda row: (row["mean_ber"], row["q"]))
    return dict(selected_q=ranking[0]["q"], runner_up_q=ranking[1]["q"],
                best_extension_q=extra["q"], q32_mean_ber=next(r["mean_ber"] for r in validation if r["q"] == 32),
                selected_mean_ber=ranking[0]["mean_ber"],
                selection_rule="minimum mean full-matrix BER over 0:5:30 dB on validation; ties use lower Q",
                test_scope="Q32, validation-selected Q, best Q33..40, and validation runner-up")


def finish(output, metadata, selection, validation, test, checks, concentration, basis_checks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = dict(complete=True, metadata=metadata, selection=selection, validation=validation, test=test,
                  q32_reproduction=checks, basis_diagnostics=basis_checks,
                  dpss_concentration_q1_to_q40=concentration.tolist())
    write_json(output/"metrics.json", report)
    with (output/"test_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q", "snr_db", "bit_errors", "bits", "ber", "mean_nmse_db"])
        for row in test:
            for snr, errors, ber, nmse in zip(SNR_DB, row["bit_errors"], row["ber"], row["mean_nmse_db"]):
                writer.writerow([row["q"], snr, errors, row["bits_per_snr"], ber, nmse])
    for metric in ("ber", "nmse"):
        fig, ax = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
        for row in test:
            y = row["ber"] if metric == "ber" else row["mean_nmse_db"]
            ax.plot(SNR_DB, y, "o-", linewidth=2 if row["q"] == selection["selected_q"] else 1.1,
                    label=f"Q={row['q']}" + (" selected" if row["q"] == selection["selected_q"] else ""))
        if metric == "ber":
            ax.set_yscale("log")
        ax.set(title="NCS1 raw CSI | held-out test", xlabel="SNR (dB)",
               ylabel="BER" if metric == "ber" else "Mean raw CSI NMSE (dB)", xticks=SNR_DB)
        ax.grid(alpha=.25, which="both")
        ax.legend()
        fig.savefig(output/f"{metric}_snr.png", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    axes[0].plot([r["q"] for r in validation], [100*r["mean_ber"] for r in validation], "o-")
    axes[1].plot([r["q"] for r in validation],
                 [10*np.log10(np.mean(r["mean_nmse_linear"])) for r in validation], "o-")
    for ax, label in zip(axes, ("Mean BER (%)", "Mean raw CSI NMSE (dB)")):
        ax.set(xlabel="Q", ylabel=label, xticks=ORDERS)
        ax.grid(alpha=.25)
    fig.suptitle("NCS1 validation | no compensation | 12 Hz / L=176")
    fig.savefig(output/"validation_q_search.png", dpi=170)
    plt.close(fig)


def run(source, baseline, output, archive_path, smoke, backend):
    source, baseline, output = (Path(p).resolve() for p in (source, baseline, output))
    if output == source or output == baseline:
        raise ValueError("Extension requires a separate output directory")
    metadata = build_manifest(source, baseline, archive_path, smoke, backend)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"source_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output directory belongs to a different source or run")
    write_json(manifest, metadata)
    prepare_physical_wrapper(source, output, archive_path, metadata, smoke)
    orders = (32, 33, 40) if smoke else ORDERS
    validation = [evaluate_candidate(CHANNEL, output, output, metadata, q, "validation", backend) for q in orders]
    if smoke:
        print("Raw NCS1 Q32/Q33/Q40 smoke completed; no research selection", flush=True)
        return
    baseline_validation = json.loads((baseline/"ncs1/validation_q32.json").read_text(encoding="utf-8"))
    checks = dict(validation=verify_baseline(validation[0], baseline_validation))
    selection = select(validation)
    selection_path = output/"selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise ValueError("Frozen selection changed")
    write_json(selection_path, selection)
    print(f"Frozen NCS1 raw selection: Q={selection['selected_q']}, "
          f"best Q33..40={selection['best_extension_q']}", flush=True)
    test_orders = sorted({REFERENCE_ORDER, selection["selected_q"], selection["best_extension_q"],
                          selection["runner_up_q"]})
    test = [evaluate_candidate(CHANNEL, output, output, metadata, q, "test", backend) for q in test_orders]
    baseline_test_path = baseline/"ncs1/test_q32.json"
    checks["test"] = verify_baseline(next(r for r in test if r["q"] == REFERENCE_ORDER),
                                     json.loads(baseline_test_path.read_text(encoding="utf-8")))
    checks["previous_test_q32_sha256"] = file_sha256(baseline_test_path)
    if file_sha256(source/CHANNEL.lower()/"physical.h5") != metadata["physical_sha256"]:
        raise ValueError("Physical waveform source changed during the run")
    with h5py.File(output/CHANNEL.lower()/"physical.h5", "r") as h:
        concentration = h["concentration"][:]
        basis_checks = json.loads(h.attrs["basis_diagnostics_json"])
    finish(output, metadata, selection, validation, test, checks, concentration, basis_checks)
