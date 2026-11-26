"""Four-data comb-pilot physical inputs and raw-CSI Q sweep."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
from scipy import linalg, sparse
from scipy.signal.windows import dpss

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import FROZEN_SUPPORT_CONFIGS, ROOT, digest, load_legacy_dddd_config
from data_generation.src.receiver import pdd
from data_generation.src.receiver.chain import ReceiverChain, relative_energy, resample
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.transmitter.ofdm import carrier_masks, decisions, fft, make_frame as transmit_frame, receive_symbols, window_indices
from src.common.experiment import file_sha256, write_json


ORDERS = tuple(range(16, 33))
BANDWIDTH_HZ = 12.0
LENGTH = 176
SNR_DB = (0, 5, 10, 15, 20, 25, 30)
PRIOR = [(r, s) for r in range(51, 56) for s in (2, 14)]
VALIDATION = [(r, s) for r in (13, 21, 29, 37, 45) for s in (2, 14)]
TEST = [(r, s) for r in (11, 15, 19, 23, 27, 31, 35, 39, 43, 47) for s in (2,)]
CODE = ("src/evaluation/four_data_q_sweep.py", "scripts/evaluate/four_data_q_sweep.py",
        "data_generation/src/receiver/chain.py", "data_generation/src/receiver/dpss.py",
        "data_generation/src/receiver/estimation.py", "data_generation/src/receiver/pdd.py",
        "data_generation/src/transmitter/ofdm.py", "data_generation/src/channel/watermark/replay.py",
        "src/algorithms/full_matrix_lmmse/equalizer.py", "src/algorithms/likelihood.py",
        "data_generation/src/config.py", *FROZEN_SUPPORT_CONFIGS)


def config(channel):
    c = load_legacy_dddd_config(channel)
    if (c["frame_structure"] != "DDDD" or c["delay_length"] != LENGTH
            or c["dpss_half_bandwidth_hz"] != BANDWIDTH_HZ
            or (c["data_count"], c["pilot_count"]) != (512, 256)):
        raise ValueError("Four-data configuration mismatch")
    return c


def masks(c):
    """Retain 64 edge nulls; use P,D,D,P,D,D,zero on the 896 interior bins."""
    n, edge = c["nfft"], c["null_per_edge"]
    if n != 1024 or edge != 64 or (n - 2 * edge) % 7:
        raise ValueError("This experiment assumes FFT1024 and 64 nulls per edge")
    result = carrier_masks(c)
    if (result["data"].sum(), result["pilot"].sum(), result["protection"].sum()) != (512, 256, 128):
        raise AssertionError("Unexpected carrier allocation")
    return result


def make_frame(c, record, start, split):
    return transmit_frame(c, record, start, split, payload_repeat=0)


def basis(c):
    total = 4 * (c["nfft"] + c["cp"])
    nw = total * BANDWIDTH_HZ / c["fs_baseband"]
    vectors, concentration = dpss(total, nw, Kmax=max(ORDERS), sym=True, norm=2, return_ratios=True)
    return vectors.T.copy(), concentration


@contextmanager
def four_data_estimator_layout():
    """Temporarily expose the four data symbols to the existing joint estimator."""
    before = (pdd.DATA_SYMBOL_INDICES, pdd.TRAINING_SYMBOL_INDICES)
    pdd.DATA_SYMBOL_INDICES, pdd.TRAINING_SYMBOL_INDICES = (0, 1, 2, 3), ()
    try:
        yield
    finally:
        pdd.DATA_SYMBOL_INDICES, pdd.TRAINING_SYMBOL_INDICES = before


class BlockedReceiver:
    """The same physical RF chain, with bounded native-channel interpolation."""
    def __init__(self, c, block=256):
        self.chain = ReceiverChain(c)
        self.block = block
        self.ui_conj = self.chain.ui.conjugate()

    def native_rows(self, sounding, origin, indices):
        taps = sounding.interpolate(origin + indices/sounding.fs_delay)
        cols = indices[:, None] - np.arange(taps.shape[1])
        valid = cols >= 0
        counts = np.minimum(indices+1, taps.shape[1])
        indptr = np.r_[0, np.cumsum(counts)]
        return sparse.csr_matrix((taps[:, ::-1][valid[:, ::-1]], cols[:, ::-1][valid[:, ::-1]], indptr),
                                 shape=(len(indices), self.chain.u.shape[0]))

    def reference_rows(self, sounding, origin, indices):
        out = []
        for begin in range(0, len(indices), self.block):
            selected = indices[begin:begin+self.block]
            forward, image = self.chain.r[selected], self.chain.ri[selected]
            support = np.union1d(forward.indices, image.indices)
            native = self.native_rows(sounding, origin, support)
            direct = (forward[:, support] @ native) @ self.chain.u
            mixed = (image[:, support] @ native.conjugate()) @ self.ui_conj
            block = (.5 * (direct+mixed)).tocsr()
            block.eliminate_zeros()
            out.append(block)
        return sparse.vstack(out, format="csr")

    def prior(self, sounding, start, full_basis):
        c = self.chain.c
        origin = start-c["guard_samples"]/c["fs_baseband"]
        result = np.zeros((full_basis.shape[1], LENGTH), complex)
        for begin in range(0, len(full_basis), self.block):
            end = min(begin+self.block, len(full_basis))
            indices = c["guard_samples"]+np.arange(begin, end)
            operator = self.reference_rows(sounding, origin, indices)
            cols = indices[:, None]+c["fft_advance"]-np.arange(LENGTH)
            rows = np.broadcast_to(np.arange(end-begin)[:, None], cols.shape)
            valid = (cols >= 0) & (cols < operator.shape[1])
            taps = np.zeros(cols.shape, complex)
            taps[valid] = np.asarray(operator[rows[valid], cols[valid]]).ravel()
            result += full_basis[begin:end].conj().T @ taps
        return result

    def reference(self, sounding, start, symbol):
        c, n = self.chain.c, self.chain.n
        origin = start-c["guard_samples"]/c["fs_baseband"]
        op = self.reference_rows(sounding, origin, window_indices(c, symbol)).tocoo()
        offset = op.col-(c["guard_samples"]+symbol*(n+c["cp"]))
        inside = (offset >= 0) & (offset < n+c["cp"])
        rows = op.row[inside]
        delays = (rows-(offset[inside]-c["cp"])) % n
        taps = np.zeros((n, n), complex)
        np.add.at(taps, (rows, delays), op.data[inside])
        leak = float(np.sum(abs(op.data[~inside])**2)/max(np.sum(abs(op.data)**2), 1e-300))
        return taps, leak

    def replay(self, sounding, origin, frame):
        c, chain = self.chain.c, self.chain
        tx = chain.transmit(frame.baseband)
        native = resample(tx*chain.phase_carrier.conj(), 1, 6, c)
        length = sounding.h.shape[1]
        delayed = np.lib.stride_tricks.sliding_window_view(np.pad(native, (length-1, 0)), length)[:, ::-1]
        received = np.empty_like(native)
        for begin in range(0, len(native), self.block):
            end = min(begin+self.block, len(native))
            impulse = sounding.interpolate(origin+np.arange(begin, end)/sounding.fs_delay)
            received[begin:end] = np.einsum("ij,ij->i", impulse, delayed[begin:end])
        return np.real(resample(received, 6, 1, c)[:len(tx)]*chain.phase_carrier)


def identities(smoke):
    return (PRIOR[:2], VALIDATION[:1], TEST[:1]) if smoke else (PRIOR, VALIDATION, TEST)


def prepare(channel, output, archive_path, smoke):
    c = config(channel)
    prior_ids, validation_ids, test_ids = identities(smoke)
    assert not ({r for r, _ in prior_ids} & {r for r, _ in validation_ids+test_ids})
    assert not ({r for r, _ in validation_ids} & {r for r, _ in test_ids})
    run = output/channel.lower()
    run.mkdir(parents=True, exist_ok=True)
    path = run/"physical.h5"
    if path.exists():
        with h5py.File(path, "r") as h:
            if h.attrs.get("complete") and h.attrs.get("identity") == output_identity(channel, archive_path, smoke):
                print(f"{channel}: verified existing physical inputs", flush=True)
                return
        raise ValueError(f"Incomplete or mismatched physical cache: {path}")
    archive, receiver = WatermarkArchive(archive_path, channel), BlockedReceiver(c)
    full_basis, concentration = basis(c)
    partial = path.with_suffix(".partial.h5")
    with h5py.File(partial, "a") as h:
        identity = output_identity(channel, archive_path, smoke)
        if "identity" in h.attrs and h.attrs["identity"] != identity:
            raise ValueError(f"Partial cache belongs to another experiment: {partial}")
        h.attrs.update(identity=identity, complete=False, channel=channel)
        if "basis" not in h:
            h.create_dataset("basis", data=full_basis)
            h.create_dataset("concentration", data=concentration)
            h.create_dataset("unit_noise", data=receiver.chain.noise_covariance)
            h.create_dataset("known", data=make_frame(c, *validation_ids[0], 1).known)
            h.create_dataset("prior", (len(prior_ids), max(ORDERS), LENGTH), dtype=np.complex128)
            h.create_dataset("prior_complete", data=np.zeros(len(prior_ids), dtype=bool))
        np.testing.assert_allclose(h["basis"][:], full_basis, rtol=0, atol=1e-13)
        prior = h["prior"]
        for i, (record, start) in enumerate(prior_ids):
            if h["prior_complete"][i]:
                continue
            t = time.perf_counter()
            prior[i] = receiver.prior(archive.load(record), start, full_basis)
            h.flush()
            h["prior_complete"][i] = True
            h.flush()
            print(f"{channel}: prior {i+1}/{len(prior_ids)} {record}@{start}s {time.perf_counter()-t:.1f}s", flush=True)
        for split, split_code, ids in (("validation", 1, validation_ids), ("test", 2, test_ids)):
            for i, (record, start) in enumerate(ids):
                key = f"{split}/{i:04d}"
                if key in h and h[key].attrs.get("complete"):
                    continue
                if key in h:
                    del h[key]
                t = time.perf_counter()
                sounding = archive.load(record)
                frame = make_frame(c, record, start, split_code)
                origin = start-c["guard_samples"]/c["fs_baseband"]
                clean = receiver.replay(sounding, origin, frame)
                y = receive_symbols(receiver.chain.receive(clean), c)
                noise_seed = [c["seed"], split_code, record, round(start*1000), 1, 0]
                rng = np.random.default_rng(np.random.SeedSequence(noise_seed))
                standard = receive_symbols(receiver.chain.receive(rng.standard_normal(len(clean))), c)
                power = float(np.mean(abs(y)**2))
                variance = power/receiver.chain.noise_gain/10**(np.asarray(SNR_DB)/10)
                refs = [receiver.reference(sounding, start, d) for d in range(4)]
                truth = np.stack([item[0] for item in refs])
                predicted = fft(np.sum(truth*input_windows(frame.x, c["nfft"]), axis=-1))
                model_error = [relative_energy(a-b, a) for a, b in zip(y, predicted, strict=True)]
                support = (np.sum(abs(truth[..., LENGTH:])**2, axis=(1, 2)) /
                           np.sum(abs(truth)**2, axis=(1, 2)))
                g = h.create_group(key)
                g.attrs.update(record=record, start_seconds=start, rf_power=power,
                               payload_seed=json.dumps(frame.seed), sounding_sha256=sounding.sha256,
                               noise_seed=json.dumps([c["seed"], split_code, record, round(start*1000), 1, 0]),
                               intersymbol_leakage=json.dumps([v for _, v in refs]),
                               model_error=json.dumps(model_error), support_residual=json.dumps(support.tolist()))
                for name, value in dict(truth=truth, clean=y, standard=standard, variance=variance, bits=frame.bits).items():
                    g.create_dataset(name, data=value, compression="lzf" if name == "truth" else None)
                h.flush()
                g.attrs["complete"] = True
                h.flush()
                print(f"{channel}: {split} {i+1}/{len(ids)} {record}@{start}s "
                      f"leak={max(v for _, v in refs):.3g} {time.perf_counter()-t:.1f}s", flush=True)
        h.attrs["complete"] = True
    partial.replace(path)


@lru_cache(maxsize=8)
def output_identity(channel, archive_path, smoke):
    return digest(dict(schema="four-data-256-pilot-q-v1", channel=channel, config=config(channel),
                       smoke=smoke, ids=identities(smoke), orders=ORDERS,
                       archive_sha256=file_sha256(archive_path),
                       code_sha256={name: file_sha256(ROOT/name) for name in CODE}))


def raw_nmse(model, coeff, truth):
    """Full-matrix NMSE by the unitary Fourier identity, including all omitted delays."""
    n, length = model.c["nfft"], model.length
    basis4 = model.sampled_basis.reshape(4, n, model.count)
    result = np.empty((len(coeff), 4))
    for d in range(4):
        estimated = np.einsum("nq,sql->snl", basis4[d], coeff, optimize=True)
        energy = np.sum(abs(truth[d])**2)
        residual = np.sum(abs(estimated-truth[d, :, :length])**2, axis=(1, 2))
        result[:, d] = (residual+np.sum(abs(truth[d, :, length:])**2))/energy
    return result


def time_noise_covariance(noise):
    unshifted = np.fft.ifftshift(noise, axes=(0, 1))
    return np.fft.ifft(np.fft.fft(unshifted, axis=1, norm="ortho"), axis=0, norm="ortho")


def estimate_split(h, c, model, order, split, backend):
    from src.evaluation.four_data_raw_q_sweep import detect_raw
    known, noise = h["known"][:], h["unit_noise"][:]
    groups = list(h[split].values())
    observations = np.stack([g["clean"][:][None]+np.sqrt(g["variance"][:])[:, None, None]*g["standard"][:][None]
                             for g in groups])
    variances = np.stack([g["variance"][:] for g in groups])
    prior = FrozenPrior.fit_coefficients(model, h["prior"][:, :order], 0, 1, range(51, 56),
                                         regularizer=c["relative_regularizer"], delay_block_size=32)
    m = masks(c)
    with four_data_estimator_layout():
        estimator = pdd.JointEstimator(prior, known, m, noise, backend)
        coeff = estimator.estimate_coefficients(observations.reshape(-1, 4, c["nfft"]), known, m,
                                                variances.ravel()).reshape(len(groups), len(SNR_DB), order, LENGTH)
    del estimator, prior
    if backend.torch:
        backend.torch.cuda.empty_cache()
    rows = []
    for i, g in enumerate(groups):
        t = time.perf_counter()
        detected = detect_raw(model, coeff[i], observations[i], known, variances[i], noise, m["data"], backend)
        errors = np.count_nonzero(decisions(detected) != g["bits"][:][None], axis=(-1, -2))
        raw = raw_nmse(model, coeff[i], g["truth"][:])
        rows.append(dict(record=int(g.attrs["record"]), start_seconds=float(g.attrs["start_seconds"]),
                         errors=errors.tolist(), raw_nmse=raw.tolist()))
        print(f"{c['channel']} {split} Q={order} frame {i+1}/{len(groups)} {time.perf_counter()-t:.1f}s", flush=True)
    error = np.asarray([r["errors"] for r in rows]).sum(axis=(0, 2))
    nmse = np.asarray([r["raw_nmse"] for r in rows]).mean(axis=(0, 2))
    count = len(groups)*4*512*2
    return dict(q=order, split=split, frames=rows, input_view="raw", bit_errors=error.tolist(),
                bits_per_snr=count, ber=(error/count).tolist(), mean_ber=float(np.mean(error/count)),
                mean_nmse_db=(10*np.log10(nmse)).tolist())


def run_candidate(channel, output, archive_path, smoke, order, split, backend):
    c = config(channel)
    run = output/channel.lower()
    out = run/f"{split}_q{order:02d}.json"
    with h5py.File(run/"physical.h5", "r") as h:
        identity = output_identity(channel, archive_path, smoke)
        if not h.attrs["complete"] or h.attrs["identity"] != identity:
            raise ValueError("Physical inputs do not match the current experiment")
        if out.exists():
            row = json.loads(out.read_text(encoding="utf-8"))
            if row["identity"] != identity or row["q"] != order or row["split"] != split:
                raise ValueError("Candidate result identity mismatch")
            return row
        model = DPSSModel(h["basis"][:, :order], BANDWIDTH_HZ, 1e-5, LENGTH, c)
        row = estimate_split(h, c, model, order, split, backend)
        row.update(identity=identity, concentration=h["concentration"][:order].tolist())
        write_json(out, row)
    print(f"{channel} {split} Q={order}: mean BER={100*row['mean_ber']:.4f}%", flush=True)
    return row


def choose(validation):
    if set(validation) != {"NOF1", "NCS1"} or any({r["q"] for r in rows} != set(ORDERS) for rows in validation.values()):
        raise ValueError("Both channels must complete the entire Q=16..32 validation sweep")
    by_channel = {channel: min(rows, key=lambda r: (r["mean_ber"], r["q"]))["q"]
                  for channel, rows in validation.items()}
    joint = {str(q): float(np.mean([next(r["mean_ber"] for r in rows if r["q"] == q)
                                   for rows in validation.values()])) for q in ORDERS}
    return dict(by_channel=by_channel, shared=min(ORDERS, key=lambda q: (joint[str(q)], q)),
                joint_validation_mean_ber=joint,
                rule="minimum mean full-matrix BER across 0:5:30 dB; ties use lower Q; both channels equal weight for shared Q")


def finalize(output, selection, validation, test, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = dict(complete=True, selection=selection, metadata=metadata, validation=validation, test=test)
    write_json(output/"metrics.json", report)
    for metric in ("ber", "nmse"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
        for ax, channel in zip(axes, ("NOF1", "NCS1"), strict=True):
            for row in test[channel]:
                vals = row["ber"] if metric == "ber" else row["mean_nmse_db"]
                ax.plot(SNR_DB, vals, "o-", label=f"Q={row['q']}")
            ax.set(title=channel, xlabel="SNR (dB)", ylabel="BER" if metric == "ber" else "Mean CSI NMSE (dB)", xticks=SNR_DB)
            if metric == "ber":
                ax.set_yscale("log")
            ax.grid(alpha=.25, which="both")
            ax.legend()
        fig.suptitle("4D, 256 comb pilots, 12 Hz, L=176 | held-out test")
        fig.savefig(output/f"{metric}_snr.png", dpi=170)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for channel, rows in validation.items():
        ax.plot([r["q"] for r in rows], [100*r["mean_ber"] for r in rows], "o-", label=channel)
    ax.set(xlabel="Q", ylabel="Validation mean BER (%)", xticks=ORDERS)
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output/"validation_q_search.png", dpi=170)
    plt.close(fig)


def run(output, archive_path, smoke, backend, *, stage="all"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    archive_sha256=file_sha256(archive_path), code_sha256={name:file_sha256(ROOT/name) for name in CODE},
                    orders=list(ORDERS), snr_db=list(SNR_DB), bandwidth_hz=BANDWIDTH_HZ, length=LENGTH,
                    frame="D1-D2-D3-D4; no full-pilot symbol", carrier_pattern="P,D,D,P,D,D,zero; 64 edge nulls",
                    observations="256 pilot + 128 interior null per D; four D jointly", smoke=smoke,
                    csi_view="raw receiver-estimated CSI; no reference-assisted adjustment",
                    source_splits={name: [[r, s] for r, s in ids]
                                   for name, ids in zip(("prior", "validation", "test"), identities(smoke))},
                    equalizer="all 1024 receive rows, 512 data columns, full colored noise covariance",
                    precision="complex128", device=backend.device,
                    source_notice="New physical waveforms; no PDDP CSI or cached evaluations reused")
    manifest = output/"source_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output directory belongs to a different run")
    write_json(manifest, metadata)
    if stage in ("all", "prepare"):
        for channel in ("NOF1", "NCS1"):
            prepare(channel, output, archive_path, smoke)
    if stage == "prepare":
        return
    orders = (16, 17) if smoke else ORDERS
    validation = {channel: [run_candidate(channel, output, archive_path, smoke, q, "validation", backend)
                            for q in orders] for channel in ("NOF1", "NCS1")}
    if smoke:
        print("Smoke candidate sweep completed; no research selection", flush=True)
        return
    selection_path = output/"selection.json"
    selected = choose(validation)
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selected:
        raise ValueError("Frozen selection differs from validation results")
    write_json(selection_path, selected)
    print(f"Frozen validation selection: {selected['by_channel']}; shared Q={selected['shared']}", flush=True)
    test = {channel: [run_candidate(channel, output, archive_path, smoke, q, "test", backend)
                      for q in sorted({selected['by_channel'][channel], selected['shared']})]
            for channel in ("NOF1", "NCS1")}
    finalize(output, selected, validation, test, metadata)
