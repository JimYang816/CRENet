"""Independent-prior/validation audit of 12-Hz 4D representations and pilots."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy import linalg
from scipy.signal.windows import dpss

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT
from data_generation.src.receiver.estimation import input_windows
from data_generation.src.receiver.four_data import BlockedFourDataReceiver, fixed_four_data_model
from data_generation.src.transmitter.ofdm import qpsk
from src.common.experiment import file_sha256, write_json
from src.evaluation.four_data_mmse import experiment_config


ORDERS = (8, 12, 16, 20, 24, 27)
LENGTHS = (64, 96, 128, 176, 224, 256, 384, 512, 736, 1024)
MODES = (16, 24, 32, 40, 48, 56)
LAYOUTS = {"current": ("DDZPZDD", False),
           "more_pilots_same_data": ("DDPPZDD", True),
           "more_pilots_more_observations": ("DDZPPZD", True)}


def pilot_layout(config, name):
    """All four D use public independent QPSK pilots; optional staggered masks."""
    pattern, stagger = LAYOUTS[name]
    n, edge = config["nfft"], config["null_per_edge"]
    active = n-2*edge
    if active % len(pattern):
        raise ValueError("Pilot pattern must tile the occupied band")
    base = np.tile(list(pattern), active//len(pattern))
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 11]))
    known, rows, counts = np.zeros((4, n), complex), [], []
    for symbol in range(4):
        layout = np.roll(base, (0, 1, 3, 5)[symbol] if stagger else 0)
        pilots = edge+np.flatnonzero(layout == "P")
        selected = edge+np.flatnonzero(layout != "D")
        known[symbol, pilots] = qpsk(rng.integers(0, 2, (len(pilots), 2), dtype=np.uint8))
        rows.append(selected)
        counts.append(dict(pilots=len(pilots), protection=int(np.count_nonzero(layout == "Z")),
                           data=int(np.count_nonzero(layout == "D")), observations=len(selected)))
    return known, rows, counts


def measurement_matrix(basis, delay_basis, known, rows):
    """Channel coefficients C=D U^H; construct the public-pilot map of vec(D)."""
    symbols, n, q = basis.shape
    delayed = input_windows(known, len(delay_basis)) @ delay_basis.conj()
    atoms = (basis[:, :, :, None]*delayed[:, :, None, :]).reshape(symbols, n, -1)
    frequency = np.fft.fftshift(np.fft.fft(atoms, axis=1, norm="ortho"), axes=1)
    return np.concatenate([frequency[s, selected] for s, selected in enumerate(rows)])


def projection_error(basis, coefficients, cross, energy, support, delay_basis=None):
    """Per-D linear relative errors without materializing the reconstructed taps."""
    fitted = coefficients[:, :support]
    if delay_basis is not None:
        fitted = (fitted @ delay_basis) @ delay_basis.conj().T
    result = []
    for symbol in range(4):
        gram = basis[symbol].conj().T @ basis[symbol]
        prediction = float(np.sum(fitted.conj()*(gram @ fitted)).real)
        correlation = float(np.sum(fitted.conj()*cross[symbol, :, :support]).real)
        result.append(max(float(energy[symbol])+prediction-2*correlation, 0)/float(energy[symbol]))
    return result


def collect(channel, archive, path, *, smoke=False):
    """Use only independent prior 051-055 and estimator validation 056-060."""
    config = experiment_config(channel)
    model = fixed_four_data_model(config, count=max(ORDERS), length=1024)
    prior = [(record, start) for record in config["prior_records"] for start in (2, 14)]
    validation = [(record, start) for record in config["validation_records"] for start in (2, 14)]
    if smoke:
        prior, validation = prior[:2], validation[:1]
    if path.exists():
        with h5py.File(path, "r") as saved:
            if saved.attrs["channel"] != channel or bool(saved.attrs["smoke"]) != smoke:
                raise ValueError("Design cache identity mismatch")
            if saved.attrs.get("complete"):
                return
        raise ValueError("Incomplete design cache; use a fresh output directory")
    dataset, receiver = WatermarkArchive(archive, channel), BlockedFourDataReceiver(config)
    sampled = model.sampled_basis
    pins = {q: linalg.pinv(sampled[:, :q], check_finite=False) for q in ORDERS}
    with h5py.File(path, "w") as saved:
        saved.attrs.update(channel=channel, smoke=smoke, complete=False)
        saved.create_dataset("basis", data=model.basis)
        for split, identities in (("prior", prior), ("validation", validation)):
            for index, (record, start) in enumerate(identities):
                sounding = dataset.load(record)
                taps = np.stack([receiver.reference_taps(sounding, start, symbol)[0] for symbol in range(4)])
                group = saved.create_group(f"{split}/{index:03d}")
                group.attrs.update(record=record, start_seconds=start, source_sha256=sounding.sha256)
                group.create_dataset("energy", data=np.sum(abs(taps)**2, axis=(1, 2)))
                group.create_dataset("delay_energy", data=np.sum(abs(taps)**2, axis=1))
                for q in ORDERS:
                    g = group.create_group(f"Q{q}")
                    g.create_dataset("coefficients", data=pins[q] @ taps.reshape(-1, 1024))
                    v = sampled[:, :q].reshape(4, config["nfft"], q)
                    g.create_dataset("cross", data=np.einsum("dnq,dnl->dql", v.conj(), taps, optimize=True))
                saved.flush()
                print(f"{channel} representation {split} {index+1}/{len(identities)}: {record:03d} @ {start}s", flush=True)
        saved.attrs["complete"] = True


def audit(channel, path, output, *, ranks=True):
    config = experiment_config(channel)
    result = dict(channel=channel, representation=[], reduced_representation=[], sparse_support=[], pilot_rank=[],
                  prior_records=config["prior_records"], validation_records=config["validation_records"])
    with h5py.File(path, "r") as saved:
        total = len(saved["basis"])
        positions = np.concatenate([s*(config["nfft"]+config["cp"])+config["cp"]-config["fft_advance"]
                                    +np.arange(config["nfft"]) for s in range(4)])
        full_basis = saved["basis"][:]
        time_energy = 2*total*12/config["fs_baseband"]
        _, eigenvalues = dpss(total, time_energy/2, Kmax=max(ORDERS), norm=2, return_ratios=True)
        result["time_basis"] = dict(samples=total, seconds=total/config["fs_baseband"],
                                    two_W_T=time_energy, tail_by_q={str(q):float(1-eigenvalues[:q].sum()/time_energy) for q in ORDERS})
        validation = list(saved["validation"].values())
        for q in ORDERS:
            basis = full_basis[positions, :q].reshape(4, config["nfft"], q)
            for length in LENGTHS:
                values = [projection_error(basis, g[f"Q{q}/coefficients"][:], g[f"Q{q}/cross"][:],
                                           g["energy"][:], length) for g in validation]
                mean = float(np.mean(values))
                result["representation"].append(dict(q=q, length=length, parameters=q*length,
                                                       mean_nmse_db=10*np.log10(mean)))
        # Learn delay modes from prior coefficients only. L736 preserves a
        # shared long physical support for both channels; Q27 covers 12 Hz.
        q, length = 27, 736
        training = np.concatenate([g[f"Q{q}/coefficients"][:, :length] for g in saved["prior"].values()])
        # EVR also handles the locally reproduced Windows complex-SVD failure.
        scale = float(np.linalg.norm(training))
        covariance = (training/scale).conj().T @ (training/scale)
        values, vectors = linalg.eigh((covariance+covariance.conj().T)*0.5,
                                      check_finite=False, driver="evr")
        delay_modes = vectors[:, ::-1]
        singular = scale*np.sqrt(np.maximum(values[::-1], 0))
        basis = full_basis[positions, :q].reshape(4, config["nfft"], q)
        for count in MODES:
            modes = delay_modes[:, :count]
            values = [projection_error(basis, g[f"Q{q}/coefficients"][:], g[f"Q{q}/cross"][:],
                                       g["energy"][:], length, modes) for g in validation]
            mean = float(np.mean(values))
            result["reduced_representation"].append(dict(q=q, length=length, modes=count, parameters=q*count,
                prior_retained_energy=float(np.sum(singular[:count]**2)/np.sum(singular**2)),
                mean_nmse_db=float(10*np.log10(mean))))
        prior_profile = np.mean([g["delay_energy"][:]/g["energy"][:][:, None]
                                  for g in saved["prior"].values()], axis=(0, 1))
        prior_order = np.argsort(prior_profile[:length])[::-1]
        for count in (32, 48, 56, 64, 96, 128, 176):
            selected = np.sort(prior_order[:count])
            values, oracle_tail = [], []
            for g in validation:
                energy = g["energy"][:]
                profile = np.mean(g["delay_energy"][:]/energy[:, None], axis=0)
                oracle_tail.append(max(1-float(np.sort(profile[:length])[-count:].sum()), 0))
                modes = np.eye(length)[:, selected]
                values.append(projection_error(basis, g[f"Q{q}/coefficients"][:],
                                               g[f"Q{q}/cross"][:], energy, length, modes))
            result["sparse_support"].append(dict(q=q, length=length, retained_taps=count,
                parameters=q*count, prior_selected_delays=selected.tolist(),
                prior_selected_mean_nmse_db=float(10*np.log10(np.mean(values))),
                oracle_support_tail_db=float(10*np.log10(np.mean(oracle_tail)))))
        np.savez(output/f"{channel.lower()}_delay_basis.npz", basis=delay_modes[:, :max(MODES)],
                 temporal_basis=full_basis, prior_singular_values=singular)
        if ranks:
            for name in LAYOUTS:
                known, rows, counts = pilot_layout(config, name)
                for count in (32, 48):
                    matrix = measurement_matrix(basis, delay_modes[:, :count], known, rows)
                    norms = np.linalg.norm(matrix, axis=0)
                    if np.any(norms == 0):
                        raise ValueError("Zero public-pilot channel atom")
                    matrix /= norms
                    singular = linalg.svdvals(matrix, check_finite=False)
                    ratio = float(singular[-1]/singular[0])
                    row = dict(layout=name, counts_per_symbol=counts[0], observations=len(matrix),
                               unknowns=q*count, modes=count,
                               numerical_rank_1e_10=int(np.count_nonzero(singular > singular[0]*1e-10)),
                               effective_rank_1e_6=int(np.count_nonzero(singular > singular[0]*1e-6)),
                               min_to_max_singular_value=ratio, column_normalized_condition=1/ratio)
                    result["pilot_rank"].append(row)
                    print(f"{channel} {name} Q27 K{count}: {row['observations']} x {row['unknowns']}, "
                          f"rank={row['numerical_rank_1e_10']}, condition={1/ratio:.3g}", flush=True)
    write_json(output/f"{channel.lower()}_design.json", result)
    return result


def run_design(output, archive, *, smoke=False, ranks=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for channel in ("NOF1", "NCS1"):
        cache = output/f"{channel.lower()}_projection_cache.h5"
        collect(channel, archive, cache, smoke=smoke)
        results[channel] = audit(channel, cache, output, ranks=ranks)
    report = dict(schema="four-data-12hz-design-audit-v1", complete=True, smoke=smoke,
                  source=dict(code_sha256=file_sha256(Path(__file__)), git_head=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()),
                  note="Representation uses oracle projection on independent validation, not receiver NMSE or BER; rank uses public pilots only",
                  channels=results)
    write_json(output/"design.json", report)
    return report
