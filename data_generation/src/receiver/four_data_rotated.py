"""Power-matched full-band 4D waveforms and independent-prior temporal models."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from data_generation.src.receiver.chain import resample_operator
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.receiver.four_data import make_four_data_frame
from data_generation.src.transmitter.ofdm import Frame, carrier_masks, ifft, qpsk


def waveform(frequency, config):
    time = ifft(frequency)
    return np.pad(np.concatenate((time[:, -config["cp"]:], time), axis=1).ravel(),
                   (config["guard_samples"], config["guard_samples"]))


def full_band_layout(config):
    n = config["nfft"]
    if n % 8:
        raise ValueError("The full-band pattern needs a multiple of eight FFT bins")
    base = np.tile(list("DDZPPPPZ"), n//8)
    public = np.random.default_rng(np.random.SeedSequence([config["seed"], 11]))
    masks = {name: np.zeros((4, n), bool) for name in ("data", "pilot", "protection", "tracking")}
    known = np.zeros((4, n), dtype=complex)
    for symbol, shift in enumerate((0, 1, 3, 5)):
        pattern = np.roll(base, shift)
        for name, label in (("data", "D"), ("pilot", "P"), ("protection", "Z")):
            masks[name][symbol] = pattern == label
        masks["tracking"][symbol] = pattern != "D"
        known[symbol, masks["pilot"][symbol]] = qpsk(public.integers(0, 2, (n//2, 2), dtype=np.uint8))
    return known, masks


class TransmitPower:
    """Exact ensemble RF energy for proper independent QPSK and fixed pilots."""

    def __init__(self, chain):
        self.chain = chain
        up = resample_operator(chain.length, 16, 1, chain.c)
        gram = (up.T @ up).tocsr()
        rows, columns = gram.nonzero()
        width = int(np.max(abs(rows-columns)))
        config = chain.c
        step, guard = config["nfft"]+config["cp"], config["guard_samples"]
        self.weights = np.zeros((4, width+1))
        for lag in range(width+1):
            diagonal = gram.diagonal(lag)
            for symbol in range(4):
                start = guard+symbol*step
                self.weights[symbol, lag] = diagonal[start:start+max(step-lag, 0)].sum()

    def data_energy(self, masks):
        masks = np.broadcast_to(np.asarray(masks, bool), (4, self.chain.c["nfft"]))
        correlation = np.fft.ifft(np.fft.ifftshift(masks.astype(float), axes=-1), axis=-1)
        total = 0.0
        for symbol in range(4):
            weights = self.weights[symbol]
            value = weights[0]*correlation[symbol, 0].real
            value += 2*np.sum(weights[1:]*correlation[symbol, np.arange(1, len(weights)) % masks.shape[1]].real)
            total += value*.5
        return float(total)

    def known_energy(self, known):
        return float(np.sum(self.chain.transmit(waveform(known, self.chain.c))**2))

    def match(self, original_config, unit_known, data_masks):
        old = make_four_data_frame(original_config, 11, 2)
        old_data = self.data_energy(carrier_masks(original_config)["data"])
        old_pilot = self.known_energy(old.known)
        new_data, new_pilot = self.data_energy(data_masks), self.known_energy(unit_known)
        pilot_energy = (old_data+old_pilot-new_data)/new_pilot
        if not np.isfinite(pilot_energy) or pilot_energy <= 0:
            raise ValueError("The requested transmit power cannot support these unit-energy data symbols")
        samples = len(self.chain.phase_carrier)
        return dict(pilot_symbol_energy=float(pilot_energy), data_symbol_energy=1.0,
                    old_expected_rf_power=(old_data+old_pilot)/samples,
                    new_expected_rf_power=(new_data+pilot_energy*new_pilot)/samples,
                    rf_samples=samples, definition="ensemble over independent proper unit-energy QPSK payload; fixed public pilots; includes CP and transmit filtering")


def make_full_band_frame(config, record, start, known, masks):
    old = make_four_data_frame(config, record, start)
    count = int(masks["data"][0].sum())
    bits = old.bits[:, :count].copy()
    frequency = known.copy()
    for symbol in range(4):
        frequency[symbol, masks["data"][symbol]] = qpsk(bits[symbol])
    return Frame(frequency, known.copy(), bits, waveform(frequency, config), old.seed)


def load_rotated_prior(design, channel, config, count, length, *, smoke=False):
    design = Path(design)
    with np.load(design/f"{channel.lower()}_temporal_basis.npz", allow_pickle=False) as source:
        basis = source["full_basis"][:, :count].copy()
        rotation = source["rotation"][:, :count].copy()
    model = DPSSModel(basis, 12.0, 8.22586010451598e-6, length, config)
    np.testing.assert_allclose(model.sampled_basis.conj().T @ model.sampled_basis,
                                np.eye(count), atol=1e-10, rtol=0)
    coefficients, identities = [], []
    with h5py.File(design/f"{channel.lower()}_projection_cache.h5", "r") as source:
        if not source.attrs.get("complete") or bool(source.attrs["smoke"]) != smoke:
            raise ValueError("Temporal prior design cache is incomplete or has a different sample scope")
        if source.attrs["channel"] != channel:
            raise ValueError("Temporal prior channel mismatch")
        np.testing.assert_allclose(source["basis"][:] @ rotation, basis, atol=1e-12, rtol=1e-12)
        for group in source["prior"].values():
            identities.append((int(group.attrs["record"]), float(group.attrs["start_seconds"])))
            cross = group["Q27/cross"][:, :, :length].sum(axis=0)
            coefficients.append(rotation.conj().T @ cross)
    if not {record for record, _ in identities} <= set(config["prior_records"]):
        raise ValueError("Temporal prior contains records outside the independent prior split")
    fitted = FrozenPrior.fit_coefficients(model, np.stack(coefficients), 0, 1,
        sorted({record for record, _ in identities}), regularizer=config["relative_regularizer"], delay_block_size=32)
    fitted = replace(fitted, fit_domain="four_fft_windows_with_full_frame_temporal_basis")
    return model, fitted, identities
