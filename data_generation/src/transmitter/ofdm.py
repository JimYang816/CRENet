from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from data_generation.src.pdd import FRAME_STRUCTURE, SYMBOL_COUNT, DATA_SYMBOL_INDICES


def fft(x):
    return np.fft.fftshift(np.fft.fft(x, axis=-1, norm="ortho"), axes=-1)


def ifft(x):
    return np.fft.ifft(np.fft.ifftshift(x, axes=-1), axis=-1, norm="ortho")


def qpsk(bits):
    b = np.asarray(bits)
    if b.shape[-1] != 2 or not np.all((b == 0) | (b == 1)):
        raise ValueError("QPSK requires binary pairs")
    return ((1 - 2*b[..., 0].astype(float)) + 1j*(1 - 2*b[..., 1].astype(float))) / np.sqrt(2)


def decisions(symbols):
    return np.stack((np.asarray(symbols).real < 0, np.asarray(symbols).imag < 0), axis=-1).astype(np.uint8)


def carrier_masks(c):
    if c.get("frame_structure") not in (FRAME_STRUCTURE, "DDDD"):
        raise ValueError("Unknown four-symbol frame layout")
    n = c["nfft"]
    middle = n-2*c["null_per_edge"]
    if c["frame_structure"] == "DDDD":
        group = [2, 1, 1] if middle == c["data_count"]+c["pilot_count"] else [2, 1, 1, 2, 1, 1, 0]
    else:
        group = [1, 1, 0, 2, 0, 1, 1]
    if middle <= 0 or middle % len(group):
        raise ValueError("The interior FFT bins must contain whole carrier groups")
    pattern = np.tile(group, middle//len(group))
    layout = np.pad(pattern, c["null_per_edge"])
    data, pilot = layout == 1, layout == 2
    active = np.zeros(n, bool)
    active[c["null_per_edge"]:n-c["null_per_edge"]] = True
    if data.sum() != c["data_count"] or pilot.sum() != c["pilot_count"]:
        raise ValueError("Frame carrier counts do not match")
    protection = active & ~(data | pilot)
    return dict(active=active, pilot=pilot, data=data, null=~(data | pilot),
                protection=protection, edge=~active, tracking=pilot | protection)


def window_indices(c, symbol):
    return (c["guard_samples"] + symbol*(c["nfft"]+c["cp"]) + c["cp"]
            - c["fft_advance"] + np.arange(c["nfft"]))


def receive_symbols(baseband, c):
    return fft(np.stack([baseband[window_indices(c, s)] for s in range(SYMBOL_COUNT)]))


@dataclass
class Frame:
    x: np.ndarray
    known: np.ndarray
    bits: np.ndarray
    baseband: np.ndarray
    seed: tuple


def make_frame(c, record, start, split=2, *, payload_repeat=None):
    # Each identity has its own stream; iteration order and SNR never change it.
    seed = (c["seed"], split, int(record), round(float(start)*1000), 0)
    if payload_repeat is not None:
        seed += (int(payload_repeat),)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    m = carrier_masks(c)
    if c["frame_structure"] == "DDDD":
        public = np.random.default_rng(np.random.SeedSequence([c["seed"], 11]))
        known = np.zeros((SYMBOL_COUNT, c["nfft"]), np.complex128)
        known[:, m["pilot"]] = qpsk(public.integers(0, 2, (SYMBOL_COUNT, c["pilot_count"], 2), dtype=np.uint8))
        bits = rng.integers(0, 2, (SYMBOL_COUNT, c["data_count"], 2), dtype=np.uint8)
        x = known.copy()
        x[:, m["data"]] = qpsk(bits)
        time = ifft(x)
        blocks = np.concatenate((time[:, -c["cp"]:], time), axis=1).ravel()
        return Frame(x, known, bits, np.pad(blocks, (c["guard_samples"], c["guard_samples"])), seed)
    x = np.zeros((SYMBOL_COUNT, c["nfft"]), np.complex128)
    training_rng = np.random.default_rng(np.random.SeedSequence([c["seed"], 11]))
    x[:, m["active"]] = qpsk(training_rng.integers(0, 2, (SYMBOL_COUNT, m["active"].sum(), 2), dtype=np.uint8))
    for s in DATA_SYMBOL_INDICES:
        x[s, ~(m["data"] | m["pilot"])] = 0
        x[s, m["data"]] = qpsk(rng.integers(0, 2, (m["data"].sum(), 2), dtype=np.uint8))
    bits = decisions(x[list(DATA_SYMBOL_INDICES)][:, m["data"]])
    known = x.copy()
    for s in DATA_SYMBOL_INDICES:
        known[s, m["data"]] = 0
    u = ifft(x)
    blocks = np.concatenate((u[:, -c["cp"]:], u), axis=1).ravel()
    bb = np.pad(blocks, (c["guard_samples"], c["guard_samples"]))
    return Frame(x, known, bits, bb, seed)


def overhead(c):
    n, cp = c["nfft"], c["cp"]
    data_symbols = SYMBOL_COUNT if c.get("frame_structure") == "DDDD" else len(DATA_SYMBOL_INDICES)
    return dict(nondata_grid_fraction=1-c["data_count"]*data_symbols/(SYMBOL_COUNT*n),
                cp_fraction_of_transmitted_samples=cp/(n+cp),
                cp_to_useful_ratio=cp/n, useful_frame_samples=SYMBOL_COUNT*(n+cp),
                useful_passband_samples=n*16, cp_passband_samples=cp*16,
                subcarrier_spacing_hz=c["fs_baseband"]/n)
