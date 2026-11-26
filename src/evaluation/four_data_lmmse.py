"""Banded LMMSE detection for four-symbol time-varying channel taps."""

import numpy as np
from src.algorithms.banded_lmmse.equalizer import equalize_batch
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener


def channel_matrices(taps, backend):
    """Unitary F H_time F^H in FFT-shift order, batched over the first axis."""
    b = backend
    count, n, length = taps.shape
    if length > n:
        raise ValueError("Physical support must not exceed the FFT size")
    rows = np.arange(n)[:, None]
    cols = (rows - np.arange(length)) % n
    if b.torch:
        t = b.torch
        circular = t.zeros((count, n, n), dtype=t.complex128, device=b.device)
        circular[:, b.array(rows), b.array(cols)] = b.array(
            np.asarray(taps, np.complex128)
        )
        return t.fft.fftshift(
            t.fft.ifft(t.fft.fft(circular, dim=-2, norm="ortho"), dim=-1, norm="ortho"),
            dim=(-2, -1),
        )
    circular = np.zeros((count, n, n), np.complex128)
    circular[:, rows, cols] = taps
    return np.fft.fftshift(
        np.fft.ifft(np.fft.fft(circular, axis=-2, norm="ortho"), axis=-1, norm="ortho"),
        axes=(-2, -1),
    )


def detect(
    taps,
    observations,
    known,
    variances,
    noise,
    data_mask,
    backend,
    *,
    half_bandwidth=16,
):
    """Detect with supplied CSI; no bits or reference CSI enter this routine."""
    taps, observations = np.asarray(taps), np.asarray(observations)
    if taps.ndim == 3:
        taps = taps[None]
    count, n = len(variances), known.shape[1]
    if (
        taps.ndim != 4
        or taps.shape[0] not in (1, count)
        or taps.shape[1:3] != (4, n)
        or observations.shape != (count, 4, n)
        or known.shape != (4, n)
    ):
        raise ValueError(
            "CSI and observations do not describe the same four-symbol frame"
        )
    indices = np.flatnonzero(data_mask)
    whitener = make_whitener(noise, backend)
    symbols = np.empty((count, 4, len(indices)), np.complex128)
    for d in range(4):
        matrices = channel_matrices(taps[:, d], backend)
        if len(matrices) == 1 and count != 1:
            matrices = (
                matrices.expand(count, -1, -1)
                if backend.torch
                else np.broadcast_to(matrices, (count, n, n))
            )
        y = backend.array(observations[:, d])
        cancelled = y - matrices @ backend.array(known[d])
        batch = EqualizerBatch(
            matrices, y, cancelled, np.asarray(variances), indices, noise
        )
        symbols[:, d] = equalize_batch(
            batch, backend, whitener=whitener, half_bandwidth=half_bandwidth
        ).symbols
    return symbols
