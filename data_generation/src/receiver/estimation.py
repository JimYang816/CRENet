"""Frozen delay-local channel statistics for pilot-aided estimation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from data_generation.src.transmitter.ofdm import ifft
from data_generation.src.receiver.dpss import FULL_FRAME_DOMAIN, SINGLE_SYMBOL_DOMAIN
from src.common.numerics import hermitian


def input_windows(known, length, *, delay_indices=None):
    n = known.shape[1]
    delays = np.arange(length) if delay_indices is None else np.asarray(delay_indices)
    if delays.shape != (length,) or not np.issubdtype(delays.dtype, np.integer):
        raise ValueError("Delay indices must contain one integer per coefficient column")
    return ifft(known)[:, (np.arange(n)[:, None]-delays) % n]


@dataclass
class FrozenPrior:
    model: object
    coefficients: np.ndarray
    temporal_covariance: np.ndarray
    delay_covariance: np.ndarray
    empirical_weight: float
    scale: float
    regularizer: float
    source_records: tuple
    fit_domain: str = FULL_FRAME_DOMAIN
    delay_block_size: int = 0
    interference_scale: float = 1.0
    feedback_iterations: int = 0

    @classmethod
    def fit_coefficients(cls, model, coefficients, weight, scale, source_records,
                         regularizer=1e-6, *, fit_domain=FULL_FRAME_DOMAIN, delay_block_size=0,
                         interference_scale=1.0):
        coeff = np.asarray(coefficients, np.complex128).copy()
        if (coeff.ndim != 3 or coeff.shape[1:] != (model.count, model.length)
                or not len(coeff) or not np.all(np.isfinite(coeff))
                or np.sum(abs(coeff)**2) <= 0):
            raise ValueError("Invalid prior coefficients")
        if fit_domain not in (FULL_FRAME_DOMAIN, SINGLE_SYMBOL_DOMAIN):
            raise ValueError("Unknown prior coefficient domain")
        max_block_size = model.c["nfft"] if fit_domain == SINGLE_SYMBOL_DOMAIN else model.length
        if not isinstance(delay_block_size, (int, np.integer)) or not 0 <= delay_block_size <= max_block_size:
            raise ValueError("Invalid delay covariance block size")
        if not np.isfinite(interference_scale) or interference_scale < 0:
            raise ValueError("Invalid data interference scale")
        time = np.einsum("mql,mrl->qr", coeff, coeff.conj(), optimize=True)/len(coeff)
        delay = np.einsum("mql,mqk->lk", coeff, coeff.conj(), optimize=True)/len(coeff)
        trace = np.mean(np.sum(abs(coeff)**2, axis=(1, 2)))
        # Keep the separable and empirical second moments at the same trace.
        time += regularizer*np.trace(time).real/len(time)*np.eye(len(time))
        delay += regularizer*np.trace(delay).real/len(delay)*np.eye(len(delay))
        time /= np.trace(time).real
        delay *= trace/np.trace(delay).real
        return cls(model, coeff, hermitian(time), hermitian(delay), weight, scale,
                   regularizer, tuple(source_records), fit_domain, int(delay_block_size), float(interference_scale))

    def separable_blocks(self):
        """Retain each delay block's temporal law and within-block correlation."""
        if not self.delay_block_size or self.model.delays[-1] < self.delay_block_size:
            yield slice(0, self.model.length), self.temporal_covariance, self.delay_covariance
            return
        # Keep the original physical delay blocks when some taps are omitted.
        groups = self.model.delays//self.delay_block_size
        boundaries = np.r_[0, np.flatnonzero(np.diff(groups))+1, self.model.length]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            select = slice(start, end)
            coeff = self.coefficients[:, :, select]
            time = np.einsum("mql,mrl->qr", coeff, coeff.conj(), optimize=True)/len(coeff)
            delay = np.einsum("mql,mqk->lk", coeff, coeff.conj(), optimize=True)/len(coeff)
            energy = float(np.trace(delay).real)
            if energy == 0:
                continue
            time += self.regularizer*np.trace(time).real/len(time)*np.eye(len(time))
            delay += self.regularizer*energy/len(delay)*np.eye(len(delay))
            time /= np.trace(time).real
            delay *= energy/np.trace(delay).real
            yield select, hermitian(time), hermitian(delay)
