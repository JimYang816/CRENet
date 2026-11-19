"""Isolated four-data-symbol waveform and full-frame pilot-only CSI estimator.

The accepted PDDP dataset and its compensated CSI contract are unchanged.
All four symbols here carry unknown data; only embedded pilots and their
protection carriers enter the no-feedback channel estimate.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.signal.windows import dpss

from data_generation.src.receiver.chain import ReceiverChain, resample
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.pdd import selected_time_covariance
from data_generation.src.transmitter.ofdm import Frame, carrier_masks, ifft, qpsk, window_indices
from src.common.numerics import hermitian


def make_four_data_frame(config, record, start, *, repeat=0):
    """Generate D1-D2-D3-D4, with independent payload and public pilot streams."""
    masks = carrier_masks(config)
    seed = (config["seed"], 2, int(record), round(float(start)*1000), 0, int(repeat))
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    public = np.random.default_rng(np.random.SeedSequence([config["seed"], 11]))
    known = np.zeros((4, config["nfft"]), dtype=np.complex128)
    known[:, masks["pilot"]] = qpsk(public.integers(0, 2, (4, masks["pilot"].sum(), 2), dtype=np.uint8))
    bits = rng.integers(0, 2, (4, masks["data"].sum(), 2), dtype=np.uint8)
    x = known.copy()
    x[:, masks["data"]] = qpsk(bits)
    time = ifft(x)
    blocks = np.concatenate((time[:, -config["cp"]:], time), axis=1).ravel()
    baseband = np.pad(blocks, (config["guard_samples"], config["guard_samples"]))
    return Frame(x, known, bits, baseband, seed)


def fixed_four_data_model(config, *, bandwidth_hz=12.0, count=24, length=176):
    """Use the requested DPSS order explicitly, rather than selecting Q by a tail."""
    total = 4*(config["nfft"]+config["cp"])
    nw = total*bandwidth_hz/config["fs_baseband"]
    basis, ratios = dpss(total, nw, Kmax=count, sym=True, norm=2, return_ratios=True)
    tail = float(max(1-ratios.sum()/(2*nw), 0))
    return DPSSModel(basis.T.copy(), bandwidth_hz, tail, length, config)


class BlockedFourDataReceiver:
    """Bound interpolation memory when replaying NOF1's long native impulse response."""

    def __init__(self, config, *, row_block=256, native_block=256):
        self.chain = ReceiverChain(config)
        self.row_block, self.native_block = row_block, native_block
        self._u_conj = self.chain.ui.conjugate()

    def _native_rows(self, sounding, origin, indices):
        taps = sounding.interpolate(origin+indices/sounding.fs_delay)
        cols = indices[:, None]-np.arange(taps.shape[1])
        valid = cols >= 0
        counts = np.minimum(indices+1, taps.shape[1])
        indptr = np.concatenate(([0], np.cumsum(counts)))
        return sparse.csr_matrix((taps[:, ::-1][valid[:, ::-1]],
                                  cols[:, ::-1][valid[:, ::-1]], indptr),
                                 shape=(len(indices), self.chain.u.shape[0]))

    def reference_rows(self, sounding, origin, indices):
        indices = np.asarray(indices, dtype=np.intp)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.chain.length):
            raise ValueError("Reference rows must lie in the receive waveform")
        blocks = []
        for start in range(0, len(indices), self.row_block):
            selected = indices[start:start+self.row_block]
            forward, image = self.chain.r[selected], self.chain.ri[selected]
            support = np.union1d(forward.indices, image.indices)
            native = self._native_rows(sounding, origin, support)
            direct = (forward[:, support] @ native) @ self.chain.u
            mixed = (image[:, support] @ native.conjugate()) @ self._u_conj
            block = (0.5*(direct+mixed)).tocsr()
            block.eliminate_zeros()
            blocks.append(block)
        return sparse.vstack(blocks, format="csr")

    def replay_frame(self, sounding, origin, frame):
        config = self.chain.c
        if sounding.carrier_hz != config["carrier_hz"]:
            raise ValueError("Incompatible carrier frequencies")
        tx = self.chain.transmit(frame.baseband)
        native = resample(tx*self.chain.phase_carrier.conj(), 1, 6, config)
        length = sounding.h.shape[1]
        delayed = np.lib.stride_tricks.sliding_window_view(np.pad(native, (length-1, 0)), length)[:, ::-1]
        received = np.empty_like(native)
        for begin in range(0, len(native), self.native_block):
            end = min(begin+self.native_block, len(native))
            taps = sounding.interpolate(origin+np.arange(begin, end)/sounding.fs_delay)
            received[begin:end] = np.einsum("ij,ij->i", taps, delayed[begin:end])
        return np.real(resample(received, 6, 1, config)[:len(tx)]*self.chain.phase_carrier)

    def prior_coefficients(self, sounding, start, model):
        """Project physical kernels across all useful and CP samples, without labels."""
        config = self.chain.c
        origin = start-config["guard_samples"]/config["fs_baseband"]
        result = np.zeros((model.count, model.length), dtype=np.complex128)
        for begin in range(0, len(model.basis), self.row_block):
            end = min(begin+self.row_block, len(model.basis))
            indices = config["guard_samples"]+np.arange(begin, end)
            operator = self.reference_rows(sounding, origin, indices)
            columns = indices[:, None]+config["fft_advance"]-np.arange(model.length)
            valid = (columns >= 0) & (columns < operator.shape[1])
            rows = np.broadcast_to(np.arange(end-begin)[:, None], columns.shape)
            taps = np.zeros(columns.shape, dtype=np.complex128)
            taps[valid] = np.asarray(operator[rows[valid], columns[valid]]).ravel()
            result += model.basis[begin:end].conj().T @ taps
        return result

    def reference_taps(self, sounding, start, symbol):
        """Keep the full circular reference and separately measure omitted-symbol leakage."""
        config = self.chain.c
        n = config["nfft"]
        origin = start-config["guard_samples"]/config["fs_baseband"]
        operator = self.reference_rows(sounding, origin, window_indices(config, symbol)).tocoo()
        offset = operator.col-(config["guard_samples"]+symbol*(n+config["cp"]))
        inside = (offset >= 0) & (offset < n+config["cp"])
        samples = operator.row[inside]
        delays = (samples-(offset[inside]-config["cp"])) % n
        taps = np.zeros((n, n), dtype=np.complex128)
        np.add.at(taps, (samples, delays), operator.data[inside])
        leakage = float(np.sum(abs(operator.data[~inside])**2)/max(np.sum(abs(operator.data)**2), 1e-300))
        return taps, leakage


class FourDataEstimator:
    """One shared Q x L coefficient solve from the four embedded-pilot observations."""

    def __init__(self, prior: FrozenPrior, known, masks, unit_noise, backend, *, capture_moments=False):
        self.prior, self.backend = prior, backend
        self.known = np.asarray(known, dtype=np.complex128).copy()
        n, q, length = prior.model.c["nfft"], prior.model.count, prior.model.length
        data_masks = np.broadcast_to(np.asarray(masks["data"], bool), (4, n))
        tracking_masks = np.broadcast_to(np.asarray(masks["tracking"], bool), (4, n))
        if self.known.shape != (4, n) or np.any(self.known[data_masks] != 0):
            raise ValueError("All four symbols must hide their unknown data in the public waveform")
        if (prior.empirical_weight, prior.scale, prior.interference_scale, prior.feedback_iterations) != (0, 1, 1, 0):
            raise ValueError("Use the fixed delay-block prior without feedback")
        self.rows = [np.flatnonzero(mask) for mask in tracking_masks]
        observation_offsets = np.cumsum([0]+[len(rows) for rows in self.rows])
        size = int(observation_offsets[-1])
        v = prior.model.sampled_basis.reshape(4, n, q)
        x = input_windows(self.known, length)
        correlation = np.fft.ifft(np.fft.ifftshift(data_masks.astype(float), axes=-1), axis=-1)
        differences = (np.arange(n)[:, None]-np.arange(n)) % n
        covariance = backend.array(np.zeros((size, size), dtype=np.complex128))
        cross = np.zeros((q, length, size), dtype=np.complex128)
        interference = np.zeros((4, n, n), dtype=np.complex128)
        for select, temporal, delay in prior.separable_blocks():
            lt = linalg.cholesky(temporal, lower=True, check_finite=False)
            ld = linalg.cholesky(delay, lower=True, check_finite=False)
            width = len(delay)
            factors = []
            for symbol in range(4):
                time = ((v[symbol] @ lt)[:, :, None]*(x[symbol, :, select] @ ld)[:, None, :]).reshape(n, q*width)
                factors.append(np.fft.fftshift(np.fft.fft(time, axis=0, norm="ortho"), axes=0)[self.rows[symbol]])
            z = np.concatenate(factors)
            dz = backend.array(z)
            covariance += dz @ dz.conj().T
            cross[:, select] = np.einsum("qa,lb,mab->qlm", lt, ld, z.conj().reshape(size, q, width), optimize=True)
            rho = np.zeros(n, dtype=np.complex128)
            offsets = np.arange(width)
            np.add.at(rho, ((offsets[:, None]-offsets) % n).ravel(), delay.ravel())
            for symbol in range(4):
                stationary = np.fft.ifft(np.fft.fft(rho)*np.fft.fft(correlation[symbol]))[differences]
                interference[symbol] += (v[symbol] @ temporal @ v[symbol].conj().T)*stationary
        captured_interference = np.zeros((size, size), complex) if capture_moments else None
        for symbol in range(4):
            select = slice(observation_offsets[symbol], observation_offsets[symbol+1])
            block = hermitian(selected_time_covariance(interference[symbol], self.rows[symbol]))
            covariance[select, select] += backend.array(block)
            if capture_moments:
                captured_interference[select, select] = block
        self.selected_noise = linalg.block_diag(*[np.asarray(unit_noise)[np.ix_(rows, rows)] for rows in self.rows])
        factor = linalg.cholesky(self.selected_noise, lower=True, check_finite=False)
        whitener = linalg.solve_triangular(factor, np.eye(size), lower=True, check_finite=False)
        w = backend.array(whitener)
        whitened = w @ covariance @ w.conj().T
        whitened = (whitened+whitened.conj().T)*0.5
        # Windows' LAPACK EVD can return inaccurate complex eigenvectors
        # without raising. EVR passes the independent dense LMMSE check.
        values, vectors = (backend.eigh(whitened) if backend.torch else
                           linalg.eigh(whitened, check_finite=False, driver="evr"))
        raw = backend.numpy(values)
        if raw.min() < -1e-8*max(raw.max(), 1e-30):
            raise RuntimeError("Four-data observation covariance is not positive semidefinite")
        values = np.maximum(raw, 0)
        self.ridge = prior.regularizer*float(values.mean())
        self.values = backend.array(values)
        self.projected = w.conj().T @ vectors
        self.gain = backend.array(cross.reshape(q*length, size)) @ self.projected
        self.diagnostics = dict(observations=size, observations_per_symbol=len(self.rows[0]),
                                min_eigenvalue=float(raw.min()), max_eigenvalue=float(raw.max()),
                                whitened_ridge=self.ridge)
        if capture_moments:
            self.captured_moments = dict(observation_covariance=backend.numpy(covariance).copy(),
                data_interference=captured_interference, cross_covariance=cross.reshape(q*length, size).copy())

    def estimate_coefficients(self, observations, variances):
        y = np.asarray(observations, dtype=np.complex128)
        variance = np.asarray(variances, dtype=float)
        if y.shape != (*variance.shape, *self.known.shape):
            raise ValueError("Expected one four-symbol observation per RF variance")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(variance)) or np.any(variance < 0):
            raise ValueError("Observations and nonnegative RF variances must be finite")
        selected = np.concatenate([y[..., symbol, rows] for symbol, rows in enumerate(self.rows)], axis=-1)
        selected = selected.reshape(-1, sum(map(len, self.rows))).T
        b = self.backend
        spectral = self.projected.conj().T @ b.array(selected)
        result = self.gain @ (spectral/(self.values[:, None]+b.array(variance.ravel())[None]+self.ridge))
        model = self.prior.model
        return b.numpy(result).T.reshape(*variance.shape, model.count, model.length)
