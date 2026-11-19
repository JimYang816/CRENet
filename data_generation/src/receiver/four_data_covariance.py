"""Independent-prior joint moments of BEM coefficients and physical 4D observations.

The physical data covariance integrates proper unit-QPSK symbols analytically.
Both A x and B x* receive paths and all four source/output blocks are retained.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.estimation import input_windows
from data_generation.src.receiver.four_data import FourDataEstimator
from data_generation.src.receiver.four_data_rotated import waveform
from data_generation.src.transmitter.ofdm import Frame, fft, receive_symbols, window_indices


def reference_taps_from_rows(operator, config, symbol):
    """Same self-symbol circular reference as the original NMSE definition."""
    n, cp = config["nfft"], config["cp"]
    rows = operator[symbol*n:(symbol+1)*n].tocoo()
    offset = rows.col-(config["guard_samples"]+symbol*(n+cp))
    inside = (offset >= 0) & (offset < n+cp)
    result = np.zeros((n, n), complex)
    np.add.at(result, (rows.row[inside], (rows.row[inside]-offset[inside]+cp) % n), rows.data[inside])
    return result


def selected_fft(values, config, rows):
    n = config["nfft"]
    transformed = values.reshape(4, n, *values.shape[1:])
    transformed = np.fft.fftshift(np.fft.fft(transformed, axis=1, norm="ortho"), axes=1)
    return np.concatenate([transformed[s, r] for s, r in enumerate(rows)], axis=0)


def physical_data_columns(direct, image, config, masks, rows, symbol):
    """Map one D's data bins to every receive D, including CP/ISI and images."""
    n, cp, guard = config["nfft"], config["cp"], config["guard_samples"]
    carriers = np.flatnonzero(masks["data"][symbol])
    frequency = np.eye(n, dtype=complex)[:, carriers]
    time = np.fft.ifft(np.fft.ifftshift(frequency, axes=0), axis=0, norm="ortho")
    tones = np.concatenate((time[-cp:], time), axis=0)
    start = guard+symbol*(n+cp)
    block = slice(start, start+n+cp)
    return (selected_fft(direct[:, block] @ tones, config, rows),
            selected_fft(image[:, block] @ tones.conj(), config, rows))


def collect_joint_moments(prior, known, masks, noise, receiver, dataset, identities, backend):
    """Calibration is restricted to the declared independent prior records."""
    if not identities or not {int(r) for r, _ in identities} <= set(prior.source_records):
        raise ValueError("Joint covariance calibration must use independent prior records")
    config, b = prior.model.c, backend
    n, q, length = config["nfft"], prior.model.count, prior.model.length
    basis = prior.model.sampled_basis.reshape(4, n, q)
    np.testing.assert_allclose(prior.model.sampled_basis.conj().T @ prior.model.sampled_basis,
                                np.eye(q), atol=1e-10, rtol=0)
    base = FourDataEstimator(prior, known, masks, noise, b, capture_moments=True)
    rows = base.rows
    offsets = np.cumsum([0]+list(map(len, rows)))
    size, count = offsets[-1], len(identities)
    data_covariance = b.array(np.zeros((size, size), complex))
    predictions, residuals, targets, diagnostics = [], [], [], []
    public_waveform = waveform(known, config)
    sample_indices = np.concatenate([window_indices(config, s) for s in range(4)])
    for index, (record, start) in enumerate(identities):
        sounding = dataset.load(int(record))
        origin = float(start)-config["guard_samples"]/config["fs_baseband"]
        direct, image = receiver.reference_parts(sounding, origin, sample_indices)
        truth = np.stack([reference_taps_from_rows(direct, config, s) for s in range(4)])
        coefficients = np.einsum("dnq,dnl->ql", basis.conj(), truth[..., :length], optimize=True)
        expected = prior.coefficients[index]
        relative = float(np.linalg.norm(coefficients-expected)/np.linalg.norm(expected))
        if relative > 2e-8:
            raise AssertionError(f"Physical target projection differs from frozen prior: {relative}")
        target = coefficients.ravel()
        projected = basis @ coefficients
        predicted = fft(np.sum(projected*input_windows(known, length), axis=-1))
        predicted = np.concatenate([predicted[s, r] for s, r in enumerate(rows)])
        physical = selected_fft(direct @ public_waveform+image @ public_waveform.conj(), config, rows)
        operator_error = None
        if index == 0:
            frame = Frame(known, known, np.zeros((4, n//4, 2), np.uint8), public_waveform, ())
            replay = receiver.replay_frame(sounding, origin, frame)
            replay = receive_symbols(receiver.chain.receive(replay), config)
            replay = np.concatenate([replay[s, r] for s, r in enumerate(rows)])
            operator_error = float(np.linalg.norm(replay-physical)/np.linalg.norm(replay))
            if operator_error > 2e-10:
                raise AssertionError(f"Joint physical operator disagrees with RF replay: {operator_error}")
        actual_power, inside_power, outside_power, cross_power, cross_block_power = 0., 0., 0., 0., 0.
        for s in range(4):
            physical_d, conjugate_d = physical_data_columns(direct, image, config, masks, rows, s)
            in_space = np.zeros_like(physical_d)
            carriers = np.flatnonzero(masks["data"][s])
            target_rows = slice(offsets[s], offsets[s+1])
            in_space[target_rows] = taps_to_matrix(projected[s])[np.ix_(rows[s], carriers)]
            outside = physical_d-in_space
            d, dc = b.array(physical_d), b.array(conjugate_d)
            data_covariance += (d @ d.conj().T+dc @ dc.conj().T)/count
            actual_power += float(np.sum(abs(physical_d)**2)+np.sum(abs(conjugate_d)**2))
            inside_power += float(np.sum(abs(in_space)**2))
            outside_power += float(np.sum(abs(outside)**2)+np.sum(abs(conjugate_d)**2))
            cross_power += float(2*np.vdot(in_space, outside).real)
            cross_block_power += float(np.sum(abs(physical_d[:offsets[s]])**2)
                +np.sum(abs(physical_d[offsets[s+1]:])**2)
                +np.sum(abs(conjugate_d[:offsets[s]])**2)+np.sum(abs(conjugate_d[offsets[s+1]:])**2))
        np.testing.assert_allclose(actual_power, inside_power+outside_power+cross_power, rtol=2e-12, atol=1e-20)
        predictions.append(predicted)
        residuals.append(physical-predicted)
        targets.append(target)
        diagnostics.append(dict(record=int(record), start_seconds=float(start), projection_relative_error=relative,
            operator_replay_relative_error=operator_error, data_physical_trace=actual_power,
            data_in_space_trace=inside_power, data_outside_trace=outside_power,
            data_in_out_cross_trace=cross_power, data_cross_symbol_trace=cross_block_power,
            pilot_model_residual_power=float(np.sum(abs(physical-predicted)**2))))
        print(f"{dataset.channel} joint prior {index+1}/{count} {record:03d}@{start}s; "
              f"cross-D data fraction={cross_block_power/max(actual_power,1e-300):.5g}", flush=True)
    moments = dict(base_pilot_covariance=base.captured_moments["observation_covariance"]-base.captured_moments["data_interference"],
        base_cross_covariance=base.captured_moments["cross_covariance"],
        physical_data_covariance=b.numpy(data_covariance), coefficients=np.stack(targets, axis=-1),
        model_pilot=np.stack(predictions, axis=-1), pilot_residual=np.stack(residuals, axis=-1),
        selected_noise=base.selected_noise)
    data = moments["physical_data_covariance"]
    off = data.copy()
    for s in range(4):
        off[offsets[s]:offsets[s+1], offsets[s]:offsets[s+1]] = 0
    metadata = dict(prior_windows=identities, frames=diagnostics, ridge=base.ridge,
        physical_data_off_diagonal_D_fraction=float(np.linalg.norm(off)/np.linalg.norm(data)),
        target="unchanged self-symbol full-reference projection onto frozen Qeff/L",
        data_integration="exact proper unit-QPSK covariance: all source/output D blocks, direct and conjugate paths",
        statistics="uncentered second moments for zero-intercept complex-linear LMMSE")
    return moments, metadata


class JointMoments:
    """Noise-whitened frozen moments, reused across validation shrinkage choices."""

    def __init__(self, prior, known, masks, moments, metadata, backend):
        self.prior, self.known, self.backend = prior, np.asarray(known), backend
        self.rows = [np.flatnonzero(m) for m in masks["tracking"]]
        self.ridge = metadata["ridge"]
        b = backend
        offsets = np.cumsum([0]+list(map(len, self.rows)))
        whiteners = []
        for i in range(4):
            sl = slice(offsets[i], offsets[i+1])
            factor = linalg.cholesky(moments["selected_noise"][sl, sl], lower=True, check_finite=False)
            whiteners.append(linalg.solve_triangular(factor, np.eye(len(factor)), lower=True, check_finite=False))
        self.whitener = b.array(linalg.block_diag(*whiteners))
        w = self.whitener
        self.base_pilot = w @ b.array(moments["base_pilot_covariance"]) @ w.conj().T
        self.data = w @ b.array(moments["physical_data_covariance"]) @ w.conj().T
        self.base_cross = b.array(moments["base_cross_covariance"]) @ w.conj().T
        self.pilot = w @ b.array(moments["model_pilot"])
        self.residual = w @ b.array(moments["pilot_residual"])
        self.coefficients = b.array(moments["coefficients"])
        self.count, self.size = self.pilot.shape[1], self.pilot.shape[0]
        self.residual_variance = float(b.numpy((self.residual.conj()*self.residual).real.sum())/(self.count*self.size))


class JointFourDataEstimator(FourDataEstimator):
    """PSD joint-moment shrinkage; residual energy stays fixed across choices.

    c=sqrt(alpha)*c_emp+sqrt(1-alpha)*c_struct;
    r=sqrt(1-rho)*r_emp+sqrt(rho)*r_isotropic.
    The empirical c and r share a latent sample index, retaining C_cr.
    """

    def __init__(self, moments, alpha, rho):
        if not 0 <= alpha <= 1 or not 0 <= rho <= 1:
            raise ValueError("Joint shrinkage must be in [0,1]")
        self.prior, self.backend = moments.prior, moments.backend
        self.known, self.rows, self.ridge = moments.known, moments.rows, moments.ridge
        b = self.backend
        sample_y = np.sqrt(alpha)*moments.pilot+np.sqrt(1-rho)*moments.residual
        covariance = (1-alpha)*moments.base_pilot+sample_y @ sample_y.conj().T/moments.count+moments.data
        covariance += b.array(np.eye(moments.size, dtype=complex))*(rho*moments.residual_variance)
        covariance = .5*(covariance+covariance.conj().T)
        cross = (1-alpha)*moments.base_cross+np.sqrt(alpha)*moments.coefficients @ sample_y.conj().T/moments.count
        values, vectors = (b.eigh(covariance) if b.torch else linalg.eigh(covariance, check_finite=False, driver="evr"))
        raw = b.numpy(values)
        if raw.min() < -1e-8*max(raw.max(), 1e-30):
            raise RuntimeError("Calibrated joint observation covariance is not PSD")
        self.values = b.array(np.maximum(raw, 0))
        self.projected = moments.whitener.conj().T @ vectors
        self.gain = cross @ vectors
        self.diagnostics = dict(alpha=float(alpha), residual_shape_shrinkage=float(rho),
            residual_whitened_variance=moments.residual_variance,
            observations=moments.size, min_eigenvalue=float(raw.min()), max_eigenvalue=float(raw.max()),
            whitened_ridge=self.ridge, coefficient_residual_cross_weight=float(np.sqrt(alpha*(1-rho))))
