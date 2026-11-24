"""Check physical delay positions against literal dense receiver calculations."""
from __future__ import annotations

import numpy as np
import pytest
import h5py
import json
from scipy import linalg
from scipy import sparse
from types import SimpleNamespace

from data_generation.src.receiver.chain import full_frame_kernels
from data_generation.src.receiver.delay_support import SelectedDelayDPSSModel, profile_full_frame
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.receiver.pdd import JointEstimator, selected_components
from data_generation.src.config import ROOT, digest, load_config
from src.common.numerics import DenseBackend
from src.evaluation.ncs1_delay_support import freeze_supports, full_nmse, load_study_config


def fixture():
    rng = np.random.default_rng(311)
    c = dict(nfft=16, cp=6, fft_advance=1, frame_structure="DDDD", test_records=[1])
    basis, _ = np.linalg.qr(rng.normal(size=(88, 3)))
    model = SelectedDelayDPSSModel(basis, 12, 1e-5, 4, c, np.array([1, 3, 9, 10]))
    coeff = rng.normal(size=(6, 3, 4))+1j*rng.normal(size=(6, 3, 4))
    return rng, model, coeff


def test_selected_model_keeps_physical_positions_and_both_projection_domains():
    _, model, coeff = fixture()
    selected = np.einsum("nq,mql->mnl", model.basis, coeff)
    full = model.expand_taps(selected)
    np.testing.assert_allclose(model.coefficients_full_frame(full), coeff, atol=4e-15)
    sampled = model.expand_taps(np.einsum("nq,mql->mnl", model.sampled_basis, coeff)).reshape(6, 4, 16, -1)
    np.testing.assert_allclose(model.coefficients(sampled), coeff, atol=6e-15)
    np.testing.assert_allclose(model.reconstruct(coeff, 2), sampled[:, 2], atol=4e-15)
    np.testing.assert_array_equal(full[..., [0, 2, 4, 5, 6, 7, 8]], 0)
    for delays in ([1, 1, 9, 10], [3, 1, 9, 10], [-1, 3, 9, 10], [1, 3, 9, 16], [1., 3., 9., 10.]):
        with pytest.raises(ValueError, match="Selected delays"):
            SelectedDelayDPSSModel(model.basis, 12, 1e-5, 4, model.c, np.asarray(delays))


def test_selected_covariance_matches_literal_four_data_design_and_unknown_data_ici():
    rng, model, coeff = fixture()
    prior = FrozenPrior.fit_coefficients(model, coeff, 0, 1, [51], delay_block_size=4)
    blocks = list(prior.separable_blocks())
    assert [(sl.start, sl.stop) for sl, _, _ in blocks] == [(0, 2), (2, 4)]
    n, q, length = model.c["nfft"], model.count, model.length
    data = np.isin(np.arange(n), [2, 4, 5, 8])
    tracking = np.isin(np.arange(n), [0, 1, 3, 7, 10, 12])
    known = rng.normal(size=(4, n))+1j*rng.normal(size=(4, n))
    known[:, data] = 0
    masks = dict(data=data, tracking=tracking)
    rc = np.zeros((q, length, q, length), complex)
    for sl, temporal, delay in blocks:
        rc[:, sl, :, sl] = np.einsum("qr,lk->qlrk", temporal, delay)
    rc = rc.reshape(q*length, q*length)

    def design(symbols):
        time = np.fft.ifft(np.fft.ifftshift(symbols, axes=-1), axis=-1, norm="ortho")
        rows = []
        for s in range(4):
            columns = []
            for qi in range(q):
                for delay in model.delays:
                    response = model.sampled_basis[s*n:(s+1)*n, qi]*time[s, (np.arange(n)-delay) % n]
                    columns.append(np.fft.fftshift(np.fft.fft(response, norm="ortho"))[tracking])
            rows.append(np.stack(columns, axis=-1))
        return np.concatenate(rows)

    a = design(known)
    expected = a@rc@a.conj().T
    for s in range(4):
        for k in np.flatnonzero(data):
            unit = np.zeros((4, n), complex)
            unit[s, k] = 1
            interference = design(unit)
            expected += interference@rc@interference.conj().T
    actual, cross, _, _ = selected_components(prior, known, masks, DenseBackend())
    np.testing.assert_allclose(actual, expected, atol=8e-14)
    np.testing.assert_allclose(cross, rc@a.conj().T, atol=5e-14)
    estimator = JointEstimator(prior, known, masks, np.eye(n), DenseBackend())
    y = rng.normal(size=(4, n))+1j*rng.normal(size=(4, n))
    variance = .2
    expected_coeff = cross@np.linalg.solve(expected+(variance+estimator.ridge)*np.eye(len(expected)), y[:, tracking].ravel())
    estimated = estimator.estimate_coefficients(y, known, masks, [variance])
    np.testing.assert_allclose(estimated.ravel(), expected_coeff, atol=4e-13)


def test_omitted_taps_are_included_in_full_matrix_nmse():
    rng, model, _ = fixture()
    truth = rng.normal(size=(4, 16, 16))+1j*rng.normal(size=(4, 16, 16))
    selected = rng.normal(size=(2, 4, 16, 4))+1j*rng.normal(size=(2, 4, 16, 4))
    dense = np.zeros((2, 4, 16, 16), complex)
    dense[..., model.delays] = selected
    expected = np.sum(abs(dense-truth[None])**2, axis=(2, 3))/np.sum(abs(truth)**2, axis=(1, 2))[None]
    np.testing.assert_allclose(full_nmse(selected, truth, model.delays), expected, atol=4e-16)


def test_full_frame_profile_matches_explicit_physical_kernels_including_boundaries():
    rng = np.random.default_rng(165)
    c = dict(nfft=8, cp=4, guard_samples=2, fft_advance=1, fs_baseband=6000)
    operator = sparse.csr_matrix(rng.normal(size=(60, 60))+1j*rng.normal(size=(60, 60)))
    basis, _ = np.linalg.qr(rng.normal(size=(48, 3)))
    receiver = SimpleNamespace(chain=SimpleNamespace(c=c), block=7,
                               reference_rows=lambda sounding, origin, indices: operator[indices])
    coefficients, power = profile_full_frame(receiver, None, 2, basis)
    taps = full_frame_kernels(operator, c, c["nfft"])
    np.testing.assert_allclose(coefficients, linalg.pinv(basis)@taps, atol=2e-15)
    np.testing.assert_allclose(power, np.mean(abs(taps)**2, axis=0), atol=9e-16)


def test_default_192_support_is_frozen_independent_and_keeps_physical_positions(tmp_path):
    study = load_study_config()
    support = study["frozen_support"]
    c = load_config(ROOT/"configs/data/ncs1/dddd.json")
    assert study["tap_count"] == support["tap_count"] == len(support["indices"]) == 192
    assert support["frame_config_sha256"] == digest(c)
    np.testing.assert_array_equal(support["indices"], np.arange(32, 224))
    assert {record for record, _ in support["prior_windows"]} == set(c["prior_records"])
    assert not ({record for record, _ in support["prior_windows"]} & set(c["learning_dataset"]["records"]["test"]))
    path = tmp_path/"physical.h5"
    with h5py.File(path, "w") as h:
        # A smaller smoke prior cannot silently change the frozen receiver support.
        h.create_dataset("prior_power", data=np.ones((2, 1024)))
        h.create_group("test")
    supports, report = freeze_supports(path, tmp_path, study)
    assert set(supports) == {"selected192"}
    np.testing.assert_array_equal(supports["selected192"], support["indices"])
    assert report["candidates"]["selected192"]["span"] == 224


def test_study_rejects_duplicate_frozen_delays(tmp_path):
    study = load_study_config()
    support = study.pop("frozen_support")
    support["indices"][1] = support["indices"][0]
    support_path = tmp_path/"bad_support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    study["support_file"] = str(support_path)
    config_path = tmp_path/"study.json"
    config_path.write_text(json.dumps(study), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted unique"):
        load_study_config(config_path)


@pytest.mark.parametrize("count", [64, 176, 184, 208, 192.0, True])
def test_study_rejects_other_tap_configurations(tmp_path, count):
    study = load_study_config()
    study.pop("frozen_support")
    study["tap_count"] = count
    path = tmp_path/"obsolete.json"
    path.write_text(json.dumps(study), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed 192-tap"):
        load_study_config(path)
