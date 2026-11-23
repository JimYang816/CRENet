"""Checks for fresh reduced-order estimation and paired baseline validation."""
from copy import deepcopy
import json

import h5py
import numpy as np
import pytest
from scipy.signal.windows import dpss
from threadpoolctl import threadpool_limits

from data_generation.src.pdd import load_pdd_config
from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.transmitter.ofdm import make_frame
from src.common.numerics import DenseBackend
from src.evaluation.four_data_lmmse import channel_config, evaluate, model_basis
from src.evaluation.four_data_q24_compare import validate_source_config, verify_q32, verify_reference


def test_q24_uses_same_first_dpss_vectors_and_reports_tail_limit():
    c = channel_config("NOF1")
    full, concentration = model_basis(c)
    nw = len(full)*c["dpss_half_bandwidth_hz"]/c["fs_baseband"]
    reduced = dpss(len(full), nw, Kmax=24, sym=True, norm=2).T
    np.testing.assert_allclose(full[:, :24], reduced, rtol=0, atol=2e-12)
    tail = 1-np.sum(concentration[:24])/(2*nw)
    assert tail == pytest.approx(.0000559585884206681, rel=1e-8)
    assert tail > c["dpss_tail_ratio"]


def test_historical_q32_source_may_differ_only_in_order():
    c = channel_config("NOF1")
    validate_source_config("NOF1", dict(c, dpss_expected_count=32))
    with pytest.raises(ValueError, match="beyond DPSS order"):
        validate_source_config("NOF1", dict(c, dpss_expected_count=32, delay_length=175))
    with pytest.raises(ValueError, match="beyond DPSS order"):
        validate_source_config("NOF1", dict(c, dpss_expected_count=32, cp=384))
    with pytest.raises(ValueError, match="beyond DPSS order"):
        validate_source_config("NOF1", c)


def synthetic_source(source):
    rng = np.random.default_rng(729)
    c = load_pdd_config()
    c.update(channel="NOF1", frame_structure="DDDD", nfft=14, cp=6, guard_samples=4,
             fft_advance=1, null_per_edge=0, data_count=8, pilot_count=4, dpss_expected_count=3,
             delay_length=4, delay_block_size=2, snr_db=[0, 10])
    basis, _ = np.linalg.qr(rng.normal(size=(80, 3)))
    complex_random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    prior = complex_random((5, 3, 4))
    model = DPSSModel(basis, c["dpss_half_bandwidth_hz"], c["dpss_tail_ratio"], 4, c)
    frame = make_frame(c, 11, 2, 2, payload_repeat=0)
    truth = np.stack([np.pad(model.reconstruct(prior[0], d), ((0, 0), (0, 10))) for d in range(4)])
    clean = np.stack([taps_to_matrix(truth[d])@frame.x[d] for d in range(4)])
    directory = source/"nof1"
    directory.mkdir(parents=True)
    with h5py.File(directory/"physical.h5", "w") as h:
        h.attrs["complete"] = True
        h.create_dataset("basis", data=basis)
        h.create_dataset("prior", data=prior)
        h.create_dataset("known", data=frame.known)
        h.create_dataset("unit_noise", data=np.eye(14))
        g = h.create_group("test/0000")
        g.attrs.update(record=11, start_seconds=2, model_error=json.dumps([0.]*4),
                       intersymbol_leakage=json.dumps([0.]*4), support_residual=json.dumps([0.]*4))
        for name, value in dict(truth=truth, clean=clean, standard=complex_random((4, 14)),
                                variance=np.array([.5, .05]), bits=frame.bits).items():
            g.create_dataset(name, data=value)
    return dict(configs={"NOF1": c}, options=dict(prior_records=[51, 52, 53, 54, 55]))


def test_reduced_order_is_reestimated_without_using_test_reference(tmp_path):
    source = tmp_path/"source"
    metadata = synthetic_source(source)
    with threadpool_limits(2):
        q2 = evaluate("NOF1", tmp_path/"q2", metadata, DenseBackend(), physical_root=source, order=2)
        q3 = evaluate("NOF1", tmp_path/"q3", metadata, DenseBackend(), physical_root=source, order=3)
    assert (q2["q"], q3["q"], q2["bits_per_snr"]) == (2, 3, 64)
    verify_reference(q2, q3)
    with h5py.File(source/"nof1/physical.h5", "r+") as h:
        h["test/0000/truth"][:] *= -1
    with threadpool_limits(2):
        altered = evaluate("NOF1", tmp_path/"changed_truth", metadata, DenseBackend(), physical_root=source, order=2)
    for key in ("raw_single", "raw_full"):
        assert altered["curves"][key] == q2["curves"][key]
    assert altered["mean_raw_nmse_db"] != q2["mean_raw_nmse_db"]
    assert altered["curves"]["reference_full"] != q2["curves"]["reference_full"]


def test_q32_reproduction_rejects_changed_frame_errors_and_nmse():
    frame = dict(record=11, start_seconds=2., raw_single_errors=[[1, 2, 3, 4]],
                 raw_full_errors=[[4, 3, 2, 1]], reference_single_errors=[[1, 1, 1, 1]],
                 reference_full_errors=[[0, 0, 0, 0]], raw_nmse_linear=[[.1, .2, .3, .4]])
    old = dict(channel="NOF1", snr_db=[0], bits_per_snr=64, frames=[frame])
    current = dict(deepcopy(old), q=32)
    assert verify_q32(old, current)["bit_errors_identical"]
    broken = deepcopy(current)
    broken["frames"][0]["raw_full_errors"][0][0] += 1
    with pytest.raises(ValueError, match="bit errors"):
        verify_q32(old, broken)
    broken = deepcopy(current)
    broken["frames"][0]["raw_nmse_linear"][0][0] += .1
    with pytest.raises(ValueError, match="NMSE"):
        verify_q32(old, broken)
