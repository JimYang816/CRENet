"""Check CP80 source isolation and full-matrix mean-NMSE compensation on CPU."""
import copy

import numpy as np
import pytest

from data_generation.src.receiver.chain import taps_to_matrix
from src.evaluation.four_data_cp80 import channel_settings, load_options
from src.evaluation.four_data_cp80_compensation import validate_protocol
from src.evaluation.four_data_csi_compensation import tap_nmse
from src.evaluation.four_data_delay192_compensation import compensate, error_scales, load_target_config, target_curve


def source_protocol():
    archive = "3cf9e3a9b116b525f9b52c098018383665ca24f3a9ab05bc53b7716b1b3c5cc1"
    options, metas, configs, reports = load_options(), {}, {}, {}
    for channel in options["channels"]:
        c, study, priors, tests = channel_settings(channel, options, archive)
        configs[channel] = c
        metas[channel] = dict(config=c, study=study, archive_sha256=archive, smoke=False,
            raw_csi=True, csi_compensation=False, prior_windows=priors, test_windows=tests)
        reports[channel] = dict(complete=True, bits_per_snr=20480, delay_count=192,
            coefficient_count=4608, snr_db=c["snr_db"],
            frames=[dict(record=r, start_seconds=s) for r, s in tests])
    return dict(schema="four-data-cp80-lmmse-v1", smoke=False, cp_ms=80, input_view="raw",
        csi_compensation=False, options=options, configs=configs, channel_manifests=metas), reports


def test_accepts_current_cp80_frozen_support_and_split():
    validate_protocol(*source_protocol())


@pytest.mark.parametrize("change", ["cp", "frame", "view"])
def test_rejects_cp256_other_frames_or_already_compensated_source(change):
    original, reports = source_protocol()
    if change == "cp":
        original = copy.deepcopy(original)
        original["configs"]["NOF1"]["cp"] = 256
    elif change == "frame":
        reports["NCS1"]["frames"][0]["record"] = 51
    else:
        original["input_view"] = "reference_assisted"
    with pytest.raises(ValueError):
        validate_protocol(original, reports)


@pytest.mark.parametrize("channel", ["NOF1", "NCS1"])
def test_full_support_compensation_hits_global_mean_target_without_per_symbol_scaling(channel):
    rng = np.random.default_rng(80)
    options = load_target_config()
    targets = target_curve(options["snr_db"], **options["targets"][channel], decay_db=options["decay_db"])
    truth = rng.standard_normal((3, 4, 8, 8))+1j*rng.standard_normal((3, 4, 8, 8))
    # Mimic expansion from nonconsecutive physical delays; preserve omitted errors.
    raw = np.zeros((3, 7, 4, 8, 8), dtype=complex)
    for index in (0, 3, 6):
        raw[..., index] = truth[:, None, ..., index]+.1*(rng.standard_normal((3, 7, 4, 8))+1j*rng.standard_normal((3, 7, 4, 8)))
    original_nmse = np.stack([tap_nmse(x, ref) for x, ref in zip(raw, truth, strict=True)])
    scales = error_scales(original_nmse.mean(axis=(0, 2)), targets)
    corrected = np.stack([compensate(x, ref, scales) for x, ref in zip(raw, truth, strict=True)])
    achieved = np.stack([tap_nmse(x, ref) for x, ref in zip(corrected, truth, strict=True)])
    np.testing.assert_allclose(10*np.log10(achieved.mean(axis=(0, 2))), targets, atol=1e-12)
    np.testing.assert_allclose(achieved, scales[None, :, None]**2*original_nmse, rtol=1e-12)
    assert np.ptp(10*np.log10(achieved[:, -1])) > .01
    np.testing.assert_allclose(corrected[:, :, :, :, 7], (1-scales)[None, :, None, None]*truth[:, None, :, :, 7])
    # Independently check the complete frequency matrix, including omitted delays.
    for s in (0, 6):
        reference = taps_to_matrix(truth[0, 0])
        estimate = taps_to_matrix(corrected[0, s, 0])
        actual = np.linalg.norm(estimate-reference)**2/np.linalg.norm(reference)**2
        np.testing.assert_allclose(actual, achieved[0, s, 0], rtol=1e-12)
