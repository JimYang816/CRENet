"""Formal split identities and exact selected-delay/full-reference calibration."""
import copy

import numpy as np
import pytest
import h5py

from data_generation.src.dataset.four_data_math import (
    assisted_taps, coefficient_nmse, reconstruct_taps, reference_statistics, sampled_grams,
)
from data_generation.src.dataset.four_data_protocol import channel_config, counts, load_options, validate_plan
from data_generation.src.dataset.four_data_targets import channel_targets, error_scales, load_target_config, target_curve
from data_generation.src.dataset.four_data_export import calibrate
from src.common.experiment import file_sha256


@pytest.mark.parametrize("channel", ["NOF1", "NCS1"])
def test_formal_record_counts_and_four_symbol_samples(channel):
    c = channel_config(channel)
    p = counts(c)
    assert [len(p[s]["records"]) for s in ("train", "validation", "test")] == [35, 5, 10]
    assert [p[s]["physical_windows"] for s in ("train", "validation", "test")] == [525, 75, 150]
    assert [p[s]["frames_all_snr"] for s in ("train", "validation", "test")] == [3675, 525, 4200]
    assert c["delay_length"] == 192 and all(p[s]["data_symbols_per_frame"] == 4 for s in p)


@pytest.mark.parametrize("kind", ["records", "starts", "repeats"])
def test_reject_split_leakage_and_changed_formal_plan(kind):
    c = copy.deepcopy(channel_config("NOF1"))
    if kind == "records":
        c["learning_dataset"]["records"]["validation"][0] = 1
    elif kind == "starts":
        c["learning_dataset"]["starts_seconds"][1] = 2.2
    else:
        c["learning_dataset"]["repeats"]["test"] = 1
    with pytest.raises(ValueError):
        validate_plan(c)


def test_sufficient_statistics_match_actual_full_support_errors():
    rng = np.random.default_rng(504)
    sampled = rng.normal(size=(4, 12, 3))
    delays = np.array([1, 4, 9])
    truth = rng.normal(size=(4, 12, 12))+1j*rng.normal(size=(4, 12, 12))
    coeff = rng.normal(size=(7, 3, 3))+1j*rng.normal(size=(7, 3, 3))
    cross, energy = reference_statistics(sampled, truth, delays)
    measured = coefficient_nmse(coeff, sampled_grams(sampled), cross, energy)
    raw = reconstruct_taps(sampled, coeff, delays, full_length=12)
    direct = np.sum(abs(raw-truth[None])**2, axis=(2, 3))/energy[None]
    np.testing.assert_allclose(measured, direct, rtol=1e-12)
    # Omitted taps remain in the denominator and error, even with perfect retained taps.
    assert np.all(energy > np.sum(abs(truth[..., delays])**2, axis=(1, 2)))


def test_split_average_targets_do_not_force_each_frame_to_the_same_error():
    rng = np.random.default_rng(88)
    reference = rng.normal(size=(2, 4, 12, 12))+1j*rng.normal(size=(2, 4, 12, 12))
    raw = rng.normal(size=(2, 7, 4, 12, 8))+1j*rng.normal(size=(2, 7, 4, 12, 8))
    padded = np.pad(raw, ((0,0),(0,0),(0,0),(0,0),(0,4)))
    denominator = np.sum(abs(reference)**2, axis=(2,3))
    errors = np.sum(abs(padded-reference[:,None])**2, axis=(3,4))/denominator[:,None]
    targets = load_target_config()
    curve = target_curve(targets["snr_db"], **targets["targets"]["NOF1"], decay_db=targets["decay_db"])
    alpha = error_scales(errors.mean(axis=(0,2)), curve)
    corrected = np.stack([assisted_taps(raw[i],reference[i],alpha) for i in range(2)])
    actual = np.sum(abs(corrected-reference[:,None])**2,axis=(3,4))/denominator[:,None]
    np.testing.assert_allclose(10*np.log10(actual.mean(axis=(0,2))),curve,atol=1e-12)
    np.testing.assert_allclose(actual/errors,np.broadcast_to(alpha[None,:,None]**2,actual.shape),rtol=1e-12)
    assert not np.allclose(actual[0],actual[1])


def test_default_dataset_view_and_calibration_are_explicitly_reference_assisted():
    options = load_options()
    assert options["default_input_view"] == "reference_assisted"
    assert options["compensation_calibration"] == "per_split"


def test_repeated_target_calibration_keeps_completed_hdf5_files_identical(tmp_path):
    for i, split in enumerate(("train", "validation", "test")):
        with h5py.File(tmp_path/f"{split}.h5", "w") as h:
            h.attrs["raw_complete"] = True
            h["samples/raw_nmse_linear"] = np.full((7, 2+i, 4), .5+i*.1)
    c, options, targets = channel_config("NCS1"), load_options(), load_target_config()
    calibrate(tmp_path,c,options,targets)
    before = {p.name:file_sha256(p) for p in tmp_path.glob('*.h5')}
    calibrate(tmp_path,c,options,targets)
    assert before == {p.name:file_sha256(p) for p in tmp_path.glob('*.h5')}


@pytest.mark.parametrize("channel, expected", [
    ("NOF1", [-11.41,-13.22,-14.31,-14.97,-15.38,-15.62,-15.77]),
    ("NCS1", [-5.37,-6.84,-7.73,-8.27,-8.60,-8.80,-8.92]),
])
def test_requested_decimal_targets_decrease_steeply_then_gently(channel, expected):
    actual = channel_targets(load_target_config(), channel)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual != np.round(actual))
    improvement = -np.diff(actual)
    assert np.all(improvement > 0) and np.all(np.diff(improvement) < 0)


@pytest.mark.parametrize("channel", ["NOF1", "NCS1"])
def test_recalibration_changes_only_csi_metadata_and_is_idempotent(tmp_path, channel):
    from data_generation.src.dataset.four_data_recalibration import recalibrate_metadata
    sources, payloads = {}, {}
    rng = np.random.default_rng(271)
    for i, split in enumerate(("train","validation","test")):
        path = tmp_path/f"{split}.h5"
        payloads[split] = rng.normal(size=(7,2+i,3))+1j*rng.normal(size=(7,2+i,3))
        with h5py.File(path,"w") as h:
            h.attrs.update(raw_complete=True,model_identity="original-generation")
            h["samples/raw_nmse_linear"] = np.full((7,2+i,4),.4+i*.1)
            h["samples/raw_coefficients"] = payloads[split]
        sources[path.name] = dict(sha256=file_sha256(path))
    c, options, targets = channel_config(channel), load_options(), load_target_config()
    identities = recalibrate_metadata(tmp_path,c,options,targets,"new-calibration",sources)
    after = {p.name:file_sha256(p) for p in tmp_path.glob("*.h5")}
    assert recalibrate_metadata(tmp_path,c,options,targets,"new-calibration",sources) == identities
    assert after == {p.name:file_sha256(p) for p in tmp_path.glob("*.h5")}
    for split in payloads:
        with h5py.File(tmp_path/f"{split}.h5","r") as h:
            np.testing.assert_array_equal(h["samples/raw_coefficients"][:],payloads[split])
            assert h.attrs["model_identity"] == "original-generation"
            assert h.attrs["calibration_identity"] == identities[split]
            assert h["compensation"].attrs["source_file_sha256"] == sources[f"{split}.h5"]["sha256"]
            np.testing.assert_allclose(10*np.log10(h["compensation/assisted_mean_nmse_linear"][:]),
                                       channel_targets(targets,channel),atol=1e-12)
