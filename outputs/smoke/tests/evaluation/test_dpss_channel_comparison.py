import os
import numpy as np
import pytest

from data_generation.src.dataset.compact_export import generate
from data_generation.src.dataset.compact_math import coefficient_nmse
from data_generation.src.receiver.chain import taps_to_matrix
from outputs.smoke.data_generation_tests.test_compact import fixture
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.dpss_channel_comparison import ReferenceScore, restore_receiver_lmmse, run


def test_reference_scoring_matches_full_matrix_nmse():
    c, prior, chain, archive = fixture()
    model = prior.model
    rng = np.random.default_rng(16)
    truth = rng.normal(size=(2, c["nfft"], c["nfft"]))+1j*rng.normal(size=(2, c["nfft"], c["nfft"]))
    reference = np.stack([taps_to_matrix(t) for t in truth])
    coefficients = prior.coefficients[:3]
    np.testing.assert_allclose(ReferenceScore(model, reference).nmse(coefficients),
                               coefficient_nmse(model, coefficients, reference), rtol=1e-12, atol=1e-12)


def test_full_diagnostic_uses_every_validation_and_test_sample(tmp_path):
    folders = {}
    for channel in ("NOF1", "NCS1"):
        config, prior, chain, archive = fixture()
        config["channel"] = channel
        folder, logs = tmp_path/channel, tmp_path/(channel+"_log")
        logs.mkdir()
        generate(folder, logs, config, archive, chain, prior, DenseBackend(), {"commit": "test"})
        (folder/"manifest.json").write_text("{}", encoding="utf-8")
        folders[channel] = folder
    controls = dict(group_counts=[1, 2], ridge_multipliers=[.1, 1.0],
                    source_files=["src/algorithms/dpss_bomp/estimator.py", "src/evaluation/dpss_channel_comparison.py"])
    report = run(folders, tmp_path/"report", controls, DenseBackend())
    assert report["complete"]
    for channel, folder in folders.items():
        result = report["channels"][channel]
        assert result["frame_count_per_snr"] == 2
        assert result["bit_count_per_snr"] == 2*2*8*2
        for method in ("joint_lmmse", "group_omp"):
            assert len(result["curves"][method]["nmse_db"]) == 7
            assert len(result["curves"][method]["ber"]) == 7
        with CompactDataset(folder/"test.h5") as ds:
            receiver = restore_receiver_lmmse(ds, DenseBackend())
            y = ds._h["samples/clean_observations"][0][None]
            var = np.array([ds._h["samples/rf_noise_variance"][0, 0]])
            coeff = receiver.estimate_coefficients(y, ds._known, ds._masks, var)
            assert coeff.shape == (1, ds._model.count, ds._model.length)
    assert (tmp_path/"report/ber_snr.png").exists()
    assert (tmp_path/"report/nmse_snr.png").exists()
    # Resume only this diagnostic's own identity-bound state.
    again = run(folders, tmp_path/"report", controls, DenseBackend())
    assert again["channels"]["NOF1"]["curves"]["group_omp"]["ber"] == report["channels"]["NOF1"]["curves"]["group_omp"]["ber"]


@pytest.mark.skipif(os.environ.get("CRENET_TEST_CUDA") != "1", reason="explicit server GPU verification")
def test_cuda_detector_matches_full_frequency_domain_cpu(tmp_path):
    from src.evaluation.dpss_channel_comparison import FullMatrixDetector
    config, prior, chain, archive = fixture()
    folder, logs = tmp_path/"data", tmp_path/"log"
    logs.mkdir()
    generate(folder, logs, config, archive, chain, prior, DenseBackend(), {"commit": "test"})
    with CompactDataset(folder/"test.h5") as ds:
        coefficients = prior.coefficients[:2]
        observations = ds._h["samples/clean_observations"][:2]
        variances = ds._h["samples/rf_noise_variance"][2, :2]
        expected = FullMatrixDetector(ds, DenseBackend()).detect(coefficients, observations, variances)
        got = FullMatrixDetector(ds, DenseBackend("cuda")).detect(coefficients, observations, variances)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)
