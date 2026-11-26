"""Physical-delay, colored-noise, full-support NMSE and split-boundary checks."""
from types import SimpleNamespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from data_generation.src.receiver.chain import taps_to_matrix
from src.data.four_data_training import FourDataTrainingSet
from src.evaluation.four_data_crenet import banded_lmmse, evaluate
from src.training.four_data_crenet import train


def complex_random(rng, shape):
    return rng.normal(size=shape)+1j*rng.normal(size=shape)


def fake_reader(monkeypatch):
    opened = []
    rng = np.random.default_rng(24)
    n, q, length, frames = 12, 3, 3, 2
    basis = complex_random(rng,(4,n,q))*.1
    delays = np.array([1,5,10])
    reference = complex_random(rng,(frames,4,n,n))*.02
    coefficients = complex_random(rng,(7,frames,q,length))*.1
    mask = np.arange(n)%3 == 0
    pilot = np.arange(n)%3 == 1
    known = np.tile(pilot.astype(complex),(4,1))
    bits = rng.integers(0,2,(frames,4,int(mask.sum()),2))
    clean = complex_random(rng,(frames,4,n))*.1
    noise = complex_random(rng,(frames,4,n))*.1
    variance = np.ones((7,frames))*.01
    alpha = np.linspace(.1,.2,7)
    storage = {"samples/window_index":np.arange(frames), "windows/record":np.array([1,2]),
        "samples/clean_observations":clean, "samples/standard_noise":noise,
        "samples/rf_noise_variance":variance, "samples/raw_coefficients":coefficients,
        "samples/bits":bits, "compensation/error_scales":alpha,
        "compensation/assisted_mean_nmse_linear":np.ones(7), "windows/reference_taps":reference}

    class Storage(dict):
        attrs = {"model_identity":"synthetic", "schema":"synthetic-DDDD"}

    class Reader:
        def __init__(self,path,**kwargs):
            self.split = path.stem
            opened.append(self.split)
            self.channel, self.input_view = "NOF1",kwargs["input_view"]
            self.config, self.snr_db = dict(nfft=n,data_count=int(mask.sum()),
                                          cp=480,fs_baseband=6000),np.arange(0,35,5)
            self.frame_count, self.sampled_basis, self.delays = frames,basis,delays
            self.known, self.masks = known,dict(data=mask,pilot=pilot)
            self._model = {"unit_rf_noise_covariance":np.eye(n,dtype=complex)}
            self._h = Storage(storage)
            self._h["windows/record"] = np.array([1,2])+10*("train","validation","test").index(self.split)
        def identity(self,i):
            return dict(record=int(self._h["windows/record"][i]),start_seconds=2.,repeat=0)
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
    monkeypatch.setattr("src.data.four_data_training.FourDataDataset",Reader)
    return SimpleNamespace(opened=opened,n=n,basis=basis,delays=delays,reference=reference,
        coefficients=coefficients,alpha=alpha,bits=bits,clean=clean,noise=noise,variance=variance)


def test_four_blocks_noncontiguous_delays_and_supervision_are_aligned(monkeypatch,tmp_path):
    source = fake_reader(monkeypatch)
    dataset = FourDataTrainingSet(tmp_path/"train.h5")
    seen = set()
    for inputs, targets, scoring in dataset.batches(5,seed=6):
        assert set(inputs) == {"observations","initial_channel","known","data_mask","pilot_mask",
                               "null_mask","carrier_positions","unit_covariance","rf_variance"}
        for j,(si,fi,di) in enumerate(scoring["keys"]):
            key = (si,fi,di)
            assert key not in seen
            seen.add(key)
            taps = np.zeros((source.n,source.n),complex)
            taps[:,source.delays] = source.basis[di]@source.coefficients[si,fi]
            expected = taps_to_matrix(source.reference[fi,di])
            expected += source.alpha[si]*(taps_to_matrix(taps)-expected)
            np.testing.assert_allclose(inputs["initial_channel"][j],expected,rtol=1e-5,atol=3e-8)
            np.testing.assert_array_equal(targets["bits"][j],source.bits[fi,di])
            np.testing.assert_allclose(inputs["observations"][j],source.clean[fi,di]
                + np.sqrt(source.variance[si,fi])*source.noise[fi,di],rtol=1e-6,atol=2e-8)
    assert seen == {(s,f,d) for s in range(7) for f in range(2) for d in range(4)}


def test_three_stage_training_never_opens_test_and_final_test_counts_all_bits(monkeypatch,tmp_path):
    torch.set_num_threads(1)
    source = fake_reader(monkeypatch)
    config = dict(input_view="reference_assisted", model=dict(layers=2,bandwidth=2,
        window_size=4,center_size=2,hidden_size=4,detector_mode="budgeted",noise_mode="diagonal",
        initial_soft_damping=.8), training=dict(seed=2,stage_epochs=[1,1,1],patience=4,batch_size=16,
        learning_rate=.001,stage_b_learning_rate=.001,stage_c_learning_rate=.0001,
        stage_c_correction_learning_rate=.00001,lambda_h=.1),evaluation=dict(batch_size=5))
    run = tmp_path/"run"
    report = train(config,tmp_path,run,"test-identity",frame_limit=2)
    assert report["complete"] and set(source.opened) == {"train","validation"}
    before = (run/"model.pt").read_bytes()
    repeated = train(config,tmp_path,run,"test-identity",frame_limit=2)
    assert repeated == report and (run/"model.pt").read_bytes() == before
    scored = evaluate(config,tmp_path,run,"test-identity",frame_limit=2)
    assert scored["cp"] == 480 and scored["fs_baseband"] == 6000
    assert scored["bits_per_snr"] == 2*4*4*2
    assert len(scored["curves"]) == 3 and scored["frames_per_snr"] == 2
    metrics = np.load(run/"frame_metrics.npz")
    assert metrics["complete"].all() and metrics["nmse_linear"].shape == (7,2,4,2)
    for name,curve in scored["curves"].items():
        np.testing.assert_array_equal(curve["errors"],metrics[name].sum(axis=(1,2)))
    cross = evaluate(config,tmp_path,tmp_path/"ncs1","cross-test-identity",frame_limit=2,
                     checkpoint_path=run/"model.pt",training_identity="test-identity")
    assert cross["training_identity"] == "test-identity"
    assert cross["checkpoint_sha256"] == scored["checkpoint_sha256"]
    assert (run/"model.pt").read_bytes() == before
    with pytest.raises(ValueError,match="identity mismatch"):
        evaluate(config,tmp_path,tmp_path/"wrong","different-test",frame_limit=2,
                 checkpoint_path=run/"model.pt",training_identity="wrong-training")


def test_banded_baseline_matches_production_with_gaps_and_colored_noise():
    from src.algorithms.banded_lmmse.equalizer import equalize_batch
    from src.algorithms.interface import EqualizerBatch
    from src.common.numerics import DenseBackend
    rng = np.random.default_rng(7)
    n, count = 48, 3
    h = complex_random(rng,(count,n,n))*.1
    y = complex_random(rng,(count,n))
    known = np.zeros((count,n),complex)
    known[:,::5] = (1+1j)/np.sqrt(2)
    data = (np.arange(n)%5 != 0) & (np.arange(n)%7 != 0)
    root = complex_random(rng,(n,n))
    covariance = root@root.conj().T+np.eye(n)
    variance = np.array([.01,.1,1.])
    cancelled = y-(h@known[...,None])[...,0]
    batch = EqualizerBatch(h,y,cancelled,variance,np.flatnonzero(data),covariance)
    expected = equalize_batch(batch,DenseBackend()).symbols
    actual = banded_lmmse(*(torch.as_tensor(v) for v in (h,y,known,variance,data)),
                          torch.linalg.cholesky(torch.as_tensor(covariance)))
    np.testing.assert_allclose(actual.numpy(),expected,rtol=2e-11,atol=2e-12)


def test_cli_writes_portable_manifest_for_relative_output(monkeypatch,tmp_path):
    import scripts.run.four_data_experiment as cli
    base = json.loads((cli.ROOT/"configs/training/four_data_crenet.json").read_text())
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(base))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,"argv",["experiment","--config",str(config_path),"--device","cpu",
        "--output","outputs/smoke/relative-run","--frame-limit","2"])
    trained = []
    tested = []
    monkeypatch.setattr(cli,"preflight",lambda *a:{"accepted":True,"basis_sha256":"shared"})
    def fake_train(config,directory,run,*args,**kwargs):
        trained.append(str(directory))
        (run/"model.pt").write_bytes(b"one frozen NOF1 model")
    def fake_evaluate(config,directory,*args,**kwargs):
        tested.append((str(directory),kwargs["checkpoint_path"],kwargs["training_identity"]))
        return {"channel":"NOF1" if "nof1" in str(directory) else "NCS1"}
    monkeypatch.setattr(cli,"train",fake_train)
    monkeypatch.setattr(cli,"evaluate",fake_evaluate)
    monkeypatch.setattr(cli,"plot_reports",lambda *a:None)
    monkeypatch.setattr(cli.subprocess,"check_output",lambda *a,**kw:"test-commit\n")
    monkeypatch.setattr(cli.torch,"set_num_interop_threads",lambda n:None)
    cli.main()
    report = json.loads(Path("outputs/smoke/relative-run/experiment.json").read_text())
    assert report["reports"] == {"NOF1":"nof1/metrics.json","NCS1":"ncs1/metrics.json"}
    assert len(trained) == 1 and "nof1" in trained[0]
    assert len(tested) == 2 and tested[0][1:] == tested[1][1:]
    assert report["train_channel"] == report["validation_channel"] == "NOF1"


def test_sparse_frame_schedule_preserves_original_indices(monkeypatch,tmp_path):
    source = fake_reader(monkeypatch)
    dataset = FourDataTrainingSet(tmp_path/"validation.h5")
    batches = list(dataset.batches(3,snr_indices=[6],frame_indices=[1]))
    keys = np.concatenate([scoring["keys"] for _,_,scoring in batches])
    np.testing.assert_array_equal(keys,np.array([[6,1,d] for d in range(4)]))
    for _,targets,scoring in batches:
        for bits,(_,fi,di) in zip(targets["bits"],scoring["keys"]):
            np.testing.assert_array_equal(bits,source.bits[fi,di])
    with pytest.raises(ValueError,match="Frame indices"):
        list(dataset.batches(4,frame_indices=[1,1]))
