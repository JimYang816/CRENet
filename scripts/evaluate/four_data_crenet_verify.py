"""Verify shared NOF1 weights, exhaustive counts, NMSE, and independent B16 solves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_generation.src.config import ROOT
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend, default_device
from src.data.four_data import FourDataDataset
from src.data.four_data_training import FourDataTrainingSet
from src.evaluation.four_data_crenet import ALGORITHMS, banded_lmmse
from src.evaluation.four_data_lmmse import detect


def verify_channel(directory, run, *, device="cpu"):
    report = json.loads((run/"metrics.json").read_text(encoding="utf-8"))
    with np.load(run/"frame_metrics.npz") as saved:
        assert saved["complete"].all()
        assert saved["complete"].shape == (7,report["frames_per_snr"],4)
        if report["formal"]:
            assert report["frames_per_snr"] == 600
            assert report["bits_per_snr"] == 2457600
        for key in ALGORITHMS:
            counts = saved[key].sum(axis=(1,2))
            np.testing.assert_array_equal(counts,report["curves"][key]["errors"])
            np.testing.assert_array_equal(counts/report["bits_per_snr"],report["curves"][key]["ber"])
        measured = saved["nmse_linear"].mean(axis=(1,2))
        np.testing.assert_allclose(measured[:,0],report["initial_nmse_linear"],rtol=1e-13)
        np.testing.assert_allclose(measured[:,1],report["corrected_nmse_linear"],rtol=1e-13)
        np.testing.assert_allclose(10*np.log10(measured[:,0]),report["initial_nmse_db"],atol=1e-12)
        np.testing.assert_allclose(10*np.log10(measured[:,1]),report["corrected_nmse_db"],atol=1e-12)
        first_errors = {key:saved[key][:,0].copy() for key in ALGORITHMS}
    subset = FourDataTrainingSet(directory/"test.h5",device=device,frame_limit=1,
                                input_view=report["input_view"],reference_precision="complex128")
    factor = torch.linalg.cholesky(subset.covariance)
    keys = ("banded_lmmse","perfect_csi_banded_lmmse")
    estimates = {key:np.empty((7,4,int(subset.masks["data"].sum())),complex) for key in keys}
    with torch.no_grad():
        for inputs,targets,scoring in subset.batches(4):
            for key,h in zip(keys,(scoring["channel"],scoring["reference"])):
                symbols = banded_lmmse(h,scoring["observations"],scoring["known"],
                    scoring["rf_variance"],inputs["data_mask"],factor)
                si,_,di = scoring["keys"].T
                estimates[key][si,di] = symbols.cpu().numpy()
    backend = DenseBackend(device)
    maximum = 0.
    with FourDataDataset(directory/"test.h5",input_view=report["input_view"]) as ds:
        observations = np.stack([ds.observations(si,0) for si in range(7)])
        taps = np.stack([ds.csi_taps(si,0) for si in range(7)])
        variance = ds._h["samples/rf_noise_variance"][:,0]
        noise = ds._model["unit_rf_noise_covariance"][:]
        for key,value in zip(keys,(taps,ds.reference_taps(0))):
            expected = detect(value,observations,ds.known,variance,noise,ds.masks["data"],backend)
            actual = estimates[key]
            maximum = max(maximum,float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)))
            np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-7)
            bits = ds.labels(0)["bits"]
            errors = np.count_nonzero(np.stack((expected.real<0,expected.imag<0),axis=-1) != bits[None],axis=(-1,-2))
            np.testing.assert_array_equal(errors,first_errors[key])
    verification = dict(complete=True,channel=report["channel"],all_symbols_counted=True,
        exact_error_count_aggregation=True,linear_nmse_before_db=True,
        independent_b16_all_snr_four_symbols=True,b16_max_relative_symbol_difference=maximum,
        formal_full_test=report["formal"],bits_per_snr=report["bits_per_snr"])
    write_json(run/"verification.json",verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default=default_device())
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((ROOT/"configs/training/four_data_crenet.json").read_text(encoding="utf-8"))
    output = args.output or ROOT/config["output"]
    experiment = json.loads((output/"experiment.json").read_text(encoding="utf-8"))
    training = json.loads((output/"training/nof1/training.json").read_text(encoding="utf-8"))
    checkpoint = output/"training/nof1/model.pt"
    checkpoint_hash = file_sha256(checkpoint)
    assert training["complete"] and training["channel"] == "NOF1"
    assert not training["test_used_in_training"]
    assert experiment["train_channel"] == experiment["validation_channel"] == "NOF1"
    assert experiment["checkpoint_sha256"] == checkpoint_hash
    reports = []
    for channel in experiment["channels"]:
        run = output/channel.lower()
        provenance = json.loads((run/"provenance.json").read_text(encoding="utf-8"))
        metrics = json.loads((run/"metrics.json").read_text(encoding="utf-8"))
        assert metrics["train_channel"] == "NOF1"
        assert metrics["training_identity"] == training["identity"] == experiment["training_identity"]
        assert metrics["checkpoint_sha256"] == checkpoint_hash == provenance["checkpoint_sha256"]
        reports.append(verify_channel(ROOT/provenance["config"]["datasets"][channel],run,device=args.device))
    result = dict(complete=True,train_channel="NOF1",validation_channel="NOF1",
        shared_frozen_checkpoint_sha256=checkpoint_hash,test_used_in_training=False,channels=reports)
    write_json(output/"verification.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
