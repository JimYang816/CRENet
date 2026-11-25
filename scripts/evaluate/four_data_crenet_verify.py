"""Independent checks of complete test counts and the established tap-domain LMMSE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_generation.src.config import ROOT
from src.common.experiment import write_json
from src.common.numerics import DenseBackend, default_device
from src.data.four_data import FourDataDataset
from src.data.four_data_training import FourDataTrainingSet
from src.evaluation.four_data_crenet import full_matrix_lmmse, bit_errors, ALGORITHMS
from src.evaluation.four_data_lmmse import detect


def verify_channel(directory, run, *, device="cpu"):
    report = json.loads((run/"metrics.json").read_text(encoding="utf-8"))
    saved = np.load(run/"frame_metrics.npz")
    assert saved["complete"].all()
    assert saved["complete"].shape == (7,report["frames_per_snr"],4)
    for key in ALGORITHMS:
        counts = saved[key].sum(axis=(1,2))
        np.testing.assert_array_equal(counts,report["curves"][key]["errors"])
        np.testing.assert_array_equal(counts/report["bits_per_snr"],report["curves"][key]["ber"])
    measured = saved["nmse_linear"].mean(axis=(1,2))
    np.testing.assert_allclose(measured[:,0],report["initial_nmse_linear"],rtol=1e-13)
    np.testing.assert_allclose(measured[:,1],report["corrected_nmse_linear"],rtol=1e-13)
    # Compare the new batched frequency solve with the established independently
    # assembled time-coordinate detector on frame 0 at every SNR and all 4 symbols.
    subset = FourDataTrainingSet(directory/"test.h5",device=device,frame_limit=1,
                                  reference_precision="complex128")
    factor = torch.linalg.cholesky(subset.covariance)
    estimates = {key:np.empty((7,4,int(subset.masks["data"].sum())),complex)
                 for key in ("full_matrix_lmmse","perfect_csi_full_matrix_lmmse")}
    with torch.no_grad():
        for inputs,targets,scoring in subset.batches(4):
            for key,h in (("full_matrix_lmmse",scoring["channel"]),
                          ("perfect_csi_full_matrix_lmmse",scoring["reference"])):
                value = full_matrix_lmmse(h,scoring["observations"],scoring["known"],
                    scoring["rf_variance"],inputs["data_mask"],factor)
                si,_,di = scoring["keys"].T
                estimates[key][si,di] = value.cpu().numpy()
    backend = DenseBackend(device)
    maximum_relative_error = 0.
    with FourDataDataset(directory/"test.h5",input_view=report["input_view"]) as ds:
        observation = np.stack([ds.observations(si,0) for si in range(7)])
        taps = np.stack([ds.csi_taps(si,0) for si in range(7)])
        variance = ds._h["samples/rf_noise_variance"][:,0]
        noise,known,mask = ds._model["unit_rf_noise_covariance"][:],ds.known,ds.masks["data"]
        for key,value in (("full_matrix_lmmse",taps),("perfect_csi_full_matrix_lmmse",ds.reference_taps(0))):
            single,expected = detect(value,observation,known,variance,noise,mask,backend)
            actual = estimates[key]
            relative = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            maximum_relative_error = max(maximum_relative_error,relative)
            np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-7)
            bits = ds.labels(0)["bits"]
            error = np.count_nonzero(np.stack((expected.real<0,expected.imag<0),axis=-1) != bits[None],axis=(-1,-2))
            np.testing.assert_array_equal(error,saved[key][:,0])
            if key == "full_matrix_lmmse":
                single_errors = np.count_nonzero(np.stack((single.real<0,single.imag<0),axis=-1) != bits[None],axis=(-1,-2))
                np.testing.assert_array_equal(single_errors,saved["single_tap_lmmse"][:,0])
    verification = dict(complete=True,channel=report["channel"],all_symbols_counted=True,
        exact_error_count_aggregation=True,linear_nmse_before_db=True,
        independent_tap_domain_lmmse_all_snr_four_symbols=True,
        lmmse_max_relative_symbol_difference=maximum_relative_error,
        formal_full_test=report["formal"],bits_per_snr=report["bits_per_snr"])
    write_json(run/"verification.json",verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/experiment/four_data_crenet_20261125")
    parser.add_argument("--channel",choices=("NOF1","NCS1","all"),default="all")
    parser.add_argument("--device",choices=("cpu","cuda"),default=default_device())
    args = parser.parse_args()
    torch.set_num_threads(4)
    channels = ("NOF1","NCS1") if args.channel == "all" else (args.channel,)
    reports = [verify_channel(ROOT/f"data/processed/{channel.lower()}_dddd",
                args.output/channel.lower(),device=args.device) for channel in channels]
    write_json(args.output/"verification.json",dict(complete=True,channels=reports))
    print(json.dumps(reports,indent=2))


if __name__ == "__main__":
    main()
