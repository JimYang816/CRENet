"""Original-waveform 4D experiment: fixed masks, fair three-way CSI comparison."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from data_generation.src.channel.watermark.replay import WatermarkArchive
from data_generation.src.config import ROOT, digest
from data_generation.src.receiver.four_data import BlockedFourDataReceiver, FourDataEstimator
from data_generation.src.receiver.four_data_original import (
    assert_original_protocol, make_original_frame, original_protocol,
)
from data_generation.src.receiver.four_data_rotated import load_rotated_prior
from src.algorithms.likelihood import make_whitener
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.evaluation.four_data_covariance import ALPHAS, RHOS, calibrate, channel_report, comparison
from src.evaluation.four_data_mmse import ALGORITHMS, _summary, experiment_config, window_ids
from src.evaluation.four_data_rotated import PARAMETERS, baseline_report, score


CODE = ("data_generation/src/receiver/four_data_original.py", "src/evaluation/four_data_original.py",
        "scripts/evaluate/four_data_covariance.py", "data_generation/src/receiver/four_data.py",
        "data_generation/src/receiver/four_data_covariance.py", "src/evaluation/four_data_covariance.py",
        "src/evaluation/four_data_rotated.py", "src/evaluation/four_data_rotated_audit.py",
        "data_generation/src/receiver/chain.py", "data_generation/src/receiver/estimation.py",
        "data_generation/src/transmitter/ofdm.py", "src/algorithms/single_tap_mmse/equalizer.py",
        "data_generation/src/receiver/four_data_rotated.py", "data_generation/src/receiver/dpss.py",
        "data_generation/src/channel/watermark/replay.py",
        "src/algorithms/full_matrix_lmmse/equalizer.py", "src/algorithms/likelihood.py", "src/common/numerics.py")


def restored_parameters(channel):
    q, length = PARAMETERS[channel]
    return dict(frame="D1-D2-D3-D4", bandwidth_hz=12, dictionary_q=27, effective_q=q, length=length,
        pilots_per_d=128, protection_per_d=256, data_per_d=512, empty_edges_per_d=128,
        edge_bins_each_side=64, observations=1536, unknown_coefficients=q*length,
        pilot_system_underdetermined=q*length > 1536, pilot_rank_upper_bound=1536,
        delay_block_size=32, csi_view="direct receiver CSI", ideal_compensation_db=0,
        layout="64 empty | DDZPZDD repeated 128 | 64 empty; identical masks in all four D",
        data_symbol_energy=1., pilot_symbol_energy=1., tx_gain=1.,
        power="identical original waveform and RF energy; original Q24/L176 receive cache reused",
        estimation="Bayesian LMMSE with independent prior; no claim of full pilot column rank",
        csi_definition="unchanged self-symbol circular CSI, projected into frozen Qeff/L")


def run_channel(channel, archive_path, basis_source, baseline, output, backend, source, *, smoke):
    config = experiment_config(channel)
    _, original = baseline_report(baseline, channel)
    prior_ids, test_ids = window_ids(config, smoke)
    validation_ids = [(record, start) for record in range(56, 61) for start in (2., 14.)]
    if smoke:
        validation_ids = validation_ids[:1]
    if original["smoke"] != smoke or [(f["record"],f["start_seconds"]) for f in original["frames"]] != test_ids:
        raise ValueError("The source original-waveform experiment must match this test scope")
    if ({r for r,_ in prior_ids} & {r for r,_ in validation_ids}
            or {r for r,_ in prior_ids+validation_ids} & {r for r,_ in test_ids}):
        raise ValueError("Independent calibration, validation and test records overlap")
    identity = digest(dict(channel=channel, source=source, smoke=smoke, prior=prior_ids,
        validation=validation_ids, test=test_ids, alphas=ALPHAS, rhos=RHOS, carrier_contract="original-fixed-v1"))
    run = output/channel.lower()
    run.mkdir(parents=True, exist_ok=True)
    path = run/"metrics.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError("Refusing to mix another protocol or source into restored-layout results")
        if result["complete"]:
            before = dict(summary=result["structured_summary"])
            return result, before, original
    else:
        result = dict(schema="four-data-original-fixed-receiver-v1", identity=identity, complete=False,
            channel=channel, smoke=smoke, source=source, prior_windows=prior_ids, validation_windows=validation_ids,
            test_windows=test_ids, parameters=restored_parameters(channel), frames=[],
            source_receiver_identity=original["identity"],
            comparison="All three curves use exactly the original carrier masks, bits, pilots, waveform and RF noise")
    known, masks = original_protocol(config)
    source_cache = baseline/channel.lower()/"receiver_estimates.h5"
    with h5py.File(source_cache, "r") as saved:
        if not saved.attrs.get("complete") or saved.attrs["identity"] != original["identity"]:
            raise ValueError("Original receive cache is incomplete or belongs to another experiment")
        assert_original_protocol(config, saved["known"][:], masks)
        noise = saved["unit_noise_covariance"][:]
        np.testing.assert_array_equal(saved["snr_db"][:], config["snr_db"])
    q, length = PARAMETERS[channel]
    model,prior,ids = load_rotated_prior(basis_source,channel,config,q,length,smoke=smoke)
    if ids != [(int(r),float(s)) for r,s in prior_ids]:
        raise ValueError("Independent temporal prior source identities changed")
    prior_path = run/"frozen_prior.h5"
    if not prior_path.exists():
        with h5py.File(prior_path,"w") as saved:
            saved.attrs.update(identity=identity,complete=True,independent_records_json=json.dumps(prior_ids))
            saved.create_dataset("basis",data=model.basis)
            saved.create_dataset("coefficients",data=prior.coefficients)
    receiver, archive = BlockedFourDataReceiver(config), WatermarkArchive(archive_path, channel)
    print(f"{channel} original masks fixed: 128P/256Z/512D, 64 empty bins per edge; "
          f"{q*length} coefficients > 1536 pilot/protection observations (Bayesian estimate)", flush=True)
    estimator, selection = calibrate(run, identity, prior, known, masks, noise, receiver, archive,
        prior_ids, validation_ids, backend, smoke=smoke, frame_factory=make_original_frame)
    result["selected_parameters"] = selection["selected"]
    result["estimator"] = estimator.diagnostics
    result["frozen_estimator_sha256"] = file_sha256(run/"frozen_estimator.h5")
    structured = FourDataEstimator(prior, known, masks, noise, backend)
    result["structured_estimator"] = structured.diagnostics
    whitener = make_whitener(noise, backend)
    write_json(path, result)
    done = {(f["record"],f["start_seconds"]) for f in result["frames"]}
    with h5py.File(source_cache, "r") as received, h5py.File(run/"receiver_estimates.h5", "a") as saved:
        if "identity" in saved.attrs and saved.attrs["identity"] != identity:
            raise ValueError("Restored-layout CSI cache identity mismatch")
        saved.attrs.update(identity=identity, complete=False, source_receiver_identity=original["identity"],
            estimator_sha256=result["frozen_estimator_sha256"], layout="original-fixed-v1")
        if "known" not in saved:
            saved.create_dataset("known", data=known)
            saved.create_dataset("unit_noise_covariance", data=noise)
            for name,value in masks.items():
                saved.create_dataset(f"masks/{name}", data=value)
        for index,(record,start) in enumerate(test_ids):
            if (record,start) in done:
                continue
            begin = time.perf_counter()
            group = received[f"frames/{index:04d}"]
            frame = make_original_frame(config, record, start, known, masks)
            np.testing.assert_array_equal(frame.bits, group["bits"][:])
            if json.loads(group.attrs["payload_seed"]) != list(frame.seed):
                raise ValueError("Original payload seed changed")
            tx_energy = float(np.sum(receiver.chain.transmit(frame.baseband)**2))
            variances = group["rf_variances"][:]
            y = group["clean_observations"][:][None]+np.sqrt(variances[:,None,None])*group["standard_noise"][:][None]
            # Both CSI estimates are frozen before this first test read.
            fitted = estimator.estimate_coefficients(y, variances)
            fitted_before = structured.estimate_coefficients(y, variances)
            sounding = archive.load(record)
            refs = [receiver.reference_taps(sounding,start,s) for s in range(4)]
            truth = np.stack([t for t,_ in refs])
            nmse, errors = score(model,fitted,truth,y,known,frame.bits,variances,masks,noise,backend,whitener,verify=index==0)
            nmse_before, errors_before = score(model,fitted_before,truth,y,known,frame.bits,variances,masks,noise,backend,whitener,verify=index==0)
            key = f"frames/{index:04d}"
            if key in saved:
                del saved[key]
            target = saved.create_group(key)
            target.attrs.update(record=record,start_seconds=start,tx_gain=1.,rf_tx_energy=tx_energy,
                payload_seed=json.dumps(frame.seed),noise_seed=group.attrs["noise_seed"])
            arrays = dict(coefficients=fitted,structured_coefficients=fitted_before,rf_variances=variances,
                nmse_linear=nmse,structured_nmse_linear=nmse_before,
                **{f"errors_{name}":v for name,v in errors.items()},
                **{f"structured_errors_{name}":v for name,v in errors_before.items()})
            for name,value in arrays.items():
                target.create_dataset(name,data=value)
            saved.flush()
            item = dict(record=record,start_seconds=start,payload_seed=frame.seed,
                noise_seed=json.loads(group.attrs["noise_seed"]),tx_gain=1.,rf_tx_energy=tx_energy,
                old_rf_tx_energy=tx_energy,new_rf_tx_energy=tx_energy,rf_power_ratio=1.,
                original_payload_identical=True,nmse_linear_per_d=nmse.tolist(),
                bit_errors={name:v.tolist() for name,v in errors.items()},
                structured_nmse_linear_per_d=nmse_before.tolist(),
                structured_bit_errors={name:v.tolist() for name,v in errors_before.items()},
                intersymbol_leakage=[v for _,v in refs],seconds=time.perf_counter()-begin)
            result["frames"].append(item)
            result["summary"] = _summary(result["frames"],config["snr_db"],frame.bits.size)
            before_frames = [dict(nmse_linear_per_d=f["structured_nmse_linear_per_d"],bit_errors=f["structured_bit_errors"]) for f in result["frames"]]
            result["structured_summary"] = _summary(before_frames,config["snr_db"],frame.bits.size)
            write_json(path,result)
            print(f"{channel} original-layout {index+1}/{len(test_ids)} {record:03d}@{start}s: "
                f"30dB NMSE structured/joint={10*np.log10(nmse_before[-1].mean()):.3f}/{10*np.log10(nmse[-1].mean()):.3f} dB; "
                f"joint BER single/full={errors['single_tap_mmse'][-1].sum()/frame.bits.size:.5f}/"
                f"{errors['full_matrix_mmse'][-1].sum()/frame.bits.size:.5f}; masks fixed",flush=True)
        saved.attrs["complete"] = True
    if file_sha256(run/"frozen_estimator.h5") != result["frozen_estimator_sha256"]:
        raise AssertionError("Frozen covariance estimator changed during testing")
    assert_original_protocol(config,known,masks)
    result["complete"] = True
    write_json(path,result)
    return result,dict(summary=result["structured_summary"]),original


def run_experiment(archive,basis_source,baseline,output,*,channels,device,smoke=False,report_only=False):
    basis_source,baseline,output = Path(basis_source),Path(baseline),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if report_only:
        results = {}
        for channel in channels:
            corrected = channel_report(output,channel)[1]
            if corrected.get("schema") != "four-data-original-fixed-receiver-v1":
                raise ValueError("This entry only reports original-carrier-layout experiments")
            results[channel] = corrected,dict(summary=corrected["structured_summary"]),baseline_report(baseline,channel)[1]
        return comparison(output,results,next(iter(results.values()))[0]["source"],original_layout=True)
    source = dict(git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        code_sha256={name:file_sha256(ROOT/name) for name in CODE},archive_sha256=file_sha256(archive),
        temporal_prior={channel:{name:file_sha256(basis_source/f"{channel.lower()}_{name}") for name in
            ("temporal_basis.npz","projection_cache.h5")} for channel in channels},
        original_waveform={channel:{name:file_sha256(baseline/channel.lower()/name) for name in
            ("metrics.json","receiver_estimates.h5")} for channel in channels},
        carrier_contract="original-fixed-v1",waveform_source="original Q24/L176 cache only; no full-band observations",
        basis_source_role="independent temporal functions and prior coefficients only; no carrier masks or test data",
        device=device,smoke=smoke,precision="complex128",validation_records=list(range(56,61)))
    write_json(output/"source_manifest.json",source)
    backend = DenseBackend(device)
    results = {channel:run_channel(channel,archive,basis_source,baseline,output,backend,source,smoke=smoke) for channel in channels}
    return comparison(output,results,source,original_layout=True)
