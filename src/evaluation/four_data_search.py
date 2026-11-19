"""Prior/validation-only Qeff/L search under the immutable original pilot mask."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy import linalg

from data_generation.src.config import ROOT,digest
from data_generation.src.receiver.estimation import input_windows
from data_generation.src.receiver.four_data import FourDataEstimator,BlockedFourDataReceiver
from data_generation.src.receiver.four_data_covariance import JointMoments,JointFourDataEstimator
from data_generation.src.receiver.four_data_original import original_protocol,assert_original_protocol,make_original_frame
from data_generation.src.receiver.four_data_rotated import load_rotated_prior
from data_generation.src.transmitter.ofdm import fft,receive_symbols,decisions
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.common.experiment import file_sha256,write_json
from src.common.numerics import DenseBackend
from src.evaluation.four_data_covariance import (
    ALPHAS,RHOS,validation_score,save_estimator,load_estimator,comparison,channel_report,
)
from src.evaluation.four_data_mmse import experiment_config,window_ids,ALGORITHMS
from src.evaluation.four_data_original import run_channel,CODE as ORIGINAL_CODE
from src.evaluation.four_data_rotated import baseline_report


ORDERS=tuple(range(1,28))
LENGTHS=tuple(range(16,1025,16))
RANK_TOL=1e-10
MAX_CONDITION=1/RANK_TOL
SHORTLIST=12
MAX_PER_ORDER=3
CODE=tuple(dict.fromkeys((*ORIGINAL_CODE,"src/evaluation/four_data_search.py")))


def budget_grid(orders=ORDERS,lengths=LENGTHS,observations=1536):
    return [(int(q),int(length)) for q in orders for length in lengths if q*length<=observations]


def pilot_matrix(basis,known,rows,length):
    """Independent public-pilot map in q-major, delay-minor coefficient order."""
    delayed=input_windows(known,length)
    atoms=(basis[:,:,:,None]*delayed[:,:,None,:]).reshape(4,known.shape[1],-1)
    transformed=np.fft.fftshift(np.fft.fft(atoms,axis=1,norm="ortho"),axes=1)
    return np.concatenate([transformed[s,r] for s,r in enumerate(rows)])


def rank_check(matrix,backend,*,tol=RANK_TOL,max_condition=MAX_CONDITION):
    norm=np.linalg.norm(matrix,axis=0)
    if matrix.shape[1]>matrix.shape[0] or np.any(norm==0):
        return dict(observations=matrix.shape[0],unknowns=matrix.shape[1],numerical_rank=0,
                    column_normalized_condition=None,accepted=False,reason="dimension budget or zero column")
    normalized=matrix/norm
    singular=(backend.numpy(backend.torch.linalg.svdvals(backend.array(normalized))) if backend.torch
              else linalg.svdvals(normalized,check_finite=False))
    rank=int(np.count_nonzero(singular>singular[0]*tol))
    condition=float(singular[0]/max(singular[-1],np.finfo(float).tiny))
    return dict(observations=matrix.shape[0],unknowns=matrix.shape[1],numerical_rank=rank,
        relative_rank_tolerance=tol,column_normalized_condition=min(condition,1e300),
        min_to_max_singular_value=float(singular[-1]/singular[0]),
        accepted=rank==matrix.shape[1] and condition<=max_condition,
        condition_limit=max_condition)


def projection_curve(basis,cross,energy,lengths):
    """Full-reference per-D projection error including omitted delay energy."""
    coeff=cross.sum(axis=1)
    grams=np.einsum("dnq,dnr->dqr",basis.conj(),basis,optimize=True)
    power=np.einsum("bql,dqr,brl->bdl",coeff.conj(),grams,coeff,optimize=True).real
    correlation=np.einsum("bql,bdql->bdl",coeff.conj(),cross,optimize=True).real
    cumulative=np.cumsum(power-2*correlation,axis=-1)
    return {int(length):float((np.maximum(energy+cumulative[...,length-1],0)/energy).mean())
            for length in lengths}


def design_arrays(channel,design,*,smoke):
    config=experiment_config(channel)
    with np.load(design/f"{channel.lower()}_temporal_basis.npz",allow_pickle=False) as source:
        full,rotation=source["full_basis"].copy(),source["rotation"].copy()
    prior_ids,_=window_ids(config,smoke)
    validation_ids=[(r,s) for r in config["validation_records"] for s in (2.,14.)]
    if smoke:
        validation_ids=validation_ids[:1]
    result=dict(full=full,rotation=rotation,prior_ids=prior_ids,validation_ids=validation_ids)
    with h5py.File(design/f"{channel.lower()}_projection_cache.h5","r") as saved:
        if not saved.attrs.get("complete") or bool(saved.attrs["smoke"])!=smoke or saved.attrs["channel"]!=channel:
            raise ValueError("Independent physical projection scope mismatch")
        np.testing.assert_allclose(saved["basis"][:]@rotation,full,atol=1e-12,rtol=1e-12)
        for split,expected in (("prior",prior_ids),("validation",validation_ids)):
            groups=list(saved[split].values())
            actual=[(int(g.attrs["record"]),float(g.attrs["start_seconds"])) for g in groups]
            if actual!=expected:
                raise ValueError("Only the declared independent prior/validation windows may enter search")
            result[split+"_cross27"]=np.stack([g["Q27/cross"][:] for g in groups])
            result[split+"_energy"]=np.stack([g["energy"][:] for g in groups])
    positions=np.concatenate([s*(config["nfft"]+config["cp"])+config["cp"]-config["fft_advance"]
                             +np.arange(config["nfft"]) for s in range(4)])
    result["sampled"]=full[positions]
    np.testing.assert_allclose(result["sampled"].conj().T@result["sampled"],np.eye(27),atol=1e-10,rtol=0)
    return result


def rank_and_screen(channel,arrays,known,masks,backend,output,*,smoke):
    orders=(2,4) if smoke else ORDERS
    lengths=(64,128) if smoke else LENGTHS
    rows=[np.flatnonzero(m) for m in masks["tracking"]]
    record=dict(complete=False,channel=channel,observations=1536,orders=list(orders),lengths=list(lengths),
        grid_count=len(budget_grid(orders,lengths)),rank_tolerance=RANK_TOL,condition_limit=MAX_CONDITION,
        algorithm="SVD bisection of nested column sets; smaller normalized-column subsets cannot have a larger condition number",
        checked=[],candidates=[],shortlist=[])
    for q in orders:
        allowed=[length for length in lengths if q*length<=1536]
        if not allowed:
            continue
        maximum=max(allowed)
        basis=arrays["sampled"][:,:q].reshape(4,1024,q)
        maximum_matrix=pilot_matrix(basis,known,rows,maximum)
        cache={}
        def check(index):
            if index not in cache:
                length=allowed[index]
                ix=(np.arange(q)[:,None]*maximum+np.arange(length)).ravel()
                cache[index]=dict(effective_q=q,length=length,**rank_check(maximum_matrix[:,ix],backend))
                record["checked"].append(cache[index])
            return cache[index]
        low,high=-1,len(allowed)
        while high-low>1:
            mid=(low+high)//2
            if check(mid)["accepted"]:
                low=mid
            else:
                high=mid
        eligible=allowed[:low+1]
        if eligible:
            transformed=np.einsum("rq,bdrl->bdql",arrays["rotation"][:,:q].conj(),arrays["validation_cross27"],optimize=True)
            curve=projection_curve(basis,transformed,arrays["validation_energy"],eligible)
            for length in eligible:
                record["candidates"].append(dict(effective_q=q,length=length,unknowns=q*length,
                    validation_projection_nmse_linear=curve[length],validation_projection_nmse_db=float(10*np.log10(curve[length])),
                    rank_eligibility="nested subset of checked full-rank frontier",frontier_length=eligible[-1]))
        print(f"{channel} fixed-mask rank Qeff={q}: max admissible L={eligible[-1] if eligible else 'none'}",flush=True)
        write_json(output/"rank_screen.json",record)
    counts={}
    limit=2 if smoke else SHORTLIST
    ranked=sorted(record["candidates"],key=lambda v:(v["validation_projection_nmse_linear"],v["unknowns"]))
    for row in ranked:
        q=row["effective_q"]
        if counts.get(q,0)>=MAX_PER_ORDER:
            continue
        record["shortlist"].append(row)
        counts[q]=counts.get(q,0)+1
        if len(record["shortlist"])==limit:
            break
    if not record["shortlist"]:
        raise ValueError("No full-column-rank candidate under the fixed carrier contract")
    # Independently SVD every actual shortlisted matrix, not just the frontier.
    for row in record["shortlist"]:
        q,length=row["effective_q"],row["length"]
        row["pilot_rank"]=rank_check(pilot_matrix(arrays["sampled"][:,:q].reshape(4,1024,q),known,rows,length),backend)
        if not row["pilot_rank"]["accepted"]:
            raise AssertionError("Nested-rank eligibility was not reproduced by direct SVD")
    record["complete"]=True
    write_json(output/"rank_screen.json",record)
    return record


def independent_physics(channel,calibration,arrays,known,masks):
    run=calibration/channel.lower()
    with h5py.File(run/"frozen_moments.h5","r") as saved:
        if not saved.attrs.get("complete"):
            raise ValueError("Independent physical moments must be complete")
        assert_original_protocol(experiment_config(channel),saved["known"][:],{k:saved["masks/"+k][:] for k in masks})
        metadata=json.loads(saved.attrs["metadata_json"])
        if metadata["prior_windows"]!=[list(x) for x in arrays["prior_ids"]]:
            raise ValueError("Physical covariance source must use independent prior records only")
        physical_pilot=saved["moments/model_pilot"][:]+saved["moments/pilot_residual"][:]
        data=saved["moments/physical_data_covariance"][:]
        noise=saved["unit_noise_covariance"][:]
    with h5py.File(run/"validation.h5","r") as saved:
        if not saved.attrs.get("complete") or json.loads(saved.attrs["windows_json"])!=[list(x) for x in arrays["validation_ids"]]:
            raise ValueError("Validation observation source must contain only records 056-060")
        clean=saved["clean"][:]
        variances=saved["variances"][:]
    receiver=BlockedFourDataReceiver(experiment_config(channel))
    standard,bits=[],[]
    for record,start in arrays["validation_ids"]:
        frame=make_original_frame(receiver.chain.c,record,start,known,masks)
        seed=(receiver.chain.c["seed"],2,int(record),round(start*1000),1,0)
        rf=np.random.default_rng(np.random.SeedSequence(seed)).standard_normal(len(receiver.chain.phase_carrier))
        standard.append(receive_symbols(receiver.chain.receive(rf),receiver.chain.c))
        bits.append(frame.bits)
    validation=dict(clean=clean,variances=variances,standard_noise=np.stack(standard),bits=np.stack(bits),energy=arrays["validation_energy"])
    return dict(pilot=physical_pilot,data=data,noise=noise,metadata=metadata),validation


def candidate_moments(prior,known,masks,physical,backend):
    base=FourDataEstimator(prior,known,masks,physical["noise"],backend,capture_moments=True)
    q,length=prior.model.count,prior.model.length
    basis=prior.model.sampled_basis.reshape(4,prior.model.c["nfft"],q)
    predicted=fft(np.sum((basis[None]@prior.coefficients[:,None])*input_windows(known,length)[None],axis=-1))
    predicted=np.concatenate([predicted[:,s,r] for s,r in enumerate(base.rows)],axis=-1).T
    residual=physical["pilot"]-predicted
    raw=dict(base_pilot_covariance=base.captured_moments["observation_covariance"]-base.captured_moments["data_interference"],
        base_cross_covariance=base.captured_moments["cross_covariance"],physical_data_covariance=physical["data"],
        coefficients=prior.coefficients.reshape(len(prior.coefficients),-1).T,model_pilot=predicted,
        pilot_residual=residual,selected_noise=base.selected_noise)
    metadata=dict(prior_windows=physical["metadata"]["prior_windows"],ridge=base.ridge,
        physical_data_off_diagonal_D_fraction=physical["metadata"]["physical_data_off_diagonal_D_fraction"],
        note="Reuse exact independent-prior physical pilot response and data covariance, which do not depend on Qeff/L; reproject targets and recompute all pilot residual/cross moments",
        pilot_residual_powers=np.sum(abs(residual)**2,axis=0).tolist(),effective_q=q,length=length)
    return raw,metadata


def validation_ber(estimator,data,known,masks,noise,backend):
    variances=data["variances"]
    y=data["clean"][:,None]+np.sqrt(variances[...,None,None])*data["standard_noise"][:,None]
    fitted=estimator.estimate_coefficients(y,variances)
    from data_generation.src.receiver.chain import taps_to_matrix
    whitener=make_whitener(noise,backend)
    errors={name:np.zeros(len(variances[0]),dtype=np.int64) for name in ALGORITHMS}
    for index in range(len(y)):
        for s in range(4):
            taps=estimator.prior.model.reconstruct(fitted[index],s)
            matrix=np.stack([taps_to_matrix(t) for t in taps])
            raw=y[index,:,s]
            cancelled=raw-(matrix@known[s,:,None])[...,0]
            batch=EqualizerBatch(backend.array(matrix),backend.array(raw),backend.array(cancelled),variances[index],
                np.flatnonzero(masks["data"][s]),noise)
            for name,equalizer in ALGORITHMS.items():
                kwargs={"whitener":whitener} if name=="full_matrix_mmse" else {}
                symbols=equalizer(batch,backend,**kwargs).symbols
                errors[name]+=np.count_nonzero(decisions(symbols)!=data["bits"][index,s],axis=(1,2))
    bits=int(data["bits"].size)
    return dict(bits_per_snr=bits,algorithms={name:dict(bit_errors=v.tolist(),ber=(v/bits).tolist()) for name,v in errors.items()})


def fit_candidate(channel,row,design,physical,validation,arrays,known,masks,backend,output,source_identity,*,smoke):
    q,length=row["effective_q"],row["length"]
    run=output/f"q{q:02d}_l{length:04d}"
    run.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(source_identity=source_identity,q=q,length=length,rank=row["pilot_rank"]))
    report_path=run/"candidate.json"
    if report_path.exists():
        result=json.loads(report_path.read_text(encoding="utf-8"))
        if result["identity"]!=identity:
            raise ValueError("Search candidate identity mismatch")
        if result["complete"]:
            return result
    model,prior,ids=load_rotated_prior(design,channel,experiment_config(channel),q,length,smoke=smoke)
    if ids!=[(int(r),float(s)) for r,s in arrays["prior_ids"]]:
        raise ValueError("Candidate temporal prior identities changed")
    raw,metadata=candidate_moments(prior,known,masks,physical,backend)
    with h5py.File(run/"frozen_moments.h5","w") as saved:
        saved.attrs.update(identity=identity,complete=False,metadata_json=json.dumps(metadata))
        for name,value in dict(basis=model.basis,known=known,unit_noise_covariance=physical["noise"]).items():
            saved.create_dataset(name,data=value)
        for name,value in masks.items():
            saved.create_dataset("masks/"+name,data=value)
        for name,value in raw.items():
            saved.create_dataset("moments/"+name,data=value)
        saved.attrs["complete"]=True
    write_json(run/"calibration.json",dict(metadata,identity=identity,complete=True))
    local_validation=dict(validation,cross=np.einsum("rq,bdrl->bdql",arrays["rotation"][:,:q].conj(),
        arrays["validation_cross27"][...,:length],optimize=True))
    with h5py.File(run/"validation.h5","w") as saved:
        saved.attrs.update(identity=identity,complete=False,windows_json=json.dumps(arrays["validation_ids"]))
        for name,value in local_validation.items():
            saved.create_dataset(name,data=value)
        saved.attrs["complete"]=True
    moments=JointMoments(prior,known,masks,raw,metadata,backend)
    del raw
    selection=dict(identity=identity,complete=False,prior_windows=arrays["prior_ids"],validation_windows=arrays["validation_ids"],
        objective="inner selection: mean linear full-reference validation NMSE over all seven SNRs, exact thermal expectation",
        grid_alpha=[.4] if smoke else list(ALPHAS),grid_rho=[.5] if smoke else list(RHOS),candidates=[])
    best,best_value=None,np.inf
    for alpha in selection["grid_alpha"]:
        for rho in selection["grid_rho"]:
            candidate=JointFourDataEstimator(moments,alpha,rho)
            value,curve=validation_score(candidate,local_validation)
            entry=dict(alpha=alpha,rho=rho,objective=value,mean_nmse_linear=curve.tolist(),mean_nmse_db=(10*np.log10(curve)).tolist(),diagnostics=candidate.diagnostics)
            selection["candidates"].append(entry)
            if value<best_value:
                best,best_value=candidate,value
                selection["selected"]=entry
            write_json(run/"selection.json",selection)
    selection["complete"]=True
    write_json(run/"selection.json",selection)
    save_estimator(run/"frozen_estimator.h5",best,identity,selection)
    ber=validation_ber(best,local_validation,known,masks,physical["noise"],backend)
    objective=float(np.mean(ber["algorithms"]["full_matrix_mmse"]["ber"]))
    result=dict(row,identity=identity,complete=True,candidate_directory=str(run.resolve()),
        candidate_identity=identity,frozen_estimator_sha256=file_sha256(run/"frozen_estimator.h5"),
        selected_hyperparameters=selection["selected"],validation_detection=ber,
        validation_full_matrix_mean_ber=objective,validation_nmse_objective=best_value)
    write_json(report_path,result)
    print(f"{channel} candidate Qeff={q},L={length},rank={row['pilot_rank']['numerical_rank']}, "
        f"validation full BER mean={objective:.6f},30dB={ber['algorithms']['full_matrix_mmse']['ber'][-1]:.6f}, "
        f"NMSE30={selection['selected']['mean_nmse_db'][-1]:.3f} dB",flush=True)
    return result


def select_channel(channel,design,calibration,output,backend,source,*,smoke):
    run=output/channel.lower()/"search"
    run.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(source=source,channel=channel,smoke=smoke,orders=ORDERS,lengths=LENGTHS,
        rank_tolerance=RANK_TOL,max_condition=MAX_CONDITION,shortlist=SHORTLIST,max_per_order=MAX_PER_ORDER,
        inner_grid=(ALPHAS,RHOS),outer_objective="mean validation full-matrix BER over seven SNRs"))
    path=run/"search.json"
    if path.exists():
        result=json.loads(path.read_text(encoding="utf-8"))
        if result["identity"]!=identity:
            raise ValueError("Search protocol or independent sources changed")
        if result["complete"]:
            return result
    arrays=design_arrays(channel,design,smoke=smoke)
    known,masks=original_protocol(experiment_config(channel))
    screen=rank_and_screen(channel,arrays,known,masks,backend,run,smoke=smoke)
    physical,validation=independent_physics(channel,calibration,arrays,known,masks)
    result=dict(identity=identity,complete=False,channel=channel,prior_windows=arrays["prior_ids"],
        validation_windows=arrays["validation_ids"],test_access_during_selection=False,shortlist_size=len(screen["shortlist"]),
        objective="mean validation full-matrix BER over all seven SNRs; NMSE then fewer coefficients break ties",
        screening="best validation projection errors among SVD-admissible candidates; at most three lengths per Qeff",candidates=[])
    for row in screen["shortlist"]:
        result["candidates"].append(fit_candidate(channel,row,design,physical,validation,arrays,known,masks,backend,run,identity,smoke=smoke))
        write_json(path,result)
        gc.collect()
    result["selected"]=min(result["candidates"],key=lambda v:(v["validation_full_matrix_mean_ber"],v["validation_nmse_objective"],v["unknowns"]))
    result["complete"]=True
    write_json(path,result)
    print(f"{channel} frozen Qeff={result['selected']['effective_q']},L={result['selected']['length']} before test access",flush=True)
    return result


def run_experiment(archive,design,calibration,baseline,output,*,channels,device,smoke=False,report_only=False):
    design,calibration,baseline,output=map(Path,(design,calibration,baseline,output))
    output.mkdir(parents=True,exist_ok=True)
    if report_only:
        results={}
        for channel in channels:
            report=channel_report(output,channel)[1]
            if not report.get("q_l_selection") or report["parameters"]["pilot_system_underdetermined"]:
                raise ValueError("Need a completed full-column-rank selected-Qeff/L experiment")
            results[channel]=report,dict(summary=report["structured_summary"]),baseline_report(baseline,channel)[1]
        return comparison(output,results,next(iter(results.values()))[0]["source"],original_layout=True)
    source=dict(git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        code_sha256={name:file_sha256(ROOT/name) for name in CODE},archive_sha256=file_sha256(archive),
        independent_design={channel:{name:file_sha256(design/f"{channel.lower()}_{name}") for name in ("temporal_basis.npz","projection_cache.h5")} for channel in channels},
        independent_physics={channel:{name:file_sha256(calibration/channel.lower()/name) for name in ("frozen_moments.h5","validation.h5")} for channel in channels},
        original_waveform={channel:{name:file_sha256(baseline/channel.lower()/name) for name in ("metrics.json","receiver_estimates.h5")} for channel in channels},
        device=device,smoke=smoke,precision="complex128",carrier_contract="original-fixed-v1",
        selection="prior and validation only; freeze both channels before entering final test evaluation")
    write_json(output/"source_manifest.json",source)
    backend=DenseBackend(device)
    searches={channel:select_channel(channel,design,calibration,output,backend,source,smoke=smoke) for channel in channels}
    frozen=dict(complete=True,source=source,channels={ch:r["selected"] for ch,r in searches.items()},
                note="Both channel selections completed and estimators serialized before final test observation/CSI reads")
    write_json(output/"model_selection.json",frozen)
    results={channel:run_channel(channel,archive,design,baseline,output,backend,source,smoke=smoke,
        selected_model=searches[channel]["selected"]) for channel in channels}
    return comparison(output,results,source,original_layout=True)
