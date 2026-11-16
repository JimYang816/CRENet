"""Inspect DJK recordings and plot explicitly hypothetical simulation references.

No equalizer, bit demapper or LDPC decoder is run. Hypothetical BER values are
not measurements, confidence intervals, or predictions of frozen-model transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from src.evaluation.registry import normalize_report_labels


ROOT = Path(__file__).resolve().parents[2]
FS = 96000
BB_FS = 12000
METHODS = ["Single-tap LMMSE", "Full-matrix LMMSE", "MMSE-DFE", "UDNet",
           "OAMP-Net2", "CRENet"]
COLORS = ["#7a8694", "#245a87", "#b27832", "#8c65a8", "#239687", "#d64d45"]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quantiles(values):
    return dict(zip(["p05", "median", "p95"], np.quantile(values, [.05, .5, .95]).tolist()))


def read_recording(path):
    words = np.memmap(path, dtype="<u4", mode="r")
    if len(words) % 8200:
        raise ValueError(f"Unexpected packet size: {path}")
    packets = words.reshape(-1, 8200)
    if not np.array_equal(packets[:, 0], np.arange(len(packets))):
        raise ValueError("Packet counters are not consecutive")
    if np.any(packets[:, 1:8]):
        raise ValueError("Unexpected nonzero header padding")
    payload = packets[:, 8:].reshape(-1).astype(np.int32)
    tags, counts = np.unique(payload >> 24, return_counts=True)
    if not set(tags).issubset({0x61, 0x63}):
        raise ValueError("Unexpected payload tags")
    samples = ((payload & 0xffffff) ^ 0x800000) - 0x800000
    info = dict(file=path.name, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                packets=len(packets), samples=len(samples), duration_s=len(samples)/FS,
                tags={hex(int(k)): int(v) for k, v in zip(tags, counts)},
                max_absolute_sample=int(np.max(np.abs(samples))),
                signed_24bit_rail_hits=int(np.sum((samples == -8388608) | (samples == 8388607))))
    return samples.astype(np.float32), info


def analyze_recordings(out):
    raw, files = [], []
    for path in sorted((ROOT / "data/raw/djk").glob("*.raw")):
        samples, info = read_recording(path)
        raw.append(samples)
        files.append(info)
    samples = np.concatenate(raw)
    mixed = (samples.reshape(-1, 4) * np.array([1, -1j, -1, 1j], np.complex64)).reshape(-1)
    bb = 2 * signal.resample_poly(mixed, 1, 8)
    # Common centered FIR for diagnostic power measurements in the 21--27 kHz band.
    inband = signal.fftconvolve(bb, signal.firwin(257, 3000, fs=BB_FS,
                               window=("kaiser", 8)), mode="same")
    chirp_t = np.arange(600)/BB_FS
    replica = np.exp(2j*np.pi*(3000*chirp_t - 60000*chirp_t**2))
    correlation = signal.correlate(bb, replica, mode="valid", method="fft")
    energy = np.maximum(signal.fftconvolve(np.abs(bb)**2, np.ones(600), mode="valid"), 1.)
    rho = np.abs(correlation)/np.sqrt(energy*600)
    peaks, _ = signal.find_peaks(rho, height=.7, prominence=.2, distance=.5*BB_FS)
    groups = np.split(peaks, np.flatnonzero(np.diff(peaks) > 1.2*BB_FS)+1)
    compressed = signal.correlate(bb, replica*signal.windows.hann(600, sym=False),
                                  mode="valid", method="fft")
    rows, profiles = [], []
    delays = np.arange(-60, 481)
    for group_index, group in enumerate(groups):
        for frame_index, peak in enumerate(group):
            def cut(a, b):
                return inband[peak+round(a*BB_FS):peak+round(b*BB_FS)]
            windows = {"pre_chirp": (-.04, -.015), "post_chirp": (.065, .09),
                       "post_training": (.36, .40), "data": (.445, .625),
                       "post_data": (.675, .695)}
            power = {k: float(np.mean(np.abs(cut(*v))**2)) for k, v in windows.items()}
            # The spectrogram/envelope shows residual chirp energy in the first
            # post-chirp gap. Keep that diagnostic but exclude it from the
            # background proxy, using three separated nominal guard windows.
            noise = (power["pre_chirp"]*.025 + power["post_training"]*.04
                     + power["post_data"]*.02)/.085
            contrast = 10*np.log10(power["data"]/noise)
            chirp_tail_noise = (power["post_chirp"]*.025 + power["post_training"]*.04)/.065
            tone = cut(.108, .175)
            t = np.arange(len(tone))/BB_FS
            fit = np.polyfit(t, np.unwrap(np.angle(tone)), 1, w=np.abs(tone))
            shape = compressed[peak+delays]
            profile = np.abs(shape)**2
            main = (delays >= -6) & (delays <= 6)
            late = (delays >= 12) & (delays < 360)
            rows.append(dict(round=group_index+1, index_in_round=frame_index+1,
                             sample_concatenated_lfm_start_s=float(peak/BB_FS),
                             normalized_lfm_correlation=float(rho[peak]),
                             tone_offset_hz=float(fit[0]/(2*np.pi)),
                             data_to_quiet_power_db=float(contrast), window_powers=power,
                             background_proxy_windows=["pre_chirp", "post_training", "post_data"],
                             with_chirp_tail_power_contrast_db=float(10*np.log10(power["data"]/chirp_tail_noise)),
                             late_to_main_compressed_energy_db=float(10*np.log10(profile[late].sum()/profile[main].sum())),
                             strongest_late_peak_db=float(10*np.log10(profile[late].max()/profile[main].max())),
                             has_payload_energy=bool(contrast > 6)))
            if contrast > 6:
                profiles.append(profile/profile.max())
    full = [r for r in rows if r["has_payload_energy"]]
    empty = [r for r in rows if not r["has_payload_energy"]]
    result = dict(kind="recording_signal_diagnostics_not_equalizer_evaluation",
                  format_interpretation="8 little-endian uint32 header words (counter + seven zeros), then 8192 tagged words; sign-extend low 24 bits",
                  files=files, fs_hz=FS, baseband_fs_hz=BB_FS,
                  sample_concatenated_duration_s=len(samples)/FS,
                  detected_lfm_headers=len(rows), round_sizes=[len(g) for g in groups],
                  payload_energy_candidates=len(full), header_only_candidates=len(empty),
                  payload_energy_threshold_db=6.,
                  summaries={key: quantiles([r[key] for r in full]) for key in
                             ["normalized_lfm_correlation", "tone_offset_hz", "data_to_quiet_power_db",
                              "late_to_main_compressed_energy_db", "strongest_late_peak_db"]},
                  frame_interval_s=quantiles(np.concatenate([np.diff(g)/BB_FS for g in groups if len(g)>1])),
                  per_frame=rows,
                  limitations=[
                      "Container interpretation inferred from consistent counters, tags, and recovered waveforms; no vendor format document was provided.",
                      "Sample-concatenated time is not verified wall-clock time. A near-750 ms chirp interval crosses the file boundary.",
                      "99 payload-energy candidates are not 99 verified decodes or verified bit-label alignments.",
                      "Power contrast pools pre-chirp, post-training and post-data guard windows. Post-chirp guard is excluded due to visible reverberation. Remaining windows may also contain echoes or interference; this is not calibrated SNR.",
                      "Hann-weighted chirp compression is a delay-response proxy, not calibrated CSI, and includes the replica autocorrelation and noise.",
                      "Tone offset includes oscillator and common Doppler effects and is not Doppler spread.",
                      "No BER, FER, channel NMSE, or frozen-model transfer performance has been measured."])
    dump(out/"recording_analysis.json", result)

    fig, ax = plt.subplots(2, 2, figsize=(10, 6.6), constrained_layout=True)
    p = int(groups[0][0]); start = p-round(.05*BB_FS); segment = bb[start:start+round(.8*BB_FS)]
    f, t, z = signal.stft(segment, BB_FS, nperseg=256, noverlap=224,
                          return_onesided=False, boundary=None)
    f = np.fft.fftshift(f)+24000; z = np.fft.fftshift(z, axes=0)
    power_db = 20*np.log10(np.maximum(np.abs(z), 1e-9)); power_db -= power_db.max()
    im=ax[0,0].pcolormesh(t, f/1000, power_db, cmap="magma", vmin=-50, vmax=0, shading="auto")
    ax[0,0].set(ylim=(20,28), xlabel="Time from nominal frame start (s)", ylabel="Frequency (kHz)", title="(a) Recovered first-frame spectrogram")
    fig.colorbar(im, ax=ax[0,0], label="Relative magnitude (dB)")
    profile=np.asarray(profiles); q=np.quantile(profile,[.1,.5,.9],axis=0)
    ax[0,1].fill_between(delays/12,10*np.log10(q[0].clip(1e-8)),10*np.log10(q[2].clip(1e-8)),color="#dceaf3",label="10--90% of 99 candidates")
    ax[0,1].plot(delays/12,10*np.log10(q[1].clip(1e-8)),color="#245a87",label="Median")
    ax[0,1].set(xlim=(-1,30),ylim=(-60,2),xlabel="Delay relative to strongest arrival (ms)",ylabel="Peak-normalized power (dB)",title="(b) LFM-compressed response proxy")
    ax[0,1].legend(fontsize=8,loc="upper right")
    for group_index in range(1,len(groups)+1):
        rr=[r for r in full if r["round"]==group_index]
        c=[r["data_to_quiet_power_db"] for r in rr]
        d=[r["tone_offset_hz"] for r in rr]
        ax[1,0].scatter(np.repeat(group_index,len(c)),c,s=10,color="#245a87",alpha=.45)
        ax[1,0].plot(group_index,np.median(c),"_",color="#d64d45",markersize=14,markeredgewidth=2)
        ax[1,1].scatter(np.repeat(group_index,len(d)),d,s=10,color="#239687",alpha=.45)
        ax[1,1].plot(group_index,np.median(d),"_",color="#d64d45",markersize=14,markeredgewidth=2)
    ax[1,0].set(xlabel="Transmission round",ylabel="Data / quiet power (dB)",title="(c) Band-limited power contrast, not SNR")
    ax[1,1].set(xlabel="Transmission round",ylabel="Tone offset from 24 kHz (Hz)",title="(d) Common offset, not Doppler spread")
    for a in ax.flat:
        a.grid(alpha=.2)
    fig.suptitle("DJK recording diagnostics | no equalizer or decoder evaluated",fontsize=13)
    fig.savefig(out/"djk_signal_diagnostics.png",dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,3.2),constrained_layout=True)
    for name,bounds in {"pre-chirp quiet":(-.04,-.015),"post-chirp quiet":(.065,.09),"post-training quiet":(.36,.40),"data":(.445,.625)}.items():
        ax.axvspan(bounds[0]+.05,bounds[1]+.05,alpha=.12,label=name)
    block=24; count=len(segment)//block
    time=(np.arange(count)+.5)*block/BB_FS
    v=np.sqrt(np.mean(abs(segment[:count*block].reshape(count,block))**2,axis=1))
    ax.plot(time,20*np.log10(v.clip(1)),lw=1,color="#245a87")
    ax.set(xlabel="Time from nominal frame start (s)",ylabel="Baseband RMS (dB, ADC counts)",title="Diagnostic window audit (first candidate frame)")
    ax.legend(fontsize=7,ncol=4)
    fig.savefig(out/"window_audit.png",dpi=160)
    plt.close(fig)
    return result


def analyze_watermark(out):
    from data_generation.src.channel.watermark.replay import WatermarkArchive
    rows=[]
    for name in ["NOF1","NCS1"]:
        archive=WatermarkArchive(ROOT/"outputs/smoke/compact_cpu_smoke.zip",name)
        for number in [1,2,3]:
            r=archive.load(number); power=np.mean(abs(r.h)**2,axis=0); w=power/power.sum()
            delay=np.arange(len(w))/r.fs_delay; mean=float(w@delay)
            q=np.interp([.05,.95],np.cumsum(w),delay)
            lag=round(.128*r.fs_time)
            coherence=abs(np.sum(r.h[lag:]*r.h[:-lag].conj()))/np.sqrt(np.sum(abs(r.h[lag:])**2)*np.sum(abs(r.h[:-lag])**2))
            rows.append(dict(channel=name,record=number,source=r.source,sha256=r.sha256,
                             rms_delay_ms=float(np.sqrt(w@(delay-mean)**2)*1000),
                             delay_energy90_span_ms=float((q[1]-q[0])*1000),
                             actual_correlation_lag_s=lag/r.fs_time,
                             pooled_correlation_abs=float(coherence)))
    result=dict(kind="descriptive_subset_of_existing_watermark_impulse_responses",
                records=rows,limitations=["Only records 001--003 in the local smoke archive; not statistics of the full benchmark.",
                "Delay-power metrics include the measurement floor. Neither these metrics nor their correlation estimator is directly comparable to a chirp-compression proxy."])
    dump(out/"watermark_subset_analysis.json",result)
    return result


def make_reference_scenarios(out):
    reports={}
    sources={}
    for ch in ["nof1","ncs1"]:
        p=ROOT/f"outputs/experiment/equalizer_comparison_mmse_dfe_20261031/{ch}/metrics.json"
        reports[ch]=normalize_report_labels(json.loads(p.read_text(encoding="utf-8")))
        sources[ch]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    snr=np.array(reports["nof1"]["snr_db"],float)
    # Uniform, auditable hypothetical construction; not fitted to DJK observations.
    # The interval across w is a sensitivity range, never a confidence interval.
    weights=[0.,.25,.5]
    scenario={}
    for method in METHODS:
        a=np.array(reports["nof1"]["curves"][method]["ber"])
        b=np.array(reports["ncs1"]["curves"][method]["ber"])
        scenario[method]={"NOF1_verified_simulation":a.tolist(),"NCS1_verified_simulation":b.tolist(),
                          "hypothetical_ber_by_weight":{str(w):np.exp((1-w)*np.log(a)+w*np.log(b)).tolist() for w in weights}}
    result=dict(kind="HYPOTHETICAL_SIMULATION_REFERENCE_NOT_DJK_RESULTS",
                title="Illustrative reference scenario -- not measured or validated on DJK",
                snr_db=snr.tolist(),weights=weights,nominal_weight=.25,
                formula="p_w(s) = exp((1-w)*log(p_NOF1(s)) + w*log(p_NCS1(s)))",
                weight_interpretation="Arbitrary sensitivity coordinate between existing simulation curves, not a measured channel parameter or a probability.",
                comparison_snr_db=20.,curves=scenario,
                source_reports=sources,
                limitations=["No DJK equalizer inference was run.","The 0--0.5 weight range is an explicitly chosen illustration, not estimated from the recordings.",
                "Existing curves use separately trained matched-channel models and reference-assisted compensated CSI; they cannot predict frozen-model transfer to raw receiver estimates.",
                "DJK BER may lie outside this range or reverse the ranking. FER and DJK CSI NMSE are intentionally not estimated.",
                "Nominal weight 0.25 is solely for plot layout and hypothesis discussion. This artifact is not field-test evidence."])
    dump(out/"HYPOTHETICAL_reference_scenario.json",result)
    fig,ax=plt.subplots(figsize=(5.6,3.8),constrained_layout=True)
    for method,color in zip(METHODS,COLORS):
        values=scenario[method]["hypothetical_ber_by_weight"]["0.25"]
        ax.semilogy(snr,values,"o--",markersize=3.3,lw=1.5,color=color,label=method)
    ax.set(xlabel="Assumed simulation SNR (dB)",ylabel="Hypothetical BER",xlim=(0,30),ylim=(1e-3,.4))
    ax.grid(which="both",alpha=.18);ax.legend(fontsize=7,ncol=2,loc="lower left")
    ax.set_title("ILLUSTRATIVE ONLY — NO DJK BER MEASURED",fontsize=10,color="#a6312c",weight="bold",pad=12)
    ax.text(.5,.54,"NOT FIELD-TEST RESULTS",transform=ax.transAxes,ha="center",va="center",rotation=14,
            fontsize=15,color="#a6312c",alpha=.20,weight="bold")
    fig.supxlabel("Log interpolation of existing NOF1/NCS1 simulations, w = 0.25",fontsize=8)
    fig.savefig(out/"HYPOTHETICAL_ber_reference.png",dpi=220)
    fig.savefig(out/"HYPOTHETICAL_ber_reference.svg")
    plt.close(fig)
    index=list(snr).index(20.)
    cells=[]
    for method in METHODS:
        row=scenario[method]
        cells.append([method,
                      f"{row['NOF1_verified_simulation'][index]:.3e}",
                      f"{row['NCS1_verified_simulation'][index]:.3e}",
                      f"{row['hypothetical_ber_by_weight']['0.25'][index]:.3e}"])
    fig,ax=plt.subplots(figsize=(8.3,3.1),constrained_layout=True)
    ax.axis("off")
    table=ax.table(cellText=cells,colLabels=["Method","NOF1 simulation","NCS1 simulation","Hypothetical w=0.25"],
                   cellLoc="center",loc="center",colWidths=[.25,.24,.24,.27])
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.55)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#d3dce3")
        if r==0:
            cell.set_facecolor("#e8eff5");cell.set_text_props(weight="bold")
        elif c==3:
            cell.set_facecolor("#fff0ed")
    ax.set_title("BER at assumed 20 dB | final column is NOT a DJK result",fontsize=11,color="#a6312c",weight="bold")
    fig.supxlabel("No equalizer/LDPC evaluation on DJK. Hypothetical interpolation is not a validated prediction.",fontsize=8)
    fig.savefig(out/"HYPOTHETICAL_reference_table.png",dpi=200)
    plt.close(fig)
    # Preserve the existing NMSE evidence without inventing a DJK CSI reference.
    fig,ax=plt.subplots(figsize=(5.6,3.6),constrained_layout=True)
    for ch,color in [("nof1","#245a87"),("ncs1","#d64d45")]:
        nmse=reports[ch]["nmse"]
        ax.plot(snr,nmse["mean_nmse_db"],"o--",color=color,label=f"{ch.upper()} initial CSI")
        ax.plot(snr,nmse["corrected_mean_nmse_db"],"s-",color=color,label=f"{ch.upper()} CRENet")
    ax.set(xlabel="Simulation SNR (dB)",ylabel="Mean full-matrix CSI NMSE (dB)",title="Verified simulation references — no DJK CSI truth")
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/"verified_simulation_nmse_reference.png",dpi=180)
    plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/experiment/djk_feasibility_20261101")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    measured=analyze_recordings(args.output)
    reference=analyze_watermark(args.output)
    scenario=make_reference_scenarios(args.output)
    print(json.dumps(dict(output=str(args.output),detected_headers=measured["detected_lfm_headers"],
                         payload_candidates=measured["payload_energy_candidates"],summaries=measured["summaries"],
                         watermark_records=len(reference["records"]),hypothetical_formula=scenario["formula"]),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
