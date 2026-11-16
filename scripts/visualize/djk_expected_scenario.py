"""Draw a conditional DJK planning scenario, never empirical field BER.

This is an explicit algebraic sensitivity exercise anchored to verified NOF1
simulations. Channel diagnostics motivate the qualitative scenario only; they
do not identify its BER scaling or transfer multipliers. No detector is run.
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
from src.evaluation.registry import normalize_report_labels


ROOT = Path(__file__).resolve().parents[2]
METHODS = ["Single-tap LMMSE", "Full-matrix LMMSE", "MMSE-DFE",
           "UDNet", "OAMP-Net2", "CRENet"]
LEARNED = {"UDNet", "OAMP-Net2", "CRENet"}


def load_source(path):
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())


def summarize_recording_snr(recording):
    """Recompute a guard-noise proxy; this is not calibrated detector SNR."""
    frames = [row for row in recording["per_frame"] if row["has_payload_energy"]]
    data = np.array([row["window_powers"]["data"] for row in frames])
    noise = np.array([(row["window_powers"]["pre_chirp"]*.025
                       + row["window_powers"]["post_training"]*.04
                       + row["window_powers"]["post_data"]*.02)/.085 for row in frames])
    if not len(frames) or np.any(noise <= 0) or np.any(data <= noise):
        raise ValueError("Invalid powers for the recording SNR proxy")
    per_frame_db = 10*np.log10((data-noise)/noise)
    return dict(
        frame_count=len(frames), band_hz=[21000,27000],
        estimator="10*log10((P_data-P_guard)/P_guard), using duration-weighted guard power",
        averaging="arithmetic mean of per-frame SNR in dB",
        mean_db=float(per_frame_db.mean()),
        mean_db_rounded_2dp=round(float(per_frame_db.mean()),2),
        median_db=float(np.median(per_frame_db)),
        p05_p95_db=np.quantile(per_frame_db,[.05,.95]).tolist(),
        pooled_power_snr_db=float(10*np.log10((data.sum()-noise.sum())/noise.sum())),
        background_windows=["pre_chirp","post_training","post_data"],
        calibrated_simulation_or_detector_snr=False,
    )


def interpolate_ber(grid, curve, snr_db):
    """Interpolate log10(BER) linearly in SNR dB, with no extrapolation."""
    if not grid[0] <= snr_db <= grid[-1] or np.any(np.asarray(curve) <= 0):
        raise ValueError("Reference BER interpolation requires positive in-grid values")
    if snr_db in grid:
        return float(curve[grid.index(snr_db)])
    return float(10**np.interp(snr_db,grid,np.log10(curve)))


def build_scenario(snr_db=None):
    report, report_source = load_source(ROOT/"outputs/experiment/equalizer_comparison_mmse_dfe_20261031/nof1/metrics.json")
    report = normalize_report_labels(report)
    comparison, comparison_source = load_source(ROOT/"outputs/experiment/djk_common_probe_20261101/comparison.json")
    recording, recording_source = load_source(ROOT/"outputs/experiment/djk_feasibility_20261101/recording_analysis.json")
    snr_estimate = summarize_recording_snr(recording)
    uses_recording_mean = snr_db is None
    snr_db = snr_estimate["mean_db_rounded_2dp"] if uses_recording_mean else float(snr_db)
    grid = report["snr_db"]
    at20, at30 = grid.index(20), grid.index(30)
    is_interpolated = snr_db not in grid
    baseline = report["curves"]["Full-matrix LMMSE"]["ber"]
    common = interpolate_ber(grid,baseline,snr_db)-float(baseline[at30])
    if not (0 < common < .5):
        raise ValueError("Unexpected reference contrast")
    rows = []
    for method in METHODS:
        curve = report["curves"][method]["ber"]
        residual = float(curve[at30])
        nominal = common+.25*residual
        low, high = common+.10*residual, common+.50*residual
        # Arbitrary domain-transfer sensitivity, applied symmetrically across
        # learned methods; it is not a fitted uncertainty distribution.
        transfer = [.8, 1.5] if method in LEARNED else [1., 1.]
        low *= transfer[0]
        high *= transfer[1]
        rows.append(dict(method=method, display_name=method,
                         nof1_reference_ber_at_scenario_snr=interpolate_ber(grid,curve,snr_db),
                         nof1_reference_is_interpolated=is_interpolated,
                         verified_nof1_ber_20db=float(curve[at20]), verified_nof1_ber_30db=residual,
                         conditional_scenario_ber=nominal, assumption_sensitivity_low=low,
                         assumption_sensitivity_high=high, transfer_sensitivity_factors=transfer))
    qualitative = {}
    for channel, metrics in comparison["channels"].items():
        qualitative[channel] = {key: metrics[key]["distribution"]["median"] for key in
                                ["windowed_rms_delay_ms", "energy_outside_main_1ms_fraction", "shape_correlation_0.75s"]}
    return dict(
        kind="CONDITIONAL_PLANNING_SCENARIO_NOT_DJK_MEASUREMENTS",
        equalizer_executed=False, empirical_prediction=False,
        assumed_comparable_snr_db=snr_db,
        snr_selection="recording_mean_db_rounded_2dp" if uses_recording_mean else "explicit_scenario_argument",
        recording_snr_estimate=snr_estimate,
        diagnostic_power_contrast_is_not_simulation_snr=True,
        simulation_snr_equivalence_is_assumed=True,
        nof1_reference_interpolation=dict(
            applied=is_interpolated, method="linear interpolation of log10(BER) in SNR dB",
            lower_anchor_db=max(x for x in grid if x<=snr_db),
            upper_anchor_db=min(x for x in grid if x>=snr_db),
        ),
        formula="p_scenario(a) = C + r * p_NOF1(a,30dB), C = p_NOF1(LMMSE,SNR_scenario) - p_NOF1(LMMSE,30dB)",
        common_reference_contrast=common, nominal_residual_retention=.25,
        residual_retention_sensitivity=[.10,.50], learned_transfer_sensitivity=[.8,1.5],
        rows=rows, qualitative_channel_diagnostics=qualitative,
        source_reports=[report_source,comparison_source,recording_source],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assumptions_and_limits=[
            f"Scenario operating point is {snr_db:.2f} dB; the recording-based mean SNR proxy is {snr_estimate['mean_db']:.8f} dB before rounding.",
            "The recording SNR proxy subtracts estimated guard noise from data-window total power. Guards may contain residual echoes or interference.",
            "Matching this in-band SNR proxy to simulation SNR is an unverified assumption, not a calibration.",
            "NOF1 references between sampled SNR points are log-BER interpolations, not new equalizer evaluations.",
            "C is an empirical difference between two NOF1 BER values, not an identified thermal-noise BER.",
            "30-dB BER is used as a residual reference; it is not proven to be an asymptotic error floor or an isolated CSI-error contribution.",
            "The channel analysis motivates a milder-dispersion scenario qualitatively. It does not imply 25% residual retention or any BER improvement percentage.",
            "Retention 10--50% and learned transfer 0.8--1.5 are chosen sensitivity assumptions, not fitted bounds or confidence intervals.",
            "Source learned receivers were trained for NOF1 using reference-assisted compensated CSI. These values do not predict either NOF1- or NCS1-trained frozen checkpoints on DJK.",
            "Nominal ranking is inherited from the NOF1 30-dB anchor by construction, not new evidence that CRENet is best on DJK.",
            "Actual BER can be outside all plotted ranges and method rankings can reverse.",
            "No BER, FER, decoded frame count, or channel NMSE has been measured or inferred for DJK.",
            "For experiment planning and layout preview only; do not publish as field-test results.",
        ],
    )


def draw(scenario, out):
    rows = scenario["rows"]
    snr_db = scenario["assumed_comparable_snr_db"]
    interpolated = scenario["nof1_reference_interpolation"]["applied"]
    labels = [r["display_name"] for r in rows]
    y = np.arange(len(rows))
    nominal = np.array([r["conditional_scenario_ber"] for r in rows])*1000
    low = np.array([r["assumption_sensitivity_low"] for r in rows])*1000
    high = np.array([r["assumption_sensitivity_high"] for r in rows])*1000
    reference = np.array([r["nof1_reference_ber_at_scenario_snr"] for r in rows])*1000
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9})
    fig,ax=plt.subplots(figsize=(7.1,4.0))
    fig.subplots_adjust(left=.235,right=.97,top=.81,bottom=.23)
    ax.errorbar(nominal,y,xerr=[nominal-low,high-nominal],fmt="o",color="#245a87",
                ecolor="#84aac6",capsize=3,lw=2,markersize=5,label="Conditional scenario + assumption range")
    ax.scatter(reference,y,marker="D",s=25,facecolors="none",edgecolors="#9a7652",
               label=f"NOF1 {'interpolated reference' if interpolated else 'verified simulation'}, {snr_db:.2f} dB",zorder=3)
    ax.set(yticks=y,yticklabels=labels,xlabel=r"Pre-decoding BER ($\times 10^{-3}$)",
           xlim=(0,float(max(reference.max(),high.max())*1.10)))
    ax.invert_yaxis();ax.grid(axis="x",alpha=.18);ax.spines[["top","right"]].set_visible(False)
    ax.text(.55,.5,"NOT MEASURED",transform=ax.transAxes,ha="center",va="center",
             rotation=13,color="#b34b42",alpha=.13,fontsize=23,weight="bold")
    ax.legend(loc="lower left",bbox_to_anchor=(-.01,1.01),frameon=False,fontsize=8)
    fig.suptitle(f"DJK planning scenario at {snr_db:.2f} dB",y=.98,fontsize=12,weight="bold")
    fig.text(.5,.052,"Recording mean SNR proxy; equivalence to simulation SNR is assumed.\nNo DJK equalizer run. Ranges vary assumptions; they are NOT confidence intervals.",
             ha="center",va="center",fontsize=8,color="#9d332b")
    fig.savefig(out/"DJK_EXPECTED_SCENARIO_NOT_MEASURED.png",dpi=220)
    fig.savefig(out/"DJK_EXPECTED_SCENARIO_NOT_MEASURED.svg")
    plt.close(fig)

    cells=[]
    for r in rows:
        cells.append([r["display_name"],f"{r['nof1_reference_ber_at_scenario_snr']*1000:.2f}",
                      f"{r['conditional_scenario_ber']*1000:.2f}",
                      f"{r['assumption_sensitivity_low']*1000:.2f} – {r['assumption_sensitivity_high']*1000:.2f}"])
    fig,ax=plt.subplots(figsize=(8.6,3.5))
    fig.subplots_adjust(left=.02,right=.98,top=.83,bottom=.2);ax.axis("off")
    table=ax.table(cellText=cells,colLabels=["Method","NOF1 (interp.)" if interpolated else "NOF1 verified","Conditional scenario","Assumption range"],
                   loc="center",cellLoc="center",colWidths=[.27,.22,.25,.26])
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.55)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor("#d3dce3")
        if i==0:
            cell.set_facecolor("#e9eff4");cell.set_text_props(weight="bold")
        elif j>=2:
            cell.set_facecolor("#fff3e9")
    fig.suptitle(f"SNR working point: {snr_db:.2f} dB | BER " + r"$\times 10^{-3}$" + "\nConditional scenario — NOT DJK algorithm measurements",y=.98,fontsize=11,weight="bold",color="#9d332b")
    fig.text(.5,.06,"NO DJK EQUALIZER RUN. Nominal values and ranges depend on unvalidated assumptions.\nNot a confidence interval or a validated transfer-performance prediction.",ha="center",fontsize=8,color="#9d332b")
    fig.savefig(out/"DJK_EXPECTED_TABLE_NOT_MEASURED.png",dpi=200)
    fig.savefig(out/"DJK_EXPECTED_TABLE_NOT_MEASURED.svg")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/experiment/djk_expectation_mean_snr_18p09_20261101")
    parser.add_argument("--snr-db",type=float,default=None,
                        help="Override the default recording mean SNR proxy rounded to two decimals")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    scenario=build_scenario(args.snr_db)
    (args.output/"CONDITIONAL_SCENARIO_NOT_MEASURED.json").write_text(json.dumps(scenario,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    draw(scenario,args.output)
    print(json.dumps(dict(output=str(args.output),snr_db=scenario["assumed_comparable_snr_db"],
                         common_reference_contrast=scenario["common_reference_contrast"],
                         rows=scenario["rows"]),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
