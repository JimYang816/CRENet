"""Full test-split B16 evaluation following explicit reference-assisted calibration."""

import csv
from pathlib import Path
import numpy as np

from data_generation.src.dataset.four_data_targets import channel_targets
from src.common.experiment import write_json
from src.evaluation.comparison import evaluate
from src.evaluation.registry import CURVES


def test_datasets(dataset_root, run, targets, backend, batch_size=8):
    dataset_root, run = Path(dataset_root), Path(run)
    reports = {}
    for channel in targets["channels"]:
        write_json(run / "progress.json", dict(stage="full_test", channel=channel))
        reports[channel] = evaluate(
            dataset_root / (channel.lower() + "_dddd") / "test.h5",
            run / "test" / channel.lower(),
            backend,
            batch_size=batch_size,
            cache_dir=run / "cache",
            input_view="reference_assisted",
            algorithm_parameters={"banded_lmmse": {"half_bandwidth": 16}},
        )
        np.testing.assert_allclose(
            reports[channel]["nmse"]["mean_nmse_db"],
            channel_targets(targets, channel),
            rtol=0,
            atol=1e-8,
        )
    finish(run, reports, targets)
    write_json(run / "progress.json", dict(stage="complete", complete=True))
    return reports


def finish(run, reports, targets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_json(
        run / "test_metrics.json",
        dict(
            complete=True,
            input_view="reference_assisted",
            reference_uses_full_csi=True,
            half_bandwidth=16,
            targets=targets,
            channels=reports,
        ),
    )
    with (run / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "channel",
                "snr_db",
                "frames",
                "bits",
                "bit_errors",
                "ber",
                "measured_nmse_db",
                "target_nmse_db",
            ]
        )
        for channel, r in reports.items():
            curve = r["curves"][CURVES[0]]
            target = channel_targets(targets, channel)
            for i, snr in enumerate(r["snr_db"]):
                writer.writerow(
                    [
                        channel,
                        snr,
                        r["frame_count"],
                        r["bits_per_snr"],
                        curve["errors"][i],
                        curve["ber"][i],
                        r["nmse"]["mean_nmse_db"][i],
                        target[i],
                    ]
                )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for col, (channel, r) in enumerate(reports.items()):
        x = np.asarray(r["snr_db"])
        measured = np.asarray(r["curves"][CURVES[0]]["ber"])
        floor = 0.5 / r["bits_per_snr"]
        axes[0, col].semilogy(
            x,
            np.maximum(measured, floor),
            "o-",
            color="#2166AC",
            label="Banded LMMSE B=16",
        )
        zero = measured == 0
        axes[0, col].scatter(
            x[zero],
            np.full(zero.sum(), floor),
            facecolors="white",
            edgecolors="#2166AC",
            zorder=4,
        )
        axes[0, col].set(
            title=f"{channel}: {r['frame_count']} frames / SNR", ylabel="BER"
        )
        axes[1, col].plot(
            x,
            channel_targets(targets, channel),
            "x--",
            color="black",
            label="Preset target",
            markersize=8,
        )
        axes[1, col].plot(
            x,
            r["nmse"]["mean_nmse_db"],
            "o-",
            color="#2166AC",
            label="Measured CSI NMSE",
        )
        axes[1, col].set(xlabel="SNR (dB)", ylabel="Mean full-support NMSE (dB)")
        for ax in axes[:, col]:
            ax.set_xticks(x)
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8)
    fig.suptitle(
        "Reference-assisted CSI using full reference matrices | Banded LMMSE B=16\n"
        "DDDD / CP80 ms / Q24 / 192 raw delays | complete test splits"
    )
    for extension in ("png", "pdf"):
        fig.savefig(run / f"comparison.{extension}", dpi=180)
    plt.close(fig)
    for kind in ("ber", "nmse"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        for ax, (channel, r) in zip(axes, reports.items(), strict=True):
            if kind == "ber":
                values = np.asarray(r["curves"][CURVES[0]]["ber"])
                ax.semilogy(
                    r["snr_db"],
                    np.maximum(values, 0.5 / r["bits_per_snr"]),
                    "o-",
                    label="Banded LMMSE B=16",
                )
            else:
                ax.plot(
                    r["snr_db"],
                    channel_targets(targets, channel),
                    "x--",
                    color="black",
                    label="Target",
                )
                ax.plot(r["snr_db"], r["nmse"]["mean_nmse_db"], "o-", label="Measured")
            ax.set(
                title=channel,
                xlabel="SNR (dB)",
                ylabel="BER" if kind == "ber" else "Mean full-support NMSE (dB)",
                xticks=r["snr_db"],
            )
            ax.grid(alpha=0.25, which="both")
            ax.legend()
        fig.suptitle("Full test splits | reference-assisted CSI | Banded LMMSE B=16")
        for extension in ("png", "pdf"):
            fig.savefig(run / f"{kind}_snr.{extension}", dpi=180)
        plt.close(fig)
