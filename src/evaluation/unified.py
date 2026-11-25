"""One raw-CSI report for the configured traditional/learned detectors."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from src.data.compact import CompactDataset
from src.evaluation import comparison
from src.evaluation.comparison_plots import publish_report
from src.evaluation.fingerprints import DataIdentity
from src.evaluation.registry import SUPPORTED_CURVES
from src.evaluation.result_cache import FileLock

LEARNED = ("crenet",)
ORDER = SUPPORTED_CURVES


def _preflight(dataset_path, checkpoints, backend):
    if set(checkpoints)-set(LEARNED):
        raise ValueError("Unknown learned comparator checkpoint")
    for name, path in checkpoints.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}. Train it or select --baselines-only.")
    with CompactDataset(dataset_path) as ds:
        if ds.split != "test":
            raise ValueError("Unified comparison requires the test split")
        if not ds._config.get("smoke") and (ds.frame_count != 600 or len(ds._indices) != 512):
            raise ValueError("Unified comparison requires all 600 test frames and 512 data carriers")
        for path in checkpoints.values():
            from src.algorithms.crenet.checkpoint import load_model, validate_dataset
            model, payload = load_model(path, backend.device)
            validate_dataset(payload, ds)
            del model, payload
        return bool(ds._config.get("smoke"))


def evaluate(dataset_path, run, backend, *, checkpoints=None, batch_size=8, cache_dir=None,
             algorithm_parameters=None, force=(), input_view="raw", bootstrap_samples=2000):
    if input_view != "raw":
        raise ValueError("Unified experiment requires raw CSI")
    checkpoints = dict(checkpoints or {})
    forced = {force} if isinstance(force, str) else set(force)
    smoke = _preflight(dataset_path, checkpoints, backend)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".unified_evaluation.lock"):
        guard = DataIdentity({name: Path(dataset_path).parent/name for name in ("test.h5", "model.h5")})
        baseline = comparison.evaluate(dataset_path, run/"baselines", backend, batch_size=batch_size,
            cache_dir=cache_dir, algorithm_parameters=algorithm_parameters, force=forced-{"crenet"}, input_view=input_view)
        report = dict(baseline, schema="watermark-unified-comparison-v1", smoke=smoke, curves=dict(baseline["curves"]))
        components = {}
        if "crenet" in checkpoints:
            from src.evaluation.crenet_cache import evaluate as evaluate_crenet
            components["crenet"] = evaluate_crenet(dataset_path, checkpoints["crenet"], run/"crenet", backend,
                batch_size=batch_size, cache_dir=cache_dir, bootstrap_samples=bootstrap_samples,
                force=bool(forced.intersection(("crenet", "all"))))
            crenet = components["crenet"]
            report["curves"]["CRENet"] = crenet["curves"]["CRENet"]
            report["nmse"] = dict(baseline["nmse"], corrected_mean_nmse_db=[r["mean_nmse_db"][-1] for r in crenet["per_snr"]])
            report["crenet"] = dict(checkpoint=crenet["checkpoint"], sha256=crenet["checkpoint_sha256"],
                model_config=crenet["model_config"], comparison_report=str((run/"crenet/metrics.json").resolve()),
                evaluation_cache=crenet["evaluation_cache"])
        for name, component in components.items():
            component_data = {key: component["data_identity"].get(key) for key in guard.files}
            if (component_data != guard.files or component["snr_db"] != baseline["snr_db"]
                    or component.get("input_view") != "raw"):
                raise RuntimeError(f"{name} comparison data/view/SNR identity mismatch")
            for label in set(component["curves"]).intersection(report["curves"]):
                if component["curves"][label]["total_bits"] != report["curves"][label]["total_bits"]:
                    raise RuntimeError("Comparators did not score the same number of test bits")
        if baseline["data_identity"] != guard.files:
            raise RuntimeError("Baseline data identity changed during unified evaluation")
        expected_bits = [baseline["bits_per_snr"]]*len(baseline["snr_db"])
        if any(curve["total_bits"] != expected_bits for curve in report["curves"].values()):
            raise RuntimeError("Unified comparison requires equal complete-test bit counts")
        if "crenet" in components:
            initial = [row["mean_nmse_db"][0] for row in components["crenet"]["per_snr"]]
            if not np.allclose(initial, report["nmse"]["mean_nmse_db"], rtol=0, atol=2e-5):
                raise ValueError("CRENet input NMSE disagrees with the shared full-matrix CSI definition")
        report["curves"] = {name: report["curves"][name] for name in ORDER if name in report["curves"]}
        report["comparison_algorithms"] = list(report["curves"])
        report["checkpoint_paths"] = {name: str(Path(path).resolve()) for name, path in checkpoints.items()}
        report["nmse"]["semantics"] = ("Shared input CSI. corrected_mean_nmse_db, when present, "
            "scores the full corrected CSI of CRENet.")
        guard.assert_unchanged()
        publish_report(run, report)
        _write_html(run, report)
        return report


def _write_html(run, report):
    headings = "".join(f"<th>{s:g} dB</th>" for s in report["snr_db"])
    rows = "".join("<tr><th>"+html.escape(name)+"</th>"+"".join(f"<td>{v:.6g}</td>" for v in curve["ber"])+"</tr>"
                   for name, curve in report["curves"].items())
    nmse_fields = (("输入 CSI", "mean_nmse_db"), ("CRENet", "corrected_mean_nmse_db"))
    nmse_rows = "".join(f"<tr><th>{label}</th>"+"".join(f"<td>{v:.3f}</td>" for v in report["nmse"][field])+"</tr>"
                        for label, field in nmse_fields if field in report["nmse"])
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{report['channel']} 均衡算法对比</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5dc;padding:8px}}</style>
<h1>{report['channel']}：统一均衡评测</h1><p>固定测试划分、原始估计 CSI、D1/D2，共 {report['bits_per_snr']:,} 比特 / SNR。模型选择仅使用训练集和验证集。</p>
<table><tr><th>算法 BER</th>{headings}</tr>{rows}</table><img src="ber_snr.png">
<table><tr><th>CSI 平均 NMSE (dB)</th>{headings}</tr>{nmse_rows}</table><img src="nmse_snr.png">
<p>零误码用空心三角在 0.5 / 测试比特数处显示。输入与各算法修正后的 CSI 均按逐帧、逐 D 的完整矩阵线性相对误差先平均，再转为 dB。</p>
<p>检查点、数据、代码与缓存来源见 metrics.json。</p></html>'''
    (run/"report.html").write_text(body, encoding="utf-8")
