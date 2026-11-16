"""Check frozen deployment against checkpoint inference on validation inputs."""
from src.common.numerics import default_device
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.algorithms.crenet.checkpoint import load_model, validate_dataset
from src.algorithms.crenet.inference import prepare_frozen_inference
from src.algorithms.crenet.model import CRENet
from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset
from src.data.crenet_dense_training import dense_input_batches
from src.data.crenet_training import ReferenceCache
from src.evaluation.fingerprints import DataIdentity, source_fingerprint
from src.evaluation.crenet_budgeted import EVALUATION_CODE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--double", action="store_true", help="Promote the same saved weights for an FP64 algebra check")
    parser.add_argument("--static", action="store_true", help="Also audit the optional static convolution cache")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new equivalence report path")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(args.threads), torch.inference_mode(), CompactDataset(args.dataset/"validation.h5") as ds:
        if ds.split != "validation" or ds.input_view != "compensated":
            raise ValueError("Requires accepted validation inputs")
        guard = DataIdentity({name: args.dataset/name for name in ("validation.h5", "model.h5")})
        model, payload = load_model(args.checkpoint, args.device)
        validate_dataset(payload, ds)
        if args.double:
            model = CRENet(**dict(model.config, precision="complex128")).to(args.device)
            model.load_state_dict(payload["model"], strict=True)
            model.eval().requires_grad_(False)
        optimized = prepare_frozen_inference(copy.deepcopy(model), cache_static_correction=args.static,
                                             reuse_correction_residual=True)
        batches, blocks = 0, 0
        maximum_errors = {key: 0. for key in ("probabilities", "log_probabilities", "symbols", "variances", "bands")}
        atol, rtol = (1e-4, 1e-4) if model.config["precision"] == "complex64" else (1e-12, 1e-12)
        for cpu, keys in dense_input_batches(ds, batch_size=2, frame_indices=[0],
            precision=model.config["precision"], reference_cache=ReferenceCache(64*2**20)):
            source = cpu.pop("initial_channel")
            inputs = {k: v.to(args.device) for k, v in cpu.items()}
            inputs["initial_channel"] = source.dense(args.device, basis=source.basis.to(args.device),
                                                     reference=source.reference.to(args.device))
            expected = model(**inputs, return_diagnostics=False)
            actual = optimized(**inputs, return_diagnostics=False)
            for key in ("probabilities", "log_probabilities", "symbols", "variances", "bands"):
                left, right = getattr(actual, key), getattr(expected, key)
                maximum_errors[key] = max(maximum_errors[key], float((left-right).abs().max()))
                if key == "bands":
                    scale = inputs["initial_channel"].abs().square().sum(-1).mean(-1).sqrt().clamp_min(1e-12)
                    left, right = left/scale[:, None, None, None], right/scale[:, None, None, None]
                torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
            batches += 1
            blocks += len(keys)
        guard.assert_unchanged()
        report = dict(complete=True, bitwise_equal=all(value == 0 for value in maximum_errors.values()),
            allclose=True, atol=atol, rtol=rtol, max_absolute_errors=maximum_errors,
            csi_absolute_tolerance_units="model per-sample channel normalization", cudnn_tf32=False,
            precision=model.config["precision"], diagnostic_promotion=args.double, static_cache=args.static,
            test_access=False, batches=batches, blocks=blocks,
            snr_db=ds.snr_db.tolist(), deterministic=True, purpose="equivalence only, not latency",
            checkpoint_sha256=file_sha256(args.checkpoint), data_identity=guard.files,
            source=source_fingerprint(EVALUATION_CODE), torch=str(torch.__version__), device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        print(f"{ds.channel}: {blocks} real validation blocks, all SNRs, numerically equivalent; {maximum_errors}", flush=True)


if __name__ == "__main__":
    main()
