"""Audit matched-precision receiver estimation runtime on validation inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT
from src.common.numerics import DenseBackend
from src.evaluation.dpss_precision_timing import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--windows", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    source = args.source.resolve()
    if (ROOT/"outputs/experiment").resolve() not in source.parents:
        p.error("Source must be an experiment output under this project")
    with threadpool_limits(args.threads):
        backend = DenseBackend("cuda")
        backend.torch.set_num_threads(args.threads)
        run({"NOF1": ROOT/"data/processed/nof1_compact_20261026_final",
             "NCS1": ROOT/"data/processed/ncs1_compact_20261026_nmse"}, source,
            source/"precision_runtime_audit.json", backend, windows=args.windows, repeats=args.repeats)


if __name__ == "__main__":
    main()
