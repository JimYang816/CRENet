"""Compute paired record-bootstrap intervals from the completed diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

from data_generation.src.config import ROOT
from src.evaluation.dpss_bomp_paired import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20261123)
    args = p.parse_args()
    source = args.source.resolve()
    if (ROOT/"outputs/experiment").resolve() not in source.parents:
        p.error("Source must be an experiment output under this project")
    run({"NOF1": ROOT/"data/processed/nof1_compact_20261026_final",
         "NCS1": ROOT/"data/processed/ncs1_compact_20261026_nmse"}, source,
        source/"paired_record_bootstrap.json", draws=args.draws, seed=args.seed)


if __name__ == "__main__":
    main()
