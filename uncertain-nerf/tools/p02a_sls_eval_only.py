#!/usr/bin/env python3
"""Load one existing SLS checkpoint and run its audit-only evaluation/timing path."""

from __future__ import print_function

import argparse
import os
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--lower-bound", type=float, required=True)
    parser.add_argument("--upper-bound", type=float, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    sys.path.insert(0, str(source / "examples"))
    sys.path.insert(0, str(source))
    from spotless_trainer import Config, Runner

    cfg = Config(
        disable_viewer=True,
        seed=42,
        ckpt=str(args.checkpoint.expanduser().resolve()),
        data_dir=str(args.data_dir.expanduser().resolve()),
        data_factor=1,
        normalize=True,
        result_dir=str(args.result_dir.expanduser().resolve()),
        train_keyword="clutter",
        test_keyword="extra",
        semantics=True,
        cluster=False,
        max_steps=30000,
        eval_steps=[30000],
        save_steps=[30000],
        loss_type="robust",
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        ubp=False,
    )
    runner = Runner(cfg)
    checkpoint = torch.load(cfg.ckpt, map_location=runner.device)
    for key in runner.splats.keys():
        runner.splats[key].data = checkpoint["splats"][key]
    runner.eval(step=checkpoint["step"])
    print("P02A_SLS_EVAL_ONLY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
