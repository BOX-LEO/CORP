#!/usr/bin/env python
"""CLI entry point for a YAML-driven experiment sweep.

Usage::

    python run_experiments.py --config configs/experiments/deit_sparsity_sweep.yaml
"""

import argparse
import logging
import sys

from model_prune.experiment import run_experiments


def _parse():
    p = argparse.ArgumentParser(description="Run a sweep of pruning experiments")
    p.add_argument("--config", required=True, help="YAML with an 'experiment' block")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Extra dotted override applied to every run. Repeatable.",
    )
    p.add_argument("--force", action="store_true", help="Ignore cached results")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = _parse()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_experiments(args.config, extra_overrides=args.overrides, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
