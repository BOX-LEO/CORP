#!/usr/bin/env python
"""CLI entry point for a single pruning run driven by a YAML config.

Usage::

    python run_prune.py --config configs/deit_tiny_mlp.yaml
    python run_prune.py --config configs/deit_tiny_mlp.yaml --set pruning.sparsity=0.4
    python run_prune.py --config configs/vit_base_both.yaml --eval-no-comp
"""

import argparse
import logging
import sys

from model_prune.runner import run_prune


def _parse():
    p = argparse.ArgumentParser(description="Activation-based structured pruning")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Dotted override, e.g. pruning.sparsity=0.4. Repeatable.",
    )
    p.add_argument(
        "--eval-no-comp", action="store_true",
        help="Additionally run and evaluate with compensation disabled (slow).",
    )
    p.add_argument("--log-level", default="INFO", help="Python logging level")
    return p.parse_args()


def main() -> int:
    args = _parse()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    report = run_prune(args.config, overrides=args.overrides, eval_no_comp=args.eval_no_comp)
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())
