#!/usr/bin/env python3
"""Run all FrequencyTRF example scripts and save their figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from example_backward_decoding import run_example as run_backward_example
from example_multi_trial_single_channel import run_example as run_multi_trial_example
from example_multifeature_multichannel import run_example as run_multifeature_example
from example_single_trial_single_channel import run_example as run_single_trial_example


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/examples"),
        help="Directory where all example figures should be written.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figures without opening plot windows.",
    )
    return parser


def main() -> None:
    """Run every example once."""

    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    run_single_trial_example(
        output_path=args.output_dir / "single_trial_single_channel.png",
        show=show,
    )
    run_multi_trial_example(
        output_path=args.output_dir / "multi_trial_single_channel.png",
        show=show,
    )
    run_multifeature_example(
        output_path=args.output_dir / "multifeature_multichannel.png",
        show=show,
    )
    run_backward_example(
        output_path=args.output_dir / "backward_decoding.png",
        show=show,
    )

    print(f"Saved example figures to {args.output_dir}")


if __name__ == "__main__":
    main()
