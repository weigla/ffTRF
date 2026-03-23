#!/usr/bin/env python3
"""Compare ffTRF kernels against time-domain ridge and mTRF.

Examples
--------
Run from the repository root with the local package installed:

    python examples/compare_with_mtrf.py --output artifacts/kernel_comparison.png --no-show

With Pixi:

    pixi run -e compare compare-demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from comparison_utils import compare_simulated_kernels, plot_kernel_comparison


def parse_optional_int(value: str) -> int | None:
    """Parse an integer CLI value, allowing ``none``."""

    if value.lower() == "none":
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated data, fit ffTRF and time-domain references, "
            "and plot the recovered kernels."
        )
    )
    parser.add_argument("--fs", type=float, default=1_000.0, help="Sampling rate in Hz.")
    parser.add_argument("--n-trials", type=int, default=8, help="Number of simulated trials.")
    parser.add_argument("--n-samples", type=int, default=4096, help="Samples per trial.")
    parser.add_argument("--tmin", type=float, default=0.0, help="Kernel start lag in seconds.")
    parser.add_argument("--tmax", type=float, default=0.040, help="Kernel stop lag in seconds.")
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-3,
        help="Ridge regularization passed to all fitted models.",
    )
    parser.add_argument("--noise-scale", type=float, default=0.05, help="Gaussian noise SD.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--segment-length",
        type=parse_optional_int,
        default=None,
        help="Optional segment length for ffTRF spectral estimation, or 'none'.",
    )
    parser.add_argument("--overlap", type=float, default=0.0, help="Segment overlap fraction.")
    parser.add_argument(
        "--window",
        choices=["none", "hann"],
        default="none",
        help="Window for ffTRF segments.",
    )
    parser.add_argument(
        "--skip-mtrf",
        action="store_true",
        help="Skip the optional mTRF fit even if the package is installed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kernel_comparison.png"),
        help="Path where the figure should be saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive window.",
    )
    return parser


def main() -> None:
    """Run the comparison and render the plot."""

    parser = build_parser()
    args = parser.parse_args()

    window = None if args.window == "none" else args.window
    result = compare_simulated_kernels(
        fs=args.fs,
        n_trials=args.n_trials,
        n_samples=args.n_samples,
        tmin=args.tmin,
        tmax=args.tmax,
        regularization=args.regularization,
        noise_scale=args.noise_scale,
        seed=args.seed,
        segment_length=args.segment_length,
        overlap=args.overlap,
        window=window,
        include_mtrf=not args.skip_mtrf,
    )

    print("Kernel correlations")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")
    if result.mtrf_kernel is None and not args.skip_mtrf:
        print("  mTRF was not available; only ffTRF and time-domain ridge were compared.")

    plot_kernel_comparison(
        result,
        output_path=args.output,
        show=not args.no_show,
    )
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
