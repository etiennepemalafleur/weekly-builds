"""Command-line entry point for the neural-net-from-scratch study."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import plots
from .experiment import run_all
from .gradcheck import max_relative_error


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="neural-net",
        description="A neural network from scratch: gradient-checked backprop, and "
        "when its nonlinearity actually beats a linear model.",
    )
    p.add_argument("--epochs", type=int, default=400, help="Training epochs.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    err = max_relative_error(seed=args.seed)
    print(f"\nGradient check: max relative error = {err:.2e}  "
          f"({'PASS -- backprop is correct' if err < 1e-5 else 'FAIL'})")

    comps = run_all(seed=args.seed, epochs=args.epochs)
    print("\nLinear model vs. neural net (test accuracy):")
    print(f"  {'dataset':10s} {'linear':>8s} {'neural net':>12s}")
    for name, c in comps.items():
        print(f"  {name:10s} {c.linear_acc:8.0%} {c.net_acc:12.0%}")

    gains = [c.net_acc - c.linear_acc for c in comps.values()]
    print(f"\n  On these nonlinear problems the net adds "
          f"{sum(gains) / len(gains):.0%} accuracy on average -- because a straight "
          f"line simply can't separate the classes.")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    charts = [
        plots.plot_decision_grid(comps, assets / "decision_boundaries.png"),
        plots.plot_loss(comps, assets / "loss.png"),
    ]
    print("\nSaved charts:")
    for c in charts:
        print(f"  {c}")
    print()
