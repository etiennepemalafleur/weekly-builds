"""Scaling up: the same from-scratch net on real handwritten digits.

The 2-D toys prove the network can bend a boundary; this proves it works on genuine,
high-dimensional data. Each sample is an 8x8 grayscale scan of a handwritten digit
(64 input features, 10 classes) from the classic UCI digits set -- small enough to
train in seconds with hand-written backprop, real enough to be a true classification
task, and shipped as a committed snapshot so nothing is downloaded.

The honest twist comes for free: unlike the toys, real digits are *mostly* linearly
separable, so the plain linear model is already very good and the neural net adds only
a little. Nonlinearity earns its keep when the problem is nonlinear -- not always.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import plots
from .gradcheck import max_relative_error
from .nn import MLP

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "digits.npz"


def load(path=DEFAULT_DATA):
    d = np.load(path)
    return d["X"].astype(float), d["y"].astype(int)


def prepare(seed: int = 0, test_frac: float = 0.2):
    X, y = load()
    X = X / 16.0                                   # pixels are 0..16
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int((1 - test_frac) * len(X))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def confusion_matrix(y_true, y_pred, n_classes: int = 10) -> np.ndarray:
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


def top_confusions(cm: np.ndarray, k: int = 5):
    """The k most common (true -> predicted) mistakes, off the diagonal."""
    pairs = [(i, j, cm[i, j]) for i in range(cm.shape[0]) for j in range(cm.shape[1])
             if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:k]


@dataclass
class DigitResult:
    linear_acc: float
    net_acc: float
    confusion: np.ndarray = field(repr=False)
    net: MLP = field(repr=False)
    X_test: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)
    preds: np.ndarray = field(repr=False)
    loss_history: list = field(repr=False)


def run(hidden=(100,), epochs: int = 300, lr: float = 0.1, seed: int = 0) -> DigitResult:
    Xtr, ytr, Xte, yte = prepare(seed=seed)

    linear = MLP([64, 10], seed=seed)
    linear.train(Xtr, ytr, epochs=epochs, lr=lr, seed=seed)

    net = MLP([64, *hidden, 10], seed=seed, activation="relu")
    hist = net.train(Xtr, ytr, epochs=epochs, lr=lr, seed=seed)

    preds = net.predict(Xte)
    return DigitResult(
        linear_acc=linear.accuracy(Xte, yte), net_acc=net.accuracy(Xte, yte),
        confusion=confusion_matrix(yte, preds), net=net,
        X_test=Xte, y_test=yte, preds=preds, loss_history=hist,
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="neural-net.digits",
        description="Train the from-scratch net on real handwritten digits.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=100, help="Hidden-layer width.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--assets-dir", default="assets")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    err = max_relative_error(sizes=(64, 16, 10), activation="relu", seed=args.seed)
    print(f"\nGradient check (ReLU): max relative error = {err:.2e}  "
          f"({'PASS' if err < 1e-5 else 'FAIL'})")

    res = run(hidden=(args.hidden,), epochs=args.epochs, seed=args.seed)
    print(f"\nHandwritten digits (8x8, 10 classes), test accuracy:")
    print(f"  linear model:  {res.linear_acc:.1%}")
    print(f"  neural net:    {res.net_acc:.1%}  (ReLU, {args.hidden} hidden units)")
    print(f"  -> on real digits the linear model is already strong, so the net adds "
          f"just {res.net_acc - res.linear_acc:+.1%}. Nonlinearity helps most when "
          f"the data needs it; clean digits barely do.")

    print("\nThe net's most common confusions (true -> predicted):")
    for t, p, c in top_confusions(res.confusion, k=5):
        print(f"  {t} -> {p}:  {c}")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    charts = [
        plots.plot_digit_samples(res.X_test, res.y_test, assets / "digit_samples.png"),
        plots.plot_confusion(res.confusion, assets / "digit_confusion.png",
                             accuracy=res.net_acc),
        plots.plot_misclassified(res.X_test, res.y_test, res.preds,
                                 assets / "digit_mistakes.png"),
        plots.plot_hidden_weights(res.net, assets / "digit_weights.png"),
    ]
    print("\nSaved charts:")
    for c in charts:
        print(f"  {c}")
    print()


if __name__ == "__main__":
    main()
