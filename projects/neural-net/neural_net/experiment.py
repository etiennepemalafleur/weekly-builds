"""Head-to-head: a linear model vs. the neural net on each nonlinear dataset."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .datasets import DATASETS
from .nn import MLP


@dataclass
class Comparison:
    name: str
    X_train: np.ndarray = field(repr=False)
    y_train: np.ndarray = field(repr=False)
    X_test: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)
    linear: MLP = field(repr=False)
    net: MLP = field(repr=False)
    linear_acc: float
    net_acc: float
    loss_history: list = field(repr=False)


def run(name: str, n: int = 600, hidden=(16, 16), epochs: int = 400,
        lr: float = 0.3, seed: int = 0) -> Comparison:
    """Train a linear model and a neural net on one dataset; report test accuracy."""
    X, y = DATASETS[name](n=n, seed=seed)
    cut = int(0.75 * n)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    n_classes = int(y.max()) + 1

    linear = MLP([2, n_classes], seed=seed)
    linear.train(Xtr, ytr, epochs=epochs, lr=lr, seed=seed)

    net = MLP([2, *hidden, n_classes], seed=seed)
    history = net.train(Xtr, ytr, epochs=epochs, lr=lr, seed=seed)

    return Comparison(
        name=name, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
        linear=linear, net=net,
        linear_acc=linear.accuracy(Xte, yte), net_acc=net.accuracy(Xte, yte),
        loss_history=history,
    )


def run_all(seed: int = 0, **kw) -> dict[str, Comparison]:
    return {name: run(name, seed=seed, **kw) for name in DATASETS}
