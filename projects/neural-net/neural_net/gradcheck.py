"""Prove the hand-derived gradients are right by comparing them to numerical ones.

Backprop is easy to get subtly wrong, and a wrong gradient still "trains" -- just
badly. The honest check is the definition of a derivative: nudge each parameter by a
tiny amount, see how the loss changes, and confirm that finite-difference slope
matches what backprop claims. If the relative difference is around 1e-7, the analytic
gradients are correct to machine precision.
"""

from __future__ import annotations

import numpy as np

from .nn import MLP, one_hot


def max_relative_error(sizes=(2, 8, 3), n: int = 20, eps: float = 1e-6,
                       seed: int = 0) -> float:
    """Largest relative error between analytic and numerical gradients.

    Checks every weight and bias on a small random problem. A value near 1e-7 means
    backprop is implemented correctly.
    """
    rng = np.random.default_rng(seed)
    net = MLP(list(sizes), seed=seed)
    X = rng.normal(size=(n, sizes[0]))
    y = rng.integers(0, sizes[-1], size=n)
    Y = one_hot(y, sizes[-1])

    dW, db, _ = net.gradients(X, Y)

    worst = 0.0
    params = [(net.W, dW), (net.b, db)]
    for tensors, grads in params:
        for l in range(len(tensors)):
            flat = tensors[l].ravel()
            gflat = grads[l].ravel()
            for i in range(flat.size):
                orig = flat[i]
                flat[i] = orig + eps
                plus = net.loss(net.forward(X)[0], Y)
                flat[i] = orig - eps
                minus = net.loss(net.forward(X)[0], Y)
                flat[i] = orig
                numeric = (plus - minus) / (2 * eps)
                denom = max(1e-12, abs(numeric) + abs(gflat[i]))
                worst = max(worst, abs(numeric - gflat[i]) / denom)
    return worst
