"""Small 2-D classification problems with known nonlinear structure.

Each is chosen so that *no straight line* can separate the classes -- which is the
whole point: it's where a linear model must fail and a neural net can succeed. Being
2-D, they also let us draw the decision boundary and literally see what each model
learned.
"""

from __future__ import annotations

import numpy as np


def xor(n: int = 400, noise: float = 0.3, seed: int = 0):
    """The classic XOR: two classes on opposite diagonal quadrants."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X = X + rng.normal(0, noise * 0.3, size=X.shape)
    return X, y


def moons(n: int = 400, noise: float = 0.2, seed: int = 0):
    """Two interleaving half-moons."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    outer = np.column_stack([np.cos(t1), np.sin(t1)])
    inner = np.column_stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5])
    X = np.vstack([outer, inner]) + rng.normal(0, noise, size=(n, 2))
    y = np.array([0] * n1 + [1] * (n - n1))
    return X, y


def circles(n: int = 400, noise: float = 0.1, seed: int = 0):
    """A small circle of one class inside a ring of the other."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    a1 = rng.uniform(0, 2 * np.pi, n1)
    a2 = rng.uniform(0, 2 * np.pi, n - n1)
    inner = 0.4 * np.column_stack([np.cos(a1), np.sin(a1)])
    outer = 1.0 * np.column_stack([np.cos(a2), np.sin(a2)])
    X = np.vstack([inner, outer]) + rng.normal(0, noise, size=(n, 2))
    y = np.array([0] * n1 + [1] * (n - n1))
    return X, y


def spirals(n: int = 400, noise: float = 0.2, seed: int = 0):
    """Two intertwined spiral arms."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t = np.sqrt(rng.uniform(0, 1, n1)) * 3 * np.pi
    arm0 = np.column_stack([t * np.cos(t), t * np.sin(t)]) / 10
    arm1 = np.column_stack([t * np.cos(t + np.pi), t * np.sin(t + np.pi)]) / 10
    X = np.vstack([arm0, arm1]) + rng.normal(0, noise * 0.1, size=(2 * n1, 2))
    y = np.array([0] * n1 + [1] * n1)
    return X, y


DATASETS = {"xor": xor, "moons": moons, "circles": circles, "spirals": spirals}
