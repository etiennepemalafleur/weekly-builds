"""Offline tests — the gradient check is the one that proves backprop is correct."""

from __future__ import annotations

import numpy as np

from neural_net import datasets as ds
from neural_net import gradcheck as gc
from neural_net import nn


# --- correctness: the headline test ---------------------------------------- #

def test_gradient_check_passes():
    # Analytic gradients must match numerical gradients to near machine precision.
    assert gc.max_relative_error(sizes=(2, 8, 3), seed=0) < 1e-5
    assert gc.max_relative_error(sizes=(3, 6, 5, 2), seed=1) < 1e-5


# --- building blocks -------------------------------------------------------- #

def test_softmax_is_a_distribution():
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = nn.softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_one_hot():
    oh = nn.one_hot(np.array([0, 2, 1]), 3)
    assert np.array_equal(oh, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])


def test_forward_shapes():
    net = nn.MLP([2, 5, 3], seed=0)
    probs, acts = net.forward(np.zeros((7, 2)))
    assert probs.shape == (7, 3)
    assert acts[0].shape == (7, 2) and acts[-1].shape == (7, 3)


def test_mlp_rejects_too_few_sizes():
    try:
        nn.MLP([4])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a single-size network")


# --- learning --------------------------------------------------------------- #

def test_training_reduces_loss():
    X, y = ds.moons(n=300, seed=0)
    net = nn.MLP([2, 16, 2], seed=0)
    hist = net.train(X, y, epochs=100, lr=0.3, seed=0)
    assert hist[-1] < hist[0]


def test_network_learns_xor():
    X, y = ds.xor(n=600, seed=0)
    net = nn.MLP([2, 16, 16, 2], seed=0)
    net.train(X[:450], y[:450], epochs=400, lr=0.3, seed=0)
    assert net.accuracy(X[450:], y[450:]) > 0.9


def test_network_beats_linear_on_nonlinear_data():
    X, y = ds.circles(n=600, seed=0)
    Xtr, ytr, Xte, yte = X[:450], y[:450], X[450:], y[450:]
    linear = nn.MLP([2, 2], seed=0)
    linear.train(Xtr, ytr, epochs=400, lr=0.3, seed=0)
    net = nn.MLP([2, 16, 16, 2], seed=0)
    net.train(Xtr, ytr, epochs=400, lr=0.3, seed=0)
    assert net.accuracy(Xte, yte) > linear.accuracy(Xte, yte) + 0.2


def test_zero_hidden_layer_solves_linearly_separable_data():
    # With no hidden layer the MLP is logistic regression; it should ace an easy,
    # linearly separable problem -- confirming the baseline is a real linear model.
    rng = np.random.default_rng(0)
    a = rng.normal([-2, -2], 0.5, size=(150, 2))
    b = rng.normal([2, 2], 0.5, size=(150, 2))
    X = np.vstack([a, b])
    y = np.array([0] * 150 + [1] * 150)
    linear = nn.MLP([2, 2], seed=0)
    linear.train(X, y, epochs=200, lr=0.3, seed=0)
    assert linear.accuracy(X, y) > 0.95


def test_training_is_reproducible():
    X, y = ds.moons(n=200, seed=0)
    a = nn.MLP([2, 8, 2], seed=3); a.train(X, y, epochs=50, lr=0.3, seed=3)
    b = nn.MLP([2, 8, 2], seed=3); b.train(X, y, epochs=50, lr=0.3, seed=3)
    assert np.array_equal(a.predict(X), b.predict(X))
