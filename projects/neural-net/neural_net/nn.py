"""A multilayer perceptron, forward and backward passes written by hand.

No autograd, no framework -- just numpy and the chain rule. A network is a list of
layer sizes, e.g. [2, 16, 16, 2]: two input features, two hidden layers of 16 tanh
units, two output classes. The output layer is a softmax and the loss is cross
entropy, the standard pairing for classification.

The point of doing it by hand is to *understand* backprop, and the honest way to know
the hand-derived gradients are right is to check them against numerical gradients --
see `gradcheck.py`. Setting the number of hidden layers to zero turns this exact class
into plain logistic/softmax regression, which is how the project gets its linear
baseline for free.
"""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)         # stabilize
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


class MLP:
    """A tanh-hidden, softmax-output multilayer perceptron."""

    def __init__(self, sizes: list[int], seed: int = 0):
        if len(sizes) < 2:
            raise ValueError("need at least input and output sizes")
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # --- forward ------------------------------------------------------------ #

    def forward(self, X: np.ndarray):
        """Return (probs, activations) where activations[0] = X, [-1] = probs."""
        acts = [X]
        a = X
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            a = softmax(z) if l == L - 1 else np.tanh(z)
            acts.append(a)
        return acts[-1], acts

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)

    @staticmethod
    def loss(probs: np.ndarray, y_onehot: np.ndarray) -> float:
        return float(-np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1)))

    # --- backward ----------------------------------------------------------- #

    def gradients(self, X: np.ndarray, y_onehot: np.ndarray):
        """Analytic gradients of the mean cross-entropy loss w.r.t. W and b."""
        probs, acts = self.forward(X)
        n = len(X)
        L = len(self.W)
        dW = [None] * L
        db = [None] * L

        delta = (probs - y_onehot) / n           # softmax + cross-entropy shortcut
        for l in range(L - 1, -1, -1):
            dW[l] = acts[l].T @ delta
            db[l] = delta.sum(axis=0)
            if l > 0:
                # backprop through the tanh activation that produced acts[l]
                delta = (delta @ self.W[l].T) * (1.0 - acts[l] ** 2)
        return dW, db, self.loss(probs, y_onehot)

    # --- training ----------------------------------------------------------- #

    def train(self, X, y, epochs: int = 300, lr: float = 0.1, batch_size: int = 32,
              seed: int = 0) -> list[float]:
        """Minibatch SGD. Returns the loss after each epoch."""
        rng = np.random.default_rng(seed)
        n_classes = self.sizes[-1]
        Y = one_hot(np.asarray(y), n_classes)
        n = len(X)
        history = []
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                dW, db, _ = self.gradients(X[idx], Y[idx])
                for l in range(len(self.W)):
                    self.W[l] -= lr * dW[l]
                    self.b[l] -= lr * db[l]
            probs, _ = self.forward(X)
            history.append(self.loss(probs, Y))
        return history

    def accuracy(self, X, y) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())
