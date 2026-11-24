"""Charts: decision boundaries (linear vs. net) and the training loss curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

_CMAP = plt.cm.RdBu
_POINTS = ["#b2182b", "#2166ac"]


def _boundary(ax, model, X, y, title):
    pad = 0.5
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs, _ = model.forward(grid)
    zz = probs[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap=_CMAP, alpha=0.7)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1)
    for cls in (0, 1):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=8, color=_POINTS[cls], edgecolor="white",
                   linewidth=0.2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def plot_decision_grid(comparisons: dict, out_path: str | Path) -> Path:
    """One row per dataset: the linear boundary (a line) vs. the net's curved one."""
    out_path = Path(out_path)
    names = list(comparisons)
    fig, axes = plt.subplots(len(names), 2, figsize=(7.5, 3.4 * len(names)))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(names):
        c = comparisons[name]
        _boundary(axes[row, 0], c.linear, c.X_test, c.y_test,
                  f"{name}: linear  (acc {c.linear_acc:.0%})")
        _boundary(axes[row, 1], c.net, c.X_test, c.y_test,
                  f"{name}: neural net  (acc {c.net_acc:.0%})")
    fig.suptitle("A line can't bend; a neural net can", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_digit_samples(X, y, out_path: str | Path, rows: int = 4, cols: int = 8) -> Path:
    """A grid of example digits with their labels."""
    out_path = Path(out_path)
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows + 0.5))
    for ax, img, label in zip(axes.ravel(), X, y):
        ax.imshow(img.reshape(8, 8), cmap="gray_r", interpolation="nearest")
        ax.set_title(str(int(label)), fontsize=9)
        ax.axis("off")
    fig.suptitle("Real handwritten digits (8x8 grayscale)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_confusion(cm, out_path: str | Path, accuracy: float | None = None) -> Path:
    """Confusion matrix of the digit classifier (row-normalized)."""
    out_path = Path(out_path)
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit"); ax.set_ylabel("true digit")
    title = "Digit confusion matrix"
    if accuracy is not None:
        title += f"  (accuracy {accuracy:.1%})"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of true digit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_misclassified(X, y_true, y_pred, out_path: str | Path, n: int = 16) -> Path:
    """The digits the network got wrong, labelled true -> predicted."""
    out_path = Path(out_path)
    wrong = np.where(y_true != y_pred)[0][:n]
    cols = 8
    rows = int(np.ceil(max(len(wrong), 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows + 0.5))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(X[i].reshape(8, 8), cmap="gray_r", interpolation="nearest")
        ax.set_title(f"{int(y_true[i])}→{int(y_pred[i])}", fontsize=9, color="#d1495b")
    fig.suptitle(f"Where it slips: {len(wrong)} misclassified digits (true → predicted)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_hidden_weights(net, out_path: str | Path, n: int = 24) -> Path:
    """First-layer weight vectors as 8x8 images -- what each hidden unit looks for."""
    out_path = Path(out_path)
    W = net.W[0]                                   # (64, hidden)
    n = min(n, W.shape[1])
    cols = 8
    rows = int(np.ceil(n / cols))
    lim = np.abs(W).max()
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows + 0.5))
    for ax in axes.ravel():
        ax.axis("off")
    for k, ax in zip(range(n), axes.ravel()):
        ax.imshow(W[:, k].reshape(8, 8), cmap="RdBu", vmin=-lim, vmax=lim,
                  interpolation="nearest")
    fig.suptitle("What the hidden units look for (first-layer weights)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_loss(comparisons: dict, out_path: str | Path) -> Path:
    """Training loss per epoch for the neural net on each dataset."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, c in comparisons.items():
        ax.plot(c.loss_history, label=name, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("The network learning: loss falling as it fits each dataset")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
