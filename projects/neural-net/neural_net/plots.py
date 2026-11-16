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
