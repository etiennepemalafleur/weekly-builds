"""neural_net: a multilayer perceptron from scratch, with gradient-checked backprop.

Hand-written forward and backward passes (numpy, no framework), verified against
numerical gradients, shown learning nonlinear 2-D problems a linear model cannot.
"""

from .datasets import DATASETS, circles, moons, spirals, xor
from .experiment import Comparison, run, run_all
from .gradcheck import max_relative_error
from .nn import MLP, one_hot, softmax

__all__ = [
    "MLP",
    "softmax",
    "one_hot",
    "max_relative_error",
    "xor",
    "moons",
    "circles",
    "spirals",
    "DATASETS",
    "run",
    "run_all",
    "Comparison",
]
