# Week of 2026-11-24 — Neural Net: ReLU + real handwritten digits (depth)

**Theme:** data/ML — DEPTH, not a new project. Extends week-24's from-scratch neural
net from 2-D toys to real high-dimensional data.
**Type:** depth/extension (user asked me to pick; technique breadth is complete, so I
recommended depth; this is a BUILD in weekly-builds, no new public repo).

## What's new this week
1. **ReLU** added to the from-scratch `MLP` (alongside tanh; He init), with correct
   hand-derived backprop — and gradient-checked (max rel error ~2e-6, PASS).
2. **Real handwritten digits**: the UCI 8x8 digits set (1797 samples, 64 features,
   10 classes), extracted once and committed as `data/digits.npz` (68KB, offline).
3. `digits.py` — load/split, train linear vs ReLU net, confusion matrix, top
   confusions; its own entry point (`python -m neural_net.digits`).
4. New plots: digit samples, confusion matrix, misclassified grid (true->pred), and
   first-layer weight images ("what each hidden unit looks for").
5. 5 new tests (15 total): ReLU gradient check, ReLU non-negativity + activation
   validation, digits load, confusion-matrix hand value, net>90% on real digits.

## Findings
- ReLU gradient check PASS (~2e-6) — backprop correct for the new activation too.
- Digits test accuracy: **linear 98.1% vs ReLU net 98.9%** (100 hidden units).
- **The honest payoff:** on the 2-D toys the net beat the line by ~47 points; on real
  digits by **under 1** — because real digits are mostly linearly separable (the
  linear model already gets 98%). Nonlinearity earns its keep when the problem is
  nonlinear; it is not a free upgrade. Same experiment shows the power AND the limit.
- Mistakes are human-plausible (sloppy 9->5, odd 4->7, curly 6->8); first-layer
  weights look like stroke/edge detectors.

## Why this matters for the portfolio story
Deepens the flagship ML-fundamentals project: proves the hand-coded net scales from
toys to genuine high-dimensional classification, adds ReLU (a real DL component) with
the gradient-check rigor extended to it, and sharpens the honesty thread (the net's
advantage shrinks to ~0 when a linear model already suffices). Self-contained
(committed data snapshot). Pure numpy/matplotlib, no new runtime deps (data extracted
from sklearn once, then committed).

## Deliberately deferred / next
- Full MNIST (28x28) via download; dropout/L2; a from-scratch conv layer.
- GRADUATION run (word-prints flagged 5x; neural-net now especially strong) — confirm.

## Status
Built and shipped 2026-11-24. 15 tests pass (tanh + ReLU gradient checks, digits);
verified end-to-end (2-D study + digits study). Four new digit charts.
