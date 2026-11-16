# Week of 2026-11-16 — Neural Net from scratch (deep learning fundamentals)

**Theme:** data/ML — the portfolio's first neural-network project.
**Type:** new technique (MLP + backprop from scratch, numpy, no framework); user
asked me to pick "the next one."

## The question
What's actually inside a neural network, is my hand-derived backprop correct, and
when does its nonlinearity beat a linear model?

## Scope (shipped)
- `nn.py` — `MLP` from scratch: forward (tanh hidden + softmax output), hand-derived
  backprop (`gradients`), minibatch-SGD `train`. Zero hidden layers = logistic
  regression (the linear baseline, for free).
- `gradcheck.py` — numerical vs analytic gradient comparison (central differences);
  `max_relative_error` ~2e-7 = backprop correct to machine precision. THE rigor piece.
- `datasets.py` — XOR, moons, circles, spirals: 2-D problems no straight line can
  separate (self-generated, seeded).
- `experiment.py` — train linear vs net on each, report test accuracy.
- `plots.py` — decision-boundary grid (linear line vs net's curved boundary) + loss
  curve.
- CLI; no external data (pure synthetic). 10 offline tests (gradient check first).

## Findings
- **Gradient check PASS: max relative error 2.0e-07** (backprop verified).
- Linear vs net test accuracy: xor 64%->93%, moons 87%->99%, circles 0%->100%,
  spirals 51%->99%. Net adds ~47% on average — because a line can't bend.
- Decision-boundary grid shows the net carving an X / S / ring / double-spiral.

## The honest core
- Nonlinearity only helps when the problem is nonlinear; on line-separable data the
  net wins nothing and just adds overfitting risk. The +47% is specific to problems
  chosen to be hard for a line, NOT a general "NN > linear" claim.
- Tiny clean 2-D toys isolate one idea (nonlinear boundaries); real problems bring
  dimensions/noise/scale where architecture + regularization + optimization matter.
- "From scratch" is for understanding; use PyTorch/JAX for real work.

## Why this matters for the portfolio story
First neural network / deep-learning-fundamentals project, written from scratch
(hand-coded backprop) — demonstrates understanding, not a library call. The
gradient-check rigor and the "compared to what / when is it worth it" framing are the
portfolio's honesty thread. Pure numpy/matplotlib, no new deps.

## Deliberately deferred / next
- ReLU + deeper nets; a real dataset (MNIST subset); dropout/regularization;
  a small conv net.
- GRADUATE one of the many ungraduated projects (confirm first).

## Status
Built and shipped 2026-11-16. 10 tests pass (gradient check included); verified
end-to-end on the four 2-D datasets. Two charts.
