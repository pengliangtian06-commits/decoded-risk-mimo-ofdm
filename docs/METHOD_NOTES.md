# Exploratory mechanism notes — 2026-10-31

## Why the original scalar uncertainty story needs care

Under iid zero-mean conditional CSI error and unit-energy independent streams,
the error enters linear detection through an effective disturbance variance.
A residual baseline must receive that same variance. A method cannot claim a
new advantage merely because its baseline was denied CSI-error correction.

## Two monitored numerical errors

1. Cheap Euclidean upper bound: ||x_k-x_*||^2 <= ||b-Ax_k||^2 / lambda^2,
   for A=G+lambda I, G positive semidefinite. This can be very conservative.
2. Gauss-Radau energy-error monitor, inherited from numerical linear algebra:
   bound the remaining A-energy error using the CG/Lanczos coefficients and a
   valid spectral lower bound. For symmetric Jacobi scaling the regularization
   gives lambda_min(D^-1/2 A D^-1/2) >= lambda/max(diag(A)).

An exploratory stopping budget compares remaining numerical A-energy with
theta * effective_variance * U. Under the stated Gaussian conditional linear
model, the expected statistical LMMSE error in the A metric is
effective_variance * U. This budget alone does not guarantee coded BLER.

Gauss-Radau itself is NOT a claimed invention. Closest numerical references
include Meurant and Tichy, *The behaviour of the Gauss-Radau upper bound of the
error norm in CG*, arXiv:2209.14601, and their 2024 SIAM book *Error Norm
Estimation in the Conjugate Gradient Algorithm*. Verify the formal journal
version and bibliographic metadata before citation in the final manuscript.

Primary sources inspected for the method choice:

- https://arxiv.org/abs/2209.14601
- https://epubs.siam.org/doi/10.1137/1.9781611977868.ch7
- https://www.hsl.rl.ac.uk/specs/hsl_mi31.pdf

Published warnings about finite-precision loss of bound tightness are material.
Every monitor is compared with direct-solve errors in tests, numerical failure
disables early stopping, and a numerical bound is not labelled an unconditional
floating-point certificate.

## Feasibility boundary

The reference trajectory stores iterates and recomputes true residuals for
offline diagnosis. Those costs must not be used to represent an optimized
receiver. Performance is not established until the production C++ implementation
is measured, including monitoring, preparation, soft output and reuse.
