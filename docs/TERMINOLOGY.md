# Terminology ledger

| Canonical term | First-use expansion / meaning | Rule |
|---|---|---|
| MIMO-OFDM | multiple-input multiple-output orthogonal frequency-division multiplexing | One spelling throughout |
| CSI | channel state information | Distinguish true, estimated and oracle CSI |
| LMMSE | linear minimum mean-square error | Distinguish statistical estimator from numerical solver |
| CG | conjugate gradient | Complex Hermitian implementation |
| PCG | preconditioned conjugate gradient | Jacobi is the initial fixed preconditioner |
| LLR | log-likelihood ratio | Define bit-sign convention in code and manuscript |
| BLER | block error rate | Specify user codeword/block as sample unit |
| BER | bit error rate | State before or after decoding |
| numerical residual | b - A z_k | Never equate with receive noise |
| channel estimation error | H - Hhat | True error unavailable at runtime |
| reliability-aware stopping | Proposed mechanism under investigation | Descriptive wording, no invented branded acronym |
| preparation time | Gram matrix, factorization/preconditioning and reusable statistics | Count once per actual coherence/reuse unit |
| detection time | Preparation amortization + solve + necessary soft-output processing | Same accounting across methods |
| effective goodput | Correct useful information bits per stated resource/time | State whether radio airtime or CPU time is denominator |

No terminology collisions are currently known. New terms require a ledger update, not stylistic synonym substitution.
