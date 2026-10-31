# Scientific protocol v0.1 — before feasibility experiments

## Research question and hypothesis

Can a receiver-observable uncertainty-dependent stopping rule improve the coded-reliability/computation Pareto frontier of iterative LMMSE MIMO-OFDM detection compared with well-tuned conventional stopping?

The hypothesis is unvalidated. The first experiments are exploratory and cannot later be relabelled confirmatory.

## Mathematical separation

Use y = Hx + n, available Hhat and receiver-estimated error statistics. An effective disturbance covariance accounts for noise and channel uncertainty under clearly stated independence/model assumptions. Solve the resulting positive-definite regularized linear system A z = b using CG/PCG.

The algebraic residual r_k = b - A z_k measures numerical solution error, not the received noise or channel error directly. For any valid lower bound m on lambda_min(A), ||z_k-z_exact||_2 <= ||r_k||_2/m. Whether a cheap bound is informative enough for stopping is an empirical and analytical feasibility question. Soft-output accuracy must be checked separately.

## Baselines and fairness

- Optimized Cholesky-based LMMSE with factorization reuse.
- Fixed-iteration CG and Jacobi-PCG across a validation-selected iteration grid.
- Relative-residual stopping CG/PCG across a validation-selected tolerance grid.
- Proposed uncertainty-dependent stopping with and without decision/soft-output stability.
- Closest published methods when reproducible and directly applicable; list adaptations explicitly.

All use identical normalization, Hhat, available error/noise estimates, coding, observations and optimization effort. Oracle H or true error values are diagnostic-only variants labelled oracle. All methods receive equivalent reuse opportunities.

## Data partitions

Development seed families: 1100–1199. Validation: 2100–2199. Final: 3100–3199, reserved and untouched until method/configuration lock. Repeated blocks within a coherent channel are not independent channel realizations; retain channel-level IDs for clustered uncertainty estimates.

## Initial configurations

Feasibility begins with normalized Rayleigh/Toeplitz-correlated channels and antenna/user pairs 64/16 and 64/32, then 128/32. Use 16-QAM then 64-QAM. Vary controlled CSI error for debugging, then require pilot-derived estimates. Vary matrix reuse count because direct factorization may beat iterative solvers when reuse is large.

Final waveform experiments add OFDM, a fixed public LDPC configuration, real pilot cost, and TDL-A/C from a fixed official TR 38.901 version. Exact waveform, code and stopping grids must be logged before final evaluation.

## Metrics and uncertainty

Retain errors and denominators, not only rounded rates. Report uncoded BER as diagnostic and coded BLER/BER as primary. Report goodput after pilot/cyclic-prefix/code overhead. Measure CPU elapsed time after warm-up with fixed thread counts and identical batch sizes; distinguish preparation, solve, LLR, decoding and complete receiver.

Compute confidence intervals using independent-channel clustering where needed. Report upper bounds for zero observed failures. Use paired comparisons across methods. Tune on validation only and freeze before final seeds.

## Mechanism-revision gate

The initial internal target is <=0.2 dB loss at BLER 1e-2 and >=20% detector-time improvement over the strongest applicable tuned stopping baseline on at least two representative configurations. This is a research prioritization threshold, not a promised result or journal criterion. If the cheap bound is ineffective, or Cholesky/ordinary stopping dominates after all costs, preserve the result and revise the mechanism before scaling up.
