# Prospective joint-budget study, version 0.2

Status: prospective protocol recorded before calibration root 2100 or test root
3100 is used. Machine-readable policy/cost/source lock will be written after the
serial development cost screen finishes and before either independent stage.

## Bounded contribution and reason for refinement

Initial scalar/bound-based stopping did not beat strong conventional stopping.
Native-library and complete-receiver timing substantially narrowed apparent
gains. We now test an algorithm-agnostic *joint numerical and soft-output budget
selection procedure* against an optimized exact receiver, including direct
solvers with sparse gain calculations among the candidates. The central question
is whether candidate budgets can be selected with a reproducible limit on
additional decoded errors, rather than assuming that an algebraic residual or
an iteration count certifies BLER.

This uses established CG, frequency interpolation, exact binomial bounds and
finite-family testing. No first-CG, first-interpolation, first-wireless-conformal
or new statistical-theorem claim is made. The prospective contribution is the
specific decoded excess-risk object, auditable independent sampling, fair
complete receiver implementation, and measured trade-offs/failure boundaries
for the joint computational family. Publication novelty remains subject to
closest-work comparison and final internal review.

The original <=0.2 dB / >=20% internal feasibility target was not established.
It is retained as a failed prioritization gate. This study asks the revised
risk-budget question explicitly; no numerical target is retrospectively
reported as passed. A small or null speed gain must be reported as such.

## Frozen family and source distribution

Seventeen pipelines comprise one exact reference and sixteen candidates. The
explicit recipes are in `configs/joint_budget_development.json`: four direct
gain strides, conventional fixed/residual/noise stopping with exact/diagonal/
interpolated gains, two published approximate-CG soft-output adaptations and
matrix-free PCG. Every family member gets the same calibration sample size and
the same one-sided confidence allocation.

The exact direct backend is selected by a development-only dimension rule:
native LAPACK up to 16 users, SciPy/LAPACK above 16, based on serial benchmark
002. This reference and all direct sparse-gain candidates share that rule.
All methods use the same pilot estimates, scalar effective covariance,
complex128 arithmetic, optimized separable max-log demapper, LDPC and CRC.
The Yin baseline uses its ordinary-CG diagonal-polynomial approximate gain and
SINR, substituting the common effective disturbance variance for its original
perfect-CSI noise term. It is not a full paper-experiment reproduction.

Source frames are iid uniform over twelve cells: (B,U)=(64,16)/(64,32), receive
Toeplitz correlation 0/0.7, and Es/N0=10/12/14 dB. Each draw uses a fresh
independent eight-tap exponentially decaying block channel, new payloads and
noise. One comb-pilot plus one payload OFDM symbol uses FFT512/CP32, unit
symbol energy, 16-QAM and 1024 information bits per user. FFT normalization,
channel power and pilot energy follow the component-tested implementation.

The primary BLER estimand is the mean within-frame user error fraction under
this uniform *frame* mixture. Since U differs by cell, a raw pooled count ratio
has a different user-weighted estimand and is only secondary descriptive data.

## Sample sizes, risk and selection

- Development: 48 fixed balanced frames, root1137, one worker, used to rank mean
  complete receiver costs. Existing exploratory screens are not confirmatory.
- Calibration: 4096 iid mixture frames, root2100. Fixed sample size, no interim
  threshold selection or optional stopping. Four workers; these timings do not
  support final speed claims.
- Test: 2048 iid mixture frames, root3100, generated only after calibration and
  primary policy selection are complete. All family members are retained for
  transparent ablations; the primary selected policy is frozen before testing.

For each frame draw one uniform audit user before decoding outcomes. Its harm
event is candidate wrong and exact reference correct. This is one Bernoulli
observation per independent frame, regardless of U. For K=16 candidates, use
one-sided Clopper--Pearson limits with failure probability delta/K, delta=0.05.
The primary excess-risk budget is alpha=0.01, an absolute one percentage point
margin in source-mixture BLER. This does not certify absolute BLER=0.01 or a
relative 1% loss. Also report alpha=0.002/0.005/0.02 using the same simultaneous
bounds; these are secondary operating-point analyses.

Choose the eligible pipeline with lowest development mean complete receiver
cost, including the identically-zero-excess-risk exact reference as a candidate.
If the reference is cheaper or no approximation qualifies, select it. These
finite-family guarantees are inherited from established testing machinery.
They assume independent calibration frames and the same future mixture, permit
arbitrary within-frame user dependence, and require no monotone error/iteration
relationship. They do not protect conditional cells or shifted channels.

## Primary evaluation and uncertainty

The paired difference in mean within-frame BLER and paired receiver wall time
are the principal measured endpoints. Use channel/frame-cluster bootstrap
intervals for within-frame proportions and paired differences, with bootstrap
RNG independent of simulation. For the one-audit-user Bernoulli loss, report
an exact one-sided limit, including nonzero limits when no failures occur.
Per-cell BLER curves are descriptive, with explicit independent frame counts.
No rare-event or URLLC performance is inferred from a few zero-error cells.

Publication timing must be a separate serial replay benchmark on held-out
waveforms, with method order randomized, warm-up, preparation/gain/LLR/decoder
components, complete wall time and P50/P95. Replays are not independent channel
or BLER samples. Include exact native and SciPy alternatives and state the
development backend rule. Net goodput counts correctly delivered information
bits per waveform sample, including comb-pilot and CP overhead; CPU throughput
is distinct from radio goodput. No energy or RF real-time claim follows from
CPU software timings.

## Transfer and engineering boundaries

Preserve independent roots for transfer and record the full receiver prior.
Use quantized TDL-A/C power-delay profiles from TR38.901 v18.0.0, with actual
quantized RMS delay reported, block-static Rayleigh taps and explicitly
synthetic spatial correlation. This is not a full 3GPP channel-conformance
claim. Test prior mismatch, longer delay, modulation and/or reuse transfer
without applying a source-mixture certificate to those distributions. Any
target recalibration must use a separate calibration/test split and be labelled.

Source, compiled extension and configuration hashes must match the lock before
and after each stage. Keep raw counts, individual frame/user error vectors,
gain-clipping counts, stopping reasons, random sample IDs and all failed family
members. No final frame is removed because it is unfavorable. If a substantive
implementation bug is found, record it and invalidate/restart affected stages
under a new version rather than silently editing the evidence.
