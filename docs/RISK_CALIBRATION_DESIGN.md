# Candidate refinement: decoded excess-risk calibration

Status: implemented primitives and development screening; scientific contribution
and confirmatory performance remain unestablished. This is an engineering use
of existing finite-family hypothesis testing and exact binomial bounds. It is
not a new conformal-prediction theorem.

## Loss and sampling unit

Let a frame contain U co-scheduled users under a single independent channel
draw. Before observing any decoding outcomes, choose J uniformly from those
users using an independent random draw. For candidate policy p define

L_p = 1{candidate p is wrong for J and exact reference is correct for J}.

Then P(candidate error) - P(reference error) <= E[L_p] for the declared frame
mixture and uniform user selection. This controls one-sided excess BLER; it
does not claim the reference has zero BLER or impose an absolute URLLC target.
An observed user vector provides clustered descriptive BLER information, but
only one Bernoulli observation per independent frame enters this exact bound.

For K frozen candidates and n independent calibration frames per candidate,
compute the one-sided Clopper--Pearson upper bound at failure probability
delta/K for each candidate. Bonferroni then covers all candidates simultaneously.
Select the cheapest of the eligible candidates and exact reference, ranked
using separate development observations. If none qualifies, or the exact
reference is cheaper, select the reference.
No monotonicity of risk versus solver iteration or soft-gain stride is assumed.
Policies, condition sampling distribution, K, alpha, delta and cost ordering
must be frozen before independent calibration outcomes are inspected.

With zero events, the upper bound is 1-(delta/K)^(1/n). Small sample sizes
therefore cannot certify rare excess risk. This algebra is included in tests.

## Scope of validity

The probability statement is over repeated independent calibration datasets
from the same mixture as future frames. It is not per-channel certification,
conditional-on-SNR certification, a guarantee under distribution shift, nor
evidence of unconditional floating-point stability. If stratum-wise claims are
desired, each stratum requires its own independent observations and a confidence
budget shared across both policies and strata. No final choice has been locked.

## Engineering candidate family

Candidate policies can jointly specify solver backend, stopping tolerance,
and inverse-diagonal soft-gain sampling stride. Exact gain calculations use
only estimated channels and effective noise; remaining tones use periodic
linear interpolation. The diagonal gain approximation is an explicit baseline.
All calibration cost, preprocessing, Gram construction, solves, LLRs and
decoding must enter receiver timing. Sparse interpolation and CG are known
ingredients; a defensible contribution would require their reproducible coded
reliability/cost trade-off and a precise distinction from closest prior work.

## Closest-work warning

- Yin et al. (GLOBECOM 2014, DOI 10.1109/GLOCOM.2014.7037382) already develops
  exact and approximate CG soft outputs. Do not claim the first soft-output CG.
- Borgmann and Boelcskei (Asilomar 2004, DOI 10.1109/ACSSC.2004.1399503)
  already uses interpolation for MIMO-OFDM inverses. Do not claim the first
  frequency-domain interpolation receiver.
- Learn then Test (DOI 10.1214/24-AOAS1998, journal route verified; pagination pending) is
  the statistical framework to cite, not a newly invented risk-control result.
- QuaCE-Guard (Physical Communication, DOI 10.1016/j.phycom.2026.103396)
  explicitly concerns joint user/budget conformal pilot acquisition. Its title
  and publication are verified; primary abstract/introduction/section snippets
  were inspected. Complete methods access is unresolved. No assertion that it
  lacks a feature may be based only on its title or snippets.
- CoDE (arXiv:2608.15751, 2026 preprint; not counted as a journal article)
  already concerns reliability-controlled early decisions in a spiking short
  packet receiver. Numerical solver/soft-output budgeting needs its own narrow
  positioning, without a broad first-reliable-compute claim.

Scite search and full-text tools returned Unknown tool on 2026-10-31, so direct
primary sources/Crossref remain the fallback. That failure is not evidence that
a paper or a method does not exist.
