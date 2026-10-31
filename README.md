# Decoded-risk MIMO-OFDM receiver study

Version 1.0.0: research software, simulation records and reproducible analyses.
The receiver is implemented in Python and C++17, with time-domain CP-OFDM,
pilot/noise estimation, Gray-QAM soft outputs, several direct/iterative numerical
recipes, CRC16 and BG2 LDPC decoding. The waveform is custom, not NR conformance.

## Engineering question and measured outcome

Which numerical/soft-output approximation saves complete receiver computation
while controlling additional decoded-block errors relative to the same exact
numerical reference? Residual thresholds alone do not answer that question.

Sixteen frozen approximations plus the exact fallback share the same received
waveform. Calibration uses 4096 independent frames and one uniformly selected
audit user per frame. The selected direct solver samples gain-related inverse
diagonals every 32 tones; it still solves symbols on every tone. Its simultaneous
one-sided additional-error limit is 0.3993% against a 1-percentage-point source
mixture budget. Cost ranking was determined on separate development frames.

An independent 2048-frame source test gives mean within-frame BLER 17.245% for
the reference and 17.232% for the selected policy. The paired difference is
-0.014 percentage points (95% frame interval -0.076 to +0.046); this does not
establish an accuracy improvement or rare-error performance. Serial timing of
48 held-out waveforms, five measured calls per policy after warm-up, gives
38.22 versus 34.83 ms mean complete CPU latency: an 8.9% saving (95% interval
8.4%–9.4%). Timing includes FFT through LDPC/CRC and wrapper overhead; transmit
encoding and channel generation are outside it. Twenty transfer cells retain
failures under long-delay prior mismatch. The source certificate does not
extend to each condition or shifted operation.

CG, frequency interpolation and simultaneous binomial testing are established
ingredients. This release provides their implemented decoded-risk/full-cost
interface, independent evaluation and failure boundaries. It claims no new
statistical theorem, universal detector superiority, hardware energy saving
or RF real-time operation.

## Reproduce the archived study

Tested platform: macOS arm64, Apple M4, Python 3.12, Apple clang/Accelerate.
Use the exact requirements-lock.txt. Other machines/BLAS/platforms need
validation and may need linker adaptation. A fresh source build was checked
on the original machine; this is not independent-machine replication.

From a full release-archive extraction:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/reproduce_archive.py
```

The last command checksum-verifies bundled Eigen, compiles the extension,
runs 57 tests, verifies 25 locked sources, reconstructs development ranking and
calibration, replays all 17 policies on the first held-out draw in each of 12
source conditions, and regenerates every field of the three principal analyses.
It never overwrites an existing reproduction output. Use --output NEW_DIRECTORY
for another run. Historical timing records are reanalysed; new absolute clock
times are not expected to match. See docs/ARCHIVE_REPRODUCTION.md.

For a Git checkout, first run `python scripts/fetch_release_data.py`. This downloads and SHA-256 verifies the complete release into a new `release-v1.0.0/` directory; run its reproduction script there. No existing directory is overwritten.

The versioned release archive contains all large raw records and dependencies.
The Git repository may omit large raw records; use the full release asset for
the self-contained reproduction command above. ARCHIVE_MANIFEST.json hashes
all payload files. Retained earlier pilots are exploratory and include failed
ideas; they do not replace the frozen independent tests.

## Evidence map

| Evidence | Location |
|---|---|
| Prospective source/selection lock | research/protocols/study_002_lock.json |
| Calibration and frozen selection | results/calibration/joint_budget_002 |
| Independent source outcomes | results/final/joint_budget_002 |
| Serial complete-receiver timing | results/final/receiver_timing_002 |
| Shifted evaluations | results/final/transfer_002 |
| Canonical source/timing/transfer summaries | results/analysis |
| Native/kernel, demapper, coding and risk checks | tests |
| Fresh-extraction verification | results/reproduction/archive_release_002/report.json |

## License, attribution and provenance

Original software is licensed under AGPL-3.0-only (LICENSE). Original simulated
outcome/benchmark records and quantitative source data are licensed under
CC BY 4.0 (LICENSE-DATA.txt). LICENSE_SCOPE.md describes the directory scope
and third-party exceptions. Dependencies keep their original notices; the
unchanged Eigen source archive and py3gpp notice are retained. No third-party
paper full texts, article manuscript, author photographs or submission forms
are distributed here. CITATION.cff identifies this software/data release.

AI assistance was used for planning, literature discovery, implementation,
analysis/figure code and documentation. Human authors approved the release;
the tests, source lock and raw records make the reported checks inspectable.
The publication manuscript separately discloses assistance and human review.
