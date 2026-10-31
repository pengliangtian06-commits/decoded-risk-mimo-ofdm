# Engineering and reproducibility

The tested system is macOS arm64, Apple M4, ten CPU cores and 16 GB memory.
Python 3.12 orchestrates C++17 Eigen/Accelerate kernels and SciPy LAPACK.
Scientific package versions are pinned in requirements-lock.txt. Other
platforms may need linker changes and have not been independently validated.

The receiver implements unit-energy Gray-QAM; time-domain CP-OFDM convolution;
orthogonal comb-pilot and residual-noise estimation; configured-PDP shrinkage;
Cholesky, CG, Jacobi-PCG and matrix-free PCG; exact, diagonal, periodic
interpolated and published approximate-CG soft-output adaptations; and CRC16
plus BG2 LDPC with an independently implemented normalized-min-sum decoder.
The waveform is custom and is not a complete NR conformance implementation.

The direct dimension rule was chosen on development data. Native LAPACK is
used at at most sixteen users and SciPy/LAPACK above that. Direct reference and
interpolated-gain candidates share this rule. Costs include preparation,
anchor factorizations, gains, demapping, decoding and wrapper overhead.

The prospective lock is research/protocols/study_002_lock.json. Do not edit its
listed scientific sources/configuration or replace raw results. A substantive
scientific correction requires a new protocol version and affected-stage rerun.
Numerical stopping criteria and statistical decoded-risk calibration are
separate modules. The latter uses established exact binomial and finite-family
testing machinery, not a new theorem.

Use docs/ARCHIVE_REPRODUCTION.md for the complete fresh-extraction command.
The archive validation runs 57 tests, checks all locked sources, reconstructs
development costs/calibration, replays twelve source cells for seventeen
pipelines and exactly regenerates all principal analysis fields. Historical
clock observations are reanalysed; matching absolute new times is not claimed.

Unsuccessful early bound-based heuristics and superseded unoptimized timings
remain under results/pilot. They are not baseline speedup evidence. Final
serial timing and transfer studies live under results/final; canonical
summaries are under results/analysis. Source and target certificate scopes,
user/frame aggregation, standard-PDP quantization, easy zero-event cells and
same-machine reproduction limits are stated explicitly in the article.
