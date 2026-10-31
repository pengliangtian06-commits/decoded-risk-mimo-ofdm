# Reproducing the code/data archive

The tested platform is macOS arm64 with Apple M4, 10 CPU cores and 16 GB memory.
Use Python 3.12 and the exact `requirements-lock.txt` versions. Other CPU/OS,
BLAS and fixed-point targets have not been validated. The current Linux build
link settings may require porting; no cross-platform reproduction is claimed.

From the extracted archive directory, create/activate an environment, install
the pinned dependencies, then run:

```sh
python scripts/reproduce_archive.py
```

The command extracts the checksum-verified bundled Eigen source, compiles the
extension, runs all 57 component tests, verifies every locked scientific
source, reconstructs all development costs and calibration selections,
replays the first source-test frame in each of twelve predefined conditions
for every policy, and regenerates the source, timing and transfer analyses.
It compares every analysis field against the archived canonical reports.
Results and logs go to a new `results/reproduced_archive_001` directory;
existing output is never overwritten. Use `--output` for another new path.

The rebuilt extension is allowed to have a different binary hash only after
the locked source checks pass. Imports are checked to originate within the
extracted archive. No original workspace source is required. The validation
uses the same pinned Python environment and machine, so it is not an
independent-investigator or independent-machine replication. Historical wall
times are analysed from raw serial observations; new absolute wall times are
not expected to match them.

The release includes raw development observations and the direct-backend
benchmark that informed the native/SciPy dimension rule. Original exploratory
studies remain labelled as such. They are not confirmatory extensions of the
locked source-mixture certificate. All source, transfer and gain-clipping
outcomes remain present, including failed policies.

This versioned code/data release was approved by the authors for public GitHub deposition under the documented software/data terms. Repository: https://github.com/pengliangtian06-commits/decoded-risk-mimo-ofdm. No DOI has been assigned.
