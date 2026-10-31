#!/usr/bin/env python3
"""Exploratory preparation/reuse crossovers; NOT coded receiver timing."""
import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(name, "1")
import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from mimo_receiver import _core
from mimo_receiver.channels import conditional_rayleigh, complex_normal


def benchmark(output, repeats=30):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "timing.csv").exists():
        raise FileExistsError("Do not overwrite benchmark records")
    rng = np.random.default_rng(1108)
    records = []
    for antennas, users in [(32, 8), (64, 16), (64, 32), (128, 32), (256, 64), (256, 128)]:
        for rho in [0.0, 0.7]:
            _, h, error = conditional_rayleigh(rng, antennas, users, rho, 0.02)
            variance = 0.01 + users * error
            for reuse in [1, 14, 128]:
                y = complex_normal(rng, (antennas, reuse))
                exact = np.linalg.solve(h.conj().T @ h + variance * np.eye(users), h.conj().T @ y)
                for backend, mode, threshold in [("cholesky", "fixed", 1),
                                                  ("pcg", "relative_residual", 0.001),
                                                  ("pcg", "radau", 0.001),
                                                  ("matrixfree_pcg", "relative_residual", 0.001),
                                                  ("pcg", "decision_residual", 0.0001),
                                                  ("matrixfree_pcg", "decision_residual", 0.0001),
                                                  ("scipy_cholesky", "fixed", 1)]:
                    for repeat in range(-3, repeats):
                        start = perf_counter()
                        if backend == "scipy_cholesky":
                            a = h.conj().T @ h + variance * np.eye(users)
                            factor = cho_factor(a, lower=True, check_finite=False)
                            prepared_at = perf_counter()
                            x = cho_solve(factor, h.conj().T @ y, check_finite=False)
                            solved_at = perf_counter()
                            iterations = 0
                        else:
                            prepared = _core.PreparedSystem(h, variance, backend)
                            prepared_at = perf_counter()
                            result = prepared.solve(y, mode, threshold, 2 * users, 2)
                            solved_at = perf_counter()
                            x = result["symbols"]
                            iterations = sum(result["iterations"]) / reuse
                        if repeat >= 0:
                            records.append({"antennas": antennas, "users": users, "rho": rho,
                                            "reuse": reuse, "backend": backend, "mode": mode,
                                            "threshold": threshold, "repeat": repeat,
                                            "preparation_seconds": prepared_at-start,
                                            "solve_batch_seconds": solved_at-prepared_at,
                                            "total_seconds": solved_at-start,
                                            "mean_iterations": iterations,
                                            "relative_solution_error": np.linalg.norm(x-exact)/np.linalg.norm(exact)})
                print(f"B={antennas} U={users} rho={rho} reuse={reuse}", flush=True)
    with (output / "timing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (output / "metadata.json").write_text(json.dumps({
        "stage": "exploratory", "seed": 1108, "repeats": repeats, "warmups": 3,
        "timing": "Python wall time including conversion; prep and batched solve separated",
        "arithmetic": "complex128", "threads": 1,
        "excludes": ["pilots", "LLRs", "coding", "decoding", "OFDM", "RF hardware"],
        "comparison_limit": "Rules are not yet calibrated to matched coded reliability; this is an implementation diagnostic"
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    benchmark(args.output, args.repeats)
