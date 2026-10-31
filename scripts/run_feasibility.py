#!/usr/bin/env python3
"""Record exploratory solver trajectories and stopping trade-offs.

This is not the frozen final study. It deliberately records failures as well as
favorable curves. No final-evaluation seed is touched.
"""
import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")

import argparse
import csv
import hashlib
import itertools
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import scipy
from mimo_receiver.channels import conditional_rayleigh, complex_normal, linear_system
from mimo_receiver.qam import modulate, hard_demodulate
from mimo_receiver.solvers import cg_trace, cholesky_solve, choose_iteration


def run(config_path: Path, output: Path):
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    if config["seed_start"] >= 3000 or config["seed_start"] + config["seeds"] > 1200:
        raise ValueError("Feasibility must use only reserved development seed family")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise FileExistsError("Choose a new output directory rather than overwrite an experiment")
    metadata = {"config": config, "config_sha256": hashlib.sha256(raw_config).hexdigest(),
                "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                "platform": platform.platform(), "started_unix": time.time(),
                "stage": "exploratory", "oracle_inputs": ["conditional_error_variance", "exact_debiasing_gain"],
                "claims_not_supported": ["coded BLER", "production speedup", "confirmed novelty"]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    trace_rows, summary_rows = [], []
    widths = int(np.log2(config["qam_order"]))
    conditions = list(itertools.product(config["antenna_user_pairs"], config["correlations"],
                                       config["snr_db"], config["error_fractions"]))
    for condition_index, ((antennas, users), rho, snr, error_fraction) in enumerate(conditions):
        for seed in range(config["seed_start"], config["seed_start"] + config["seeds"]):
            rng = np.random.default_rng(np.random.SeedSequence([seed, condition_index]))
            h, hbar, error_var = conditional_rayleigh(rng, antennas, users, rho, error_fraction)
            vectors = config["vectors_per_channel"]
            bits = rng.integers(0, 2, (users, vectors * widths), dtype=np.uint8)
            symbols = modulate(bits, config["qam_order"])
            noise_var = 10 ** (-snr / 10)
            received = h @ symbols + np.sqrt(noise_var) * complex_normal(rng, (antennas, vectors))
            a, rhs, lam = linear_system(hbar, received, noise_var, error_var)
            exact = cholesky_solve(a, rhs)
            diag_inv = cholesky_solve(a, np.eye(users)).diagonal().real
            gain = np.maximum(1 - lam * diag_inv, 1e-10)
            exact_bits = hard_demodulate(exact / gain[:, None], config["qam_order"])
            common = {"antennas": antennas, "users": users, "correlation": rho,
                      "snr_db": snr, "error_fraction": error_fraction, "seed": seed}
            trajectories = [cg_trace(a, rhs[:, vector], lam, max_iterations=2 * users,
                                     qam_order=config["qam_order"])
                            for vector in range(vectors)]
            for vector, history in enumerate(trajectories):
                for item in history:
                    delta = item.solution - exact[:, vector]
                    true_energy = float(np.vdot(delta, a @ delta).real)
                    predicted = hard_demodulate((item.solution / gain)[None, :], config["qam_order"]).reshape(users, widths)
                    truth = bits[:, vector * widths:(vector + 1) * widths]
                    exact_vector_bits = exact_bits[:, vector * widths:(vector + 1) * widths]
                    trace_rows.append({**common, "vector": vector, "iteration": item.index,
                                       "relative_residual": item.relative_residual,
                                       "radau_upper": item.radau_energy_upper,
                                       "true_energy_error": true_energy,
                                       "normalized_true_energy": true_energy / (lam * users),
                                       "decision_score": item.decision_score,
                                       "stable_decisions": item.stable_decisions,
                                       "bound_violation": int(true_energy > item.radau_energy_upper * (1 + 1e-6) + 1e-10),
                                       "bit_errors": int(np.sum(predicted != truth)),
                                       "bits": users * widths,
                                       "bit_difference_from_exact": int(np.sum(predicted != exact_vector_bits))})
            rules = [("relative_residual", t) for t in config["relative_residual_thresholds"]]
            rules += [(mode, t) for mode in ["noise_residual", "radau", "euclidean_bound", "decision_residual"]
                      for t in config["energy_budget_thresholds"]]
            rules += [("fixed", t) for t in config["fixed_iterations"] if t <= users]
            for mode, threshold in rules:
                selected = [choose_iteration(history, mode, threshold, lam, a) for history in trajectories]
                estimates = np.stack([item.solution for item in selected], axis=1)
                decisions = hard_demodulate(estimates / gain[:, None], config["qam_order"])
                summary_rows.append({**common, "mode": mode, "threshold": threshold,
                                     "vectors": vectors,
                                     "iterations": sum(item.index for item in selected),
                                     "bit_errors": int(np.sum(decisions != bits)),
                                     "exact_bit_errors": int(np.sum(exact_bits != bits)),
                                     "bit_difference_from_exact": int(np.sum(decisions != exact_bits)),
                                     "bits": bits.size,
                                     "squared_difference_from_exact": float(np.sum(abs(estimates - exact) ** 2))})
        print(f"condition {condition_index + 1}/{len(conditions)} complete: B={antennas} U={users} rho={rho} SNR={snr} error={error_fraction}", flush=True)
    for filename, rows in [("trajectories.csv", trace_rows), ("stopping_results.csv", summary_rows)]:
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata["elapsed_seconds"] = time.time() - metadata["started_unix"]
    metadata["completed"] = True
    metadata["independent_channels"] = len(conditions) * config["seeds"]
    metadata["vectors"] = metadata["independent_channels"] * config["vectors_per_channel"]
    metadata["bound_violations"] = sum(row["bound_violation"] for row in trace_rows)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: metadata[key] for key in ["elapsed_seconds", "independent_channels", "vectors", "bound_violations"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/feasibility.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
