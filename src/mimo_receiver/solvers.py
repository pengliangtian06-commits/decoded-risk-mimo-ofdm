"""Complex CG/PCG reference and exploratory numerical-error monitoring.

Gauss-Radau quadrature is established numerical analysis, not a new method of
this study. See docs/METHOD_NOTES.md. Bounds are exact-arithmetic statements;
floating-point behavior is checked against direct solves, never assumed.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class Iteration:
    index: int
    solution: np.ndarray
    residual_norm2: float
    relative_residual: float
    radau_energy_upper: float
    euclidean_error_upper2: float
    step_norm2: float
    decision_score: float = float("inf")
    stable_decisions: int = 0


def cholesky_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cho_solve(cho_factor(a, lower=True, check_finite=True), b, check_finite=True)


def _radau_energy_upper(diag: list[float], off: list[float], next_off: float,
                        lower: float, initial_norm2: float,
                        gauss: float) -> float:
    """Gauss-Radau upper bilinear form minus the current CG energy.

    `off` contains len(diag)-1 off-diagonals; next_off couples an appended
    prescribed-node row. Two tridiagonal continued fractions take O(k) work.
    Return infinity rather than approving stopping if the numerical recurrence
    becomes invalid. lower must be a strict positive eigenvalue lower bound.
    """
    if lower <= 0 or len(off) != len(diag) - 1:
        raise ValueError("Invalid quadrature dimensions or spectral lower bound")
    pivot = np.longdouble(diag[0]) - lower
    for i in range(1, len(diag)):
        if pivot <= 0:
            return float("inf")
        pivot = np.longdouble(diag[i]) - lower - np.longdouble(off[i - 1]) ** 2 / pivot
    if pivot <= 0:
        return float("inf")
    denominator = np.longdouble(lower) + np.longdouble(next_off) ** 2 / pivot
    for i in range(len(diag) - 1, -1, -1):
        coupling = next_off if i == len(diag) - 1 else off[i]
        if denominator <= 0:
            return float("inf")
        denominator = np.longdouble(diag[i]) - np.longdouble(coupling) ** 2 / denominator
    if denominator <= 0:
        return float("inf")
    upper = np.longdouble(initial_norm2) / denominator
    floor = 64 * np.finfo(float).eps * max(float(abs(upper)), abs(gauss), 1.0)
    gap = float(upper - gauss)
    if gap < -floor:
        return float("inf")
    return max(gap, floor)


def cg_trace(a: np.ndarray, b: np.ndarray, regularization: float,
             precondition: bool = True, max_iterations: int | None = None,
             rtol: float = 1e-13, qam_order: int = 16) -> list[Iteration]:
    """Return the full numerical trajectory for *offline development*.

    regularization is a verified lower bound from A=G+lambda I with G PSD.
    Diagnostic storage and explicit true-residual recomputation are not a
    production timing implementation. The C++ implementation measures its own
    real monitoring costs separately.
    """
    a, b = np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)
    n = len(b)
    if a.shape != (n, n) or not np.allclose(a, a.conj().T):
        raise ValueError("Require a Hermitian matrix and compatible vector")
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("Regularization must be finite and positive")
    diagonal = a.diagonal().real
    if np.any(diagonal <= 0):
        raise ValueError("System diagonal must be positive")
    scale = np.sqrt(diagonal) if precondition else np.ones(n)
    transformed = a / scale[:, None] / scale[None, :]
    bt = b / scale
    lower = 0.99 * regularization / np.max(scale ** 2)
    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return [Iteration(0, np.zeros(n, complex), 0., 0., 0., 0., 0.)]
    z = np.zeros(n, complex)
    r, p = bt.copy(), bt.copy()
    rr = float(np.vdot(r, r).real)
    initial_rr = rr
    lanczos_diag, lanczos_off = [], []
    previous_alpha = previous_beta = None
    gauss = 0.0
    history = []
    from .qam import hard_demodulate
    side = int(np.sqrt(qam_order))
    normalization = np.sqrt(2 * (qam_order - 1) / 3)
    boundaries = np.arange(-side + 2, side, 2) / normalization
    approximate_gain = np.maximum(1 - regularization / diagonal, 1e-3)
    posterior_proxy = regularization / diagonal / approximate_gain ** 2
    previous_decisions = None
    stable_decisions = 0
    for k in range(max_iterations or (2 * n)):
        ap = transformed @ p
        curvature = float(np.vdot(p, ap).real)
        if curvature <= 0 or not np.isfinite(curvature):
            raise ArithmeticError("Nonpositive CG curvature")
        alpha = rr / curvature
        old_z = z.copy()
        z += alpha * p
        r -= alpha * ap
        next_rr = float(np.vdot(r, r).real)
        beta = next_rr / rr
        if k == 0:
            lanczos_diag.append(1 / alpha)
        else:
            lanczos_diag.append(1 / alpha + previous_beta / previous_alpha)
            lanczos_off.append(np.sqrt(previous_beta) / previous_alpha)
        gauss += alpha * rr
        upper = _radau_energy_upper(lanczos_diag, lanczos_off,
                                     np.sqrt(beta) / alpha, lower, initial_rr, gauss)
        x = z / scale
        true_residual = b - a @ x
        residual_norm2 = float(np.vdot(true_residual, true_residual).real)
        relative = np.sqrt(residual_norm2) / norm_b
        unbiased = x / approximate_gain
        margin = np.minimum(np.min(abs(unbiased.real[:, None] - boundaries), axis=1),
                            np.min(abs(unbiased.imag[:, None] - boundaries), axis=1))
        sensitivity = posterior_proxy / (posterior_proxy + margin ** 2)
        decision_score = float(np.sum(abs(true_residual) ** 2 / diagonal * sensitivity)
                               / (regularization * n))
        decisions = hard_demodulate(unbiased[None, :], qam_order)
        stable_decisions = (stable_decisions + 1
                            if previous_decisions is not None and np.array_equal(decisions, previous_decisions)
                            else 0)
        previous_decisions = decisions
        history.append(Iteration(k + 1, x.copy(), residual_norm2, relative, upper,
                                 residual_norm2 / regularization ** 2,
                                 float(np.linalg.norm((z - old_z) / scale) ** 2),
                                 decision_score, stable_decisions))
        if relative <= rtol:
            break
        p = r + beta * p
        rr = next_rr
        previous_alpha, previous_beta = alpha, beta
    return history


def choose_iteration(history: list[Iteration], mode: str, threshold: float,
                     effective_variance: float, a: np.ndarray,
                     min_iterations: int = 2) -> Iteration:
    """Offline selection. The noise-residual baseline has equivalent variance info."""
    if threshold < 0:
        raise ValueError("Threshold must be nonnegative")
    n = a.shape[0]
    for item in history:
        if item.index < min_iterations:
            continue
        if mode == "relative_residual":
            stop = item.relative_residual <= threshold
        elif mode == "noise_residual":
            stop = item.residual_norm2 <= threshold * effective_variance * np.trace(a).real
        elif mode == "radau":
            stop = item.radau_energy_upper <= threshold * effective_variance * n
        elif mode == "decision_residual":
            stop = (item.decision_score <= threshold and item.relative_residual <= 0.2
                    and item.stable_decisions >= 1)
        elif mode == "euclidean_bound":
            statistical_lower = effective_variance * np.sum(1 / a.diagonal().real)
            stop = item.euclidean_error_upper2 <= threshold * statistical_lower
        elif mode == "fixed":
            stop = item.index >= int(threshold)
        else:
            raise ValueError(f"Unknown stopping mode: {mode}")
        if stop:
            return item
    return history[-1]
