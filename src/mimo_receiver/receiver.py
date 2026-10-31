"""Measured soft-output receiver using only estimated channel/noise inputs.

Sparse inverse-diagonal interpolation is an experimental engineering option,
not an asserted new interpolation technique or exact truncated-PCG SINR.
"""
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import zpotri
from . import _core
from .qam import maxlog_llr


@dataclass(frozen=True)
class DetectorPolicy:
    name: str = "exact"
    backend: str = "cholesky"
    mode: str = "relative_residual"
    threshold: float = 1e-3
    soft: str = "exact"
    anchor_stride: int = 1


def periodic_interpolate(values: np.ndarray, anchors: np.ndarray, tones: int) -> np.ndarray:
    """Periodic piecewise-linear interpolation with no extrapolation."""
    values = np.asarray(values, float)
    anchors = np.asarray(anchors)
    if (tones < 1 or values.ndim != 2 or anchors.ndim != 1 or
            values.shape[1] != len(anchors) or not len(anchors) or
            anchors[0] != 0 or np.any(np.diff(anchors) <= 0) or
            anchors[-1] >= tones or not np.all(np.isfinite(values))):
        raise ValueError("Invalid periodic interpolation grid")
    extended = np.r_[anchors, tones]
    return np.stack([np.interp(np.arange(tones), extended, np.r_[row, row[0]])
                     for row in values])


def detect_soft_reference(channel_mean: np.ndarray, received_data: np.ndarray,
                effective_variance: float, policy: DetectorPolicy,
                qam_order: int, valid_symbols: int,
                diagonal_backend: str = 'solve') -> dict:
    """Return LLRs and complete detection/soft-output wall-clock components.

    Shapes H=(B,U,N), Y=(B,S,N); all S vectors share each tone's preparation.
    No payload, true channel or true noise is available to this function.
    FFT/estimation and LDPC timing belong to the outer receiver, separately.
    """
    if diagonal_backend not in ('solve', 'lapack'):
        raise ValueError('Unknown exact diagonal backend')
    h = np.asarray(channel_mean, complex)
    y = np.asarray(received_data, complex)
    if (h.ndim != 3 or y.ndim != 3 or h.shape[0] != y.shape[0] or
            h.shape[2] != y.shape[2] or h.shape[0] < h.shape[1] or
            not np.all(np.isfinite(h)) or not np.all(np.isfinite(y)) or
            not np.isfinite(effective_variance) or effective_variance <= 0 or
            not 1 <= valid_symbols <= y.shape[1] * y.shape[2] or
            policy.soft not in ("exact", "interpolate", "diagonal") or
            policy.backend not in ("cholesky", "pcg", "cg", "matrixfree_pcg") or
            not isinstance(policy.anchor_stride, int) or policy.anchor_stride < 1):
        raise ValueError("Invalid receiver observations or detector policy")
    users, tones = h.shape[1:]
    identity = np.eye(users)
    detected = np.empty((users, y.shape[1], tones), complex)
    anchors = np.arange(0, tones, policy.anchor_stride if policy.soft == "interpolate" else 1)
    anchor_set = set(anchors.tolist())
    diagonal_values = np.empty((users, len(anchors)))
    approximate_diagonal = np.empty((users, tones))
    preparation = solving = soft_calibration = 0.
    iterations, reasons, residuals = [], [], []
    start = perf_counter()
    anchor_index = 0
    for tone in range(tones):
        mean = np.asfortranarray(h[:, :, tone])
        observation = np.asfortranarray(y[:, :, tone])
        if policy.backend == "cholesky":
            tick = perf_counter()
            system = mean.conj().T @ mean + effective_variance * identity
            factor = cho_factor(system, lower=True, check_finite=False)
            preparation += perf_counter() - tick
            tick = perf_counter()
            detected[:, :, tone] = cho_solve(factor, mean.conj().T @ observation, check_finite=False)
            solving += perf_counter() - tick
            iterations.extend([0] * y.shape[1]); reasons.extend(["direct"] * y.shape[1])
        else:
            tick = perf_counter()
            prepared = _core.PreparedSystem(mean, effective_variance, policy.backend)
            preparation += perf_counter() - tick
            tick = perf_counter()
            result = prepared.solve(observation, policy.mode, policy.threshold, 2 * users, 2, qam_order)
            solving += perf_counter() - tick
            detected[:, :, tone] = result["symbols"]
            iterations.extend(result["iterations"]); reasons.extend(result["stop_reasons"])
            residuals.extend(result["recursive_residual_norm2"])
        tick = perf_counter()
        if policy.soft == "diagonal":
            approximate_diagonal[:, tone] = 1 / (np.sum(abs(mean)**2, axis=0) + effective_variance)
        elif tone in anchor_set:
            if policy.backend != "cholesky":
                factor = cho_factor(mean.conj().T @ mean + effective_variance * identity,
                                    lower=True, check_finite=False)
            if diagonal_backend == 'lapack':
                inverse, info = zpotri(factor[0], lower=factor[1], overwrite_c=False)
                if info != 0:
                    raise RuntimeError(f'LAPACK inverse failure: {info}')
                diagonal_values[:, anchor_index] = inverse.diagonal().real
            else:
                diagonal_values[:, anchor_index] = cho_solve(factor, identity, check_finite=False).diagonal().real
            anchor_index += 1
        soft_calibration += perf_counter() - tick
    tick = perf_counter()
    if policy.soft == "exact":
        inverse_diagonal = diagonal_values
    elif policy.soft == "interpolate":
        inverse_diagonal = periodic_interpolate(diagonal_values, anchors, tones)
    else:
        inverse_diagonal = approximate_diagonal
    # For converged LMMSE under its scalar covariance model this is exact.
    # With truncated solves/interpolation it is a proxy, evaluated after LDPC.
    gains = np.clip(1 - effective_variance * inverse_diagonal, 1e-8, 1 - 1e-12)
    variances = np.maximum(1 / gains - 1, 1e-12)
    soft_calibration += perf_counter() - tick
    tick = perf_counter()
    unbiased = (detected / gains[:, None, :]).reshape(users, -1)[:, :valid_symbols]
    variance = np.broadcast_to(variances[:, None, :], detected.shape).reshape(users, -1)[:, :valid_symbols]
    llr = maxlog_llr(unbiased, variance, qam_order)
    llr_seconds = perf_counter() - tick
    return {"llr": llr, "symbols": detected, "gain_proxy": gains,
            "variance_proxy": variances, "iterations": np.asarray(iterations),
            "stop_reasons": reasons, "recursive_residual_norm2": residuals,
            "preparation_seconds": preparation, "solve_seconds": solving,
            "soft_calibration_seconds": soft_calibration, "llr_seconds": llr_seconds,
            "detection_total_seconds": perf_counter() - start,
            "exact_diagonal_tones": 0 if policy.soft == "diagonal" else len(anchors)}


def detect_soft(channel_mean: np.ndarray, received_data: np.ndarray,
                effective_variance: float, policy: DetectorPolicy,
                qam_order: int, valid_symbols: int) -> dict:
    """Native batched implementation; reference implementation above is retained."""
    h = np.asarray(channel_mean, complex)
    y = np.asarray(received_data, complex)
    if (h.ndim != 3 or y.ndim != 3 or h.shape[0] != y.shape[0] or
            h.shape[2] != y.shape[2] or h.shape[0] < h.shape[1] or
            h.shape[1] < 1 or h.shape[2] < 1 or y.shape[1] < 1 or
            not np.all(np.isfinite(h)) or not np.all(np.isfinite(y)) or
            not np.isfinite(effective_variance) or effective_variance <= 0 or
            not 1 <= valid_symbols <= y.shape[1] * y.shape[2] or
            not isinstance(policy.anchor_stride, int) or policy.anchor_stride < 1):
        raise ValueError("Invalid receiver observations or detector policy")
    users, tones = h.shape[1:]
    if policy.backend == 'auto_cholesky':
        # Choice frozen from development benchmark 002 before calibration.
        if users > 16:
            return detect_soft_reference(h,y,effective_variance,
                replace(policy,backend='cholesky'),qam_order,valid_symbols,
                diagonal_backend='lapack')
        policy = replace(policy,backend='lapack_cholesky')
    start = perf_counter()
    result = _core.detect_tones([h[:, :, tone] for tone in range(tones)],
        [y[:, :, tone] for tone in range(tones)], effective_variance,
        policy.backend, policy.mode, policy.threshold, 2*users, qam_order,
        policy.soft, policy.anchor_stride)
    detected = np.stack(result.pop('symbols_by_tone'), axis=-1)
    inverse_diagonal = result.pop('inverse_diagonal')
    tick = perf_counter()
    if policy.soft == 'yin_approx':
        raw_gains = np.stack(result.pop('approximate_gains_by_tone'), axis=-1)
        gains = np.maximum(raw_gains, 1e-8)
        variances = effective_variance / np.maximum(np.sum(abs(h)**2, axis=0),1e-12)
        variances = np.broadcast_to(variances[:,None,:], detected.shape)
    elif policy.soft == 'interpolate':
        anchors = np.arange(0, tones, policy.anchor_stride)
        inverse_diagonal = periodic_interpolate(inverse_diagonal[:, anchors], anchors, tones)
    if policy.soft != 'yin_approx':
        result.pop('approximate_gains_by_tone')
        raw_gains = 1-effective_variance*inverse_diagonal
        gains = np.clip(raw_gains, 1e-8, 1-1e-12)
        variances = np.maximum(1/gains-1, 1e-12)
    result['soft_calibration_seconds'] += perf_counter()-tick
    tick = perf_counter()
    gains_broadcast = gains[:,None,:] if gains.ndim == 2 else gains
    variances_broadcast = variances[:,None,:] if variances.ndim == 2 else variances
    unbiased = (detected/gains_broadcast).reshape(users, -1)[:, :valid_symbols]
    variance = np.broadcast_to(variances_broadcast, detected.shape).reshape(users, -1)[:, :valid_symbols]
    llr = maxlog_llr(unbiased, variance, qam_order)
    llr_seconds = perf_counter()-tick
    result.update(llr=llr, symbols=detected, gain_proxy=gains, variance_proxy=variances,
        iterations=np.asarray(result['iterations']), llr_seconds=llr_seconds,
        detection_total_seconds=perf_counter()-start,
        gain_clip_count=int(np.count_nonzero(raw_gains < 1e-8)),
        exact_diagonal_tones=0 if policy.soft in ('diagonal','yin_approx') else
            len(np.arange(0, tones, policy.anchor_stride if policy.soft == 'interpolate' else 1)))
    return result
