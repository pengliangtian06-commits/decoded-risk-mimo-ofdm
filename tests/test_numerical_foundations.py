import numpy as np
import pytest
from mimo_receiver.qam import constellation, modulate, hard_demodulate, maxlog_llr
from mimo_receiver.channels import conditional_rayleigh, complex_normal, linear_system
from mimo_receiver.solvers import cg_trace, cholesky_solve, choose_iteration


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_qam_all_labels_roundtrip_energy_and_llr_sign(order):
    points, labels = constellation(order)
    symbols = modulate(labels, order)
    np.testing.assert_allclose(symbols[:, 0], points)
    np.testing.assert_array_equal(hard_demodulate(symbols, order), labels)
    np.testing.assert_array_equal(maxlog_llr(symbols, 0.001, order) < 0, labels)
    assert np.isclose(np.mean(abs(points) ** 2), 1)
    side = int(np.sqrt(order))
    grid = labels.reshape(side, side, -1)
    assert np.all(np.sum(grid[:, 1:] != grid[:, :-1], axis=-1) == 1)
    assert np.all(np.sum(grid[1:] != grid[:-1], axis=-1) == 1)


@pytest.mark.parametrize("precondition", [False, True])
@pytest.mark.parametrize("correlation", [0.0, 0.8])
def test_complex_cg_matches_direct_and_monitors_energy(precondition, correlation):
    rng = np.random.default_rng(1101)
    h, hbar, err = conditional_rayleigh(rng, 32, 12, correlation, 0.02)
    y = h @ complex_normal(rng, (12,)) + 0.1 * complex_normal(rng, (32,))
    a, b, lam = linear_system(hbar, y, 0.01, err)
    exact = cholesky_solve(a, b)
    history = cg_trace(a, b, lam, precondition=precondition)
    np.testing.assert_allclose(history[-1].solution, exact, atol=2e-10, rtol=2e-10)
    errors = []
    for item in history:
        delta = exact - item.solution
        energy = np.vdot(delta, a @ delta).real
        errors.append(energy)
        assert energy <= item.radau_energy_upper * (1 + 1e-7) + 1e-11
        assert np.vdot(delta, delta).real <= item.euclidean_error_upper2 * (1 + 1e-7) + 1e-11
    assert np.all(np.diff(errors) <= 1e-10)


def test_stopping_uses_no_truth_and_zero_rhs_is_safe():
    a = np.array([[2, 0.2j], [-0.2j, 1.1]])
    zero = cg_trace(a, np.zeros(2), 0.5)
    assert zero[0].index == 0
    h = cg_trace(a, np.array([1 + 2j, 2 - 1j]), 0.5)
    for mode in ["relative_residual", "noise_residual", "radau", "euclidean_bound", "fixed"]:
        selected = choose_iteration(h, mode, 0.01, 0.5, a)
        assert selected.index >= 2


def test_conditionally_generated_error_has_expected_variance():
    rng = np.random.default_rng(1102)
    errors = []
    for _ in range(200):
        h, hbar, error = conditional_rayleigh(rng, 16, 4, 0.3, 0.1)
        errors.append(np.mean(abs(h - hbar) ** 2))
    assert np.isclose(np.mean(errors), error, rtol=0.06)


def test_invalid_signal_parameters_are_rejected():
    with pytest.raises(ValueError):
        constellation(8)
    with pytest.raises(ValueError):
        modulate(np.array([0, 1, 2, 0]), 16)
    with pytest.raises(ValueError):
        maxlog_llr(np.array([1j]), -1, 4)
