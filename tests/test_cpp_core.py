import numpy as np
import pytest
from mimo_receiver.channels import conditional_rayleigh, complex_normal, linear_system
from mimo_receiver.solvers import cg_trace, choose_iteration, cholesky_solve

core = pytest.importorskip("mimo_receiver._core")


@pytest.mark.parametrize("backend", ["cg", "pcg", "matrixfree_pcg", "cholesky", "lapack_cholesky"])
def test_compiled_direct_and_iterative_solvers(backend):
    rng = np.random.default_rng(1104)
    h, mean, err = conditional_rayleigh(rng, 32, 8, 0.6, 0.03)
    y = h @ complex_normal(rng, (8, 5)) + 0.1 * complex_normal(rng, (32, 5))
    a, b, lam = linear_system(mean, y, 0.01, err)
    prepared = core.PreparedSystem(mean, lam, backend)
    result = prepared.solve(y, "relative_residual", 1e-11, 32, 2)
    np.testing.assert_allclose(prepared.system, a, atol=1e-13)
    np.testing.assert_allclose(result["symbols"], cholesky_solve(a, b), atol=1e-9, rtol=1e-9)
    assert result["solve_seconds"] > 0
    assert len(result["stop_reasons"]) == 5


@pytest.mark.parametrize("mode", ["relative_residual", "noise_residual", "radau", "euclidean_bound", "decision_residual", "fixed"])
def test_compiled_stopping_agrees_with_reference(mode):
    rng = np.random.default_rng(1105)
    h, mean, err = conditional_rayleigh(rng, 48, 16, 0.7, 0.02)
    y = h @ complex_normal(rng, (16, 4)) + 0.1 * complex_normal(rng, (48, 4))
    a, b, lam = linear_system(mean, y, 0.01, err)
    threshold = 6 if mode == "fixed" else 0.01
    result = core.PreparedSystem(mean, lam).solve(y, mode, threshold, 32, 2)
    for j in range(y.shape[1]):
        history = cg_trace(a, b[:, j], lam)
        selected = choose_iteration(history, mode, threshold, lam, a)
        assert result["iterations"][j] == selected.index
        np.testing.assert_allclose(result["symbols"][:, j], selected.solution, atol=1e-10)


def test_compiled_zero_and_invalid_input():
    prepared = core.PreparedSystem(np.eye(3, dtype=complex), 0.1)
    result = prepared.solve(np.zeros((3, 2), complex))
    assert result["iterations"] == [0, 0]
    assert result["stop_reasons"] == ["zero_rhs", "zero_rhs"]
    with pytest.raises(ValueError):
        prepared.solve(np.zeros((2, 2), complex))
    with pytest.raises(ValueError):
        core.PreparedSystem(np.eye(3), -1)
