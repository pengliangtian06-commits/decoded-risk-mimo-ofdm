import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve
from mimo_receiver.risk import binomial_upper, select_policy
from mimo_receiver.receiver import DetectorPolicy, periodic_interpolate, detect_soft, detect_soft_reference


def test_zero_event_limit_and_familywise_fallback():
    assert binomial_upper(0, 100, 0.05) == pytest.approx(1 - 0.05**0.01)
    assert binomial_upper(100, 100, 0.05) == 1
    events = {"cheap": np.zeros(100, int), "slow": np.zeros(100, int)}
    result = select_policy(events, {"cheap": 1., "slow": 2., "exact": 1.5}, 0.02)
    assert result.selected == "exact"
    assert select_policy(events, {"cheap": 1., "slow": 2., "exact": 1.5}, 0.05).selected == "cheap"
    assert select_policy(events, {"cheap": 1., "slow": 2., "exact": .5}, 0.05).selected == "exact"
    with pytest.raises(ValueError):
        select_policy({"bad": np.array([0.5])}, {"bad": 1, "exact": 2}, 0.01)


def test_periodic_interpolation_wrap_and_identity():
    value = periodic_interpolate(np.array([[1., 3.]]), np.array([0, 4]), 8)
    np.testing.assert_allclose(value, [[1, 1.5, 2, 2.5, 3, 2.5, 2, 1.5]])
    np.testing.assert_allclose(periodic_interpolate(value, np.arange(8), 8), value)


def test_receiver_full_stride_matches_direct_reference():
    rng = np.random.default_rng(1130)
    h = (rng.normal(size=(16, 4, 8)) + 1j*rng.normal(size=(16, 4, 8))) / np.sqrt(32)
    y = rng.normal(size=(16, 2, 8)) + 1j*rng.normal(size=(16, 2, 8))
    exact = detect_soft(h, y, 0.1, DetectorPolicy(), 16, 16)
    for tone in range(8):
        factor = cho_factor(h[:, :, tone].conj().T @ h[:, :, tone] + .1*np.eye(4), lower=True)
        np.testing.assert_allclose(exact['symbols'][:, :, tone],
            cho_solve(factor, h[:, :, tone].conj().T @ y[:, :, tone]))
    iterative = detect_soft(h, y, .1, DetectorPolicy('pcg', 'pcg', threshold=1e-12,
                            soft='interpolate', anchor_stride=1), 16, 16)
    np.testing.assert_allclose(exact['llr'], iterative['llr'], atol=1e-9)
    assert exact['exact_diagonal_tones'] == 8
    assert exact['detection_total_seconds'] >= sum(exact[k] for k in
        ['preparation_seconds', 'solve_seconds', 'soft_calibration_seconds', 'llr_seconds'])


@pytest.mark.parametrize('backend', ['cholesky', 'pcg', 'cg', 'matrixfree_pcg'])
@pytest.mark.parametrize('soft', ['exact', 'diagonal', 'interpolate'])
def test_native_batch_matches_independent_reference(backend, soft):
    rng = np.random.default_rng(1132)
    h = (rng.normal(size=(24, 8, 16)) + 1j*rng.normal(size=(24, 8, 16)))/np.sqrt(48)
    y = rng.normal(size=(24, 3, 16)) + 1j*rng.normal(size=(24, 3, 16))
    policy = DetectorPolicy('test', backend, threshold=.01, soft=soft, anchor_stride=4)
    expected = detect_soft_reference(h, y, .2, policy, 64, 45)
    actual = detect_soft(h, y, .2, policy, 64, 45)
    np.testing.assert_allclose(expected['llr'], actual['llr'], atol=1e-9)
    np.testing.assert_allclose(expected['gain_proxy'], actual['gain_proxy'], atol=1e-13)
    np.testing.assert_array_equal(expected['iterations'], actual['iterations'])
    assert expected['stop_reasons'] == actual['stop_reasons']
    lapack = detect_soft_reference(h, y, .2, policy, 64, 45, diagonal_backend='lapack')
    np.testing.assert_allclose(expected['llr'], lapack['llr'], atol=1e-9)


def test_native_lapack_direct_soft_output_parity():
    rng = np.random.default_rng(1135)
    h = (rng.normal(size=(48, 32, 12)) + 1j*rng.normal(size=(48, 32, 12)))/np.sqrt(96)
    y = rng.normal(size=(48, 2, 12)) + 1j*rng.normal(size=(48, 2, 12))
    reference = detect_soft_reference(h,y,.05,DetectorPolicy(),16,24)
    result = detect_soft(h,y,.05,DetectorPolicy('lapack',backend='lapack_cholesky'),16,24)
    np.testing.assert_allclose(reference['llr'],result['llr'],atol=1e-9)
    np.testing.assert_allclose(reference['symbols'],result['symbols'],atol=1e-12)


@pytest.mark.parametrize('iterations',[2,4,8])
def test_published_cg_soft_output_against_three_level_recursion(iterations):
    from mimo_receiver.soft_reference import yin_diagonal_cg
    rng=np.random.default_rng(1136)
    h=(rng.normal(size=(32,12,6))+1j*rng.normal(size=(32,12,6)))/8
    y=rng.normal(size=(32,2,6))+1j*rng.normal(size=(32,2,6))
    policy=DetectorPolicy('yin','cg','fixed',iterations,'yin_approx')
    result=detect_soft(h,y,.15,policy,16,12)
    for tone in range(6):
        for vector in range(2):
            x,gain,variance=yin_diagonal_cg(h[:,:,tone],y[:,vector,tone],.15,iterations)
            np.testing.assert_allclose(result['symbols'][:,vector,tone],x,atol=1e-11)
            np.testing.assert_allclose(result['gain_proxy'][:,vector,tone],np.maximum(gain,1e-8),atol=1e-10)
            np.testing.assert_allclose(result['variance_proxy'][:,vector,tone],variance,atol=1e-12)
