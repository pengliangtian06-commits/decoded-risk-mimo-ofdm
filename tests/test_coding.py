import numpy as np
import pytest
from mimo_receiver.coding import LDPCCodec, LDPCConfig
from mimo_receiver.qam import modulate, maxlog_llr
import py3gpp
from mimo_receiver import _core


@pytest.mark.parametrize("order", [4, 16, 64])
def test_ldpc_crc_rate_matching_modulation_noiseless_roundtrip(order):
    rng = np.random.default_rng(1110 + order)
    codec = LDPCCodec(LDPCConfig(qam_order=order, max_iterations=8))
    payload = rng.integers(0, 2, 1024, dtype=np.uint8)
    coded = codec.encode(payload)
    symbols = modulate(coded[None, :], order)
    llr = maxlog_llr(symbols, 1e-3, order)[0]
    decoded, crc_failed, iterations = codec.decode(llr)
    np.testing.assert_array_equal(decoded, payload)
    assert not crc_failed
    assert iterations <= 8


def test_ldpc_crc_detects_forced_llr_corruption():
    codec = LDPCCodec(LDPCConfig(max_iterations=3))
    payload = np.random.default_rng(1111).integers(0, 2, 1024, dtype=np.uint8)
    transmitted = codec.encode(payload)
    wrong_llrs = -100 * (1 - 2 * transmitted.astype(float))
    decoded, crc_failed, _ = codec.decode(wrong_llrs)
    assert crc_failed
    assert np.any(decoded != payload)


@pytest.mark.parametrize("seed", [1115, 1116, 1117])
def test_cpp_decoder_matches_reference_at_fixed_iterations(seed):
    codec = LDPCCodec(LDPCConfig(max_iterations=4))
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 2, 1024, dtype=np.uint8)
    transmitted = codec.encode(payload)
    llr = (1-2*transmitted.astype(float))*2.0 + 1.5*rng.standard_normal(len(transmitted))
    recovered = py3gpp.nrRateRecoverLDPC(llr,1024,0.5,0,'16QAM',1)
    expected, _ = py3gpp.nrLDPCDecode(recovered,2,4,Algorithm='Normalized min-sum',ScalingFactor=0.75)
    result = _core.decode_ldpc(recovered,codec.base_graph,int(codec.info['Zc']),4,0.75,False)
    np.testing.assert_array_equal(result['bits'],expected)
    assert result['iterations'] == [4]
