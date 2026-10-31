import numpy as np
import pytest
from mimo_receiver.qam import maxlog_llr,maxlog_llr_exhaustive
from mimo_receiver.tdl import PROFILES,quantized_pdp


@pytest.mark.parametrize('order',[4,16,64,256])
def test_separable_demapper_matches_exhaustive_metric(order):
    rng=np.random.default_rng(1139)
    z=rng.normal(size=(8,137))+1j*rng.normal(size=(8,137))
    variance=np.exp(rng.normal(size=z.shape))
    np.testing.assert_allclose(maxlog_llr(z,variance,order),
                               maxlog_llr_exhaustive(z,variance,order),rtol=1e-10,atol=1e-10)


def test_standard_profile_facts_and_quantization_conservation():
    assert len(PROFILES['TDL-A'][0])==23
    assert len(PROFILES['TDL-C'][0])==24
    assert PROFILES['TDL-A'][0][-1]==9.6586
    assert PROFILES['TDL-C'][1][-1]==-22.8
    for name in PROFILES:
        pdp,meta=quantized_pdp(name,100.,7.68e6)
        assert pdp.sum()==pytest.approx(1.)
        assert np.all(pdp>=0)
        assert meta['max_delay_samples']<=8
        # Nearest-sample quantization perturbs RMS; it is reported, not hidden.
        assert abs(meta['quantized_rms_delay_ns']-100.)<40.
