"""Quantized NLOS TDL-A/C PDPs from TR 38.901 v18.0.0 Tables 7.7.2-1/-3.

This provides normalized powers and delays only. A static block with Toeplitz
spatial correlation is an explicit link-study adaptation, not full 3GPP channel
or Doppler-conformance implementation. Original normalized delays are scaled
then rounded to the nearest sample; powers in coincident bins are summed.
"""
import numpy as np

PROFILES={
    'TDL-A':(
        [0,.3819,.4025,.5868,.4610,.5375,.6708,.5750,.7618,1.5375,1.8978,
         2.2242,2.1718,2.4942,2.5119,3.0582,4.0810,4.4579,4.5695,4.7966,
         5.0066,5.3043,9.6586],
        [-13.4,0,-2.2,-4,-6,-8.2,-9.9,-10.5,-7.5,-15.9,-6.6,-16.7,
         -12.4,-15.2,-10.8,-11.3,-12.7,-16.2,-18.3,-18.9,-16.6,-19.9,-29.7]),
    'TDL-C':(
        [0,.2099,.2219,.2329,.2176,.6366,.6448,.6560,.6584,.7935,.8213,.9336,
         1.2285,1.3083,2.1704,2.7105,4.2589,4.6003,5.4902,5.6077,6.3065,
         6.6374,7.0427,8.6523],
        [-4.4,-1.2,-3.5,-5.2,-2.5,0,-2.2,-3.9,-7.4,-7.1,-10.7,-11.1,
         -5.1,-6.8,-8.7,-13.2,-13.9,-13.9,-15.8,-17.1,-16,-15.7,-21.6,-22.8])}


def quantized_pdp(profile: str,rms_delay_ns: float,sampling_rate_hz: float):
    if profile not in PROFILES or not np.isfinite(rms_delay_ns) or rms_delay_ns<=0 or not np.isfinite(sampling_rate_hz) or sampling_rate_hz<=0:
        raise ValueError('Unknown profile or invalid physical scale')
    delay,power=map(np.asarray,PROFILES[profile])
    physical=delay*rms_delay_ns*1e-9
    power=10.**(power/10);power/=power.sum()
    bins=np.floor(physical*sampling_rate_hz+.5).astype(int)
    pdp=np.bincount(bins,weights=power)
    times=np.arange(len(pdp))/sampling_rate_hz
    mean=float(pdp@times)
    actual=float(np.sqrt(pdp@(times-mean)**2)*1e9)
    return pdp,{'profile':profile,'nominal_rms_delay_ns':rms_delay_ns,
        'quantized_rms_delay_ns':actual,'sampling_rate_hz':sampling_rate_hz,
        'max_delay_samples':int(bins.max()),'original_paths':len(delay),
        'occupied_bins':int(np.count_nonzero(pdp)),'quantization':'nearest sample; coincident powers summed',
        'source':'ETSI TR 138 901 V18.0.0 (2024-05), Tables 7.7.2-1 and 7.7.2-3'}
