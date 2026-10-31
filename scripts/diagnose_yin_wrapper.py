#!/usr/bin/env python3
"""Post-lock sensitivity check of one equivalent Yin wrapper norm reduction.

Does not replace any locked source, policy, calibration or primary outcome.
An AST transform changes only sum(abs(H)**2, axis=0) to a real/imag einsum.
This probes implementation overhead, not a new detector or a SOTA comparison.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import ast,hashlib,inspect,itertools,json,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from mimo_receiver import receiver
from mimo_receiver.coding import LDPCCodec,LDPCConfig
from mimo_receiver.qam import modulate
from mimo_receiver.ofdm import (OFDMConfig,exponential_pdp,sample_taps,make_grid,
    modulate_grid,time_channel,demodulate_waveform,estimate_from_pilots)
from run_locked_study import source_check

def channel_power(h):
    return np.einsum('bun,bun->un',h.real,h.real)+np.einsum('bun,bun->un',h.imag,h.imag)

def variant():
    original=inspect.getsource(receiver.detect_soft);tree=ast.parse(original)
    expression=ast.dump(ast.parse('np.sum(abs(h)**2, axis=0)',mode='eval').body)
    class ReplaceNorm(ast.NodeTransformer):
        changed=0
        def visit_Call(self,node):
            if ast.dump(node)==expression:
                self.changed+=1;return ast.copy_location(ast.parse('channel_power(h)',mode='eval').body,node)
            return self.generic_visit(node)
    transform=ReplaceNorm();tree=transform.visit(tree);assert transform.changed==1
    namespace=dict(receiver.__dict__);namespace['channel_power']=channel_power
    exec(compile(ast.fix_missing_locations(tree),'<equivalent norm reduction diagnostic>','exec'),namespace)
    return namespace['detect_soft'],hashlib.sha256(original.encode()).hexdigest()

def main():
    lock=json.loads((ROOT/'research/protocols/study_002_lock.json').read_text());source_check(lock)
    out=ROOT/'results/diagnostics/yin_wrapper_001';out.mkdir(parents=True,exist_ok=False)
    optimized,digest=variant();settings=lock['settings'];codec=LDPCCodec(LDPCConfig())
    policy={p['name']:receiver.DetectorPolicy(**p) for p in settings['policies']}
    names=['exact','direct-i32','yin8','yin8-norm-variant']
    conditions=list(itertools.product(settings['antenna_user_pairs'],settings['correlations'],settings['snr_db']))
    order=np.random.default_rng(6150);rows=[];parity=[]
    tasks=[(c,d) for c in range(12) for d in range(2)]
    tasks=[tasks[i] for i in order.permutation(len(tasks))]
    for ci,draw in tasks:
        (b,u),rho,snr=conditions[ci];cfg=OFDMConfig(b,u,512,32,8)
        rng=np.random.default_rng(np.random.SeedSequence([6100,ci,draw]))
        payload=rng.integers(0,2,(u,1024),dtype=np.uint8)
        encoded=np.stack([codec.encode(p) for p in payload]);grid,valid=make_grid(modulate(encoded,16),cfg)
        pdp=exponential_pdp(8);taps=sample_taps(rng,cfg,pdp,rho)
        waveform=time_channel(modulate_grid(grid,cfg),taps,10**(-snr/10),rng)
        f=demodulate_waveform(waveform,cfg);e=estimate_from_pilots(f,cfg,pdp)
        args=(e['channel_mean'],f[:,1:],e['effective_variance'],policy['yin8'],16,valid)
        baseline=receiver.detect_soft(*args);alternative=optimized(*args)
        for key in ['llr','symbols','gain_proxy','variance_proxy']:
            np.testing.assert_allclose(alternative[key],baseline[key],rtol=2e-12,atol=1e-10)
        baseline_bits=np.stack([codec.decode(v)[0] for v in baseline['llr']])
        alt_bits=np.stack([codec.decode(v)[0] for v in alternative['llr']]);assert np.array_equal(alt_bits,baseline_bits)
        parity.append({'frame_id':f'{ci}:{draw}','llr_max_abs_difference':float(np.max(abs(alternative['llr']-baseline['llr']))),'decoded_bits_identical':True})
        known={}
        for rep in range(-3,5):
            for index in order.permutation(len(names)):
                name=names[index];p=policy['yin8'] if name.endswith('variant') else policy[name]
                fun=optimized if name.endswith('variant') else receiver.detect_soft
                start=perf_counter();frequency=demodulate_waveform(waveform,cfg);est=estimate_from_pilots(frequency,cfg,pdp)
                detected=fun(est['channel_mean'],frequency[:,1:],est['effective_variance'],p,16,valid)
                decoded=np.stack([codec.decode(v)[0] for v in detected['llr']]);elapsed=perf_counter()-start
                errors=np.sum(decoded!=payload,axis=1).tolist()
                if name in known:assert known[name]==errors
                else:known[name]=errors
                if rep>=0:rows.append({'frame_id':f'{ci}:{draw}','condition_id':ci,'repetition':rep,'method':name,
                    'receiver_ms':elapsed*1000,'soft_calibration_ms':detected['soft_calibration_seconds']*1000})
        assert known['yin8']==known['yin8-norm-variant']
        print('Equivalent wrapper check',len(parity),'/24',flush=True)
    (out/'timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    means={name:float(np.mean([r['receiver_ms'] for r in rows if r['method']==name])) for name in names}
    soft={name:float(np.mean([r['soft_calibration_ms'] for r in rows if r['method']==name])) for name in names}
    source_check(lock)
    report={'completed':True,'status':'post-lock implementation sensitivity; not confirmatory policy selection',
        'fresh_seed_root':6100,'independent_frames':24,'balanced_conditions':12,'warmups':3,'calls_per_frame':5,
        'locked_function_source_sha256':digest,'transform':'only channel power: sum(abs(h)**2) to real/imag einsum',
        'locked_implementation_unchanged':True,'mean_receiver_ms':means,'mean_gain_stage_ms':soft,
        'parity':parity,'maximum_llr_abs_difference':max(r['llr_max_abs_difference'] for r in parity),
        'independent_machine_replication':False}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(means),flush=True)
if __name__=='__main__':main()
