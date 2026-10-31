"""Independent reference for Yin et al. GLOBECOM 2014 Eqs. (13)--(15).

Ordinary CG with their three-level approximate equalizer recursion. The
production C++ uses an algebraically equivalent two-state companion form.
For imperfect CSI our study substitutes the common effective variance for
their noise variance. This is an explicit model adaptation, not a reproduction
of the paper's perfect-CSI WINNER/convolutional-code experiments.
"""
import numpy as np


def yin_diagonal_cg(channel, received, effective_variance, iterations):
    h=np.asarray(channel,complex); y=np.asarray(received,complex)
    a=h.conj().T@h+effective_variance*np.eye(h.shape[1])
    b=h.conj().T@y
    x=np.zeros_like(b); r=b.copy(); p=r.copy()
    old=[np.zeros(len(b)) for _ in range(3)]
    alphas=[1.,1.]; betas=[0.,0.]
    diagonal=a.diagonal().real
    for k in range(iterations):
        ap=a@p
        alpha=np.vdot(r,r).real/np.vdot(p,ap).real
        x+=alpha*p; new_r=r-alpha*ap
        beta=np.vdot(new_r,new_r).real/np.vdot(r,r).real
        if k==0:
            polynomial=np.full(len(b),alpha)
        else:
            polynomial=(old[0]+(alpha*(1+betas[0])/alphas[0]-alpha*diagonal)*(old[0]-old[1])
                        -alpha*betas[1]/alphas[1]*(old[1]-old[2]))
        old=[polynomial,old[0],old[1]]
        alphas=[alpha,alphas[0]]; betas=[beta,betas[0]]
        p=new_r+beta*p; r=new_r
    gain=polynomial*(diagonal-effective_variance)
    variance=effective_variance/(diagonal-effective_variance)
    return x,gain,variance
