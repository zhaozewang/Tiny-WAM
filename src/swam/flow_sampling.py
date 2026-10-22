"""Autoregressive flow integration without ground-truth future latent access."""
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

@torch.inference_mode()
def next_frame(model,history,times,goals,goal_times,*,task='full',steps=8,generator=None,terminal=False):
    """history contains only available past frames, including the supplied a_0.

    The new frame starts at independent Gaussian noise and is Euler-integrated
    from tau=0 to 1. No noisy copy of the ground-truth target is accepted here.
    """
    b=history[0].shape[0];t=history[0].shape[1]+1;device=history[0].device
    assert steps>=1 and times.shape==(b,t)
    clean=tuple(torch.cat((v,torch.zeros_like(v[:,:1])),1) for v in history)
    z=[torch.randn(b,1,v.shape[-1],device=device,generator=generator) for v in history]
    known=torch.ones(b,t,3,device=device,dtype=torch.bool);known[:,-1]=False
    valid=torch.ones_like(known)
    if terminal:valid[:,-1,2]=False
    for k in range(steps):
        tau=torch.ones(b,t,device=device);tau[:,-1]=k/steps
        noisy=tuple(torch.cat((history[m],z[m]),1) for m in range(3))
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]),torch.autocast(device.type,dtype=torch.bfloat16):
            out=model(clean,noisy,tau,times,known,valid,task=task,goals=goals,goal_times=goal_times)
        for m in range(3):
            if (m==0 and task=='lowdim') or (m==2 and terminal):continue
            z[m]=z[m]+out.velocity[m][:,-1:].float()/steps
    if task=='lowdim':z[0]=torch.zeros_like(z[0])
    if terminal:z[2]=torch.zeros_like(z[2])
    if not all(torch.isfinite(v).all() for v in z):raise RuntimeError('Non-finite rollout')
    return tuple(z)

@torch.inference_mode()
def rollout(model,initial,times,goals,goal_times,*,task='full',steps=8,generator=None,progress=None):
    """Initial is exactly one frame. Future ground truth cannot enter this API."""
    assert all(v.shape[1]==1 for v in initial)
    history=tuple(v.clone() for v in initial)
    for t in range(1,times.shape[1]):
        z=next_frame(model,history,times[:,:t+1],goals,goal_times,task=task,steps=steps,generator=generator,terminal=t==times.shape[1]-1)
        history=tuple(torch.cat((a,b),1) for a,b in zip(history,z))
        if progress:progress(t)
    return history
