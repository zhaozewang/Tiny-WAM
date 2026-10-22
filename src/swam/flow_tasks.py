"""Assemble the two agreed training tasks from frozen, normalized latents.

Normalization and tokenizer fitting belong outside this module. A training
window has 241 observations and 240 actions at 24 Hz. Goal coordinates are
normalized qpos, not the qpos/qvel proprioception VAE latent. The returned goal
role is represented by the model's single shared condition embedding.
"""
import torch
from swam.flow_model import make_noisy


def prepare_window(images, proprio, actions, joint_positions, times, *, task=None,
                   camera=None, generator=None):
    """images [B,T,V,2048], proprio [B,T,512], actions [B,T-1,512].

    Select one camera for the entire window; select 2 or 3 future position goals.
    Task is sampled once per batch (50/50) to avoid padding removed image tokens.
    Pass a seeded torch.Generator on the same device for reproducible sampling.
    """
    b,t,views,_=images.shape
    assert proprio.shape[:2]==joint_positions.shape[:2]==times.shape==(b,t)
    assert actions.shape[:2]==(b,t-1) and t>=4
    device=images.device
    if task is None:
        task='full' if torch.rand((),device=device,generator=generator).item()<.5 else 'lowdim'
    if task not in ('full','lowdim'):raise ValueError(task)
    if camera is None:camera=torch.randint(views,(b,),device=device,generator=generator)
    assert camera.shape==(b,) and bool(((camera>=0)&(camera<views)).all())
    # Advanced indexing retains camera identity at every frame of each sample.
    image=images[torch.arange(b,device=device),:,camera,:]
    action=torch.cat((actions,torch.zeros_like(actions[:,:1])),1)
    clean=(image,proprio,action)
    valid=torch.ones(b,t,3,device=device,dtype=torch.bool);valid[:,-1,2]=False
    known=torch.zeros_like(valid);known[:,0]=True
    indices=torch.zeros(b,3,device=device,dtype=torch.long)
    goal_valid=torch.zeros(b,3,device=device,dtype=torch.bool)
    for n in range(b):
        count=int(torch.randint(2,4,(),device=device,generator=generator))
        indices[n,:count]=(torch.randperm(t-1,device=device,generator=generator)[:count]+1).sort().values
        goal_valid[n,:count]=True
    goals=joint_positions.gather(1,indices[...,None].expand(-1,-1,joint_positions.shape[-1]))
    goal_times=times.gather(1,indices)
    tau=torch.rand(b,t,device=device,generator=generator)
    noise=tuple(torch.randn(x.shape,device=device,dtype=x.dtype,generator=generator) for x in clean)
    noisy,target=make_noisy(clean,tau,known,valid,noise)
    inputs=dict(clean=clean,noisy=noisy,tau=tau,times=times,known=known,valid=valid,
                task=task,goals=goals,goal_times=goal_times,goal_valid=goal_valid)
    return inputs,target,dict(camera=camera,goal_indices=indices,goal_valid=goal_valid,task=task)
