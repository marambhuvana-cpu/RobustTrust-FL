from __future__ import annotations
import numpy as np
import torch

DATA_ATTACKS={'label_flip','backdoor'}
UPDATE_ATTACKS={'sign_flip','gaussian','model_replacement','byzantine'}


def assign_malicious(client_states, ratio: float, attack: str, seed: int):
    rng=np.random.default_rng(seed); n=max(0,int(round(ratio*len(client_states))))
    ids=set(rng.choice(len(client_states),n,replace=False).tolist()) if n else set()
    kinds=['label_flip','sign_flip','gaussian','model_replacement','backdoor','byzantine']
    for c in client_states:
        c.malicious=c.client_id in ids
        if not c.malicious: c.attack_type='none'
        elif attack=='mixed': c.attack_type=kinds[c.client_id%len(kinds)]
        else: c.attack_type=attack


def apply_update_attack(update: torch.Tensor, attack_type: str, rng: np.random.Generator,
                        benign_std: float, sign_scale: float=5.0, replacement_scale: float=10.0):
    u=update.clone()
    if attack_type=='sign_flip': return -sign_scale*u
    if attack_type=='gaussian':
        return u + torch.from_numpy(rng.normal(0,max(1e-8,5*benign_std),size=u.numel()).astype(np.float32))
    if attack_type=='model_replacement': return replacement_scale*u
    if attack_type=='byzantine':
        mode=int(rng.integers(0,3))
        if mode==0: return -sign_scale*u
        if mode==1: return replacement_scale*u
        return torch.from_numpy(rng.normal(0,max(1e-8,8*benign_std),size=u.numel()).astype(np.float32))
    return u
