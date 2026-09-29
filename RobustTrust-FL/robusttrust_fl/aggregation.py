from __future__ import annotations
import numpy as np
import torch


def fedavg(updates, sample_counts):
    w=np.asarray(sample_counts,dtype=float); w=w/max(w.sum(),1e-12)
    return torch.sum(torch.stack(updates)*torch.tensor(w,dtype=torch.float32)[:,None],dim=0),w


def coordinate_median(updates):
    s=torch.stack(updates); return torch.median(s,dim=0).values, np.ones(len(updates))/len(updates)


def trimmed_mean(updates, trim_ratio=0.20):
    s=torch.stack(updates); n=len(updates); k=int(np.floor(trim_ratio*n))
    vals,_=torch.sort(s,dim=0); core=vals[k:n-k] if n-2*k>0 else vals
    return torch.mean(core,dim=0),np.ones(n)/n


def _pairwise_sq(updates):
    s=torch.stack(updates); return torch.cdist(s,s,p=2)**2


def krum(updates, byzantine_f=1):
    n=len(updates)
    if n<=2*byzantine_f+2: return fedavg(updates,[1]*n)
    D=_pairwise_sq(updates); scores=[]; keep=n-byzantine_f-2
    for i in range(n): scores.append(float(torch.sort(D[i][torch.arange(n)!=i]).values[:keep].sum()))
    idx=int(np.argmin(scores)); w=np.zeros(n); w[idx]=1; return updates[idx].clone(),w


def multi_krum(updates,byzantine_f=1,m=None):
    n=len(updates)
    if n<=2*byzantine_f+2:return fedavg(updates,[1]*n)
    D=_pairwise_sq(updates); keep=n-byzantine_f-2; scores=[]
    for i in range(n):scores.append(float(torch.sort(D[i][torch.arange(n)!=i]).values[:keep].sum()))
    m=m or max(1,n-byzantine_f-2); chosen=np.argsort(scores)[:m]; w=np.zeros(n); w[chosen]=1/len(chosen)
    return torch.mean(torch.stack([updates[i] for i in chosen]),dim=0),w


def bulyan(updates,byzantine_f=1):
    n=len(updates)
    if n<4*byzantine_f+3:return trimmed_mean(updates,0.20)
    target=n-2*byzantine_f; remaining=list(range(n)); selected=[]
    while len(selected)<target and remaining:
        subset=[updates[i] for i in remaining]; agg,w=krum(subset,min(byzantine_f,max(0,(len(subset)-3)//2)))
        j=int(np.argmax(w)); selected.append(remaining.pop(j))
    sel=[updates[i] for i in selected]
    agg,_=trimmed_mean(sel,min(0.25,byzantine_f/max(1,len(sel))))
    w=np.zeros(n); w[selected]=1/len(selected); return agg,w


def robusttrust(updates, sample_counts, trust, reliability, freshness, quality, risk, accepted,
                max_weight=0.30):
    n=len(updates); scores=np.zeros(n,dtype=float)
    for i in range(n):
        if accepted[i]:
            scores[i]=max(0,float(sample_counts[i]))*max(0,trust[i])*max(0,reliability[i])*max(0,freshness[i])*max(0,quality[i])*max(0,1-risk[i])
    if scores.sum()<=1e-15:
        valid=np.asarray(accepted,dtype=bool)
        if not valid.any(): return torch.zeros_like(updates[0]), scores
        scores=valid.astype(float)
    w=scores/scores.sum()
    if max_weight and max_weight<1:
        # Iterative capped-simplex redistribution.
        for _ in range(20):
            over=w>max_weight+1e-12
            if not over.any(): break
            excess=float((w[over]-max_weight).sum()); w[over]=max_weight
            under=(~over)&(w>0)
            if under.any(): w[under]+=excess*w[under]/w[under].sum()
            else: break
        w=w/max(w.sum(),1e-12)
    agg=torch.sum(torch.stack(updates)*torch.tensor(w,dtype=torch.float32)[:,None],dim=0)
    return agg,w
