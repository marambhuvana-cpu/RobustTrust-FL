from __future__ import annotations
import numpy as np
import torch


def _minmax(values):
    a=np.asarray(values,dtype=float); lo=np.nanmin(a); hi=np.nanmax(a)
    if hi-lo<1e-12: return np.zeros_like(a)
    return (a-lo)/(hi-lo)

class AttackShield:
    def __init__(self, threshold=0.60, weights=(0.25,0.25,0.25,0.25), reference='mean'):
        self.threshold=float(threshold); self.weights=np.asarray(weights,dtype=float); self.weights/=self.weights.sum()
        self.reference=reference

    def analyze(self, client_ids, updates, previous_updates):
        if not updates: return {}
        stack=torch.stack(updates)
        if self.reference=='median': ref=torch.median(stack,dim=0).values
        else: ref=torch.mean(stack,dim=0)
        norms=torch.linalg.vector_norm(stack,dim=1).cpu().numpy(); mu=float(np.mean(norms)); sd=float(np.std(norms))+1e-12
        dev=[]; cosdis=[]; ganom=[]; temp=[]
        for cid,u,n in zip(client_ids,updates,norms):
            dev.append(float(torch.linalg.vector_norm(u-ref)))
            denom=float(torch.linalg.vector_norm(u)*torch.linalg.vector_norm(ref))+1e-12
            cos=float(torch.dot(u,ref))/denom; cosdis.append(np.clip((1-cos)/2,0,1))
            ganom.append(abs(float(n)-mu)/sd)
            pu=previous_updates.get(cid)
            if pu is None: temp.append(0.5)
            else:
                d=float(torch.linalg.vector_norm(u)*torch.linalg.vector_norm(pu))+1e-12
                c=float(torch.dot(u,pu))/d; temp.append(np.clip((1-c)/2,0,1))
        devn=_minmax(dev); gnorm=_minmax(ganom); t=np.asarray(temp); c=np.asarray(cosdis)
        risk=self.weights[0]*devn+self.weights[1]*c+self.weights[2]*gnorm+self.weights[3]*t
        out={}
        for j,cid in enumerate(client_ids):
            out[cid]={'deviation':float(devn[j]),'cosine_dissimilarity':float(c[j]),
                      'norm_anomaly':float(gnorm[j]),'temporal_inconsistency':float(t[j]),
                      'risk':float(np.clip(risk[j],0,1)),
                      'accepted':bool(risk[j] <= self.threshold)}
        return out
