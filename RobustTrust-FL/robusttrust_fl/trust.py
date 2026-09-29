from __future__ import annotations
import numpy as np

class TrustShieldNet:
    def __init__(self, weights=(0.25,0.20,0.15,0.15,0.25), memory=0.80, penalty=0.50,
                 high=0.70, low=0.30):
        self.w=np.asarray(weights,dtype=float); self.w/=self.w.sum(); self.memory=float(memory)
        self.penalty=float(penalty); self.high=float(high); self.low=float(low)

    def update(self,state,quality,risk,accepted):
        evidence=(self.w[0]*quality+self.w[1]*state.reliability+self.w[2]*state.participation+
                  self.w[3]*state.effectiveness+self.w[4]*(1-risk))
        evidence=float(np.clip(evidence,0,1))
        if accepted: new=self.memory*state.trust+(1-self.memory)*evidence
        else: new=self.penalty*state.trust
        state.trust=float(np.clip(new,0,1))
        if state.trust>=self.high: label='trusted'
        elif state.trust>=self.low: label='uncertain'
        else: label='restricted'
        return {'evidence':evidence,'trust':state.trust,'state':label}
