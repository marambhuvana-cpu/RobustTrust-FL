from __future__ import annotations
import math
import numpy as np
import torch

class PrivacyGuard:
    def __init__(self, clipping=1.0, noise_multiplier=0.8, delta=1e-5,
                 max_epsilon=8.0, enabled=True):
        self.clipping=float(clipping); self.noise_multiplier=float(noise_multiplier)
        self.delta=float(delta); self.max_epsilon=float(max_epsilon); self.enabled=bool(enabled)

    def protect(self, update: torch.Tensor, state, sampling_rate: float, round_number: int,
                rng: np.random.Generator):
        if not self.enabled:
            return update.clone(), True, state.privacy_epsilon
        norm=float(torch.linalg.vector_norm(update))
        scale=min(1.0,self.clipping/(norm+1e-12)); clipped=update*scale
        noise=torch.from_numpy(rng.normal(0,self.noise_multiplier*self.clipping,
                                          size=clipped.numel()).astype(np.float32))
        protected=clipped+noise
        # Reproducible analytical estimate; conservative simulation metric, not a formal accountant.
        if self.noise_multiplier>0:
            eps=sampling_rate*math.sqrt(2*round_number*math.log(1/max(self.delta,1e-12)))/self.noise_multiplier
        else: eps=float('inf')
        state.privacy_epsilon=max(state.privacy_epsilon,float(eps))
        return protected, state.privacy_epsilon <= self.max_epsilon, state.privacy_epsilon
