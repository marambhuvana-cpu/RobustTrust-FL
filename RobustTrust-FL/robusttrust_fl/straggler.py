from __future__ import annotations
import math, numpy as np

class StragglerSync:
    def __init__(self,freshness_decay=0.30,max_age=3,min_sync_ratio=0.70,max_wait_s=30.0):
        self.decay=float(freshness_decay); self.max_age=int(max_age); self.min_sync=float(min_sync_ratio); self.max_wait=float(max_wait_s)

    def communication_quality(self, profile):
        bw=np.clip((np.log(max(profile.bandwidth_mbps,1e-6))-np.log(2))/(np.log(100)-np.log(2)),0,1)
        return float(np.clip(0.45*bw+0.30*(1-profile.packet_loss)+0.25*(1-min(profile.jitter_ms/100,1)),0,1))

    def simulate_delay(self, profile, update_bytes: int, local_samples: int, epochs: int, rng):
        compute=max(0.01,(local_samples*epochs/25000)/profile.compute_factor)
        tx=(update_bytes*8)/(max(profile.bandwidth_mbps,0.1)*1e6)
        network=profile.latency_ms/1000 + abs(float(rng.normal(0,profile.jitter_ms/1000)))
        loss_penalty=tx*profile.packet_loss*3
        return compute+tx+network+loss_penalty

    def assess(self,state,delay_s,age,accepted):
        freshness=float(math.exp(-self.decay*age)); temporal=bool(age<=self.max_age)
        cq=self.communication_quality(state.profile)
        delay_norm=min(delay_s/max(self.max_wait,1e-9),1.0)
        priority=float(np.clip(0.35*state.trust+0.30*state.reliability+0.20*state.participation+0.15*(1-delay_norm),0,1))
        coord=float(np.clip(0.45*priority+0.30*freshness+0.25*cq,0,1)); state.coordination=coord
        return {'delay_s':float(delay_s),'age':int(age),'freshness':freshness,'temporal_eligible':temporal,
                'communication_quality':cq,'coordination':coord,'usable':bool(accepted and temporal)}
