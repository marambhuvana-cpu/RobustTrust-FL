from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch

@dataclass
class ClientProfile:
    compute_factor: float
    bandwidth_mbps: float
    latency_ms: float
    packet_loss: float
    jitter_ms: float
    availability: float

@dataclass
class ClientState:
    client_id: int
    indices: np.ndarray
    profile: ClientProfile
    trust: float = 0.50
    reliability: float = 0.50
    participation: float = 0.50
    effectiveness: float = 0.50
    coordination: float = 0.50
    previous_update: torch.Tensor | None = None
    privacy_epsilon: float = 0.0
    selected_count: int = 0
    success_count: int = 0
    participated_count: int = 0
    contribution_history: list[float] = field(default_factory=list)
    malicious: bool = False
    attack_type: str = "none"
    last_generated_round: int | None = None


def make_client_profiles(n: int, seed: int) -> list[ClientProfile]:
    rng=np.random.default_rng(seed)
    out=[]
    for _ in range(n):
        out.append(ClientProfile(
            compute_factor=float(rng.uniform(0.5,2.0)),
            bandwidth_mbps=float(np.exp(rng.uniform(np.log(2),np.log(100)))),
            latency_ms=float(rng.uniform(10,500)),
            packet_loss=float(rng.uniform(0,0.10)),
            jitter_ms=float(rng.uniform(0,80)),
            availability=float(rng.uniform(0.70,1.00)),
        ))
    return out
