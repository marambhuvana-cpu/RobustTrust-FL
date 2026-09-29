from __future__ import annotations
import json, random, hashlib
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def load_parameter_vector(model: torch.nn.Module, vector: torch.Tensor) -> None:
    vector = vector.detach().to(next(model.parameters()).device)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vector[offset:offset+n].view_as(p))
            offset += n
    if offset != vector.numel():
        raise ValueError("Parameter vector length does not match model")


def parameter_hash(vector: torch.Tensor) -> str:
    arr = vector.detach().cpu().numpy().astype(np.float32, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)


def ensure_dir(path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p
