"""Standard zero-initialized LoRA; shared frozen bases, independent adapters.

This implements the paper's low-rank parameterization, not the author's runtime.
Only A/B matrices train. Bias, normalization, embeddings, and output head freeze.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=64, alpha=128):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or alpha <= 0:
            raise ValueError('LoRA requires a Linear, positive rank and alpha')
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.enabled = True
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        # Explicit FP32 adapters avoid vanishing BF16 updates. Base output stays
        # native dtype. Disable outer autocast for this small branch.
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.A), self.B) * (self.alpha / self.rank)
        return y + delta.to(y.dtype)


def attach(module, rank=64, alpha=128):
    """Replace every Linear below one transformer block, excluding nothing else."""
    names = []
    for path, child in list(module.named_modules()):
        if path and isinstance(child, nn.Linear):
            parent_path, _, name = path.rpartition('.')
            parent = module.get_submodule(parent_path) if parent_path else module
            setattr(parent, name, LoRALinear(child, rank, alpha))
            names.append(path)
    if not names:
        raise ValueError('No Linear modules found in transformer block')
    return names


def clone_with_shared_frozen_parameters(module):
    if any(p.requires_grad for p in module.parameters()):
        raise ValueError('Only frozen parameter storage may be shared')
    memo = {id(p): p for p in module.parameters()}
    memo.update({id(b): b for b in module.buffers()})
    return copy.deepcopy(module, memo=memo)


@contextlib.contextmanager
def disabled(module):
    layers = [m for m in module.modules() if isinstance(m, LoRALinear)]
    previous = [m.enabled for m in layers]
    try:
        for m in layers:
            m.enabled = False
        yield
    finally:
        for m, enabled in zip(layers, previous):
            m.enabled = enabled


def frozen_digest(module):
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def adapter_state(module):
    return {n: p.detach().cpu().clone() for n, p in module.named_parameters() if p.requires_grad}


def load_adapter_state(module, state):
    params = {n: p for n, p in module.named_parameters() if p.requires_grad}
    if set(params) != set(state):
        raise ValueError('Adapter parameter set does not match')
    with torch.no_grad():
        for name, p in params.items():
            x = state[name]
            if x.shape != p.shape or not torch.isfinite(x).all():
                raise ValueError('Invalid adapter tensor: ' + name)
            p.copy_(x)
