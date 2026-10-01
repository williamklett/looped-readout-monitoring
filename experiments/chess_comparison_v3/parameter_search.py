"""Antithetic parameter-space search scored ONLY by greedy episode rewards.

Unlike a sampled-token policy gradient, this evaluates the actual requested
greedy policy. A fresh direction is regenerated for each tensor, so no full
second model or optimizer moments need be stored. Candidate weights are restored
exactly even if an environment evaluation raises. The caller must use matched
boards and hidden-coin seeds for the two candidates and baseline.

This is a development optimizer, not evidence of convergence or an experiment
result. Acceptance on training episodes requires separate held-out evaluation.
"""
from __future__ import annotations
import math
import torch


@torch.no_grad()
def search_step(model, evaluate, *, seed, scale, directions=4):
    """Accept the best strictly improving candidate; ties leave weights intact.

    evaluate() returns mean flat greedy-episode reward. scale is the standard
    deviation of independent per-parameter Gaussian perturbations to trainable
    LoRA tensors. Frozen tensors must be excluded via requires_grad=False.
    """
    if not math.isfinite(scale) or scale <= 0 or directions < 1:
        raise ValueError('Positive finite scale and directions required')
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('No trainable parameters')
    initial = {name: p.detach().clone() for name, p in parameters}
    if not all(torch.isfinite(value).all().item() for value in initial.values()):
        raise ValueError('Nonfinite starting parameters')

    def restore():
        for name, p in parameters:
            p.copy_(initial[name])

    def perturb(direction, sign):
        restore()
        for index, (name, p) in enumerate(parameters):
            generator = torch.Generator(device=p.device)
            generator.manual_seed(seed + 1000003*direction + 9176*index)
            noise = torch.randn(p.shape, generator=generator, device=p.device, dtype=p.dtype)
            p.add_(noise, alpha=sign*scale)
            if not torch.isfinite(p).all().item():
                raise ValueError('Nonfinite candidate parameters')

    accepted = None
    success = False
    records = []
    try:
        baseline = float(evaluate())
        if not math.isfinite(baseline):
            raise ValueError('Nonfinite baseline reward')
        best = baseline
        for direction in range(directions):
            for sign in (-1, 1):
                perturb(direction, sign)
                reward = float(evaluate())
                if not math.isfinite(reward):
                    raise ValueError('Nonfinite candidate reward')
                records.append(dict(direction=direction, sign=sign, reward=reward))
                if reward > best:
                    best, accepted = reward, (direction, sign)
        restore()
        if accepted is not None:
            perturb(*accepted)
        change = max(float((p-initial[name]).abs().max()) for name, p in parameters)
        if accepted is not None and change == 0:
            raise RuntimeError('Improvement claimed without a parameter change')
        success = True
        return dict(baseline_reward=baseline, accepted_reward=best,
                    accepted=accepted, max_parameter_change=change,
                    directions=directions, scale=scale, seed=seed, candidates=records)
    finally:
        if not success:
            restore()
