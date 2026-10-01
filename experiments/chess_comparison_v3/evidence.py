"""CPU-readable tensor fingerprints and strict greedy-return selection checks."""
import hashlib
import json
import math

import torch


def tensor_sha(state):
    digest = hashlib.sha256()
    assert state
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        assert torch.isfinite(tensor).all(), 'Nonfinite adapter tensor'
        header = json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode()
        digest.update(len(header).to_bytes(8,'big'));digest.update(header)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    return digest.hexdigest()


def model_tensor_sha(model):
    return tensor_sha({n:p for n,p in model.named_parameters() if p.requires_grad})


def verify_choice(rewards, result):
    assert len(rewards) == 1+2*result['directions']
    assert all(math.isfinite(r) for r in rewards)
    assert result['baseline_reward'] == rewards[0]
    assert len(result['candidates']) == len(rewards)-1
    best, selected = rewards[0], None
    for i,(reward,record) in enumerate(zip(rewards[1:],result['candidates'])):
        direction,sign = i//2,(-1 if i%2 == 0 else 1)
        assert record == dict(direction=direction,sign=sign,reward=reward)
        if reward > best:
            best,selected = reward,[direction,sign]
    assert result['accepted'] == selected and result['accepted_reward'] == best
    assert math.isfinite(result['max_parameter_change'])
    assert (result['max_parameter_change'] > 0) if selected is not None else (result['max_parameter_change'] == 0)
    assert (result['adapter_tensor_sha_before'] != result['adapter_tensor_sha_after']) == (selected is not None)
