"""Lossless XOR-delta checkpoints against the recorded initialization."""
from __future__ import annotations

import hashlib
import os
import zlib
import tempfile
from pathlib import Path

import torch


def reference_hash(initial):
    h=hashlib.sha256()
    for name,tensor in sorted(initial.items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_delta(path,state,initial):
    if set(state)!=set(initial):raise ValueError('Delta checkpoint parameter mismatch')
    packed={}
    for name,tensor in state.items():
        x=tensor.detach().cpu().contiguous();ref=initial[name].contiguous()
        if x.dtype!=ref.dtype or x.shape!=ref.shape:raise ValueError('Delta reference shape/dtype mismatch')
        delta=torch.bitwise_xor(x.view(torch.uint8),ref.view(torch.uint8))
        packed[name]=zlib.compress(delta.numpy().tobytes(),level=6)
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save({'format':'xor-zlib-v1','reference_sha256':reference_hash(initial),'parameters':packed},tmp)
    os.replace(tmp,path)


def load_state(path,initial):
    path=Path(path)
    if not path.exists() and path.with_suffix(path.suffix+'.remote.json').exists():
        from .checkpoint_archive import restore
        with tempfile.TemporaryDirectory(prefix='loop-checkpoint-restore-',dir='/tmp') as folder:
            local=Path(folder)/'checkpoint.pt'
            restore(path.with_suffix(path.suffix+'.remote.json'),local)
            return load_state(local,initial)
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('format')!='xor-zlib-v1':return state
    if state['reference_sha256']!=reference_hash(initial):
        raise ValueError('Delta checkpoint initialization hash mismatch')
    if set(state['parameters'])!=set(initial):raise ValueError('Delta checkpoint parameter mismatch')
    decoded={}
    for name,packed in state['parameters'].items():
        ref=initial[name].contiguous()
        delta=torch.frombuffer(bytearray(zlib.decompress(packed)),dtype=torch.uint8)
        if delta.numel()!=ref.numel()*ref.element_size():raise ValueError('Delta checkpoint size mismatch')
        restored=torch.bitwise_xor(delta.reshape(ref.view(torch.uint8).shape),ref.view(torch.uint8))
        decoded[name]=restored.view(ref.dtype).reshape(ref.shape)
    return decoded
