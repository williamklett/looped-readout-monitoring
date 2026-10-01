"""Stage pinned public checkpoints from login-node /tmp into job-local /tmp.

Compute nodes lack external DNS. Internal SSH is available; only public model
artifacts are transferred. Shared scratch keeps a small, auditable pointer.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

DEFAULT_ACTOR='/path/to/project/models/Ouro-1.4B'
DEFAULT_JUDGE='/path/to/project/models/Qwen2.5-3B-Instruct'


def resolve(path,job_cache):
    path=Path(path);pointer=path/'remote_model.json'
    if not pointer.exists():return str(path)
    spec=json.loads(pointer.read_text());source=spec['source']
    prefix='CLUSTER:/tmp/USER-loop-monitoring-public-models/'
    if not source.startswith(prefix) or '..' in source:
        raise ValueError('Unexpected model-cache source')
    target=Path(job_cache)/path.name
    if not (target/'download_manifest.json').exists():
        if shutil.disk_usage(job_cache).free<spec['bytes']+2*1024**3:
            raise RuntimeError('Insufficient node-local storage for public model cache')
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
                        source.rstrip('/')+'/',str(target)+'/'],check=True)
    manifest=json.loads((target/'download_manifest.json').read_text())
    if manifest['revision']!=spec['revision'] or manifest['repo']!=spec['repo']:
        raise ValueError('Staged model revision mismatch')
    return str(target)


def stage_command(command,job_cache):
    command=list(command)
    def replace(flag,default):
        if flag in command:
            i=command.index(flag)+1;command[i]=resolve(command[i],job_cache)
        else:command.extend([flag,resolve(default,job_cache)])
    if command[0] not in ['fit','audit_data']:
        replace('--model',DEFAULT_ACTOR)
    monitor=command[command.index('--monitor')+1] if '--monitor' in command else 'none'
    if command[0]=='calibrate_judges' or monitor.endswith('_judge'):
        replace('--judge-model',DEFAULT_JUDGE)
    return command
