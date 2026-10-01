"""Checksum-verified checkpoint storage outside the constrained shared quota.

Login-node /tmp is temporary, not a durable archive. Small pointer manifests and
all metrics remain on shared scratch. This module never removes the source until
the destination SHA256 matches, and only handles this study's explicit paths.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from .common import sha256,write_json

PREFIX='/tmp/USER-loop-monitoring-checkpoints/'


def remote_python(script,*args):
    cmd=shlex.join(['python3','-c',script,*map(str,args)])
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','CLUSTER',cmd],
                          check=True,capture_output=True,text=True).stdout.strip()


def archive(path,relative):
    path=Path(path);target=Path(PREFIX)/relative
    if '..' in Path(relative).parts or Path(relative).is_absolute():raise ValueError('Invalid archive path')
    digest=sha256(path);size=path.stat().st_size
    remote_python('import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); '
                  'assert not p.exists(), "Refusing archive overwrite"; '
                  'assert shutil.disk_usage(p.parent).free>int(sys.argv[2])+2*1024**3, "Archive storage full"',target,size)
    subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',str(path),
                    'CLUSTER:'+str(target)],check=True)
    actual=remote_python('import hashlib,sys; h=hashlib.sha256(); '
                         'f=open(sys.argv[1],"rb"); '
                         '[h.update(b) for b in iter(lambda:f.read(1048576),b"")]; print(h.hexdigest())',target)
    if actual!=digest:raise RuntimeError('Checkpoint archive checksum mismatch; source retained')
    record={'source':'CLUSTER:'+str(target),'sha256':digest,'bytes':size,
            'storage':'login-node temporary storage; not durable archival'}
    write_json(path.with_suffix(path.suffix+'.remote.json'),record)
    path.unlink()
    return record


def restore(pointer,target):
    record=json.loads(Path(pointer).read_text());source=record['source']
    if not source.startswith('CLUSTER:'+PREFIX) or '..' in source:raise ValueError('Unexpected archive source')
    subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',source,str(target)],check=True)
    if sha256(target)!=record['sha256']:raise RuntimeError('Restored checkpoint checksum mismatch')


if __name__=='__main__':
    import tempfile
    from .stateio import load_state,save_delta
    import torch,uuid
    initial={'w':torch.randn(8,7).bfloat16()};state={'w':initial['w']+1}
    with tempfile.TemporaryDirectory(prefix='loop-checkpoint-smoke-') as folder:
        p=Path(folder)/'final.pt';save_delta(p,state,initial)
        record=archive(p,'roundtrip-tests/'+str(uuid.uuid4())+'/final.pt')
        assert not p.exists()
        assert torch.equal(load_state(p,initial)['w'],state['w'])
        remote_python('import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.unlink(); p.parent.rmdir()',
                      record['source'].removeprefix('CLUSTER:'))
        print('ARCHIVE_ROUNDTRIP_PASSED',flush=True)
