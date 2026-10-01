"""Stream pinned Hugging Face files via login to allocated node storage.

Reads the existing HF token from stdin; never writes it or logs URLs/headers.
No model or repository code is executed by this downloader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import requests

BASE = ('meta-llama/Meta-Llama-3-8B-Instruct', '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
ADAPTER = ('AlignmentResearch/obfuscation-atlas-Meta-Llama-3-8B-Instruct-kl0.001-det1-seed1-mbpp_probe',
           '86f79c0bd99b53b0dc8d5d31224ff8fb9d3a34cb')
CONTROL = ('AlignmentResearch/obfuscation-atlas-Meta-Llama-3-8B-Instruct-kl0.001-det0-seed1',
           '23468ab1b99e887d02d9def79494b37ca4f7853e')

WRITER = '''import hashlib,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);n=int(sys.argv[2]);expected=sys.argv[3]
p.parent.mkdir(parents=True,exist_ok=True)
part=p.with_name(p.name+'.part')
h=hashlib.sha256();count=0
try:
 with part.open('xb') as f:
  while True:
   b=sys.stdin.buffer.read(1024**2)
   if not b: break
   f.write(b);h.update(b);count+=len(b)
 if count!=n or (expected!='-' and h.hexdigest()!=expected): raise ValueError('size/hash mismatch')
 if p.exists(): raise FileExistsError('refusing to overwrite existing model file')
 part.rename(p)
 print(json.dumps({'bytes':count,'sha256':h.hexdigest()}),flush=True)
except BaseException:
 part.unlink(missing_ok=True)
 raise
'''

VERIFIER = '''import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);n=int(sys.argv[2]);expected=sys.argv[3]
if not p.exists():
 print('null');sys.exit(0)
if not p.is_file() or p.is_symlink() or p.stat().st_size!=n: raise ValueError('existing file size/type mismatch')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
if expected=='-' or h.hexdigest()!=expected: raise ValueError('existing file has no matching trusted digest')
print(json.dumps({'bytes':n,'sha256':h.hexdigest()}))
'''


def node_verify(node, target, size, expected):
    import shlex
    args = [sys.executable if node == 'local' else 'python3', '-c', VERIFIER, str(target), str(size), expected or '-']
    cmd = args if node == 'local' else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node,
           ' '.join(shlex.quote(x) for x in args)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode: raise RuntimeError('Node verification failed for ' + target.name + ': ' + result.stderr[-1200:])
    return json.loads(result.stdout)


def node_write(node, target, chunks, size, expected=None):
    import shlex
    args = [sys.executable if node == 'local' else 'python3', '-c', WRITER, str(target), str(size), expected or '-']
    cmd = args if node == 'local' else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node,
           ' '.join(shlex.quote(x) for x in args)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for chunk in chunks:
            if chunk: p.stdin.write(chunk)
        p.stdin.close(); p.stdin = None
        stdout, stderr = p.communicate(timeout=120)
        if p.returncode: raise RuntimeError('Node writer failed for ' + target.name + ': ' + stderr.decode()[-1200:])
        return json.loads(stdout)
    except BaseException:
        p.kill(); p.wait()
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--node', required=True); p.add_argument('--root', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--defer-ready', action='store_true', help='Require separate safe adapter conversion before evaluation')
    a = p.parse_args()
    if not re.fullmatch(r'(adroit-h11g[123]|local)', a.node): raise ValueError('Unexpected compute node')
    durable = '/path/to/project/models/atlas-pinned-v1'
    if not (re.fullmatch(r'/(tmp|dev/shm)/USER-atlas-[0-9]+', a.root) or a.root == durable):
        raise ValueError('Unexpected staging path')
    if a.manifest.exists(): raise ValueError('Manifest already exists')
    token = sys.stdin.read().strip()
    if not token: raise ValueError('No authorized Hugging Face credential on stdin')
    headers = {'Authorization': 'Bearer ' + token, 'User-Agent': 'loop-boundary-research/1'}
    partial = a.manifest.with_name(a.manifest.name+'.partial')
    manifest = {'node': a.node, 'root': a.root, 'models': {}}
    if a.resume and partial.exists():
        manifest = json.loads(partial.read_text())
        if manifest['node'] != a.node or manifest['root'] != a.root: raise ValueError('Resume destination mismatch')
    for key, (repo, revision) in [('base', BASE), ('adapter', ADAPTER), ('control', CONTROL)]:
        response = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', headers=headers, timeout=30)
        if response.status_code != 200: raise RuntimeError('Metadata access failed: ' + str(response.status_code))
        metadata = response.json()
        if metadata['sha'] != revision: raise RuntimeError('Revision mismatch')
        chosen = []
        for entry in metadata['siblings']:
            name = entry['rfilename']
            if '/' in name or not re.fullmatch(r'[A-Za-z0-9_.-]+', name): continue
            if (name.endswith('.safetensors') or name in ('config.json','generation_config.json','model.safetensors.index.json',
                'tokenizer.json','tokenizer_config.json','special_tokens_map.json','adapter_config.json','README.md','LICENSE','USE_POLICY.md',
                'adapter_model.bin','chat_template.jinja')):
                chosen.append(entry)
        if not any(x['rfilename'].endswith('.safetensors') or x['rfilename']=='adapter_model.bin' for x in chosen): raise RuntimeError('No model weights')
        output = manifest['models'].get(key, {'repo': repo, 'revision': revision, 'files': {}})
        if output['repo'] != repo or output['revision'] != revision: raise ValueError('Resume source mismatch')
        manifest['models'][key] = output
        for entry in sorted(chosen, key=lambda x: x['rfilename']):
            name = entry['rfilename']; expected = entry.get('lfs', {}).get('sha256')
            if (name.endswith('.safetensors') or name=='adapter_model.bin') and not expected: raise RuntimeError('Unverified model weight')
            target = Path(a.root)/key/name
            previous = output['files'].get(name, {})
            details = node_verify(a.node, target, entry['size'], expected or previous.get('sha256')) if a.resume else None
            if details is None:
              with requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', headers=headers,
                              stream=True, timeout=(30, 180)) as r:
                if r.status_code != 200: raise RuntimeError('File access failed: ' + name + ' HTTP ' + str(r.status_code))
                details = node_write(a.node, target, r.iter_content(1024**2), entry['size'], expected)
            output['files'][name] = details
            print(json.dumps(dict(model=key, file=name, **details)), flush=True)
            temporary = partial.with_name(partial.name+'.tmp')
            temporary.write_text(json.dumps(manifest, indent=2)+'\n')
            temporary.replace(partial)
    payload = (json.dumps(manifest, indent=2)+'\n').encode()
    node_write(a.node, Path(a.root)/'download_manifest.json', [payload], len(payload), hashlib.sha256(payload).hexdigest())
    a.manifest.write_bytes(payload)
    if not a.defer_ready:
        if any('adapter_model.bin' in m['files'] for m in manifest['models'].values()):
            raise RuntimeError('Pickled adapters require safe conversion before publishing READY; use --defer-ready')
        node_write(a.node, Path(a.root)/'READY', [b'ready\n'], 6)
    print(json.dumps({'staging_complete': True, 'manifest': str(a.manifest)}), flush=True)


if __name__ == '__main__':
    try: main()
    except requests.RequestException as e:
        # Requests exception strings can contain signed redirect URLs.
        raise SystemExit('HTTP transport failure: ' + type(e).__name__)
