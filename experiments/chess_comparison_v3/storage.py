"""Lossless, atomic storage for large traces; small metadata stays readable JSON."""
import gzip
import io
import json
from pathlib import Path

TRACE_PREFIXES=('candidates-','development-','game-','search-','proposal-targets-')


def physical_path(path):
    path=Path(path)
    if path.name.endswith('.json.gz'):
        plain=Path(str(path)[:-3]);compressed=path
    elif path.suffix=='.json':
        plain=path;compressed=Path(str(path)+'.gz')
    else:
        return path
    if plain.exists() and compressed.exists():
        raise ValueError('Ambiguous plain/compressed evidence: '+str(plain))
    if plain.exists():return plain
    if compressed.exists():return compressed
    raise FileNotFoundError(path)


def read_json(path):
    actual=physical_path(path)
    raw=actual.read_bytes()
    if actual.name.endswith('.json.gz'):raw=gzip.decompress(raw)
    return json.loads(raw)


def save(path,value):
    logical=Path(path)
    compress=logical.suffix=='.json' and logical.name.startswith(TRACE_PREFIXES)
    actual=Path(str(logical)+'.gz') if compress else logical
    assert not (compress and logical.exists()), 'Plain trace already exists'
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if compress:
        target=io.BytesIO()
        with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as handle:handle.write(raw)
        raw=target.getvalue()
    temp=Path(str(actual)+'.tmp')
    temp.write_bytes(raw)
    temp.replace(actual)


def logical_names(folder,prefix):
    names=[]
    for path in Path(folder).iterdir():
        if not path.name.startswith(prefix):continue
        if path.name.endswith('.json.gz'):names.append(path.name[:-3])
        elif path.name.endswith('.json'):names.append(path.name)
    assert len(names)==len(set(names)), 'Duplicate plain/compressed traces'
    return sorted(names)
