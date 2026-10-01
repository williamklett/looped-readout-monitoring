import json
from pathlib import Path
from experiments.chess_oneboard_rl_v1.common import load as parent_load,sha,INITIALIZER_SHA,config
ROOT=Path(__file__).resolve().parent


def load(project):
    parent,boards=parent_load(project)
    assert sha(project/'experiments/chess_exploration_diagnostic_v1/prompts.py')=='f698a66efc0bc895177dafdc34bfddfc483c2df3cfd04495c54b39d1f39c814e'
    assert sha(project/'experiments/chess_oneboard_rl_v1/release-manifest.json')=='33d7672da08bfbe31bbf2134476c0cb3e22ea89c21b8bb08afedf2f19f3d0e4b'
    for name,digest in json.loads((ROOT/'release-manifest.json').read_text())['sha256'].items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and sha(p)==digest,name
    design=json.loads((ROOT/'design.json').read_text())
    return design,boards,parent


def provenance():
    return dict(design_sha256=sha(ROOT/'design.json'),release_manifest_sha256=sha(ROOT/'release-manifest.json'),
                initializer_sha256=INITIALIZER_SHA)
