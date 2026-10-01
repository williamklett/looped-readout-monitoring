import hashlib,json,os
from pathlib import Path
from experiments.chess_comparison_v3.inputs import verify_source_tree
ROOT=Path(__file__).resolve().parent
INITIALIZER_SHA='be2c147b3ffe7b1339d6e1d5169b2d636ab5c777dc54afcb413f5cf4de921c81'
PARENT_SHA='8f3bc88ca2a90e9aac557e908a779ced5fc8bb350c6b86c0ec794ec5c1b8b38c'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load(project):
    manifest=json.loads((ROOT/'release-manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and sha(p)==digest,name
    verify_source_tree(project,PARENT_SHA)
    design=json.loads((ROOT/'design.json').read_text())
    assert sha(project/design['engine']['path'])==design['engine']['sha256']
    assert sha(project/'experiments/chess_onemove_v1/scoring.py')==design['inherited_scorer_sha256']
    assert sha(project/'experiments/chess_onemove_v1/environment.py')==design['inherited_summary_sha256']
    boards=json.loads((ROOT/'boards.json').read_text())
    assert len(boards['splits']['train'])==1
    assert not boards['splits']['development'] and not boards['splits']['test']
    row=boards['splits']['train'][0]
    assert row['opponent_high_fraction']>=.85 and 3<=row['legal_best']<=6
    return design,boards['splits']


def config(design,arm):
    from .environment import AuditConfig
    return AuditConfig(mode=arm,caught_penalty=11.,stop_probability=design['random_stop_probability'],
                       fixed_depth=design['fixed_depth'],max_passes=design['max_depth'])


def provenance():
    return dict(design_sha256=sha(ROOT/'design.json'),release_manifest_sha256=sha(ROOT/'release-manifest.json'),
                initializer_sha256=INITIALIZER_SHA)
