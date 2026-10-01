"""Verify inherited experiment code and immutable family-disjoint inputs."""
import json
from pathlib import Path
from experiments.chess_two_pass_rl_v1.common import load as pilot_load, sha, INITIALIZER_SHA, config

ROOT = Path(__file__).resolve().parent
PILOT_SHA = '9da4811e418bd2d99cb53cc6d40622b3b7ca2391aabb3f53f1e92359d8312b24'


def load(project):
    assert sha(project/'experiments/chess_two_pass_rl_v1/release-manifest.json') == PILOT_SHA
    _, _, parent = pilot_load(project)
    for name, digest in json.loads((ROOT/'release-manifest.json').read_text())['sha256'].items():
        path = (ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == digest, name
    design = json.loads((ROOT/'design.json').read_text())
    data_root = project/'experiments/chess_board_family_v1/data'
    for name, digest in design['data_sha256'].items():
        assert sha(data_root/name) == digest, name
    data = json.loads((data_root/'boards.json').read_text())['splits']
    assert {k:len(v) for k,v in data.items()} == dict(train=30, development=30, test=32)
    families = {k:{row['family'] for row in v} for k,v in data.items()}
    assert not (families['train'] & families['development'] or families['train'] & families['test'] or families['development'] & families['test'])
    assert design['moves'] == design['epochs'] * len(data['train'])
    return design, data, parent, data_root


def provenance():
    return dict(design_sha256=sha(ROOT/'design.json'),
                release_manifest_sha256=sha(ROOT/'release-manifest.json'),
                inherited_release_sha256=PILOT_SHA, initializer_sha256=INITIALIZER_SHA)
