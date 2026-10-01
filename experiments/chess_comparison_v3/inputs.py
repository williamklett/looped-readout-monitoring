"""Pinned metadata checks, deliberately independent of GPU/model imports."""
import json
import os
from pathlib import Path

from .plan import file_sha, validate_design, matched_schedule

MODEL_MANIFEST_SHA = '2ed12fd7877766a3f36eec6c911bd5819dfcd53e4d64ac4289612511e85ed877'
REQUIRED = {'plan.py','inputs.py','runtime.py','train.py','evaluate.py','preflight.py','evidence.py','review.py','analyze.py','storage.py',
            'environment.py','rollouts.py','verify_evidence.py','integrity.py',
            'parameter_search.py','greedy-incentive-predictions.json','gradient_search.py',
            'gradient_feedback.py','gradient_integrity.py','gradient_analysis.py','packing.py','PROTOCOL.md',
            'dispatch.py','attacker_review.py','design-template.json','dispatch.sbatch',
            'train.sbatch','evaluate.sbatch','analyze.sbatch','baseline-authorization.json'}


def verify_release(project, design_path):
    design = validate_design(json.loads(Path(design_path).read_text()))
    verify_source_tree(project, design['release_manifest_sha256'])
    return design


def verify_source_tree(project, expected_manifest_sha):
    """Shared source pinning for the comparison and disposable development probe."""
    release = Path(__file__).resolve().parent
    manifest = release/'release-manifest.json'
    assert file_sha(manifest) == expected_manifest_sha
    hashes = json.loads(manifest.read_text())['sha256']
    assert REQUIRED <= hashes.keys()
    for name, expected in hashes.items():
        path = (release/name).resolve()
        assert path.is_relative_to(release) and file_sha(path) == expected, name
    model = Path(project)/'experiments/loop_replication/releases/arithmetic-allnumeric-v1'
    assert Path(os.environ.get('CHESS_MODEL_RELEASE','')).resolve() == model.resolve()
    model_manifest = model/'source_manifest.json'
    assert file_sha(model_manifest) == MODEL_MANIFEST_SHA
    for name, expected in json.loads(model_manifest.read_text()).items():
        path = (model/name).resolve()
        assert path.is_relative_to(model.resolve()) and file_sha(path) == expected, name
    return dict(release_manifest_sha256=expected_manifest_sha,
                model_source_manifest_sha256=MODEL_MANIFEST_SHA)


def load_training_data(path, design):
    assert file_sha(path) == design['training_data_sha256']
    data = json.loads(Path(path).read_text())
    assert data['source_dataset_sha256'] == design['dataset_sha256']
    assert data['excluded_splits'] == ['test'] and 'test' not in data
    assert len(data['train']) == 2048 and len(data['development']) == 128
    assert len({r['id'] for r in data['train']}) == 2048
    assert len({r['id'] for r in data['development']}) == 128
    test_seeds = set(design['evaluation']['stopping_seeds'])
    # Verify disjointness rather than trusting the design's declaration.
    for seed in design['training_seeds']:
        schedule = matched_schedule([r['id'] for r in data['train']], seed=seed,
                                    steps=design['search']['steps'], batch=design['search']['batch'])
        assert not test_seeds.intersection(s for row in schedule for s in row['hidden_seeds'])
    assert not test_seeds.intersection(range(9000,9009)), 'Test/development coin overlap'
    return data


def verify_readiness(design, initializer, capability_path):
    assert file_sha(initializer) == design['initializer_sha256']
    assert file_sha(capability_path) == design['capability_review_sha256']
    capability = json.loads(Path(capability_path).read_text())
    verify_authorized_baseline(capability)
    assert capability['adapter_sha256'] == design['initializer_sha256']
    assert capability['test_used'] is False and capability['comparison_accepted'] is False
    assert capability['dataset_sha256'] == design['training_data_sha256']
    # User requested proceeding directly to the main comparison. Successful
    # optimization is an outcome to measure, not a separate prelaunch study.
    return dict(capability_review_sha256=file_sha(capability_path),
                separate_optimizer_pilot_required=False)


def verify_authorized_baseline(capability):
    """Accept only the explicitly authorized failed baseline; never relabel it passed."""
    authorization = load_baseline_authorization()
    assert authorization['user_answer'] == 'Run the main comparison with this baseline'
    assert authorization['original_behavioral_criterion_passed'] is False
    assert capability['evidence_consistent'] is True
    assert capability['pilot_readiness'] is False
    assert capability['comparison_accepted'] is False and capability['test_used'] is False
    assert capability['adapter_sha256'] == authorization['source_adapter_sha256']
    assert capability['standard_start']['n'] == authorization['standard_games'] == 4
    assert capability['standard_start']['opponent_piece_reward_hacks'] == authorization['standard_rewarded_opponent_piece_wins'] == 4
    assert capability['development_start']['n'] == authorization['development_games'] == 8
    assert capability['development_start']['opponent_piece_reward_hacks'] == authorization['development_rewarded_opponent_piece_wins'] == 0
    return {'original_behavioral_criterion_passed':False, 'user_authorized_limited_baseline':True}


def load_baseline_authorization():
    return json.loads((Path(__file__).parent/'baseline-authorization.json').read_text())
