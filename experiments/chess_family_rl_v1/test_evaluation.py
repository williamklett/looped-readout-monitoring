"""Verify board-specific greedy caching and reward replay without a model."""
import json
from pathlib import Path
import unittest
from .evaluation import evaluate
from experiments.chess_two_pass_rl_v1.common import config


class EvaluationTests(unittest.TestCase):
    def test_all_arms_cache_by_board_and_depth_without_dropping_draws(self):
        root = Path(__file__).resolve().parents[2]
        rows = json.loads((root/'results/chess-board-family-v1/data/boards.json').read_text())['splits']['train'][:2]
        cache = json.loads((root/'results/chess-board-family-v1/data/score-cache.json').read_text())
        d = json.loads((Path(__file__).with_name('design.json')).read_text())
        for arm in ('fixed_depth_audit', 'random_audit', 'no_audit', 'execution_audit'):
            calls = set()
            def propose(fen, depth):
                self.assertNotIn((fen, depth), calls)
                calls.add((fen, depth))
                row = next(r for r in rows if r['fen'] == fen)
                action = row['legal_moves'][0]['action']
                return dict(action=action, depth=depth, token_ids=[1], token_logprobs=[0.],
                            greedy=True, temperature=1., output_mask=False)
            result = evaluate(propose, rows, 'train', config(d, arm), cache.__getitem__, d)
            self.assertEqual(result['summary']['n'], 32)
            self.assertEqual(result['summary']['legal_executions'], 32)
            self.assertEqual(result['summary']['total_detections'], 0)
            self.assertEqual([b['position_id'] for b in result['boards']], [r['id'] for r in rows])
            minimum = 2 if arm == 'fixed_depth_audit' else 8
            for row in rows:
                self.assertTrue({(row['fen'], k) for k in range(1, minimum+1)} <= calls)


if __name__ == '__main__':
    unittest.main()
