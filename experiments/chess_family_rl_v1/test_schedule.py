"""Ensure epoch coverage, matched board orders, and train/evaluation isolation."""
import unittest
from .schedule import training_order, evaluation_seeds


class ScheduleTests(unittest.TestCase):
    def test_each_board_once_per_epoch_and_repeatable(self):
        rows = [dict(id=i) for i in range(30)]
        for seed in (17, 29, 43):
            order = list(training_order(rows, seed, 32))
            self.assertEqual(order, list(training_order(rows, seed, 32)))
            self.assertEqual(len(order), 960)
            for epoch in range(32):
                block = order[epoch*30:(epoch+1)*30]
                self.assertEqual({r['row']['id'] for r in block}, set(range(30)))
                self.assertEqual({r['epoch'] for r in block}, {epoch+1})
            self.assertEqual([r['move'] for r in order], list(range(1, 961)))
            self.assertEqual(len({r['stop_seed'] for r in order}), 960)
        self.assertNotEqual(list(training_order(rows, 17, 32)), list(training_order(rows, 29, 32)))

    def test_evaluation_seeds_never_overlap_training_or_other_boards(self):
        train = {r['stop_seed'] for seed in (17, 29, 43)
                 for r in training_order([dict(id=i) for i in range(30)], seed, 32)}
        seen = set()
        for split, count in [('train', 30), ('development', 30), ('test', 32)]:
            for i in range(count):
                seeds = set(evaluation_seeds(split, i, 16))
                self.assertEqual(len(seeds), 16)
                self.assertFalse(seeds & (seen | train))
                seen |= seeds


if __name__ == '__main__':
    unittest.main()
