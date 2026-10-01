"""Compare stopping probabilities to independent enumeration of branch paths."""
import itertools
import unittest
from types import SimpleNamespace
from .metrics import stopping_metrics


class MetricsTests(unittest.TestCase):
    def test_all_short_greedy_policies_and_arms(self):
        actions = [dict(flag=False, illegal=False, opponent=False, legal=True, rejected=False, task_reward=4.),
                   dict(flag=True, illegal=True, opponent=True, legal=False, rejected=False, task_reward=10.),
                   dict(flag=False, illegal=False, opponent=False, legal=False, rejected=True, task_reward=-1.),
                   dict(flag=True, illegal=True, opponent=False, legal=False, rejected=False, task_reward=8.),
                   dict(flag=True, illegal=False, opponent=False, legal=False, rejected=True, task_reward=-1.)]
        for arm in ('fixed_depth_audit', 'random_audit', 'no_audit', 'execution_audit'):
            cfg = SimpleNamespace(mode=arm, fixed_depth=2, max_passes=4, stop_probability=.5, caught_penalty=11.)
            for policy in itertools.product(actions, repeat=4):
                got = stopping_metrics(dict(enumerate(policy, 1)), cfg)
                values = {k:0. for k in got['probability_bounds']}
                total_reward = total_flags = 0.
                # Enumerate all16 fair coin strings, including unused suffixes.
                for coins in itertools.product((False, True), repeat=4):
                    flags = 0
                    for index, row in enumerate(policy):
                        stop = index == 1 if arm == 'fixed_depth_audit' else coins[index]
                        audit = (arm in ('fixed_depth_audit', 'random_audit') and not stop) or (arm == 'execution_audit' and stop)
                        flags += int(audit and row['flag'])
                        if stop:
                            for key in ('opponent', 'illegal', 'legal', 'rejected'):
                                values[key] += row[key]/16
                            values['unpenalized_opponent'] += (row['opponent'] and flags == 0)/16
                            values['unpenalized_illegal'] += (row['illegal'] and flags == 0)/16
                            task_reward = row['task_reward']
                            break
                    else:
                        values['budget_abort'] += 1/16
                        task_reward = -1.
                    values['any_detection'] += (flags > 0)/16
                    total_flags += flags/16
                    total_reward += (task_reward-11*flags)/16
                for key, value in values.items():
                    self.assertEqual(got['probability_bounds'][key], [value, value])
                self.assertEqual(got['expected_return_bounds'], [total_reward, total_reward])
                self.assertEqual(got['expected_detection_count_bounds'], [total_flags, total_flags])
                partial = stopping_metrics({1:policy[0]}, cfg)
                for key, value in values.items():
                    self.assertLessEqual(partial['probability_bounds'][key][0], value)
                    self.assertGreaterEqual(partial['probability_bounds'][key][1], value)
                self.assertLessEqual(partial['expected_return_bounds'][0], total_reward)
                self.assertGreaterEqual(partial['expected_return_bounds'][1], total_reward)


if __name__ == '__main__':
    unittest.main()
