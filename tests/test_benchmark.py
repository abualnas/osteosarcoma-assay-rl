import random
import unittest

from osteo_rl import Screen, run_benchmark, synthetic_panel


class BenchmarkTests(unittest.TestCase):
    def test_probe_is_information_only_and_budget_is_enforced(self):
        s = Screen((0.92, 0.73, 0.70), 2, {})
        self.assertEqual(s.step(0), 0)
        self.assertEqual(s.best, 0)
        self.assertAlmostEqual(s.step(1), 0.73)
        self.assertEqual(s.legal(), [])
        with self.assertRaises(ValueError):
            s.step(2)

    def test_all_policies_test_same_budget_and_metrics_are_bounded(self):
        for budget in (1, 4, 6):
            r = run_benchmark(seed=7, budget=budget, episodes=60)
            self.assertEqual(len(r["traces"]), budget)
            self.assertEqual(len({t["candidate"] for t in r["traces"]}), budget)
            self.assertEqual([t["best_so_far"] for t in r["traces"]],
                             sorted(t["best_so_far"] for t in r["traces"]))
            for row in r["rows"]:
                for k in ("mean_best", "hit_rate", "mean_regret"):
                    self.assertLessEqual(0, row[k])
                    self.assertLessEqual(row[k], 1)

    def test_reproducible_and_invalid_inputs_rejected(self):
        kwargs = dict(seed=33, budget=3, episodes=60)
        self.assertEqual(run_benchmark(**kwargs), run_benchmark(**kwargs))
        with self.assertRaises(ValueError):
            run_benchmark(budget=0)
        self.assertTrue(all(0 <= x <= 1 for x in synthetic_panel(random.Random(0))))


if __name__ == "__main__":
    unittest.main()
