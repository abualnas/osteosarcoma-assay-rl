"""Auditable tabular RL benchmark. Entirely synthetic, not a drug recommender."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

CANDIDATES = ("SYN-PROBE",) + tuple(f"SYN-{g}{i}" for g in "ABCD" for i in (1, 2))
HIT = 0.80
EVAL_EPISODES = 300


def clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def family(arm: int) -> int:
    return (arm - 1) // 2 if arm else -1


def synthetic_panel(rng: random.Random) -> tuple[float, ...]:
    """Draw hidden profile and pre-sample shared noisy assay outcomes."""
    profile = rng.choices(range(4), weights=(4, 3, 2, 1), k=1)[0]
    probe = clip(rng.gauss(0.19 + 0.15 * profile, 0.055))
    shared = rng.gauss(0, 0.035)
    scores = [probe]
    for arm in range(1, len(CANDIDATES)):
        centre = (0.87 if arm % 2 else 0.77) if family(arm) == profile else 0.25
        scores.append(clip(rng.gauss(centre + shared, 0.105)))
    return tuple(scores)


@dataclass
class Screen:
    slate: tuple[float, ...]
    budget: int
    tested: dict[int, float]
    best: float = 0.0

    def legal(self) -> list[int]:
        return [i for i in range(len(self.slate)) if i not in self.tested] if len(self.tested) < self.budget else []

    def state(self) -> tuple[int, ...]:
        probe = self.tested.get(0)
        probe_bin = -1 if probe is None else sum(probe >= cut for cut in (0.265, 0.415, 0.565))
        lead_arm = max((i for i in self.tested if i), key=self.tested.__getitem__, default=0)
        lead_score = self.tested.get(lead_arm, 0.0) if lead_arm else 0.0
        score_bin = sum(lead_score >= cut for cut in (0.40, 0.65, HIT))
        return (len(self.tested), probe_bin, family(lead_arm) if score_bin else -1, score_bin)

    def step(self, arm: int) -> float:
        if arm not in self.legal():
            raise ValueError("Assay already used or budget exhausted")
        value = self.slate[arm]
        self.tested[arm] = value
        best_now = max(self.best, value) if arm else self.best
        reward = best_now - self.best
        self.best = best_now
        return reward


def run_benchmark(seed: int = 7, budget: int = 4, episodes: int = 240) -> dict:
    if not (1 <= budget <= 6 and 1 <= episodes <= 5000):
        raise ValueError("budget must be 1–6 and training episodes 1–5000")
    sim_train = random.Random(seed * 11 + 101)
    explore_rng = random.Random(seed * 11 + 211)
    train = [synthetic_panel(sim_train) for _ in range(episodes)]
    q: dict[tuple, float] = defaultdict(lambda: 0.0)
    visits: dict[tuple, int] = defaultdict(int)
    priors = [mean(x[i] for x in train) if i else -1.0 for i in range(len(CANDIDATES))]

    def learned(env: Screen, rng: random.Random, epsilon: float = 0.0) -> int:
        legal = env.legal()
        if rng.random() < epsilon:
            return rng.choice(legal)
        state = env.state()
        return max(legal, key=lambda a: (q.get((state, a), 0.0), priors[a]))

    # Replay training panels with exploration; never learn from the test cohort.
    for iteration in range(4 * episodes):
        env = Screen(train[iteration % episodes], budget, {})
        epsilon = max(0.12, 0.8 * (1 - iteration / (4 * episodes)))
        while env.legal():
            state = env.state()
            arm = learned(env, explore_rng, epsilon)
            gain = env.step(arm)
            future = max((q.get((env.state(), a), 0.0) for a in env.legal()), default=0.0)
            key = (state, arm)
            visits[key] += 1
            rate = visits[key] ** -0.55
            q[key] += rate * (gain + future - q[key])

    test_rng = random.Random(seed * 11 + 307)
    test = [synthetic_panel(test_rng) for _ in range(EVAL_EPISODES)]
    actions = random.Random(seed * 11 + 409)
    policies = {
        "Random": lambda e: actions.choice(e.legal()),
        "Fixed training rank": lambda e: max(e.legal(), key=priors.__getitem__),
        "Tabular Q-learning": lambda e: learned(e, actions),
    }
    rows, trace = [], []
    for label, select in policies.items():
        bests, regrets = [], []
        for j, panel in enumerate(test):
            env = Screen(panel, budget, {})
            while env.legal():
                arm = select(env)
                env.step(arm)
                if j == 0 and label == "Tabular Q-learning":
                    trace.append(dict(step=len(env.tested), candidate=CANDIDATES[arm],
                                      observed=round(env.tested[arm], 3), best_so_far=round(env.best, 3)))
            bests.append(env.best)
            regrets.append(max(panel[1:]) - env.best)
        rows.append(dict(policy=label, mean_best=mean(bests),
                         hit_rate=mean(s >= HIT for s in bests),
                         mean_regret=mean(regrets)))
    return dict(rows=rows, traces=trace, evaluation_episodes=EVAL_EPISODES,
                hit_threshold=HIT, seed=seed, budget=budget, training_episodes=episodes)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, default=7)
    cli.add_argument("--budget", type=int, default=4)
    cli.add_argument("--episodes", type=int, default=240)
    args = cli.parse_args()
    print(json.dumps(run_benchmark(args.seed, args.budget, args.episodes), indent=2))
