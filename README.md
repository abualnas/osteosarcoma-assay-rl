---
title: Osteosarcoma Assay Allocation Sandbox
emoji: 🧪
colorFrom: blue
colorTo: teal
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# Which assay would you run next?

A small, reproducible **reinforcement learning prototype for sequential assay selection**. Osteosarcoma motivates the use case; the compounds, cell-line profiles, responses and hit threshold are entirely synthetic. There are no real drug rankings, patients, wet-lab results or efficacy claims.

## Run it

```bash
python osteo_rl.py --seed 7 --budget 4 --episodes 240
python -m unittest discover -s tests -v
```

The benchmark needs only Python 3.11+. For the interactive dashboard, install its dependencies and launch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

## Why reinforcement learning?

Screening decisions are sequential. With only a few assay slots, a policy can spend one slot on an informative probe or go straight to a candidate. The simulated probe reveals a noisy clue about a hidden response profile, but can never count as a hit. Later choices can use earlier observations.

- **Observation:** step number, coarse probe reading if observed, and the family and score bin of the strongest non-probe result so far. The hidden profile and untested outcomes are not exposed. This is a compact representation of a partially observed screening problem, not an assumption that real cell lines have four subtypes.
- **Action:** choose one untested synthetic candidate or the probe. No repeated assays; all policies use the same budget.
- **Reward:** improvement in highest observed non-probe score. Across an episode, rewards telescope to the best score found.
- **Policy:** tabular Q-learning with decaying exploration, action masking and training-prior tie-breaking. No neural network, pretrained model or internet access is needed.
- **Evaluation:** frozen policy on 300 fresh simulated panels per seed. Each policy sees the same panels for a paired comparison. Baselines are uniform random and a non-adaptive ranking by average training response. Regret is the gap to the full-panel hindsight best. A hit is a synthetic score of at least 0.80.

## Reproducible result

Budget 4, 240 training panels per seed (four training passes), 300 held-out panels per seed, seeds 0–9:

| Policy | Mean best | Hit rate | Mean regret |
| --- | ---: | ---: | ---: |
| Random | 0.695 | 44.3% | 0.190 |
| Fixed training rank | 0.728 | 57.9% | 0.157 |
| Tabular Q-learning | 0.826 | 68.4% | 0.059 |

These are results about the toy simulator. The simulator deliberately makes observations informative. This is **not** evidence that RL beats a strong adaptive rule, generalises to another simulator, or finds osteosarcoma treatments. A stronger follow-up benchmark would include Bayesian or greedy adaptive policies, a shuffled or noisier probe, and tests under shifts in the response-profile distribution.

## What would make it an osteosarcoma study?

First audit osteosarcoma model and compound overlap in the Broad Institute’s [DepMap](https://depmap.org/portal/) and [PRISM public drug response screens](https://depmap.org/repurposing/). If coverage supports it, split by cell line before training and review missing measurements and assay batch effects. Retrospective replay can only score observed cell line–compound pairs; missing pairs are not negative results. Report paired uncertainty, compare to strong simple baselines, and keep the claim at **preclinical assay prioritisation**, not clinical efficacy. No public datasets or weights are bundled.

## Put it on GitHub and Hugging Face

- **GitHub:** Create a new private repository, then upload or push these files, including README.md and the tests. From this directory, if already authenticated with GitHub CLI: initialise Git, commit the files, then run `gh repo create osteosarcoma-assay-rl --private --source=. --push`. Review visibility before sharing the repository.
- **Hugging Face:** Create a private [Gradio Space](https://huggingface.co/new-space), then upload README.md, app.py, osteo_rl.py, and requirements.txt to its root and wait for the build. The configuration is already in the README header. Check your account’s [Space compute eligibility](https://huggingface.co/docs/hub/spaces-overview) before choosing hardware or accepting charges. No model repository or paid GPU is required by the code.

A fair description for a technical conversation: “I built an AI-assisted proof of concept for choosing the next assay under a fixed screening budget. The benchmark tests whether an adaptive policy can beat fixed rankings in a simulator. I can walk through the reward, evaluation split and what would need to change before using real osteosarcoma data.”
