"""Interactive Gradio wrapper for the synthetic experiment-selection benchmark."""

from __future__ import annotations

import gradio as gr

from osteo_rl import run_benchmark


INTRO = """
# Which experiment would you run next?

**Osteosarcoma-themed reinforcement learning demo** · Synthetic data only

With a fixed virtual assay budget, can a learned policy find high-scoring mock candidates
more efficiently than the baselines? Run the same benchmark at different budgets and
inspect one sequence of decisions. The labels and response scores are simulated;
there are no real compounds, experimental results, patient data, or treatment recommendations.
"""

METHODS = """
### How to read this
- **Mean best score** is the average highest score found under the budget. Higher is better.
- **Hit rate** is the share of evaluation episodes in which the policy finds a hit under the benchmark's synthetic definition. Higher is better.
- **Mean regret** is the average gap between the best available candidate score and the best candidate score found. Lower is better.
- **Decision trace** shows one illustrative learned-policy run. A probe can return a useful signal but does not count as a candidate.

Treat any improvement as a result about this simulator. It does not establish performance
in a real assay, support a drug claim, or suggest anything about patient outcomes.
"""

CSS = """
.gradio-container { max-width: 1050px !important; margin: 0 auto !important; }
#opening { max-width: 780px; }
#opening h1 { font-size: clamp(1.7rem, 4vw, 2.55rem); line-height: 1.12; }
#run-benchmark { margin-top: 0.4rem; }
"""


def display_benchmark(seed: int, budget: int, episodes: int):
    """Turn the engine's structured output into display-ready rows."""
    seed, budget, episodes = int(seed), int(budget), int(episodes)
    if not (0 <= seed <= 100 and 1 <= budget <= 6 and 60 <= episodes <= 600):
        raise gr.Error("Use a seed of 0–100, an assay budget of 1–6, and 60–600 training episodes.")
    result = run_benchmark(seed=seed, budget=budget, episodes=episodes)

    summary_rows = [
        [
            row["policy"],
            f'{row["mean_best"]:.3f}',
            f'{row["hit_rate"]:.1%}',
            f'{row["mean_regret"]:.3f}',
        ]
        for row in result["rows"]
    ]
    trace_rows = [
        [
            row["step"],
            str(row["candidate"]),
            f'{row["observed"]:.3f}',
            f'{row["best_so_far"]:.3f}',
        ]
        for row in result["traces"]
    ]
    status = (
        f"Completed with **{budget}** virtual assays per episode, "
        f"**{episodes}** training episodes, and seed **{seed}**. "
        f'Evaluated on **{result["evaluation_episodes"]}** held-out episodes; '
        f'a hit means a candidate score of at least **{result["hit_threshold"]:.2f}**.'
    )
    return status, summary_rows, trace_rows


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Which experiment would you run next?",
        css=CSS,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(INTRO, elem_id="opening")
        with gr.Row():
            seed = gr.Slider(0, 100, value=7, step=1, label="Random seed")
            budget = gr.Slider(1, 6, value=4, step=1, label="Virtual assay budget")
            episodes = gr.Slider(60, 600, value=240, step=20, label="Training episodes")
        run = gr.Button("Run benchmark", variant="primary", elem_id="run-benchmark")
        status = gr.Markdown("Run the benchmark to generate the evaluation and trace.")
        scoreboard = gr.Dataframe(
            headers=["Policy", "Mean best score", "Hit rate", "Mean regret"],
            datatype=["str", "str", "str", "str"],
            label="Evaluation across episodes",
            interactive=False,
        )
        trace = gr.Dataframe(
            headers=["Step", "Mock candidate", "Observed score", "Best score so far"],
            datatype=["number", "str", "str", "str"],
            label="One learned-policy decision trace",
            interactive=False,
        )
        with gr.Accordion("Method and limitations", open=True):
            gr.Markdown(METHODS)
        run.click(
            display_benchmark,
            inputs=[seed, budget, episodes],
            outputs=[status, scoreboard, trace],
            api_name="benchmark",
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
