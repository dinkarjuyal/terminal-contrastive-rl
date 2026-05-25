"""
Bash agent training script (Experiments 1 & 2).

Experiment 1 (pair-rate check): run with --dry_run to collect rollouts and log
  tc/positive_pair_rate without any model updates. Use this to validate the
  similarity signal before committing to a full training run.

Experiment 2 (TC training): standard training run with GRPO + TC loss.

Usage:
  # Experiment 1: just check pair rate (no training)
  python bash_agent.py --dry_run --model Qwen/Qwen2.5-1.5B-Instruct

  # Experiment 2: full training
  python bash_agent.py  (reads bash_agent_tc.toml)
  # or via verifiers CLI:
  python -m verifiers.scripts.train configs/rl/bash_agent_tc.toml
"""

import argparse
import sys
from pathlib import Path

# Allow running from the environments/bash_agent dir or from verifiers root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import verifiers as vf
from environments.bash_agent.bash_tasks import make_dataset
from verifiers.envs.local_bash_env import LocalBashEnv
from verifiers.rubrics.rubric import Rubric


class TerminalRubric(Rubric):
    """
    Placeholder rubric — rewards come from TC loss, not here.
    Returns 0.0 so GRPO advantages are driven purely by the TC signal.
    """

    async def score_rollout(self, prompt, completion, answer, state, **kwargs) -> float:
        return float(state.get("ce_implicit_reward", 0.0))


def make_env() -> LocalBashEnv:
    dataset = make_dataset()
    rubric = TerminalRubric()
    env = LocalBashEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=6,
        timeout=15,
    )
    return env


def load_environment(**kwargs) -> LocalBashEnv:
    """Entry point for vf-train / vf-rl launcher."""
    return make_env()


def dry_run(model_name: str, n_tasks: int = 5, rollouts_per_task: int = 8):
    """
    Experiment 1: collect rollouts for a few tasks and report pair stats.
    Does not train. Use to validate tc/positive_pair_rate before full training.
    """
    import asyncio
    import numpy as np
    from verifiers.rl.trainer.terminal_similarity import select_pairs, trajectory_diversity

    env = make_env()
    client_base_url = "http://0.0.0.0:8000/v1"

    print(f"\nDry run: {n_tasks} tasks × {rollouts_per_task} rollouts")
    print("=" * 60)

    async def run():
        from openai import AsyncOpenAI
        import httpx
        client = AsyncOpenAI(
            base_url=client_base_url,
            api_key="EMPTY",
            http_client=httpx.AsyncClient(timeout=300.0),
        )
        dataset = env.get_dataset().select(range(min(n_tasks, len(env.dataset))))
        repeated = dataset.repeat(rollouts_per_task)

        results = await env.a_generate(
            repeated,
            client=client,
            model=model_name,
            sampling_args={"temperature": 1.0, "max_tokens": 1024, "n": 1, "logprobs": True},
            score_rollouts=True,
            max_concurrent=32,
        )
        return results

    results = asyncio.run(run())

    N = n_tasks
    G = rollouts_per_task
    pair_rates = []
    diversities = []

    for p in range(N):
        strided_indices = [p + k * N for k in range(G)]
        stdouts = [results.state[i].get("final_stdout", "") for i in strided_indices]
        exit_codes = [results.state[i].get("exit_code", 0) for i in strided_indices]
        task_id = env.dataset[p].get("task_id", str(p))
        category = env.dataset[p].get("category", "?")

        pos, neg = select_pairs(stdouts, exit_codes)
        div = trajectory_diversity(stdouts)
        pair_rate = len(pos) / max(1, G * (G - 1) // 2)
        pair_rates.append(pair_rate)
        diversities.append(div)

        print(f"\nTask {p}: [{category}] {task_id}")
        print(f"  Positive pairs: {len(pos)}/{G*(G-1)//2} (rate={pair_rate:.2f})")
        print(f"  Negative pairs: {len(neg)}")
        print(f"  Diversity: {div:.3f}")
        for i, s in enumerate(stdouts):
            print(f"  R{i}: {s[:80].replace(chr(10), ' | ')}")

    print("\n" + "=" * 60)
    print(f"Mean positive pair rate: {np.mean(pair_rates):.3f}  (gate: > 0.30)")
    print(f"Mean diversity:          {np.mean(diversities):.3f}  (watch for collapse < 0.10)")
    if np.mean(pair_rates) > 0.30:
        print("✓ PASS — proceed to Experiment 2 (training)")
    else:
        print("✗ FAIL — tune similarity thresholds or check task variety")


def train(config_path: str = "configs/rl/bash_agent_tc.toml"):
    """Experiment 2: GRPO + TC training. Called by vf-train launcher."""
    import tomllib
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    env = make_env()
    rl_config = vf.RLConfig(**cfg["trainer"].get("args", {}))
    trainer = vf.RLTrainer(model=cfg["model"], env=env, args=rl_config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Run Experiment 1: collect rollouts, report pair stats, no training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model for dry run (ignored in full training mode)")
    parser.add_argument("--n_tasks", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=8)
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.model, n_tasks=args.n_tasks, rollouts_per_task=args.rollouts)
    else:
        train()
