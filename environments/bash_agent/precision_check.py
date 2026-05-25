"""
Precision check: does TC's positive-pair signal actually select correct rollouts?

For each task with a known ground-truth answer:
  - Collect G=8 rollouts from vLLM
  - Label each rollout correct/incorrect (ground truth check)
  - Run TC pair selection
  - Compute:
      precision  = P(both rollouts correct | TC called it positive)
      recall     = P(TC called it positive | both rollouts correct)
      false_pos  = positive pairs where at least one rollout was wrong
      accuracy   = fraction of rollouts that got the correct answer

Run against base model, then step-200 LoRA to compare.

Usage:
  # Base model (current vLLM):
  python precision_check.py

  # Trained LoRA (requires restarting vLLM with LoRA):
  python precision_check.py --lora outputs/bash-agent-tc-exp3/checkpoint-200
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import numpy as np
from openai import AsyncOpenAI

from environments.bash_agent.bash_tasks import TASKS, make_dataset
from verifiers.envs.local_bash_env import LocalBashEnv
from verifiers.rl.trainer.terminal_similarity import select_pairs
import verifiers as vf


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
G = 8
BASE_URL = "http://0.0.0.0:8000/v1"  # overridden by --port

# Fix: ls /etc/host* on this server returns bare filenames, not full paths
TASKS_GROUND_TRUTH_FIXUPS = {
    "list_etc": {
        "answer": "host.conf\nhostname\nhosts\nhosts.allow\nhosts.deny",
        "answer_type": "line_set",
    },
}


def is_correct(stdout: str, answer: str, answer_type: str) -> bool:
    if answer_type == "skip" or answer is None:
        return None  # not evaluable
    stdout_lower = stdout.lower()
    if answer_type == "contains":
        return answer.lower() in stdout_lower
    if answer_type == "line_set":
        gt_lines = set(l.strip() for l in answer.splitlines() if l.strip())
        out_lines = set(l.strip() for l in stdout.splitlines() if l.strip())
        # all ground-truth lines must appear somewhere in output
        return gt_lines.issubset(out_lines)
    return False


async def collect_rollouts(tasks_subset, port=8000):
    dataset = make_dataset()
    rubric = vf.Rubric(funcs=[lambda **kw: 0.0], weights=[1.0])
    env = LocalBashEnv(dataset=dataset, rubric=rubric, max_turns=4, timeout=15)

    base_url = f"http://0.0.0.0:{port}/v1"
    client = AsyncOpenAI(
        base_url=base_url, api_key="EMPTY",
        http_client=httpx.AsyncClient(timeout=300.0),
    )

    # Only the tasks we want to evaluate
    task_ids = [t["id"] for t in tasks_subset]
    ds = env.get_dataset().filter(lambda x: x["task_id"] in task_ids)
    repeated = ds.repeat(G)

    results = await env.a_generate(
        repeated,
        client=client,
        model=MODEL,
        sampling_args={"temperature": 1.0, "max_tokens": 512, "n": 1, "logprobs": True},
        score_rollouts=True,
        max_concurrent=1,
    )
    return results, ds


def analyze(tasks_subset, results, ds):
    N = len(tasks_subset)
    task_lookup = {t["id"]: t for t in tasks_subset}

    all_precision_num = []  # both correct in a TC-positive pair
    all_precision_den = []  # TC-positive pairs (evaluable)
    all_recall_num = []
    all_recall_den = []
    all_accuracies = []
    false_positives = []

    print(f"\n{'='*70}")
    print(f"{'Task':<20} {'Acc':>5} {'Pos':>5} {'FP':>5} {'Prec':>6} {'Recall':>7}")
    print(f"{'='*70}")

    for p in range(N):
        task = tasks_subset[p]
        task_id = task["id"]
        answer = task["answer"]
        answer_type = task["answer_type"]

        strided = [p + k * N for k in range(G)]
        stdouts = [results.state[i].get("final_stdout", "") for i in strided]
        exit_codes = [results.state[i].get("exit_code", 0) for i in strided]

        correctness = [is_correct(s, answer, answer_type) for s in stdouts]
        evaluable = [c is not None for c in correctness]

        if not any(evaluable):
            print(f"{task_id:<20} {'skip':>5}")
            continue

        acc = np.mean([c for c in correctness if c is not None])
        all_accuracies.append(acc)

        pos_pairs, neg_pairs = select_pairs(stdouts, exit_codes)

        # Only count pairs where both rollouts are evaluable
        eval_pos = [(i, j) for i, j in pos_pairs if evaluable[i] and evaluable[j]]
        both_correct = [(i, j) for i, j in eval_pos if correctness[i] and correctness[j]]
        fp = [(i, j) for i, j in eval_pos if not (correctness[i] and correctness[j])]

        # recall: of all pairs where both are correct, how many did TC find?
        all_correct_pairs = [
            (i, j) for i in range(G) for j in range(i+1, G)
            if evaluable[i] and evaluable[j] and correctness[i] and correctness[j]
        ]
        tc_found_correct = [(i,j) for i,j in all_correct_pairs if (i,j) in set(eval_pos) or (j,i) in set(eval_pos)]

        prec = len(both_correct) / max(1, len(eval_pos))
        rec = len(tc_found_correct) / max(1, len(all_correct_pairs))

        all_precision_num.append(len(both_correct))
        all_precision_den.append(len(eval_pos))
        all_recall_num.append(len(tc_found_correct))
        all_recall_den.append(len(all_correct_pairs))
        false_positives.extend(fp)

        print(f"{task_id:<20} {acc:>5.2f} {len(eval_pos):>5} {len(fp):>5} {prec:>6.2f} {rec:>7.2f}")

        # Print sample rollouts
        for i, (s, c) in enumerate(zip(stdouts, correctness)):
            mark = "✓" if c else ("✗" if c is False else "?")
            print(f"  R{i} {mark}: {s[:80].replace(chr(10), ' | ')}")

    print(f"{'='*70}")
    overall_prec = sum(all_precision_num) / max(1, sum(all_precision_den))
    overall_rec = sum(all_recall_num) / max(1, sum(all_recall_den))
    overall_acc = np.mean(all_accuracies) if all_accuracies else 0
    print(f"\nOVERALL (evaluable tasks only):")
    print(f"  Rollout accuracy:   {overall_acc:.3f}  ({100*overall_acc:.1f}% of rollouts got correct answer)")
    print(f"  TC precision:       {overall_prec:.3f}  (of TC-positive pairs, this fraction had both correct)")
    print(f"  TC recall:          {overall_rec:.3f}  (of all-correct pairs, TC found this fraction)")
    print(f"  False positives:    {len(false_positives)} TC-positive pairs with ≥1 wrong rollout")


def main(model_name=None, label="base", port=8000):
    global MODEL
    if model_name:
        MODEL = model_name

    # Apply ground-truth fixups
    evaluable_tasks = []
    for t in TASKS:
        if t["answer_type"] == "skip":
            continue
        t = dict(t)
        if t["id"] in TASKS_GROUND_TRUTH_FIXUPS:
            t.update(TASKS_GROUND_TRUTH_FIXUPS[t["id"]])
        evaluable_tasks.append(t)

    print(f"\n[{label}] model={MODEL}")
    print(f"Evaluating {len(evaluable_tasks)}/{len(TASKS)} tasks (skipping stochastic ones)")

    results, ds = asyncio.run(collect_rollouts(evaluable_tasks, port=port))
    analyze(evaluable_tasks, results, ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model name to pass to vLLM API (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--label", default="base", help="Label for this run (base / trained)")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    args = parser.parse_args()
    main(model_name=args.model, label=args.label, port=args.port)
