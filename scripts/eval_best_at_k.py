#!/usr/bin/env python3
"""best@k (pass@k) eval for the search-downstream claim (paper §8).

Generates n samples per bash task against a running vf-vllm server, scores each
rollout's final terminal output against the task's ground-truth answer, and
reports the unbiased pass@k curve for k=1..n. The headline comparison: a
scalar-GRPO policy collapses (curve flattens as samples become duplicates)
while MRPO keeps rising with k.

Run ON the pod (needs the env + a vf-vllm server hosting the policy to eval):
  python scripts/eval_best_at_k.py --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen2.5-1.5B-Instruct --n 16 --out /tmp/bestk_base.json

To eval a trained checkpoint, point --model at the served adapter/merged model.
"""
import argparse
import asyncio
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.bash_agent.bash_agent import make_env  # noqa: E402
from environments.bash_agent.bash_tasks import TASKS  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def is_correct(stdout: str, answer: str, answer_type: str) -> bool:
    if not answer or answer_type == "skip":
        return False
    out = _norm(stdout)
    if answer_type == "contains":
        # all whitespace-separated answer fragments must appear
        return all(_norm(part) in out for part in answer.split("\n") if part.strip())
    if answer_type == "line_set":
        lines = [l for l in answer.split("\n") if l.strip()]
        return all(_norm(l) in out for l in lines)
    # default: exact normalized match
    return out == _norm(answer)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: P(at least one of k samples correct) given c correct of n."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


async def _run(base_url: str, model: str, n: int):
    import httpx
    from openai import AsyncOpenAI

    env = make_env()
    answers = {t["id"]: (t.get("answer"), t.get("answer_type", "skip")) for t in TASKS}
    # only evaluate tasks with a checkable answer
    eval_ids = [tid for tid, (a, at) in answers.items() if a and at != "skip"]

    full = env.get_dataset()
    idx = [i for i in range(len(full)) if full[i]["task_id"] in eval_ids]
    dataset = full.select(idx)
    repeated = dataset.repeat(n)

    client = AsyncOpenAI(
        base_url=base_url, api_key="EMPTY",
        http_client=httpx.AsyncClient(timeout=600.0),
    )
    results = await env.a_generate(
        repeated, client=client, model=model,
        sampling_args={"temperature": 1.0, "max_tokens": 1024, "n": 1},
        score_rollouts=True, max_concurrent=32,
    )
    return env, dataset, results, eval_ids, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=16, help="samples per task")
    ap.add_argument("--out", default="/tmp/bestk.json")
    args = ap.parse_args()

    env, dataset, results, eval_ids, answers = asyncio.run(
        _run(args.base_url, args.model, args.n)
    )

    M = len(dataset)
    n = args.n
    per_task_correct = {}
    for p in range(M):
        tid = dataset[p]["task_id"]
        ans, atype = answers[tid]
        strided = [p + k * M for k in range(n)]
        c = 0
        for i in strided:
            if i >= len(results.state):
                continue
            stdout = results.state[i].get("final_stdout", "")
            if is_correct(stdout, ans, atype):
                c += 1
        per_task_correct[tid] = c

    curve = {}
    for k in range(1, n + 1):
        vals = [pass_at_k(n, per_task_correct[t], k) for t in per_task_correct]
        curve[k] = sum(vals) / len(vals) if vals else 0.0

    report = {
        "model": args.model, "n": n, "num_tasks": M,
        "per_task_correct": per_task_correct,
        "pass_at_k": curve,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nbest@k for {args.model}  ({M} tasks, n={n} samples each)")
    print(f"  pass@1  = {curve[1]:.3f}")
    print(f"  pass@4  = {curve.get(4, curve[n]):.3f}")
    print(f"  pass@{n} = {curve[n]:.3f}")
    print(f"  slope (pass@{n} - pass@1) = {curve[n] - curve[1]:+.3f}  "
          f"(flat slope => diversity collapse)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
