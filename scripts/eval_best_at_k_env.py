#!/usr/bin/env python3
"""Env-generic best@k (pass@k) for any verifiers environment whose rubric reward
encodes correctness (gsm8k, math, ...). Samples n completions per eval example
against a running vf-vllm server, scores each via the env's own rubric, and
reports the unbiased pass@k curve.

  python scripts/eval_best_at_k_env.py --env-id gsm8k --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen2.5-1.5B-Instruct --n 16 --num-examples 100 --out results/bestk_gsm8k_base.json
"""
import argparse
import asyncio
import json
import math

import verifiers as vf


def pass_at_k(n, c, k):
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


async def _run(env_id, base_url, model, n, num_examples):
    import httpx
    from openai import AsyncOpenAI

    env = vf.load_environment(env_id=env_id)
    eval_ds = env.eval_dataset if getattr(env, "eval_dataset", None) is not None else env.get_dataset()
    if num_examples > 0:
        eval_ds = eval_ds.select(range(min(num_examples, len(eval_ds))))
    M = len(eval_ds)
    repeated = eval_ds.repeat(n)
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY",
                         http_client=httpx.AsyncClient(timeout=900.0))
    res = await env.a_generate(
        repeated, client=client, model=model,
        sampling_args={"temperature": 1.0, "max_tokens": 1024, "n": 1},
        score_rollouts=True, max_concurrent=64,
    )
    return M, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--num-examples", type=int, default=100)
    ap.add_argument("--out", default="/tmp/bestk_env.json")
    a = ap.parse_args()

    M, res = asyncio.run(_run(a.env_id, a.base_url, a.model, a.n, a.num_examples))
    # correctness = env rubric reward > 0.5; strided layout: example p, sample k -> p + k*M
    correct = [0] * M
    for p in range(M):
        for k in range(a.n):
            i = p + k * M
            if i < len(res.reward) and (res.reward[i] or 0) > 0.5:
                correct[p] += 1
    curve = {}
    for k in range(1, a.n + 1):
        vals = [pass_at_k(a.n, correct[p], k) for p in range(M)]
        vals = [v for v in vals if v is not None]
        curve[k] = sum(vals) / len(vals) if vals else 0.0
    report = {"env": a.env_id, "model": a.model, "n": a.n, "num_examples": M,
              "per_example_correct": correct, "pass_at_k": curve}
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\nbest@k  env={a.env_id}  model={a.model}  ({M} examples, n={a.n})")
    for k in (1, 4, 16):
        if k in curve:
            print(f"  pass@{k:<2} = {curve[k]:.3f}")
    print(f"  slope(@{a.n}-@1) = {curve[a.n]-curve[1]:+.3f}  -> {a.out}")


if __name__ == "__main__":
    main()
