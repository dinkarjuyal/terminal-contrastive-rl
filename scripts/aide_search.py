#!/usr/bin/env python3
"""AIDE-lite: feedback-guided hill climbing under a constrained budget (paper §8.5).

best@k measures the *i.i.d.* diversity of a policy: draw k independent samples,
keep the best. That is a lower bound on what a real search extracts. Inference
agents like AIDE (Jiang et al. 2025) instead search *sequentially*: each new
candidate is proposed conditioned on the previous attempt's failure, so the
feedback channel re-injects exploration that i.i.d. sampling cannot.

This harness runs that sequential loop per task against a vf-vllm-served policy:

  round 0 : solve the task fresh.
  round r : if still unsolved, re-prompt with the previous attempt's terminal
            output appended ("that was wrong; try a different approach"), and
            propose again. Stop on solve or when the budget B of rounds is spent.

It reports, per policy, solve-rate@B (sequential) alongside i.i.d. pass@B on the
same budget, so we can see *which kind of diversity loss search compensates for*:
  - a collapsed policy (no tool use) fails both;
  - a merely sharpened policy may have flat i.i.d. pass@B but recover under
    feedback-guided search.

Run ON the pod (needs env + a vf-vllm server hosting the policy):
  python scripts/aide_search.py --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen2.5-1.5B-Instruct --budget 8 --out /tmp/aide_base.json

This is a *linear* hill climb (depth-first, beam 1). Tree/beam search is the
natural extension (branch on the k best partial attempts); noted but not run.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset  # noqa: E402

from environments.bash_agent.bash_agent import make_env  # noqa: E402
from environments.bash_agent.bash_tasks import SYSTEM_PROMPT, TASKS  # noqa: E402
from scripts.eval_best_at_k import is_correct, pass_at_k  # noqa: E402


def _revise_prompt(task_prompt: str, history: list[tuple[str, str]]) -> list[dict]:
    """Build a chat prompt that appends prior failed attempts' terminal output."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt}]
    if history:
        last_cmd, last_out = history[-1]
        fb = (f"\n\nYour previous attempt produced this terminal output:\n"
              f"```\n{last_out[:400]}\n```\n"
              f"That did not solve the task. Try a *different* command or approach.")
        msgs[-1] = {"role": "user", "content": task_prompt + fb}
    return msgs


async def _run(base_url: str, model: str, budget: int):
    import httpx
    from openai import AsyncOpenAI

    env = make_env()
    answers = {t["id"]: (t.get("answer"), t.get("answer_type", "skip"), t["prompt"])
               for t in TASKS}
    eval_ids = [tid for tid, (a, at, _) in answers.items() if a and at != "skip"]

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY",
                         http_client=httpx.AsyncClient(timeout=600.0))

    async def search_one(tid):
        ans, atype, prompt = answers[tid]
        history: list[tuple[str, str]] = []
        for _round in range(budget):
            msgs = _revise_prompt(prompt, history)
            ds = Dataset.from_list([{
                "prompt": msgs, "task_id": tid, "category": "", "answer": "",
            }])
            res = await env.a_generate(
                ds, client=client, model=model,
                sampling_args={"temperature": 1.0, "max_tokens": 1024, "n": 1},
                score_rollouts=True, max_concurrent=1,
            )
            stdout = res.state[0].get("final_stdout", "")
            cmd = ""  # best-effort: last assistant content
            comp = res.completion[0]
            if isinstance(comp, list):
                for m in reversed(comp):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        cmd = str(m.get("content", ""))[:200]
                        break
            if is_correct(stdout, ans, atype):
                return _round + 1  # solved at this many evaluations
            history.append((cmd, stdout))
        return None  # unsolved within budget

    # sequential search per task (tasks run concurrently, rounds sequential)
    solved_round = await asyncio.gather(*[search_one(t) for t in eval_ids])
    return env, eval_ids, answers, dict(zip(eval_ids, solved_round)), client, model


async def _iid_pass_at_b(env, eval_ids, answers, client, model, budget):
    """i.i.d. pass@B on the same budget, for the side-by-side comparison."""
    full = env.get_dataset()
    idx = [i for i in range(len(full)) if full[i]["task_id"] in eval_ids]
    ds = full.select(idx).repeat(budget)
    res = await env.a_generate(
        ds, client=client, model=model,
        sampling_args={"temperature": 1.0, "max_tokens": 1024, "n": 1},
        score_rollouts=True, max_concurrent=32,
    )
    M = len(idx)
    correct = {}
    for p in range(M):
        tid = full[idx[p]]["task_id"]
        ans, atype, _ = answers[tid]
        c = sum(1 for k in range(budget)
                if p + k * M < len(res.state)
                and is_correct(res.state[p + k * M].get("final_stdout", ""), ans, atype))
        correct[tid] = c
    return sum(pass_at_k(budget, correct[t], budget) for t in correct) / max(len(correct), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--budget", type=int, default=8, help="max evaluations per task")
    ap.add_argument("--out", default="/tmp/aide_search.json")
    args = ap.parse_args()

    env, eval_ids, answers, solved_round, client, model = asyncio.run(
        _run(args.base_url, args.model, args.budget))
    iid = asyncio.run(_iid_pass_at_b(env, eval_ids, answers, client, model, args.budget))

    solved = {t: r for t, r in solved_round.items() if r is not None}
    solve_rate = len(solved) / max(len(eval_ids), 1)
    avg_evals = (sum(solved.values()) / len(solved)) if solved else float("nan")
    report = {
        "model": args.model, "budget": args.budget, "num_tasks": len(eval_ids),
        "sequential_solve_rate": solve_rate,
        "iid_pass_at_budget": iid,
        "avg_evals_to_solve": avg_evals,
        "solved_round": solved_round,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAIDE-lite search for {args.model}  (B={args.budget}, {len(eval_ids)} tasks)")
    print(f"  sequential solve@B   = {solve_rate:.3f}  (avg {avg_evals:.1f} evals to solve)")
    print(f"  i.i.d.  pass@B       = {iid:.3f}")
    print(f"  search lift over iid = {solve_rate - iid:+.3f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
