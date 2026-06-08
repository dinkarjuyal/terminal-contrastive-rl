#!/usr/bin/env python3
"""Idea #4 (structured diversity), post-hoc: does pooling samples across distinct
specialist policies give higher best@k than the same budget from a single policy?

Uses the per-task correctness saved by eval_best_at_k.py (per_task_correct = #correct
of n). For an ensemble we pool all policies' samples per task (N = sum n_p,
C = sum correct_p) and compute the unbiased pass@k on the pool, vs. each single
policy's pass@k. If the ensemble curve sits above the best single policy at equal
k, directed/structured diversity beats undirected single-policy sampling.

Usage: python scripts/eval_ensemble.py results/bestk_A.json results/bestk_B.json ...
"""
import json
import math
import sys

KS = [1, 2, 4, 8, 16]


def pass_at_k(n, c, k):
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main(paths):
    reports = {}
    for p in paths:
        d = json.load(open(p))
        name = d.get("model", p).split("/")[-1]
        reports[p] = (name, d["n"], d["per_task_correct"])

    tasks = set()
    for _, _, ptc in reports.values():
        tasks |= set(ptc.keys())
    tasks = sorted(tasks)

    # pooled ensemble per task
    pooled_N = {t: 0 for t in tasks}
    pooled_C = {t: 0 for t in tasks}
    for _, n, ptc in reports.values():
        for t in tasks:
            pooled_N[t] += n
            pooled_C[t] += ptc.get(t, 0)

    def curve_from(per_task_NC):
        out = {}
        for k in KS:
            vals = [pass_at_k(N, C, k) for (N, C) in per_task_NC.values()]
            vals = [v for v in vals if v is not None]
            out[k] = sum(vals) / len(vals) if vals else None
        return out

    print("\n#4 Structured diversity: specialist ensemble vs single policies")
    print("policies pooled:", ", ".join(nm for nm, _, _ in reports.values()))
    # individual curves
    print(f"\n{'policy':<22}" + "".join(f"{'k='+str(k):>9}" for k in KS))
    best_single = {k: 0.0 for k in KS}
    for _, (nm, n, ptc) in reports.items():
        nc = {t: (n, ptc.get(t, 0)) for t in tasks}
        c = curve_from(nc)
        print(f"{nm[:21]:<22}" + "".join(f"{(c[k] if c[k] is not None else float('nan')):>9.3f}" for k in KS))
        for k in KS:
            if c[k] is not None:
                best_single[k] = max(best_single[k], c[k])
    ens = curve_from({t: (pooled_N[t], pooled_C[t]) for t in tasks})
    print(f"{'ENSEMBLE (pooled)':<22}" + "".join(f"{ens[k]:>9.3f}" for k in KS))
    print(f"{'best single':<22}" + "".join(f"{best_single[k]:>9.3f}" for k in KS))
    print(f"{'ensemble - best':<22}" + "".join(f"{ens[k]-best_single[k]:>+9.3f}" for k in KS))
    print("\n(positive bottom row at higher k => directed diversity beats single-policy sampling)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    main(sys.argv[1:])
