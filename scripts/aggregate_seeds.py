#!/usr/bin/env python3
"""Aggregate best@k across seeds → mean ± std per config per k.

Reads results/bestk_<config>*.json files. Seeds are inferred from filenames:
  bestk_mgda_nodeset.json        -> config=mgda, seed=base(42)
  bestk_mgda_s1_nodeset.json     -> config=mgda, seed=1
  bestk_mgda_s2_nodeset.json     -> config=mgda, seed=2
Prints a table of mean±std at k in {1,2,4,8,16} and the best@k slope (k16-k1).

Usage: python scripts/aggregate_seeds.py [results_dir]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

KS = [1, 2, 4, 8, 16]


def main(rdir="results"):
    # nodeset results only (exclude older Prime bestk_*.json without _nodeset)
    files = glob.glob(os.path.join(rdir, "bestk_*_nodeset.json"))
    # group by config abbrev
    by_cfg = defaultdict(dict)  # cfg -> seed -> pass_at_k dict
    for f in files:
        name = os.path.basename(f)[len("bestk_"):].rsplit(".json", 1)[0]
        name = name.replace("_nodeset", "")
        m = re.match(r"(.+?)_s(\d+)$", name)
        if m:
            cfg, seed = m.group(1), m.group(2)
        else:
            cfg, seed = name, "base"
        try:
            by_cfg[cfg][seed] = json.load(open(f))["pass_at_k"]
        except Exception as e:
            print(f"skip {f}: {e}")

    def mean_std(xs):
        if not xs:
            return (float("nan"), 0.0)
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return (m, v ** 0.5)

    order = ["base", "corr", "mgda", "ctrl", "k5"]
    cfgs = sorted(by_cfg, key=lambda c: order.index(c) if c in order else 99)

    print(f"\nbest@k across seeds  (dir={rdir})")
    hdr = "config".ljust(8) + "seeds  " + "".join(f"{'k='+str(k):>14}" for k in KS) + f"{'slope':>10}"
    print(hdr); print("-" * len(hdr))
    for cfg in cfgs:
        seeds = by_cfg[cfg]
        row_vals = {}
        for k in KS:
            xs = [seeds[s][str(k)] for s in seeds if str(k) in seeds[s]]
            row_vals[k] = mean_std(xs)
        slopes = []
        for s in seeds:
            pk = seeds[s]
            if "1" in pk and "16" in pk:
                slopes.append(pk["16"] - pk["1"])
        sm, ss = mean_std(slopes)
        cells = "".join(f"{m:>7.3f}±{sd:<5.3f}" for (m, sd) in (row_vals[k] for k in KS))
        print(f"{cfg:<8}{len(seeds):<7}{cells}{sm:>+7.3f}")
    print("\n(±std over seeds; base=42 + s1 + s2 when present)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
