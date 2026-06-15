#!/usr/bin/env python3
"""§7 reward-axis correlation gate.

Reads a JSONL dump of per-group reward vectors (written by the trainer when
reward_vector_dump_path is set) and reports the K×K Pearson correlation matrix,
the mean off-diagonal |rho|, and the effective rank of the reward code.

Decision rule (paper §5.3 / §7.2):
  • mean|off-diag rho| > 0.8  → near-collinear code; vector methods cannot beat
    a tuned scalar. Do NOT spend on the flagship run with these axes.
  • some |rho| < 0.5 / eff_rank well above 1 → non-collinear room; proceed.

Usage:
  python scripts/analyze_reward_correlation.py /tmp/mrpo_rvecs.jsonl
"""
import importlib.util
import json
import os
import sys

_TS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "verifiers", "rl", "trainer", "terminal_similarity.py",
)
_spec = importlib.util.spec_from_file_location("terminal_similarity", _TS_PATH)
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)


def main(path: str) -> int:
    rvecs: list[list[float]] = []
    axes: list[str] | None = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            axes = rec.get("axes", axes)
            rvecs.extend(rec.get("rvecs", []))

    if len(rvecs) < 2:
        print(f"Not enough reward vectors in {path} (got {len(rvecs)}).")
        return 1

    diag = _ts.reward_axis_correlation(rvecs)
    K = diag["K"]
    names = axes or [f"axis{i}" for i in range(K)]
    corr = diag["corr"]

    print(f"\nReward-axis correlation  (pooled over {diag['n']} rollouts, K={K})")
    print("axes:", ", ".join(names))
    width = max(len(n) for n in names) + 1
    print("\n" + " " * width + "".join(f"{n[:7]:>9}" for n in names))
    for a in range(K):
        row = "".join(f"{corr[a][b]:>9.2f}" for b in range(K))
        print(f"{names[a]:<{width}}{row}")

    print(f"\nmean |off-diagonal rho| = {diag['mean_abs_offdiag']:.3f}")
    print(f"effective rank          = {diag['eff_rank']:.2f} / {K}")

    mo = diag["mean_abs_offdiag"]
    if mo > 0.8:
        print("\nVERDICT: near-collinear (mean|rho|>0.8). Vector methods will NOT")
        print("         beat a tuned scalar on these axes. Redesign axes before spending.")
    elif diag["eff_rank"] > 1.5:
        print("\nVERDICT: non-collinear room (eff_rank>1.5). Flagship run is justified.")
    else:
        print("\nVERDICT: marginal. Inspect which axes are redundant before committing.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
