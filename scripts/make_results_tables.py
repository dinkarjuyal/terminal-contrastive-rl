#!/usr/bin/env python3
"""Turn experiment outputs into LaTeX snippets to fold into paper/vpo_main.tex.

Inputs (any subset):
  --corr-k5  /path/exp19b_rvecs.jsonl        (K=5 MRPO reward-vector dump)
  --corr-k3  /path/exp19_v2k3_rvecs.jsonl    (K=3 V2 reward-vector dump)
  --bestk    name=path.json [name=path.json ...]  (best@k eval reports)

Emits LaTeX to stdout (and --out file): a correlation-matrix table and a
best@k curve table. Pure-stdlib so it runs locally without the training deps.
"""
import argparse
import importlib.util
import json
import os

_TS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "verifiers", "rl", "trainer", "terminal_similarity.py",
)
_spec = importlib.util.spec_from_file_location("terminal_similarity", _TS_PATH)
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)


def _load_dump(path):
    rvecs, axes = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            axes = rec.get("axes", axes)
            rvecs.extend(rec.get("rvecs", []))
    return rvecs, axes


def corr_table(path, label):
    rvecs, axes = _load_dump(path)
    d = _ts.reward_axis_correlation(rvecs)
    K, corr = d["K"], d["corr"]
    names = axes or [f"a{i}" for i in range(K)]
    short = [n.replace("_", "\\_")[:9] for n in names]
    out = []
    out.append("\\begin{table}[h]\\centering")
    out.append(f"\\caption{{Reward-axis Pearson correlation ({label}); "
               f"pooled over {d['n']} rollouts. "
               f"mean$|\\rho_{{\\mathrm{{off}}}}|={d['mean_abs_offdiag']:.3f}$, "
               f"effective rank $={d['eff_rank']:.2f}/{K}$.}}")
    out.append("\\begin{tabular}{@{}l" + "r" * K + "@{}}\\toprule")
    out.append(" & " + " & ".join(short) + " \\\\ \\midrule")
    for a in range(K):
        row = " & ".join(f"{corr[a][b]:.2f}" for b in range(K))
        out.append(f"{short[a]} & {row} \\\\")
    out.append("\\bottomrule\\end{tabular}\\end{table}")
    return "\n".join(out), d


def bestk_table(reports):
    # reports: dict name -> json report
    ks = sorted({int(k) for r in reports.values() for k in r["pass_at_k"]})
    show = [k for k in ks if k in (1, 2, 4, 8, 16, max(ks))]
    show = sorted(set(show))
    out = []
    out.append("\\begin{table}[h]\\centering")
    out.append("\\caption{best@$k$ (unbiased pass@$k$) on held-out bash tasks. "
               "A flat curve indicates diversity collapse; a rising curve means "
               "samples remain distinct for search.}")
    out.append("\\begin{tabular}{@{}l" + "r" * len(show) + "@{}}\\toprule")
    out.append("Policy & " + " & ".join(f"$k{{=}}{k}$" for k in show) + " \\\\ \\midrule")
    for name, r in reports.items():
        vals = " & ".join(f"{r['pass_at_k'].get(str(k), r['pass_at_k'].get(k, 0)):.3f}" for k in show)
        out.append(f"{name.replace('_', ' ')} & {vals} \\\\")
    out.append("\\bottomrule\\end{tabular}\\end{table}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr-k5")
    ap.add_argument("--corr-k3")
    ap.add_argument("--bestk", nargs="*", default=[])
    ap.add_argument("--out", default="/tmp/mrpo_results.tex")
    args = ap.parse_args()

    chunks = []
    summary = []
    if args.corr_k3 and os.path.exists(args.corr_k3):
        t, d = corr_table(args.corr_k3, "K=3 V2, near-collinear")
        chunks.append(t)
        summary.append(("K=3 V2", d["mean_abs_offdiag"], d["eff_rank"], d["K"]))
    if args.corr_k5 and os.path.exists(args.corr_k5):
        t, d = corr_table(args.corr_k5, "K=5 MRPO")
        chunks.append(t)
        summary.append(("K=5 MRPO", d["mean_abs_offdiag"], d["eff_rank"], d["K"]))

    reports = {}
    for spec in args.bestk:
        name, _, path = spec.partition("=")
        if path and os.path.exists(path):
            reports[name] = json.load(open(path))
    if reports:
        chunks.append(bestk_table(reports))

    tex = "\n\n".join(chunks)
    with open(args.out, "w") as f:
        f.write(tex + "\n")
    print(tex)
    print("\n% --- summary ---")
    for name, mo, er, K in summary:
        print(f"% {name}: mean|off-diag rho|={mo:.3f}  eff_rank={er:.2f}/{K}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
