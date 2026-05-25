"""
Phase 0 overfit inspection: test similarity measures on representative terminal outputs.

Run with:
    python inspect_similarity.py [--measure gated|jaccard|line_jaccard|levenshtein]

For each task group, shows:
  - The raw terminal outputs (truncated)
  - Pairwise similarity matrix for each measure
  - Which pairs would be selected as positive / negative at various thresholds
  - Manual-labeling prompt so you can annotate which pairs should be positive

This generates no training, requires no GPU, no vLLM, no sandbox.
"""

import argparse
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

# Import terminal_similarity directly to avoid the torch dependency in __init__.py
_sim_path = Path(__file__).parent.parent.parent / "verifiers" / "rl" / "trainer" / "terminal_similarity.py"
_spec = importlib.util.spec_from_file_location("terminal_similarity", _sim_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_similarity_matrix = _mod.compute_similarity_matrix
select_pairs = _mod.select_pairs
strip_boilerplate = _mod.strip_boilerplate
trajectory_diversity = _mod.trajectory_diversity
similarity_strict = _mod.similarity_strict


# ---------------------------------------------------------------------------
# Representative terminal output groups
# Each group = one task prompt; each entry = one rollout's stdout
# These are hand-crafted to mimic what a bash agent produces on real tasks.
# ---------------------------------------------------------------------------

TASK_GROUPS = [
    {
        "name": "List Python files in project",
        "rollouts": [
            # Rollout A: find
            """\
/project/src/main.py
/project/src/utils.py
/project/tests/test_main.py
/project/tests/test_utils.py""",
            # Rollout B: ls -R
            """\
src:
main.py  utils.py

tests:
test_main.py  test_utils.py""",
            # Rollout C: glob via python
            """\
['src/main.py', 'src/utils.py', 'tests/test_main.py', 'tests/test_utils.py']""",
            # Rollout D: totally wrong approach (searches for .txt files)
            """\
/project/README.txt
/project/notes.txt""",
            # Rollout E: find with extra noise
            """\
find: warning: Unix filenames usually don't contain slashes
/project/src/main.py
/project/src/utils.py
/project/tests/test_main.py
/project/tests/test_utils.py""",
            # Rollout F: empty result (wrong directory)
            """\
(no output)""",
            # Rollout G: similar to A but with permissions
            """\
-rw-r--r-- 1 user user 2048 Jan 15 09:23 /project/src/main.py
-rw-r--r-- 1 user user  512 Jan 15 09:23 /project/src/utils.py
-rw-r--r-- 1 user user 1024 Jan 15 09:23 /project/tests/test_main.py
-rw-r--r-- 1 user user  768 Jan 15 09:23 /project/tests/test_utils.py""",
            # Rollout H: count only, no names
            """\
4""",
        ],
        "exit_codes": [0, 0, 0, 0, 0, 0, 0, 0],
        "expected_positive_pairs": [(0, 2), (0, 4), (0, 6)],  # same files, different formats
        "expected_negative_pairs": [(0, 3), (0, 5), (0, 7), (3, 5)],
    },
    {
        "name": "Count words in /etc/hosts",
        "rollouts": [
            # A: wc -w
            "42",
            # B: wc -w with filename
            "      42 /etc/hosts",
            # C: python word count
            "Word count: 42",
            # D: wrong file
            "      17 /etc/hostname",
            # E: line count instead of word count
            "      12 /etc/hosts",
            # F: wc all
            "      12      42     312 /etc/hosts",
            # G: error
            "wc: /etc/host: No such file or directory",
            # H: python with extra output
            """\
Reading /etc/hosts...
Total words: 42
Done.""",
        ],
        "exit_codes": [0, 0, 0, 0, 0, 0, 1, 0],
        "expected_positive_pairs": [(0, 1), (0, 2), (0, 5), (0, 7)],  # all show 42
        "expected_negative_pairs": [(0, 3), (0, 4), (0, 6)],
    },
    {
        "name": "Check if git repo is clean",
        "rollouts": [
            # A: git status --short (clean)
            "",
            # B: git status (clean, verbose)
            """\
On branch main
nothing to commit, working tree clean""",
            # C: git status (has changes)
            """\
On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
	modified:   src/main.py

no changes added to commit (use "git add" and/or "git commit -a")""",
            # D: custom script output (clean)
            "Repo is clean.",
            # E: custom script output (dirty)
            "Repo has uncommitted changes: src/main.py",
            # F: git diff --stat (clean = empty)
            "",
            # G: python check (clean)
            "Status: clean (0 modified files)",
            # H: python check (dirty)
            "Status: dirty (1 modified file: src/main.py)",
        ],
        "exit_codes": [0, 0, 0, 0, 0, 0, 0, 0],
        "expected_positive_pairs": [(0, 1), (0, 3), (0, 5), (0, 6)],  # all "clean"
        "expected_negative_pairs": [(0, 2), (0, 4), (0, 7)],
    },
    {
        "name": "Install numpy and verify",
        "rollouts": [
            # A: pip install (already installed)
            """\
Requirement already satisfied: numpy in /usr/local/lib/python3.11/site-packages (1.26.4)""",
            # B: pip install (fresh install)
            """\
Collecting numpy
  Downloading numpy-1.26.4-cp311-cp311-linux_x86_64.whl (17.3 MB)
     ━━━━━━━━━━━━━━━━━━━━ 17.3/17.3 MB 45.2 MB/s eta 0:00:00
Installing collected packages: numpy
Successfully installed numpy-1.26.4""",
            # C: conda install
            """\
Collecting package metadata: done
Solving environment: done

## Package Plan ##

  environment location: /opt/conda

  added / updated specs:
    - numpy


The following NEW packages will be INSTALLED:

  numpy  pkgs/main/linux-64::numpy-1.26.4-py311h08b1b3b_0

Preparing transaction: done
Verifying transaction: done
Executing transaction: done""",
            # D: uv pip install
            """\
  × No solution found when resolving dependencies for numpy""",
            # E: pip install wrong version that fails
            """\
ERROR: Could not find a version that satisfies the requirement numpy==999.0
ERROR: No matching distribution found for numpy==999.0""",
            # F: verify only (python -c)
            "1.26.4",
            # G: verify with more info
            "NumPy version: 1.26.4\nInstalled at: /usr/local/lib/python3.11/site-packages/numpy",
            # H: pip show
            """\
Name: numpy
Version: 1.26.4
Summary: Fundamental package for array computing in Python
Home-page: https://numpy.org
Author: Travis E. Oliphant et al.
License: BSD-3-Clause
Location: /usr/local/lib/python3.11/site-packages
Requires:
Required-by: pandas, scipy""",
        ],
        "exit_codes": [0, 0, 0, 1, 1, 0, 0, 0],
        "expected_positive_pairs": [(0, 1), (0, 2), (5, 6), (5, 7)],  # successful installs of 1.26.4
        "expected_negative_pairs": [(0, 3), (0, 4), (1, 3), (1, 4)],
    },
]

MEASURES = ["strict", "gated", "jaccard", "line_jaccard", "levenshtein"]
THRESH_POS = [0.5, 0.6, 0.7, 0.8, 0.85]
THRESH_NEG = 0.20


def fmt_stdout(s: str, maxlen: int = 80) -> str:
    s = s.strip().replace("\n", " ↵ ")
    return s[:maxlen] + "..." if len(s) > maxlen else s


def print_matrix(matrix: list[list[float]], n: int) -> None:
    header = "     " + "".join(f"  R{j:<2}" for j in range(n))
    print(header)
    for i in range(n):
        row = f"  R{i:<2}" + "".join(f" {matrix[i][j]:.2f}" for j in range(n))
        print(row)


def evaluate_measure(
    group: dict,
    measure: str,
    thresh_pos: float,
) -> dict:
    stdouts = group["rollouts"]
    exit_codes = group.get("exit_codes")
    matrix = compute_similarity_matrix(stdouts, exit_codes, measure)
    pos, neg = select_pairs(
        stdouts,
        exit_codes=exit_codes,
        thresh_pos=thresh_pos,
        thresh_neg=THRESH_NEG,
        measure=measure,
    )

    expected_pos = set(map(tuple, group.get("expected_positive_pairs", [])))
    expected_neg = set(map(tuple, group.get("expected_negative_pairs", [])))

    tp_pos = len(set(pos) & expected_pos)
    fp_pos = len(set(pos) - expected_pos)
    fn_pos = len(expected_pos - set(pos))

    tp_neg = len(set(neg) & expected_neg)
    fp_neg = len(set(neg) - expected_neg)

    return {
        "pos_pairs": pos,
        "neg_pairs": neg,
        "tp_pos": tp_pos,
        "fp_pos": fp_pos,
        "fn_pos": fn_pos,
        "tp_neg": tp_neg,
        "fp_neg": fp_neg,
        "matrix": matrix,
        "diversity": trajectory_diversity(stdouts, measure),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", default="all", choices=["all"] + MEASURES)
    parser.add_argument("--thresh_pos", type=float, default=None,
                        help="Single threshold to inspect (default: sweep)")
    parser.add_argument("--task", type=int, default=None,
                        help="Only show this task index (0-indexed)")
    args = parser.parse_args()

    measures = MEASURES if args.measure == "all" else [args.measure]
    thresh_sweep = [args.thresh_pos] if args.thresh_pos else THRESH_POS
    tasks = TASK_GROUPS if args.task is None else [TASK_GROUPS[args.task]]

    for task_idx, group in enumerate(tasks):
        if args.task is not None:
            task_idx = args.task
        n = len(group["rollouts"])
        print("\n" + "=" * 70)
        print(f"TASK {task_idx}: {group['name']}")
        print("=" * 70)

        print(f"\nRollouts ({n} total):")
        for i, (s, ec) in enumerate(
            zip(group["rollouts"], group.get("exit_codes", [0] * n))
        ):
            print(f"  R{i} (exit={ec}): {fmt_stdout(s)}")

        expected_pos = group.get("expected_positive_pairs", [])
        expected_neg = group.get("expected_negative_pairs", [])
        print(f"\nExpected positive pairs: {expected_pos}")
        print(f"Expected negative pairs: {expected_neg}")

        for measure in measures:
            print(f"\n--- Measure: {measure} ---")

            print("\nSimilarity matrix:")
            matrix = compute_similarity_matrix(
                group["rollouts"], group.get("exit_codes"), measure
            )
            print_matrix(matrix, n)

            print(f"\nThreshold sweep (neg_thresh={THRESH_NEG}):")
            print(f"  {'thresh_pos':>10}  {'#pos':>5}  {'#neg':>5}  "
                  f"{'tp_pos':>6}  {'fp_pos':>6}  {'fn_pos':>6}  {'tp_neg':>6}")
            for thresh in thresh_sweep:
                r = evaluate_measure(group, measure, thresh)
                print(f"  {thresh:>10.2f}  {len(r['pos_pairs']):>5}  "
                      f"{len(r['neg_pairs']):>5}  {r['tp_pos']:>6}  "
                      f"{r['fp_pos']:>6}  {r['fn_pos']:>6}  {r['tp_neg']:>6}")

            # Show best threshold pairs
            best_thresh = thresh_sweep[2] if len(thresh_sweep) > 2 else thresh_sweep[-1]
            r = evaluate_measure(group, measure, best_thresh)
            print(f"\nPairs selected at thresh_pos={best_thresh}:")
            print(f"  Positive: {r['pos_pairs']}")
            print(f"  Negative: {r['neg_pairs']}")
            print(f"  Trajectory diversity (1=fully diverse): {r['diversity']:.3f}")

    # Summary table across all tasks
    print("\n" + "=" * 70)
    print("SUMMARY: Precision/Recall across all tasks (at thresh_pos=0.70)")
    print("=" * 70)
    print(f"{'measure':>15}  {'pos_prec':>9}  {'pos_rec':>8}  {'neg_prec':>9}")
    print("-" * 50)
    for measure in measures:
        total_tp_pos = total_fp_pos = total_fn_pos = 0
        total_tp_neg = total_fp_neg = 0
        for group in TASK_GROUPS:
            r = evaluate_measure(group, measure, 0.70)
            total_tp_pos += r["tp_pos"]
            total_fp_pos += r["fp_pos"]
            total_fn_pos += r["fn_pos"]
            total_tp_neg += r["tp_neg"]
            total_fp_neg += r["fp_neg"]

        prec_pos = total_tp_pos / max(1, total_tp_pos + total_fp_pos)
        rec_pos = total_tp_pos / max(1, total_tp_pos + total_fn_pos)
        prec_neg = total_tp_neg / max(1, total_tp_neg + total_fp_neg)
        print(f"{measure:>15}  {prec_pos:>9.3f}  {rec_pos:>8.3f}  {prec_neg:>9.3f}")

    print()


if __name__ == "__main__":
    main()
