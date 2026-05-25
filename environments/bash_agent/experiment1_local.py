"""
Experiment 1 (ground-truth variant): validate terminal similarity signal using
real bash outputs from the server, without needing a running LLM.

For each of the 20 tasks, we run 6-8 DIFFERENT command approaches locally via
subprocess. This gives real terminal outputs to compute pair rates on, validating
the similarity measure on the actual target environment.

This is FASTER and MORE RELIABLE than using an LLM for validation because:
- No model inference required
- The commands are correct by construction (we know which pairs should be positive)
- We get exact ground-truth pair labels for calibration

Usage (on the GPU server, no GPU needed):
  python experiment1_local.py [--measure strict|gated|jaccard]
  python experiment1_local.py --thresh_pos 0.6  # tune threshold
"""

import argparse
import importlib.util
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

# Direct import to avoid torch dependency in __init__.py
_sim_path = Path(__file__).parent / "terminal_similarity.py"
if not _sim_path.exists():
    _sim_path = Path(__file__).parent.parent.parent / "verifiers" / "rl" / "trainer" / "terminal_similarity.py"
if not _sim_path.exists():
    _sim_path = Path(__file__).parent.parent / "verifiers" / "rl" / "trainer" / "terminal_similarity.py"
_spec = importlib.util.spec_from_file_location("terminal_similarity", _sim_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

select_pairs = _mod.select_pairs
trajectory_diversity = _mod.trajectory_diversity
compute_similarity_matrix = _mod.compute_similarity_matrix


def run(cmd: str, timeout: int = 10) -> tuple[str, int]:
    """Run a shell command, return (stdout, exit_code)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        stdout = result.stdout.strip()
        if result.stderr.strip() and not stdout:
            stdout = result.stderr.strip()
        elif result.stderr.strip():
            stdout = stdout + "\nstderr:\n" + result.stderr.strip()
        return stdout or "(no output)", result.returncode
    except subprocess.TimeoutExpired:
        return "(timeout)", 1
    except Exception as e:
        return f"error: {e}", 1


# ---------------------------------------------------------------------------
# 20 tasks × 6-8 command variants each
# Each inner list is a different approach to accomplish the same task.
# Expected: variants that succeed should form positive pairs.
# ---------------------------------------------------------------------------

TASKS = [
    # --- Numerical ---
    {
        "id": "wc_words",
        "category": "numerical",
        "desc": "Count words in /etc/hosts",
        "variants": [
            "wc -w /etc/hosts | awk '{print $1}'",
            "wc -w < /etc/hosts",
            "cat /etc/hosts | wc -w",
            "python3 -c \"print(len(open('/etc/hosts').read().split()))\"",
            "awk '{n+=NF} END{print n}' /etc/hosts",
            "grep -o '[^ ]*' /etc/hosts | wc -l",
        ],
    },
    {
        "id": "wc_lines",
        "category": "numerical",
        "desc": "Count lines in /etc/passwd",
        "variants": [
            "wc -l /etc/passwd | awk '{print $1}'",
            "wc -l < /etc/passwd",
            "cat /etc/passwd | wc -l",
            "awk 'END{print NR}' /etc/passwd",
            "grep -c '' /etc/passwd",
            "python3 -c \"print(sum(1 for _ in open('/etc/passwd')))\"",
        ],
    },
    {
        "id": "file_size",
        "category": "numerical",
        "desc": "Size in bytes of /etc/hostname",
        "variants": [
            "stat -c %s /etc/hostname",
            "wc -c < /etc/hostname",
            "ls -la /etc/hostname | awk '{print $5}'",
            "python3 -c \"import os; print(os.path.getsize('/etc/hostname'))\"",
            "du -b /etc/hostname | awk '{print $1}'",
        ],
    },
    {
        "id": "process_count",
        "category": "numerical",
        "desc": "Count running processes",
        "variants": [
            "ps aux | tail -n +2 | wc -l",
            "ps -e | tail -n +2 | wc -l",
            "ps -ef | tail -n +2 | wc -l",
            "python3 -c \"import subprocess; print(len(subprocess.check_output(['ps','-e']).decode().strip().split('\\n'))-1)\"",
        ],
    },
    {
        "id": "disk_usage_tmp",
        "category": "numerical",
        "desc": "Disk usage of /tmp",
        "variants": [
            "du -sh /tmp 2>/dev/null | awk '{print $1}'",
            "du -h --max-depth=0 /tmp 2>/dev/null | awk '{print $1}'",
            "df -h /tmp | tail -1 | awk '{print $3}'",
        ],
    },
    # --- File listing ---
    {
        "id": "list_etc_host",
        "category": "file_listing",
        "desc": "Files in /etc starting with 'host'",
        "variants": [
            "ls /etc/host*",
            "find /etc -maxdepth 1 -name 'host*'",
            "ls -1 /etc/ | grep '^host'",
            "find /etc -maxdepth 1 -name 'host*' -type f",
            "python3 -c \"import os; print('\\n'.join(sorted(f for f in os.listdir('/etc') if f.startswith('host'))))\"",
        ],
    },
    {
        "id": "find_exec_python",
        "category": "file_listing",
        "desc": "Executables in /usr/bin starting with 'python'",
        "variants": [
            "ls /usr/bin/python*",
            "find /usr/bin -maxdepth 1 -name 'python*' -executable",
            "ls -1 /usr/bin/ | grep '^python'",
            "find /usr/bin -maxdepth 1 -name 'python*'",
        ],
    },
    {
        "id": "find_recent_logs",
        "category": "file_listing",
        "desc": "Log files modified in last 24h",
        "variants": [
            "find /var/log -mtime -1 -type f 2>/dev/null | head -10",
            "find /var/log -newer /tmp -type f 2>/dev/null | head -10",
            "find /var/log -mmin -1440 -type f 2>/dev/null | head -10",
            "ls -lt /var/log/ 2>/dev/null | head -10",
        ],
    },
    # --- Data transformation ---
    {
        "id": "sort_uniq_shells",
        "category": "transformation",
        "desc": "Unique shells from /etc/passwd, sorted",
        "variants": [
            "awk -F: '{print $NF}' /etc/passwd | sort -u",
            "cut -d: -f7 /etc/passwd | sort | uniq",
            "python3 -c \"print('\\n'.join(sorted(set(l.strip().split(':')[-1] for l in open('/etc/passwd') if l.strip()))))\"",
            "grep -oP '(?<=:)[^:]+$' /etc/passwd | sort -u",
        ],
    },
    {
        "id": "grep_ip",
        "category": "transformation",
        "desc": "IP addresses from /etc/hosts (lines starting with digit)",
        "variants": [
            "grep '^[0-9]' /etc/hosts",
            "awk '/^[0-9]/' /etc/hosts",
            "sed -n '/^[0-9]/p' /etc/hosts",
            "python3 -c \"[print(l.rstrip()) for l in open('/etc/hosts') if l[0].isdigit()]\"",
        ],
    },
    {
        "id": "env_path_split",
        "category": "transformation",
        "desc": "PATH directories, one per line",
        "variants": [
            "echo $PATH | tr ':' '\\n'",
            "python3 -c \"import os; print('\\n'.join(os.environ['PATH'].split(':')))\"",
            "printenv PATH | tr ':' '\\n'",
            "awk 'BEGIN{n=split(ENVIRON[\"PATH\"],a,\":\"); for(i=1;i<=n;i++) print a[i]}'",
        ],
    },
    {
        "id": "head_hosts",
        "category": "transformation",
        "desc": "First 5 lines of /etc/hosts with line numbers",
        "variants": [
            "head -5 /etc/hosts | nl",
            "head -5 /etc/hosts | cat -n",
            "awk 'NR<=5{print NR\"\\t\"$0}' /etc/hosts",
            "python3 -c \"[print(f'{i+1}\\t{l}', end='') for i,l in enumerate(open('/etc/hosts')) if i<5]\"",
            "sed -n '1,5{=;p}' /etc/hosts | paste - -",
        ],
    },
    {
        "id": "awk_users",
        "category": "transformation",
        "desc": "Usernames from /etc/passwd, sorted",
        "variants": [
            "awk -F: '{print $1}' /etc/passwd | sort",
            "cut -d: -f1 /etc/passwd | sort",
            "python3 -c \"print('\\n'.join(sorted(l.split(':')[0] for l in open('/etc/passwd').read().strip().split('\\n'))))\"",
            "grep -oP '^[^:]+' /etc/passwd | sort",
        ],
    },
    # --- Status ---
    {
        "id": "python_version",
        "category": "status",
        "desc": "Python version number",
        "variants": [
            "python3 --version | awk '{print $2}'",
            "python3 -c 'import sys; print(sys.version.split()[0])'",
            "python3 -V 2>&1 | grep -oP '[\\d.]+'",
            "python3 -c 'import platform; print(platform.python_version())'",
        ],
    },
    {
        "id": "pwd_check",
        "category": "status",
        "desc": "Current working directory",
        "variants": [
            "pwd",
            "echo $PWD",
            "python3 -c 'import os; print(os.getcwd())'",
            "readlink -f .",
        ],
    },
    {
        "id": "env_home",
        "category": "status",
        "desc": "HOME environment variable",
        "variants": [
            "echo $HOME",
            "printenv HOME",
            "python3 -c 'import os; print(os.environ[\"HOME\"])'",
            "echo ~",
        ],
    },
    {
        "id": "hostname",
        "category": "status",
        "desc": "System hostname",
        "variants": [
            "hostname",
            "uname -n",
            "cat /etc/hostname",
            "python3 -c 'import socket; print(socket.gethostname())'",
        ],
    },
    {
        "id": "os_name",
        "category": "status",
        "desc": "OS name",
        "variants": [
            "uname -s",
            "python3 -c 'import platform; print(platform.system())'",
            "uname",
        ],
    },
    {
        "id": "user_id",
        "category": "status",
        "desc": "Current user",
        "variants": [
            "whoami",
            "echo $USER",
            "id -un",
            "python3 -c 'import os; print(os.environ.get(\"USER\", os.popen(\"whoami\").read().strip()))'",
        ],
    },
    {
        "id": "shell_check",
        "category": "status",
        "desc": "Current shell",
        "variants": [
            "echo $SHELL",
            "basename $SHELL",
            "python3 -c 'import os; print(os.environ.get(\"SHELL\", \"unknown\"))'",
        ],
    },
]


def run_task(task: dict, verbose: bool = False) -> tuple[list[str], list[int]]:
    stdouts, exit_codes = [], []
    for cmd in task["variants"]:
        stdout, ec = run(cmd)
        stdouts.append(stdout)
        exit_codes.append(ec)
        if verbose:
            status = "✓" if ec == 0 else "✗"
            print(f"    {status} {cmd[:60]:<60} → {stdout[:50].replace(chr(10), ' | ')}")
    return stdouts, exit_codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", default="strict", choices=["strict", "gated", "jaccard", "line_jaccard", "levenshtein"])
    parser.add_argument("--thresh_pos", type=float, default=0.70)
    parser.add_argument("--thresh_neg", type=float, default=0.20)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"Experiment 1: pair-rate validation (measure={args.measure}, thresh_pos={args.thresh_pos})")
    print("=" * 70)

    pair_rates, diversities = [], []
    category_stats: dict[str, list[float]] = {}

    for task in TASKS:
        stdouts, exit_codes = run_task(task, verbose=args.verbose)
        G = len(stdouts)

        pos, neg = select_pairs(
            stdouts, exit_codes,
            thresh_pos=args.thresh_pos,
            thresh_neg=args.thresh_neg,
            measure=args.measure,
        )
        div = trajectory_diversity(stdouts, measure=args.measure)
        pair_rate = len(pos) / max(1, G * (G - 1) // 2)
        pair_rates.append(pair_rate)
        diversities.append(div)
        category_stats.setdefault(task["category"], []).append(pair_rate)

        status = "✓" if pair_rate >= 0.30 else "✗"
        print(f"{status} [{task['category']:15s}] {task['id']:20s}  pos={len(pos)}/{G*(G-1)//2}  rate={pair_rate:.2f}  div={div:.2f}")
        if args.verbose and pos:
            print(f"    positive pairs: {pos}")

    # Summary
    print("\n" + "=" * 70)
    print(f"OVERALL  mean_pair_rate={np.mean(pair_rates):.3f}  mean_diversity={np.mean(diversities):.3f}")
    print()
    for cat, rates in sorted(category_stats.items()):
        print(f"  [{cat:15s}]  mean_rate={np.mean(rates):.3f}  ({sum(r>=0.3 for r in rates)}/{len(rates)} tasks pass 0.30 gate)")

    print()
    if np.mean(pair_rates) >= 0.30:
        print("✓ PASS — tc/positive_pair_rate will be adequate for TC training")
        print("  → Proceed to Experiment 2 (bash_agent.py training run)")
    else:
        print("✗ FAIL — pair rate too low; try --thresh_pos 0.5 or --thresh_pos 0.4")

    # Show threshold sweep
    print("\nThreshold sweep (to tune thresh_pos):")
    print(f"  {'thresh':>8}  {'mean_rate':>10}  {'tasks_passing':>14}")
    for thresh in [0.40, 0.50, 0.60, 0.70, 0.80]:
        rates = []
        for task in TASKS:
            stdouts, exit_codes = run_task(task)
            G = len(stdouts)
            pos, _ = select_pairs(stdouts, exit_codes, thresh_pos=thresh, thresh_neg=args.thresh_neg, measure=args.measure)
            rates.append(len(pos) / max(1, G * (G - 1) // 2))
        passing = sum(r >= 0.30 for r in rates)
        print(f"  {thresh:>8.2f}  {np.mean(rates):>10.3f}  {passing:>14}/{len(TASKS)}")


if __name__ == "__main__":
    main()
