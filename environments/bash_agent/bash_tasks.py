"""
20-task bash dataset for Experiment 1: pair-rate validation and TC training.

Tasks are chosen to cover the range where stdout similarity works:
  - Numerical output (word count, line count, file sizes) — strict measure works well
  - File listing (find, ls, glob) — works well for same-file sets
  - Data transformation (sort, filter, awk) — works for deterministic outputs
  - Status/check (git, process, disk) — works only when outputs are identical

Dataset format: HuggingFace Dataset with "prompt" column (list-of-messages format).
"""

import json
from datasets import Dataset

SYSTEM_PROMPT = """\
You are a bash assistant. When given a task, use the bash tool to accomplish it.
Run one or more commands to complete the task. Be concise and direct.
"""

TASKS = [
    # --- Numerical output (similarity works well) ---
    {
        "id": "wc_words",
        "prompt": "Count the number of words in /etc/hosts.",
        "category": "numerical",
        "answer": "28",
        "answer_type": "contains",   # ground-truth token must appear in stdout
    },
    {
        "id": "wc_lines",
        "prompt": "Count the number of lines in /etc/passwd.",
        "category": "numerical",
        "answer": "42",
        "answer_type": "contains",
    },
    {
        "id": "disk_usage",
        "prompt": "Show the total disk usage of the /tmp directory in human-readable format.",
        "category": "numerical",
        "answer": None,              # stochastic — /tmp changes; skip correctness check
        "answer_type": "skip",
    },
    {
        "id": "file_size",
        "prompt": "Find the size in bytes of /etc/hostname.",
        "category": "numerical",
        "answer": "24",
        "answer_type": "contains",
    },
    {
        "id": "process_count",
        "prompt": "Count how many processes are currently running (use ps).",
        "category": "numerical",
        "answer": None,              # stochastic — process count changes
        "answer_type": "skip",
    },
    # --- File listing (similarity works well) ---
    {
        "id": "list_etc",
        "prompt": "List all files in /etc that start with 'host'.",
        "category": "file_listing",
        "answer": "/etc/host.conf\n/etc/hostname\n/etc/hosts\n/etc/hosts.allow\n/etc/hosts.deny",
        "answer_type": "line_set",   # ground-truth lines must all appear in stdout
    },
    {
        "id": "list_hidden",
        "prompt": "List all hidden files (starting with .) in the /root or /home directory.",
        "category": "file_listing",
        "answer": None,              # user-dependent, changes
        "answer_type": "skip",
    },
    {
        "id": "find_py",
        "prompt": "Find all Python files (.py) in /usr/lib that are larger than 10KB.",
        "category": "file_listing",
        "answer": "/usr/lib/byobu/include/config.py",  # at least this must appear
        "answer_type": "contains",
    },
    {
        "id": "find_recent",
        "prompt": "Find files in /var/log that were modified in the last 24 hours.",
        "category": "file_listing",
        "answer": None,              # stochastic — changes daily
        "answer_type": "skip",
    },
    {
        "id": "find_exec",
        "prompt": "List all executable files in /usr/bin that start with 'python'.",
        "category": "file_listing",
        "answer": "/usr/bin/python3",
        "answer_type": "contains",
    },
    # --- Data transformation (similarity depends on determinism) ---
    {
        "id": "sort_uniq",
        "prompt": "Extract all unique shell names from /etc/passwd (the last colon-separated field) and sort them.",
        "category": "transformation",
        "answer": "/bin/bash\n/bin/false\n/bin/sync\n/usr/bin/bash\n/usr/sbin/nologin",
        "answer_type": "line_set",
    },
    {
        "id": "head_lines",
        "prompt": "Show the first 5 lines of /etc/hosts, with line numbers.",
        "category": "transformation",
        "answer": "127.0.0.1 localhost",  # must appear in output
        "answer_type": "contains",
    },
    {
        "id": "grep_ip",
        "prompt": "Extract all IP addresses (lines starting with a digit) from /etc/hosts.",
        "category": "transformation",
        "answer": "127.0.0.1 localhost",
        "answer_type": "contains",
    },
    {
        "id": "env_path",
        "prompt": "Show the directories in the PATH environment variable, one per line.",
        "category": "transformation",
        "answer": None,              # env-dependent
        "answer_type": "skip",
    },
    {
        "id": "awk_users",
        "prompt": "List only the usernames (first field) from /etc/passwd, sorted alphabetically.",
        "category": "transformation",
        "answer": "_apt",            # must be first alphabetically
        "answer_type": "contains",
    },
    # --- Status / check (similarity only catches identical outputs) ---
    {
        "id": "python_version",
        "prompt": "Show the installed Python version (just the version number, nothing else).",
        "category": "status",
        "answer": "3.12",
        "answer_type": "contains",
    },
    {
        "id": "os_info",
        "prompt": "Show the OS name and version.",
        "category": "status",
        "answer": "Ubuntu 22.04",
        "answer_type": "contains",
    },
    {
        "id": "pwd_check",
        "prompt": "Show the current working directory.",
        "category": "status",
        "answer": "/home/ubuntu",
        "answer_type": "contains",
    },
    {
        "id": "env_home",
        "prompt": "Show the value of the HOME environment variable.",
        "category": "status",
        "answer": "/home/ubuntu",
        "answer_type": "contains",
    },
    {
        "id": "uptime_check",
        "prompt": "Show how long the system has been running (uptime).",
        "category": "status",
        "answer": None,              # stochastic
        "answer_type": "skip",
    },
]


def make_dataset(exclude_ids=None) -> Dataset:
    """Return a HuggingFace Dataset in the format expected by verifiers environments."""
    rows = []
    for task in TASKS:
        if exclude_ids and task["id"] in exclude_ids:
            continue
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["prompt"]},
        ]
        rows.append({
            "prompt": prompt,
            "task_id": task["id"],
            "category": task["category"],
            "answer": "",  # no ground-truth answer; reward comes from TC loss
        })
    return Dataset.from_list(rows)


if __name__ == "__main__":
    ds = make_dataset()
    print(f"Dataset: {len(ds)} tasks")
    for row in ds:
        print(f"  [{row['category']:15s}] {row['task_id']}: {row['prompt'][1]['content'][:60]}")
