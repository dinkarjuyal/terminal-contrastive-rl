"""MBPP code-generation environment with a genuine multi-axis rubric.

Reward axes (the point of this task for the paper):
  tests_pass : fraction of the example's unit-test asserts that pass  (quality anchor)
  brevity    : 1=shortest in batch .. 0=longest                       (misalignable)
  format     : 1.0 if a python code block / def was produced          (validity)

tests_pass is run in a sandboxed subprocess with a wall-clock timeout so untrusted
model code cannot hang or trash the trainer. Used with reward_source="rubric": the
trainer reads these per-function scores as the reward vector.
"""
import re
import subprocess
import sys
import tempfile

import verifiers as vf
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write a single self-contained Python "
    "function that solves the task. Put the final function in one ```python ... ``` "
    "code block."
)

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _CODE_RE.findall(text or "")
    if m:
        return m[-1].strip()
    # fallback: from first 'def ' to end
    i = (text or "").find("def ")
    return text[i:].strip() if i >= 0 else ""


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for m in completion:
            if isinstance(m, dict) and m.get("role") in (None, "assistant"):
                c = m.get("content")
                if isinstance(c, str):
                    parts.append(c)
        return "\n".join(parts)
    return ""


_RUNNER = r"""
import sys
src = sys.stdin.read()
code, _, tests = src.partition("\n#@@TESTS@@\n")
ns = {}
try:
    exec(code, ns)
except Exception:
    print("PASS 0 TOTAL 0"); sys.exit(0)
passed = total = 0
for line in tests.split("\n"):
    line = line.strip()
    if not line:
        continue
    total += 1
    try:
        exec(line, ns); passed += 1
    except Exception:
        pass
print(f"PASS {passed} TOTAL {total}")
"""


def _run_tests(code: str, tests: str, timeout: int = 10) -> float:
    if not code.strip():
        return 0.0
    payload = code + "\n#@@TESTS@@\n" + tests
    try:
        out = subprocess.run(
            [sys.executable, "-c", _RUNNER], input=payload, capture_output=True,
            text=True, timeout=timeout,
        )
        m = re.search(r"PASS (\d+) TOTAL (\d+)", out.stdout)
        if not m:
            return 0.0
        p, t = int(m.group(1)), int(m.group(2))
        return p / t if t else 0.0
    except Exception:
        return 0.0


def load_environment(num_train_examples: int = -1, num_eval_examples: int = -1):
    ds = load_dataset("mbpp", "sanitized")
    train = ds["train"]
    test = ds["test"]

    def _prep(ex):
        tests = "\n".join(ex.get("test_imports", []) + ex["test_list"])
        return {
            "question": ex["prompt"],
            "answer": tests,  # carry the asserts so reward funcs can run them
        }
    train = train.map(_prep, remove_columns=train.column_names)
    test = test.map(_prep, remove_columns=test.column_names)
    if num_train_examples != -1:
        train = train.select(range(min(num_train_examples, len(train))))
    if num_eval_examples != -1:
        test = test.select(range(min(num_eval_examples, len(test))))

    def tests_pass_reward_func(completion, answer, **kwargs):
        return _run_tests(_extract_code(_completion_text(completion)), answer)

    def format_reward_func(completion, **kwargs):
        return 1.0 if _extract_code(_completion_text(completion)) else 0.0

    def brevity_reward_func(completion, **kwargs):
        # absolute proxy in [0,1]; group-relative brevity is recomputed in the trainer
        n = len(_extract_code(_completion_text(completion)))
        return max(0.0, 1.0 - n / 1200.0)

    rubric = vf.Rubric(funcs=[tests_pass_reward_func, format_reward_func, brevity_reward_func],
                       weights=[1.0, 0.0, 0.0])
    return vf.SingleTurnEnv(
        dataset=train, eval_dataset=test, system_prompt=SYSTEM_PROMPT, rubric=rubric,
    )
