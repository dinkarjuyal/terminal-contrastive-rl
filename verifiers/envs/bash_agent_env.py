"""
BashAgentEnv: extends SandboxEnv to capture terminal outputs for verifier-free training.

Adds to state after each rollout:
  state["final_stdout"]: str  — concatenated stdout of all bash calls
  state["exit_code"]: int     — exit code of the last bash call (0 = success)
  state["terminal_outputs"]: list[dict] — per-call records {cmd, stdout, exit_code}

These fields are used by terminal_similarity.select_pairs() in the Generator to
compute positive/negative pairs without any external verifier.
"""

from typing import Any

import verifiers as vf
from verifiers.envs.sandbox_env import SandboxEnv
from verifiers.types import Messages, State


class BashAgentEnv(SandboxEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Wrap the bash tool to capture outputs into state
        # We do this by overriding the bash method and tracking calls in state.
        self._orig_bash = self.bash
        self.tool_map["bash"] = self._tracked_bash

    async def _tracked_bash(self, command: str, sandbox_id: str) -> str:
        """Execute command and record stdout + exit code in state."""
        output = await self._orig_bash(command=command, sandbox_id=sandbox_id)
        # We don't have direct state access here; capture happens in post_rollout.
        # Instead, accumulate in a per-sandbox dict keyed by sandbox_id.
        if not hasattr(self, "_sandbox_outputs"):
            self._sandbox_outputs: dict[str, list[dict]] = {}
        if sandbox_id not in self._sandbox_outputs:
            self._sandbox_outputs[sandbox_id] = []
        self._sandbox_outputs[sandbox_id].append({
            "cmd": command,
            "stdout": output,
        })
        return output

    async def post_rollout(self, messages: vf.Messages, state: vf.State, **kwargs):
        """Store accumulated terminal outputs in state before sandbox is destroyed."""
        sandbox_id = state.get("sandbox_id", "")
        outputs = getattr(self, "_sandbox_outputs", {}).pop(sandbox_id, [])
        state["terminal_outputs"] = outputs

        # Derive final_stdout: concatenate all stdouts separated by newlines
        if outputs:
            state["final_stdout"] = "\n".join(o["stdout"] for o in outputs)
            # Infer exit_code: if last output is an error pattern → 1, else 0
            last_stdout = outputs[-1]["stdout"]
            state["exit_code"] = _infer_exit_code(last_stdout)
        else:
            state["final_stdout"] = ""
            state["exit_code"] = 0

        await super().post_rollout(messages, state, **kwargs)


def _infer_exit_code(stdout: str) -> int:
    """Heuristic: detect error patterns in the last stdout to infer exit code."""
    if not stdout or stdout == "(no output)":
        return 0
    lower = stdout.lower()
    error_indicators = [
        "error:", "traceback", "exception:", "fatal:", "permission denied",
        "command not found", "no such file", "segmentation fault",
        "cannot open", "failed to", "no module named",
    ]
    for indicator in error_indicators:
        if indicator in lower:
            return 1
    return 0
