"""
LocalBashEnv: StatefulToolEnv that runs bash commands via subprocess.

Avoids prime_sandboxes (no Docker, no prime login required). Commands run
directly on the host. Suitable for Experiment 2 validation.

Adds to state after each rollout:
  state["final_stdout"]: str
  state["exit_code"]: int
  state["terminal_outputs"]: list[dict]  — per-call {stdout}
"""

import asyncio
import logging
from typing import Any

import verifiers as vf
from verifiers.types import Messages, State


class LocalBashEnv(vf.StatefulToolEnv):
    def __init__(self, timeout: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.cmd_timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.add_tool(self.bash)

    async def bash(self, command: str) -> str:
        """Execute `command` locally via subprocess."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self.cmd_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return "(timeout)"

            stdout = (stdout_b or b"").decode(errors="replace").strip()
            stderr = (stderr_b or b"").decode(errors="replace").strip()
            if stdout and stderr:
                return f"{stdout}\nstderr:\n{stderr}"
            return stdout or (f"stderr:\n{stderr}" if stderr else "(no output)")
        except Exception as e:
            return f"error: {e}"

    async def setup_state(self, state: State, **kwargs) -> State:
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict[str, Any]:
        return tool_args

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        completed = await super().is_completed(messages, state, **kwargs)
        if completed:
            await self.post_rollout(messages, state, **kwargs)
        return completed

    async def post_rollout(self, messages: Messages, state: State, **kwargs) -> None:
        """Extract terminal outputs from message history into state."""
        outputs = [
            {"stdout": msg["content"]}
            for msg in messages
            if msg.get("role") == "tool"
        ]
        state["terminal_outputs"] = outputs
        state["final_stdout"] = "\n".join(o["stdout"] for o in outputs)
        last = outputs[-1]["stdout"] if outputs else ""
        state["exit_code"] = _infer_exit_code(last)


def _infer_exit_code(stdout: str) -> int:
    if stdout in ("(timeout)",):
        return 1
    lower = stdout.lower()
    for indicator in [
        "error:", "traceback", "exception:", "fatal:", "permission denied",
        "command not found", "no such file", "segmentation fault",
        "cannot open", "failed to", "no module named",
    ]:
        if indicator in lower:
            return 1
    return 0
