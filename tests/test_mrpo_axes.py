"""Local, GPU-free validation of the MRPO K=5 reward axes and the §7
correlation diagnostic. Run: python tests/test_mrpo_axes.py"""
import importlib.util
import os

# Load terminal_similarity.py directly by path to avoid the package __init__,
# which imports heavy deps (peft/torch) not needed for these pure-Python tests.
_TS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "verifiers", "rl", "trainer", "terminal_similarity.py",
)
_spec = importlib.util.spec_from_file_location("terminal_similarity", _TS_PATH)
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)

DEFAULT_AXES = _ts.DEFAULT_AXES
MRPO_AXES = _ts.MRPO_AXES
brevity_reward = _ts.brevity_reward
compute_reward_vector = _ts.compute_reward_vector
reward_axis_correlation = _ts.reward_axis_correlation
tool_format_validity = _ts.tool_format_validity


def _msg(content):
    return [{"role": "assistant", "content": content}]


def test_backward_compatible_k3():
    stdouts = ["5 files\n", "5 files\n", "12 files\n"]
    ec = [0, 0, 1]
    rv = compute_reward_vector(stdouts, exit_codes=ec)
    assert len(rv) == 3 and len(rv[0]) == 3, "default must stay K=3"
    assert rv[0][2] == 1.0 and rv[2][2] == 0.0, "exit axis wrong"
    print("ok: backward-compatible K=3", rv)


def test_brevity():
    comps = [_msg("short"), _msg("a much much much longer answer here"), _msg("mid len")]
    b = brevity_reward(comps)
    assert b[0] == 1.0 and b[1] == 0.0, f"brevity ranking wrong: {b}"
    assert 0.0 < b[2] < 1.0
    print("ok: brevity", b)


def test_tool_format():
    good = _msg('thinking <tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>')
    bad = _msg('oops <tool_call>{not valid json</tool_call>')
    none = _msg("I will just answer directly with no tool.")
    tf = tool_format_validity([good, bad, none])
    assert tf[0] == 1.0, f"valid tool call should score 1.0: {tf}"
    assert tf[1] == 0.0, f"malformed should score 0.0: {tf}"
    assert tf[2] == 0.0, f"no-tool should score 0.0: {tf}"
    print("ok: tool_format", tf)


def test_k5_shape_and_correlation():
    # Build a group where strict & jaccard are collinear (both ~ output match)
    # but brevity/tool_format vary independently.
    stdouts = ["result 42\n", "result 42\n", "result 99\n", "result 42\n"]
    ec = [0, 0, 0, 1]
    comps = [
        _msg('<tool_call>{"name":"bash","arguments":{"command":"echo 42"}}</tool_call> ' + "x" * 5),
        _msg('<tool_call>{"name":"bash","arguments":{"command":"echo 42"}}</tool_call> ' + "x" * 200),
        _msg('<tool_call>{bad</tool_call> ' + "x" * 50),
        _msg("no tool at all " + "x" * 120),
    ]
    rv = compute_reward_vector(stdouts, exit_codes=ec, completions=comps, axes=MRPO_AXES)
    assert len(rv) == 4 and len(rv[0]) == 5, f"expected 4x5, got {len(rv)}x{len(rv[0])}"
    diag = reward_axis_correlation(rv)
    print("ok: K=5 reward vectors:")
    for i, r in enumerate(rv):
        print(f"   rollout {i}: " + ", ".join(f"{name}={v:.2f}" for name, v in zip(MRPO_AXES, r)))
    print(f"   mean|off-diag rho| = {diag['mean_abs_offdiag']:.3f}  eff_rank = {diag['eff_rank']:.2f} / {diag['K']}")
    assert diag["eff_rank"] > 1.0, "K=5 set should have effective rank > 1"


if __name__ == "__main__":
    test_backward_compatible_k3()
    test_brevity()
    test_tool_format()
    test_k5_shape_and_correlation()
    print("\nALL MRPO AXIS TESTS PASSED")
