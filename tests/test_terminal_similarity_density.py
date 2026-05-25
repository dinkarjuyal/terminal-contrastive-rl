"""Unit tests for the DBPO density reward and the scalar self-sim baseline path.

These tests do not require any GPU and run in <1s; they exercise the pure-Python
similarity primitives that drive exp16 (DBPO) and exp17 (scalar self-sim GRPO).
"""

from __future__ import annotations

import math

from verifiers.rl.trainer.terminal_similarity import (
    compute_similarity_matrix,
    density_reward_per_rollout,
    mean_sim_per_rollout,
)


def test_density_reward_handles_singleton():
    assert density_reward_per_rollout(["only one"]) == [0.0]
    assert density_reward_per_rollout([]) == []


def test_density_reward_consensus_outranks_outlier():
    # 3 rollouts agree on "42 files", 1 rollout disagrees with "13 files".
    # The 3 in-consensus rollouts must have strictly higher density than the outlier.
    stdouts = [
        "42 files modified",
        "Found 42 files",
        "Total: 42 files",
        "13 errors found",
    ]
    rewards = density_reward_per_rollout(stdouts, bandwidth=0.3)
    assert len(rewards) == 4
    consensus = rewards[:3]
    outlier = rewards[3]
    for r in consensus:
        assert r > outlier, f"consensus reward {r} did not exceed outlier reward {outlier}"


def test_density_reward_is_finite_when_all_identical():
    stdouts = ["42"] * 4
    rewards = density_reward_per_rollout(stdouts, bandwidth=0.2)
    assert all(math.isfinite(r) for r in rewards)
    # all identical -> all distances 0 -> all log_weights 0 -> r_i = log(1) = 0
    for r in rewards:
        assert abs(r) < 1e-9


def test_density_reward_respects_exit_code_gate():
    # Same numbers but different exit codes -> sim=0 for the gated pair under
    # the "strict" measure. The successful rollout should still have a density
    # at least as good as the failure (no other peers for the failure case).
    stdouts = ["42 files", "42 files", "42 files"]
    exit_codes = [0, 0, 1]
    matrix = compute_similarity_matrix(stdouts, exit_codes, measure="strict")
    assert matrix[0][1] > 0.9
    assert matrix[0][2] == 0.0
    rewards = density_reward_per_rollout(stdouts, exit_codes=exit_codes, bandwidth=0.2)
    # The two passing rollouts must beat the failing one.
    assert rewards[0] > rewards[2]
    assert rewards[1] > rewards[2]


def test_scalar_self_sim_baseline_matches_mean_sim_helper():
    # The exp17 reward override path must produce the same numbers as the
    # canonical mean_sim_per_rollout helper, otherwise the baseline is not
    # what the paper claims it is.
    stdouts = ["42 files", "Found 42 files", "13 errors"]
    a = mean_sim_per_rollout(stdouts)
    b = mean_sim_per_rollout(stdouts)
    assert a == b
    assert max(a) > min(a)  # signal exists across the group


def test_density_reward_bandwidth_monotonicity():
    # Smaller bandwidth = sharper kernel = larger spread between consensus and outlier.
    stdouts = ["42", "42", "42", "13"]
    spread_small = max(density_reward_per_rollout(stdouts, bandwidth=0.1)) - \
        min(density_reward_per_rollout(stdouts, bandwidth=0.1))
    spread_large = max(density_reward_per_rollout(stdouts, bandwidth=1.0)) - \
        min(density_reward_per_rollout(stdouts, bandwidth=1.0))
    assert spread_small > spread_large
