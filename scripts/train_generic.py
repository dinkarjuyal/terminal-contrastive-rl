"""Generic RL training entrypoint for any registered verifiers environment.

Unlike environments/bash_agent/bash_agent.py (bash-specific), this loads the env
by id from the TOML so we can run standard benchmarks (gsm8k, math, ...) through
the same RLTrainer / weight-synced generator used for the bash experiments.

  python scripts/train_generic.py --config configs/rl/<task>.toml

TOML shape:
  model = "..."
  [env]            id = "gsm8k"   (optional [env.args] kwargs)
  [trainer.args]   ... RLConfig fields (micro_batch_size, max_steps, vllm_server_port, ...)
"""
import argparse
import tomllib

import verifiers as vf


def train(config_path: str):
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    env_id = cfg["env"]["id"]
    env_args = cfg.get("env", {}).get("args", {}) or {}
    env = vf.load_environment(env_id=env_id, **env_args)
    rl_config = vf.RLConfig(**cfg["trainer"].get("args", {}))
    trainer = vf.RLTrainer(model=cfg["model"], env=env, args=rl_config)
    trainer.train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)
