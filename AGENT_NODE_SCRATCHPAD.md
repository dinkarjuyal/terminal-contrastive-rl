# Agent Node Scratchpad — GPU coordination + experiment log (a3mega cluster)

**Two copies, kept in sync:**
- Local (authoritative working copy): `terminal-contrastive-rl/AGENT_NODE_SCRATCHPAD.md`
- Shared on cluster (for other agents): `/home/ubuntu/AGENT_NODE_SCRATCHPAD.md` (nodeset `34.150.192.31`)

Cluster: partition `a3mega`, 8 nodes `nucla3m-a3meganodeset-0..7`, each **8× H100 80GB**.
Slurm present. SSH IP `34.150.192.31` lands on node 7.

## Node claims
| Nodes | Status | Owner | Task | Since |
|-------|--------|-------|------|-------|
| nucla3m-a3meganodeset-**7** | **IN USE — do not use** | Claude agent (dinkarjuyal) | Terminal-Contrastive RL / MRPO | 2026-06-06 |
| nucla3m-a3meganodeset-0,1,2,3,4,5,6 | **free** | — | claim before using | |

Terminal-Contrastive uses **node 7 ONLY** (nodes 4–6 released 2026-06-06 per request).
**Important: partition `a3mega` is `OverSubscribe=EXCLUSIVE`** — each job gets a *whole node*
(all 8 GPUs / 208 CPUs), so jobs on node 7 run **sequentially**, not packed. To parallelize
you must use multiple nodes (one whole node per job).

## Standing workflow rules (this project)
1. **Smoke first, then scale.** Submit a 6-step probe (`SMOKE=1`) before any full/parallel run.
2. **Launch via Slurm `sbatch scripts/run.sbatch <config> <suffix>`** — NOT nohup/tmux (detached
   login procs get reaped on SSH disconnect here). `-w nucla3m-a3meganodeset-<4..7>`;
   `--export=ALL,SMOKE=1` for smoke. Ports auto-derived from job id (co-reside safe).
3. **Never `pkill -f vf-vllm` / `pkill -f bash_agent.py` from an SSH one-liner** — matches the
   command's own shell and kills your session. Use `scancel <jobid>` / kill by PID.
4. `CUDA_HOME=/usr/local/cuda`; 80GB → mbs=8.

## Experiment log (terminal-contrastive)  — updated 2026-06-06
| Job | Config | Node | Status |
|-----|--------|------|--------|
| 77 | exp19a_mgda (MGDA, K=3) full 200 | 7 | RUNNING |
| 78,79,80 | ctrl/k5/corr smoke | (4,5,6) | COMPLETED ✓ (6/6) |
| 81,82,83 | full runs on nodes 4–6 | — | CANCELLED (freed nodes 4–6) |
| 84 | exp19_ctrl_v2k3 (K3-V2) full 200 | 7 | QUEUED (after 77) |
| 85 | exp19b (K5) full 200 | 7 | QUEUED (after 84) |
| 86 | exp19d (corrected) full 200 | 7 | QUEUED (after 85) |

Node 7 is EXCLUSIVE so 77→84→85→86 run sequentially (~3h total). Next after runs:
best@k eval each policy (base/MGDA/K3/K5/corrected) + optional multi-seed.
Repo: `~/terminal-contrastive-rl` (uv venv). Legacy: `~/rl/verifiers`.

## RESULTS (nodeset, n=16, mbs=8) — eval job 88, 2026-06-06
best@k (pass@k): base .509→.857 (+.35) | corrected .665→.857 (BEST: high floor+ceiling) | MGDA .522→.786 | ctrl-V2 .487→.786 | K5+brevity .000 flat (collapsed).
Findings: (1) MGDA≈V2 null confirmed (collinear K3). (2) K5 collapse + corrected-dominates reproduced at mbs=8 on H100. JSONs: results/bestk_*_nodeset.json.
All training+eval jobs done (77,84,85,86 train; 88 eval). Node 7 idle now.

## 3-SEED ROBUSTNESS (seeds 42,1,2) — jobs 90 (train) + 91 (eval), 2026-06-06
best@k mean±std (k1→k16): base .509→.857 | mgda .537±.024→.857±.058 (most stable) |
corr .496±.121→.786±.058 | ctrl-V2 .420±.197→.643±.154 (noisy) | k5+brevity .000 (collapse, all seeds).
Single-seed story was partly seed-luck (corr's .665 didn't hold). K5 collapse deterministic.
Run `python scripts/aggregate_seeds.py results`. All node-7 jobs done; node idle.

## 4-IDEA CAMPAIGN (2026-06-06, 24h GPU window) — node 7
Goal: run all 4 novelty ideas, max GPU use.
- #1 Coverage reward (train FOR diversity): distinct axis added; configs/.../coverage.toml=[exit,tool_format,distinct]. vs consensus(ctrl).
- #3 MGDA reliability (variance): corr-{dirichlet(exp19d),fixed(alpha=1000),mgda} x 5 seeds.
- #2 Collapse early-warning (eval-only): eval K5 intermediate ckpts {20..200} for oracle-correctness trajectory.
- #4 Structured diversity (eval-only/post-hoc): eval_ensemble.py pools specialists' per-task correctness -> union best@k vs best single.
RUNNING: job 93 bigsweep (4 cfgs x seeds1-5 = 20 runs, 4-wide waves ~3h). Then evalsweep.sbatch (manifest of name->ckpt, 8-parallel) -> aggregate_seeds.py (#1,#3) + eval_ensemble.py (#4) + trajectory (#2).
Tools: scripts/{sweep,evalsweep}.sbatch, scripts/{aggregate_seeds,eval_ensemble}.py.

## CAMPAIGN COMPLETE (2026-06-07) — all 4 ideas + AIDE bonus run, folded into paper
#3 MGDA reliability (10 seeds): MGDA ~5x lower variance (.515±.035->.836±.046) AND highest mean vs Dirichlet(.454±.186->.707±.238)/Fixed(.397±.169->.636±.164). HEADLINE (paper sec:exp-reliability).
#2 K5 collapse cliff steps 60-80. #1 coverage=diversity knob (slope+.52, below base). #4 ensemble +.02 at k16.
#5(bonus) AIDE feedback-search (B=8): collapsed K5 unrecoverable (0->0); linear feedback NOT > i.i.d. on simple tasks (.71 vs .79). 
Jobs 92-98 done. Node 7 now IDLE. Paper vpo_main.tex updated (balances). Memory: project_mrpo_4ideas_results.

## MULTI-TASK GENERALIZATION (2026-06-07, node 7 only)
Goal: run famous tasks (GSM8K->MATH->code->multi-turn) with baselines (base, verifier-GRPO=status quo, then our consensus/vector/MGDA) to show where our methods help/fail vs status quo. Metric best@k.
Generic launcher: scripts/train_generic.py (loads any vf env by id) + scripts/run_env.sbatch (node-7 pinned).
GOTCHAS: (1) standard envs (gsm8k/math/...) must be installed first: `vf-install <env> -p environments`. (2) interactive $HOME is glitched to /home/example -> uv/HF caches fail; ALWAYS submit with `--export=ALL,HOME=/home/ubuntu` and run interactive cmds with `export HOME=/home/ubuntu`.
Status: gsm8k env installed; gsm8k_grpo.toml (verifier-GRPO baseline) smoking. TODO: gsm8k best@k eval (use env verifier), verifier-free vector axes for gsm8k (answer-consensus/format/brevity).

## MULTI-TASK STATUS (2026-06-07, node 7)
Generic multi-axis via reward_source="rubric" (env's per-func rubric scores = axes). Added scripts/train_generic.py, run_env.sbatch, sweep_env.sbatch, eval_best_at_k_env.py.
Tasks: GSM8K (single-axis ref; verifier-GRPO + answer-consensus both smoke-OK). WORDLE (multi-axis rubric: win/partial/turns; smoke OK but FLOOR EFFECT - base never wins, weak). CODE/MBPP (NEW env environments/code_mbpp; rubric tests_pass/format/brevity; smoke shows base ~37% tests_pass = REAL signal -> PRIMARY multi-axis task).
Envs need vf-install: gsm8k, wordle, code_mbpp (done). Always sbatch with --export=ALL,HOME=/home/ubuntu (HOME glitch).
RUNNING: job 107 codesweep = {grpo,vec(dir),mgda,fixed} x seeds123 (12 runs). Then eval_best_at_k_env per ckpt + aggregate_seeds. TODO after: gsm8k matrix, wordle matrix.

## ALL-TASKS CAMPAIGN (2026-06-07, node 7, queued) — "run all possible tasks today"
Feasible text tasks (Qwen-1.5B): code_mbpp, gsm8k, wordle, reasoning_gym_env.
Blocked: math (stdlib name collision), gpqa (gated HF), simpleqa (judge creds), self_reward (args), math_python (no funcs).
Queued matrices (4 configs {grpo/baseline, vec-dirichlet, mgda, fixed} x seeds 1-3 = 12 runs each):
  107 codesweep (RUNNING) -> 108 gsm8ksweep -> 109 wordlesweep -> 110 rgymsweep. Sequential (EXCLUSIVE). ~8-10h total.
After training: per-task best@k via scripts/eval_best_at_k_env.py (env-generic) + aggregate_seeds -> compare {dirichlet,mgda,fixed} variance (MGDA reliability) + collapse, across tasks.
Multi-axis tasks (real MGDA test): code (strong, base~37% tests_pass), wordle (FLOOR-base never wins). Single-axis: gsm8k, rgym.

## ALL-TASKS CAMPAIGN COMPLETE (2026-06-07) — node 7 now FREE
All 6 matrices trained (jobs 107-112) + cross-task best@k eval (113, 62 evals). Results: results/multitask/CROSS_TASK_SUMMARY.md.
Headline: MGDA reliability/collapse-avoidance replicates on signal-bearing tasks (gsm8k MGDA .93±.008 vs dirichlet/fixed ~.40; code MGDA lowest-var+best-mean); verifier-free consensus collapses gsm8k (best@1 .50->.13); MGDA fails on wordle (floor); tooltest saturated. Node idle - free for others.
