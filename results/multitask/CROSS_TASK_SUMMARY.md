# Cross-task generalization (2026-06-07, node 7, n=16 best@k, 3 seeds)
best@1 / best@16 (mean±std):
- code/MBPP:  base .051/.160 | grpo .078/.183±.017 | dirichlet .040/.107±.062 | MGDA .055/.183±.005 | fixed .045/.123±.039
- gsm8k:      base .502/.950 | grpo(verifier) .691/.953 | consensus(dir) .130/.437±.065 | MGDA .504/.930±.008 | fixed .137/.397±.061
- wordle(FLOOR): base 0/0 | grpo .007/.100 | dirichlet .028/.367 | MGDA .001/.017 | fixed .014/.183
- rgym:       grpo .109/.277 | vec .111/.243 | MGDA .042/.170(n=1) | fixed .109/.245
- tooltest:   ~1.0 all (SATURATED, no signal)

FINDINGS (honest):
1. MGDA reliability REPLICATES on signal-bearing tasks: code (lowest var ±.005 + best mean among vector methods), gsm8k (0.930±.008 vs dirichlet/fixed collapse ~0.40±.06). Strongest = gsm8k MGDA avoids the consensus collapse.
2. Verifier-free consensus COLLAPSES gsm8k (best@1 .50->.13) -> bash collapse replicates on standard benchmark.
3. MGDA FAILS on wordle (floor/degenerate axes, base never wins) -> honest boundary; consistent with "benefit needs non-degenerate signal-bearing axes".
4. tooltest saturated (trivial); rgym inconclusive (2 mgda seeds failed).
Net: nuanced cross-task generalization; axis structure governs whether MGDA helps.

## 7B SCALE CHECK (Qwen2.5-7B-Instruct, gsm8k, 3 seeds, 2026-06-08)
best@1 / best@16: base .896/.980 | verifier-GRPO .909/.973 | Dirichlet-consensus .000/.000 (TOTAL COLLAPSE) | MGDA .899±.003/.973±.005 (PRESERVED) | fixed .001/.013 (COLLAPSE).
=> MGDA collapse-avoidance HOLDS and STRENGTHENS at 7B (baselines go to ~0 vs ~0.40 at 1.5B; MGDA stays at base level). NOT a small-model artifact.

## 7B MBPP + WORDLE (Qwen2.5-7B, 3 seeds, 2026-06-09)
MBPP best@16: base .150 | grpo .160±.000 | Dirichlet .130±.008 | MGDA .150±.008 | fixed .120±.016.
=> same ordering as 1.5B (MGDA=verifier-GRPO > Dirichlet/fixed) but COMPRESSED/within-noise near task floor.
Wordle best@16: ALL methods incl base = 0.000 (total floor at 7B). Uninformative.
DECISION: paper one-liner only; GSM8K-7B stays the decisive scale anchor.
