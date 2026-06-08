"""
Similarity measures for terminal (stdout/stderr) outputs.

The core challenge: terminal outputs from the same task but different command sequences
share a lot of boilerplate (prompt strings, permission bits, sizes) but differ in key
tokens (filenames, numbers, error codes). Naive string similarity conflates these.

Four measures ranked by expected quality (to be validated via overfit inspection):
  D: exit-code gate + key-token Jaccard  (recommended starting point)
  B: key-token Jaccard (order-insensitive, boilerplate-robust)
  C: line-set Jaccard (order-insensitive)
  A: content-only Levenshtein (after boilerplate stripping)
"""

import difflib
import json
import math
import re
import string
from itertools import combinations


# ---------------------------------------------------------------------------
# Boilerplate stripping (Option A preprocessing)
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    r"^\$\s+.*$",                          # command echo: $ ls -la
    r"^total \d+$",                         # ls -la total line
    r"^[d\-lrwx]{10}\s+\d+.*\d{2}:\d{2}",  # permission + timestamp lines
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", # ISO timestamps
    r"^\[sudo\].*",                         # sudo prompts
    r"^bash:\s+line \d+:.*",               # bash line references
    r"^real\s+\d+m",                        # time output
    r"^user\s+\d+m",
    r"^sys\s+\d+m",
]
_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS), re.MULTILINE
)

_STOPWORDS = {
    "the", "a", "an", "is", "in", "on", "at", "to", "for", "of", "and",
    "or", "with", "by", "from", "as", "be", "are", "was", "were",
    "not", "no", "yes", "true", "false", "none", "null",
    # common shell words that don't carry content signal
    "error", "warning", "info", "debug", "output", "done", "ok", "success",
    "failed", "stderr", "stdout",
}


def strip_boilerplate(stdout: str) -> str:
    cleaned = _BOILERPLATE_RE.sub("", stdout)
    # collapse multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _key_tokens(text: str) -> set[str]:
    """Extract tokens that carry content signal (numbers, filenames, non-stopwords)."""
    text = strip_boilerplate(text)
    # tokenize: split on whitespace and punctuation, keep alphanumeric + dots + slashes
    raw_tokens = re.findall(r"[a-zA-Z0-9_.\/\-]+", text)
    tokens = set()
    for tok in raw_tokens:
        tok_lower = tok.lower()
        if tok_lower in _stopwords_ext():
            continue
        # always include numbers and paths
        if re.match(r"^\d+$", tok) or "/" in tok or "." in tok:
            tokens.add(tok)
            continue
        # include words longer than 3 chars that aren't stopwords
        if len(tok) > 3:
            tokens.add(tok_lower)
    return tokens


def _stopwords_ext() -> set[str]:
    return _STOPWORDS | set(string.ascii_lowercase)


def _lines(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip()}


# ---------------------------------------------------------------------------
# Option A: content-only Levenshtein ratio
# ---------------------------------------------------------------------------

def similarity_levenshtein(a: str, b: str) -> float:
    """Normalized edit distance after boilerplate removal. Slow for long outputs."""
    ca, cb = strip_boilerplate(a), strip_boilerplate(b)
    return difflib.SequenceMatcher(None, ca, cb).ratio()


# ---------------------------------------------------------------------------
# Option B: key-token Jaccard (recommended)
# ---------------------------------------------------------------------------

def similarity_key_token_jaccard(a: str, b: str) -> float:
    """Jaccard over 'interesting' tokens: numbers, paths, content words (no stopwords)."""
    ta, tb = _key_tokens(a), _key_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Option C: line-set Jaccard
# ---------------------------------------------------------------------------

def similarity_line_jaccard(a: str, b: str) -> float:
    """Jaccard over stripped lines. Order-insensitive. No boilerplate filtering."""
    la, lb = _lines(a), _lines(b)
    if not la and not lb:
        return 1.0
    if not la or not lb:
        return 0.0
    return len(la & lb) / len(la | lb)


# ---------------------------------------------------------------------------
# Option D: exit-code gate + key-token Jaccard
# ---------------------------------------------------------------------------

def similarity_gated(
    a: str,
    b: str,
    exit_code_a: int = 0,
    exit_code_b: int = 0,
) -> float:
    """
    Two-stage:
    1. Exit codes must match
    2. Key-token Jaccard on content
    """
    if exit_code_a != exit_code_b:
        return 0.0
    return similarity_key_token_jaccard(a, b)


# ---------------------------------------------------------------------------
# Option E: number-strict + containment similarity (recommended)
#
# Key insight: numbers carry the answer signal. If rollout A says "42" and
# rollout B says "12 /etc/hosts", they are wrong pairs even though they share
# the path token. Require all numbers to agree, then use containment
# (min-denominator Jaccard) so "42" scores high against "Word count: 42".
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)*\b")  # matches 42, 1.26, 1.26.4
_PATH_RE = re.compile(r"(?:/[\w.\-]+)+")       # matches /usr/local/lib/python3.11/...


def _strip_paths(text: str) -> str:
    """Remove file path tokens so path-embedded numbers (e.g. python3.11) don't pollute."""
    return _PATH_RE.sub(" ", text)


def _extract_numbers(text: str) -> frozenset[str]:
    """Extract standalone content numbers; ignores numbers embedded in file paths."""
    cleaned = strip_boilerplate(text)
    cleaned = _strip_paths(cleaned)
    return frozenset(_NUMBER_RE.findall(cleaned))


def _containment_jaccard(ta: set[str], tb: set[str]) -> float:
    """|A∩B| / min(|A|,|B|) — gives 1.0 when the smaller set is a subset."""
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def similarity_strict(
    a: str,
    b: str,
    exit_code_a: int = 0,
    exit_code_b: int = 0,
) -> float:
    """
    Option E (recommended): number-strict gate + containment key-token similarity.

    Positive pair criteria:
    1. Same exit code
    2. All numbers agree exactly (or both outputs have no numbers)
    3. Containment similarity of key tokens >= threshold

    This prevents false positives from shared path tokens ("42 /etc/hosts" vs
    "12 /etc/hosts") while allowing format variation ("42" vs "Word count: 42").
    """
    if exit_code_a != exit_code_b:
        return 0.0

    na, nb = _extract_numbers(a), _extract_numbers(b)
    # If either output has numbers, they must agree exactly
    if na or nb:
        if na != nb:
            return 0.0

    ta, tb = _key_tokens(a), _key_tokens(b)
    return _containment_jaccard(ta, tb)


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def compute_similarity_matrix(
    stdouts: list[str],
    exit_codes: list[int] | None = None,
    measure: str = "strict",
) -> list[list[float]]:
    """
    Compute pairwise similarity for a group of G rollouts.
    measure: "strict" | "gated" | "jaccard" | "line_jaccard" | "levenshtein"
    """
    n = len(stdouts)
    matrix = [[0.0] * n for _ in range(n)]
    ec = exit_codes or [0] * n

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            if measure == "strict":
                s = similarity_strict(stdouts[i], stdouts[j], ec[i], ec[j])
            elif measure == "gated":
                s = similarity_gated(stdouts[i], stdouts[j], ec[i], ec[j])
            elif measure == "jaccard":
                s = similarity_key_token_jaccard(stdouts[i], stdouts[j])
            elif measure == "line_jaccard":
                s = similarity_line_jaccard(stdouts[i], stdouts[j])
            else:
                s = similarity_levenshtein(stdouts[i], stdouts[j])
            matrix[i][j] = s
            matrix[j][i] = s
    return matrix


def select_pairs(
    stdouts: list[str],
    exit_codes: list[int] | None = None,
    commands: list[str] | None = None,
    thresh_pos: float = 0.70,
    thresh_neg: float = 0.20,
    measure: str = "strict",
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Return (positive_pairs, negative_pairs) for a rollout group.

    positive: same exit code, similar content, different commands (if commands provided)
    negative: dissimilar content (or different exit codes)
    ambiguous zone (thresh_neg <= sim < thresh_pos): excluded
    """
    n = len(stdouts)
    matrix = compute_similarity_matrix(stdouts, exit_codes, measure)

    pos_pairs, neg_pairs = [], []
    for i, j in combinations(range(n), 2):
        sim = matrix[i][j]
        if sim >= thresh_pos:
            # optional diversity check: require different command sequences
            if commands is not None and commands[i] == commands[j]:
                continue
            pos_pairs.append((i, j))
        elif sim < thresh_neg:
            neg_pairs.append((i, j))
        # else: ambiguous, skip

    return pos_pairs, neg_pairs


def trajectory_diversity(stdouts: list[str], measure: str = "strict") -> float:
    """Mean pairwise dissimilarity (1 - similarity) for a rollout group."""
    n = len(stdouts)
    if n <= 1:
        return 0.0
    matrix = compute_similarity_matrix(stdouts, measure=measure)
    pairs = list(combinations(range(n), 2))
    return 1.0 - sum(matrix[i][j] for i, j in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Deliberately-uncorrelated reward axes (MRPO / Exp19)
#
# The K=3 V2 axes (strict_sim, jaccard_sim, exit_success) are near-collinear
# (strict & jaccard both measure output similarity), which is why V2 ≈ V1.
# To test the paper's central claim — that vector rewards earn their keep only
# when the axes carry non-collinear information — we add two axes designed to
# be low-correlation with correctness:
#   • brevity      : negative completion length (a correct answer can be terse
#                    or verbose → low correlation with the similarity axes)
#   • tool_format  : fraction of the rollout's tool calls that are well-formed
#                    (a correct answer can still emit malformed tool syntax)
# These are derived from the assistant *completion*, not the terminal stdout.
# ---------------------------------------------------------------------------

# Hermes-style tool-call fragment (the parser used by vf-vllm for this env).
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OPEN_TAG_RE = re.compile(r"<tool_call>")


def _assistant_text(completion) -> str:
    """Concatenate assistant-turn content from a completion (messages list or str)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for msg in completion:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in (None, "assistant"):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):  # multimodal content blocks
                parts.extend(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            # structured tool_calls (OpenAI format) — count their args as text
            for tc in msg.get("tool_calls", []) or []:
                fn = (tc or {}).get("function", {}) if isinstance(tc, dict) else {}
                parts.append(str(fn.get("arguments", "")))
        return "\n".join(parts)
    return ""


def brevity_reward(completions: list) -> list[float]:
    """
    Per-rollout brevity in [0, 1]: 1.0 = shortest in group, 0.0 = longest.
    Normalised within the group so it is scale-free and z-scoreable downstream.
    """
    lengths = [float(len(_assistant_text(c))) for c in completions]
    lo, hi = min(lengths), max(lengths)
    if hi - lo < 1e-9:
        return [1.0] * len(lengths)
    return [1.0 - (l - lo) / (hi - lo) for l in lengths]


def tool_format_validity(completions: list) -> list[float]:
    """
    Per-rollout fraction of well-formed tool calls in [0, 1].

    A tool call is well-formed if a <tool_call>...</tool_call> block parses as
    JSON. Penalises open <tool_call> tags that never parse. A rollout with no
    tool-call attempts at all scores 0.0 (it never used the tool it was asked to).
    """
    out: list[float] = []
    for c in completions:
        text = _assistant_text(c)
        attempts = len(_OPEN_TAG_RE.findall(text))
        if attempts == 0:
            out.append(0.0)
            continue
        valid = 0
        for blob in _TOOL_CALL_RE.findall(text):
            try:
                obj = json.loads(blob)
                if isinstance(obj, dict) and ("arguments" in obj or "name" in obj):
                    valid += 1
            except (ValueError, TypeError):
                pass
        out.append(valid / max(attempts, 1))
    return out


# Registry of available axes. Each entry maps a name to a builder that returns
# a per-rollout list[float] given the group's (stdouts, exit_codes, completions).
def _axis_strict(stdouts, ec, comps):
    m = compute_similarity_matrix(stdouts, ec, "strict")
    n = len(stdouts)
    return [sum(m[i][j] for j in range(n) if j != i) / max(n - 1, 1) for i in range(n)]


def _axis_jaccard(stdouts, ec, comps):
    m = compute_similarity_matrix(stdouts, exit_codes=None, measure="jaccard")
    n = len(stdouts)
    return [sum(m[i][j] for j in range(n) if j != i) / max(n - 1, 1) for i in range(n)]


def _axis_exit(stdouts, ec, comps):
    return [1.0 if e == 0 else 0.0 for e in ec]


def _axis_distinct(stdouts, ec, comps):
    # Coverage objective (idea #1): reward a rollout for DIFFERING from the group
    # = 1 - mean strict similarity to others. Opposite of consensus. Gated by
    # quality axes (exit/tool_format) so the policy is pushed toward valid-but-
    # diverse outputs (what best@k / downstream search needs) rather than agreement.
    m = compute_similarity_matrix(stdouts, ec, "strict")
    n = len(stdouts)
    return [1.0 - (sum(m[i][j] for j in range(n) if j != i) / max(n - 1, 1)) for i in range(n)]


REWARD_AXES = {
    "strict": _axis_strict,
    "jaccard": _axis_jaccard,
    "exit": _axis_exit,
    "distinct": _axis_distinct,
    "brevity": lambda stdouts, ec, comps: brevity_reward(comps or [""] * len(stdouts)),
    "tool_format": lambda stdouts, ec, comps: tool_format_validity(
        comps or [""] * len(stdouts)
    ),
}

# Default K=3 set reproduces V2 exactly (backward compatible).
DEFAULT_AXES = ["strict", "jaccard", "exit"]
# K=5 non-collinear set for the MRPO flagship (Exp19b).
MRPO_AXES = ["strict", "jaccard", "exit", "brevity", "tool_format"]
# Coverage objective (idea #1): quality (exit, tool_format) + distinctness, NO consensus.
COVERAGE_AXES = ["exit", "tool_format", "distinct"]


def _extract_boxed(completion) -> str:
    """Final boxed answer from a completion (for answer-based tasks like gsm8k/math)."""
    try:
        from verifiers.utils.data_utils import extract_boxed_answer
    except Exception:
        return ""
    txt = _assistant_text(completion)
    try:
        return (extract_boxed_answer(txt) or "").strip()
    except Exception:
        return ""


def answer_consensus_vector(completions: list, axes: list[str] | None = None) -> list[list[float]]:
    """Verifier-free reward axes for answer-based tasks (gsm8k, math), computed
    from completions' final answers (no ground truth):
      consensus : fraction of the group sharing this rollout's answer (self-consistency)
      format    : 1.0 if a boxed answer was produced
      brevity   : 1.0=shortest in group .. 0.0=longest
      distinct  : 1 - consensus (coverage objective)
    Returns (G, len(axes)). Default axes = [consensus, format, brevity].
    """
    names = axes or ANSWER_AXES
    n = len(completions)
    ans = [_extract_boxed(c) for c in completions]
    consensus = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        same = sum(1 for j in others if ans[j] != "" and ans[j] == ans[i])
        consensus.append(same / max(len(others), 1))
    cols = {
        "consensus": consensus,
        "format": [1.0 if a != "" else 0.0 for a in ans],
        "brevity": brevity_reward(completions),
        "distinct": [1.0 - c for c in consensus],
    }
    return [[cols[ax][i] for ax in names] for i in range(n)]


ANSWER_AXES = ["consensus", "format", "brevity"]
# Coverage variant for answer tasks: reward distinctness + format, not agreement.
ANSWER_COVERAGE_AXES = ["format", "brevity", "distinct"]


def compute_reward_vector(
    stdouts: list[str],
    exit_codes: list[int] | None = None,
    completions: list | None = None,
    axes: list[str] | None = None,
) -> list[list[float]]:
    """
    Compute K reward dimensions per rollout (returns shape (G, K)).

    Default axes reproduce V2 exactly: [strict_sim, jaccard_sim, exit_success].
    Pass axes=MRPO_AXES (and completions) for the K=5 non-collinear set used by
    the MRPO flagship experiment.

    Used for vector lambda sampling (V2/MRPO): R_i = λ · reward_vector[i],
    λ ~ Dirichlet(α) sampled fresh each training step.
    """
    n = len(stdouts)
    ec = exit_codes or [0] * n
    names = axes or DEFAULT_AXES

    columns = [REWARD_AXES[name](stdouts, ec, completions) for name in names]
    # transpose columns (K x G) -> rows (G x K)
    return [[columns[k][i] for k in range(len(names))] for i in range(n)]


def reward_axis_correlation(
    reward_vectors: list[list[float]],
) -> dict:
    """
    §7 diagnostic: given a stack of per-rollout reward vectors (M x K, pooled
    across many groups), return the K×K Pearson correlation matrix, the mean
    off-diagonal |rho|, and the effective rank (exp of the entropy of the
    normalised eigenvalue spectrum of the correlation matrix).

    Pure-Python (no numpy dependency) so it can run anywhere the trainer runs.
    """
    m = len(reward_vectors)
    if m < 2:
        return {"n": m, "corr": [], "mean_abs_offdiag": 0.0, "eff_rank": 0.0}
    K = len(reward_vectors[0])
    cols = [[reward_vectors[i][k] for i in range(m)] for k in range(K)]

    def _mean(x):
        return sum(x) / len(x)

    def _std(x, mu):
        return (sum((v - mu) ** 2 for v in x) / len(x)) ** 0.5

    mus = [_mean(c) for c in cols]
    sds = [_std(c, mus[k]) or 1e-12 for k, c in enumerate(cols)]
    corr = [[0.0] * K for _ in range(K)]
    for a in range(K):
        for b in range(K):
            cov = sum(
                (cols[a][i] - mus[a]) * (cols[b][i] - mus[b]) for i in range(m)
            ) / m
            corr[a][b] = cov / (sds[a] * sds[b])

    off = [abs(corr[a][b]) for a in range(K) for b in range(K) if a != b]
    mean_abs_off = sum(off) / len(off) if off else 0.0

    # effective rank via eigenvalues of the correlation matrix (power-free:
    # use the fact that trace = K and approximate spectrum by Gershgorin-free
    # Jacobi eigenvalues for small K).
    eff_rank = _effective_rank(corr)
    return {
        "n": m,
        "K": K,
        "corr": corr,
        "mean_abs_offdiag": mean_abs_off,
        "eff_rank": eff_rank,
    }


def _effective_rank(corr: list[list[float]]) -> float:
    """Effective rank = exp(Shannon entropy of normalised eigenvalues)."""
    K = len(corr)
    # Jacobi eigenvalue iteration for a small symmetric matrix.
    a = [row[:] for row in corr]
    for _ in range(100):
        # find largest off-diagonal
        p, q, mx = 0, 1, 0.0
        for i in range(K):
            for j in range(i + 1, K):
                if abs(a[i][j]) > mx:
                    mx, p, q = abs(a[i][j]), i, j
        if mx < 1e-9:
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        phi = 0.5 * math.atan2(2 * apq, aqq - app) if abs(aqq - app) > 1e-12 else math.pi / 4
        c, s = math.cos(phi), math.sin(phi)
        for k in range(K):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(K):
            akp, akq = a[p][k], a[q][k]
            a[p][k] = c * akp - s * akq
            a[q][k] = s * akp + c * akq
    eig = [max(a[i][i], 0.0) for i in range(K)]
    tot = sum(eig) or 1e-12
    probs = [e / tot for e in eig if e > 1e-12]
    entropy = -sum(p * math.log(p) for p in probs)
    return math.exp(entropy)


def mean_sim_per_rollout(
    stdouts: list[str],
    exit_codes: list[int] | None = None,
    measure: str = "strict",
) -> list[float]:
    """
    For each rollout, compute its mean pairwise similarity to all other rollouts.

    Returns a list of floats in [0, 1], one per rollout. Used as the continuous
    reward signal for variational TC loss (V1): higher mean sim = this rollout's
    output agrees with the group consensus = more likely to be correct.
    """
    n = len(stdouts)
    if n <= 1:
        return [0.0] * n
    matrix = compute_similarity_matrix(stdouts, exit_codes, measure)
    return [
        sum(matrix[i][j] for j in range(n) if j != i) / (n - 1)
        for i in range(n)
    ]


def density_reward_per_rollout(
    stdouts: list[str],
    exit_codes: list[int] | None = None,
    measure: str = "strict",
    bandwidth: float = 0.2,
) -> list[float]:
    """
    Leave-one-out KDE log-density reward over the rollout group (threshold-free).

    Treats (1 - sim(i, j)) as a 'distance' in similarity-space and applies a
    Gaussian kernel of bandwidth h:

        r_i = log( (1 / (G - 1)) * sum_{j != i} exp( -(1 - sim(i, j))^2 / (2 h^2) ) )

    As G -> infinity, r_i -> log p(o_i | task) up to bandwidth-induced bias.
    This is the §4c "Density-Bootstrap" reward in the research plan: a single
    bandwidth hyperparameter replaces both thresh_pos and thresh_neg and yields
    a continuous, differentiable consensus signal.

    Returns a list of log-densities, one per rollout. Caller should z-score these
    across the group (matching V1) before using as the GRPO advantage.
    """
    n = len(stdouts)
    if n <= 1:
        return [0.0] * n
    matrix = compute_similarity_matrix(stdouts, exit_codes, measure)
    h2 = max(bandwidth * bandwidth, 1e-8)
    log_norm = math.log(max(n - 1, 1))
    rewards: list[float] = []
    for i in range(n):
        log_terms = [
            -((1.0 - matrix[i][j]) ** 2) / (2.0 * h2)
            for j in range(n)
            if j != i
        ]
        # numerically stable logsumexp
        m = max(log_terms)
        s = sum(math.exp(lt - m) for lt in log_terms)
        rewards.append(m + math.log(s) - log_norm)
    return rewards
