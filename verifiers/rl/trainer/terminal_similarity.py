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


def compute_reward_vector(
    stdouts: list[str],
    exit_codes: list[int] | None = None,
) -> list[list[float]]:
    """
    Compute K=3 reward dimensions per rollout (returns shape (G, 3)):
      dim 0: mean strict similarity to others  — numerical/content consensus
      dim 1: mean token-jaccard to others      — approach/structural similarity
      dim 2: exit-code success                 — 1.0 if exit_code==0 else 0.0

    Used for vector lambda sampling (V2): R_i = λ · reward_vector[i],
    λ ~ Dirichlet(α) sampled fresh each training step.
    """
    n = len(stdouts)
    ec = exit_codes or [0] * n

    strict_mat = compute_similarity_matrix(stdouts, exit_codes, "strict")
    jaccard_mat = compute_similarity_matrix(stdouts, exit_codes=None, measure="jaccard")
    exit_success = [1.0 if e == 0 else 0.0 for e in ec]

    result = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        r0 = sum(strict_mat[i][j] for j in others) / max(len(others), 1)
        r1 = sum(jaccard_mat[i][j] for j in others) / max(len(others), 1)
        r2 = exit_success[i]
        result.append([r0, r1, r2])
    return result


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
