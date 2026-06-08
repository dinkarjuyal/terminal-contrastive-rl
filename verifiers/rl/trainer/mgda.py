"""MGDA in coefficient space (paper §3, §5).

The naive Multiple-Gradient Descent Algorithm needs one backward pass per
reward axis to form the K gradients g_k. This module avoids that. Under
GRPO-style outcome advantages every axis-gradient is a reweighting of the
SAME G per-trajectory score gradients v_i = grad log pi(tau_i):

    g_k = sum_i A_k(tau_i) v_i.

Hence all g_k live in span{v_1,...,v_G} (dim <= G), and the MGDA min-norm
problem

    min_{alpha in simplex} || sum_k alpha_k g_k ||^2

reduces to a tiny K-dimensional quadratic with matrix Q = A M A^T, where
M_ij = <v_i, v_j> is the G x G Gram matrix of per-trajectory gradients and
A_{ki} = A_k(tau_i) is the (K x G) advantage matrix. The optimal mixing
weights alpha* give per-trajectory weights

    beta* = A^T alpha*   (length G),

and the MGDA update sum_k alpha*_k g_k = sum_i beta*_i v_i is exactly the
ordinary GRPO gradient with per-trajectory advantage beta*_i. So the caller
never materialises the g_k and never does a second backward: it computes M
once (one vmap per-sample-gradient pass), solves this tiny QP, and feeds
beta* as the advantage into the existing single GRPO backward.

This file is pure Python (no torch) so the solver is unit-testable on CPU;
the trainer converts its small (K<=~8, G<=~16) matrices to nested lists.
"""

from __future__ import annotations


def _matmul(X, Y):
    """X (a x b) times Y (b x c) -> (a x c), nested lists."""
    a, b, c = len(X), len(Y), len(Y[0])
    out = [[0.0] * c for _ in range(a)]
    for i in range(a):
        Xi = X[i]
        for k in range(b):
            xik = Xi[k]
            if xik == 0.0:
                continue
            Yk = Y[k]
            oi = out[i]
            for j in range(c):
                oi[j] += xik * Yk[j]
    return out


def _transpose(X):
    return [list(col) for col in zip(*X)]


def _matvec(Q, x):
    return [sum(Q[i][j] * x[j] for j in range(len(x))) for i in range(len(Q))]


def assemble_Q(A, M):
    """Q = A M A^T  (K x K), from A (K x G) and M (G x G)."""
    return _matmul(_matmul(A, M), _transpose(A))


def min_norm_on_simplex(Q, iters: int = 100, tol: float = 1e-10):
    """argmin_{alpha in simplex} alpha^T Q alpha via Frank-Wolfe with exact
    quadratic line search. Q is a K x K positive-semidefinite matrix
    (here Q = A M A^T). Returns alpha (length K, sums to 1).

    K=1 -> [1.0]. K=2 has a closed form but FW handles it fine.
    """
    K = len(Q)
    if K == 1:
        return [1.0]
    alpha = [1.0 / K] * K  # uniform start
    for _ in range(iters):
        grad = _matvec(Q, alpha)  # proportional to (1/2) d/dalpha (alpha^T Q alpha)
        # best simplex vertex = argmin_k grad_k
        t = min(range(K), key=lambda k: grad[k])
        d = [(1.0 if k == t else 0.0) - alpha[k] for k in range(K)]  # e_t - alpha
        Qd = _matvec(Q, d)
        dQd = sum(d[k] * Qd[k] for k in range(K))
        aQd = sum(alpha[k] * Qd[k] for k in range(K))
        if dQd <= tol:
            # objective linear/flat along d: step fully if it decreases, else stop
            gamma = 1.0 if aQd < -tol else 0.0
        else:
            gamma = -aQd / dQd
            gamma = 0.0 if gamma < 0.0 else (1.0 if gamma > 1.0 else gamma)
        if gamma <= tol:
            break
        alpha = [alpha[k] + gamma * d[k] for k in range(K)]
    # clean tiny negatives / renormalise
    alpha = [a if a > 0.0 else 0.0 for a in alpha]
    s = sum(alpha) or 1.0
    return [a / s for a in alpha]


def mgda_beta(A, M, iters: int = 100):
    """Given advantage matrix A (K x G) and per-trajectory Gram M (G x G),
    return (alpha, beta) where alpha solves the MGDA simplex QP and
    beta = A^T alpha is the per-trajectory advantage to feed into the
    existing single GRPO backward.
    """
    Q = assemble_Q(A, M)
    alpha = min_norm_on_simplex(Q, iters=iters)
    AT = _transpose(A)  # G x K
    beta = [sum(AT[i][k] * alpha[k] for k in range(len(alpha))) for i in range(len(AT))]
    return alpha, beta
