"""CPU unit tests for the coefficient-space MGDA solver (no torch needed).
Run: python tests/test_mgda.py"""
import importlib.util
import os

_P = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "verifiers", "rl", "trainer", "mgda.py",
)
_s = importlib.util.spec_from_file_location("mgda", _P)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

min_norm_on_simplex = _m.min_norm_on_simplex
assemble_Q = _m.assemble_Q
mgda_beta = _m.mgda_beta


def approx(a, b, tol=1e-3):
    return abs(a - b) <= tol


def test_orthogonal_unit_gradients_uniform():
    # Q = I (orthogonal equal-norm gradients) -> min alpha^T alpha on simplex = uniform.
    Q = [[1.0, 0.0], [0.0, 1.0]]
    a = min_norm_on_simplex(Q)
    assert approx(a[0], 0.5) and approx(a[1], 0.5), a
    print("ok: orthogonal -> uniform", a)


def test_conflicting_interior():
    # Negatively correlated gradients -> interior solution, here exactly [0.5,0.5].
    Q = [[1.0, -0.5], [-0.5, 1.0]]
    a = min_norm_on_simplex(Q)
    assert approx(a[0], 0.5) and approx(a[1], 0.5), a
    # objective value 0.25 < either vertex (1.0)
    val = sum(a[i] * Q[i][j] * a[j] for i in range(2) for j in range(2))
    assert approx(val, 0.25), val
    print("ok: conflicting -> interior [0.5,0.5], obj=0.25")


def test_collinear_reduces_to_smaller_norm():
    # g1 and g2 nearly parallel, g2 has larger norm. Min-norm picks the
    # combination minimising ||.||; with g2 bigger, weight shifts toward g1.
    # Q from g1=(1,0), g2=(2,0): Q=[[1,2],[2,4]]. min on simplex -> alpha=[1,0].
    Q = [[1.0, 2.0], [2.0, 4.0]]
    a = min_norm_on_simplex(Q)
    assert approx(a[0], 1.0) and approx(a[1], 0.0), a
    print("ok: collinear -> smaller-norm vertex", a)


def test_three_axes_dominant_collinear():
    # Two collinear axes + one orthogonal. MGDA should not blow up; weights valid.
    # g1=(1,0), g2=(1.0,0) (collinear), g3=(0,1). Q = G G^T with rows g_k.
    g = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    Q = [[sum(g[i][d] * g[j][d] for d in range(2)) for j in range(3)] for i in range(3)]
    a = min_norm_on_simplex(Q)
    assert approx(sum(a), 1.0) and all(x >= -1e-9 for x in a), a
    # collinear pair shares weight; orthogonal axis gets some too (min-norm interior)
    print("ok: 3-axis (2 collinear + 1 orth)", [round(x, 3) for x in a])


def test_AMAt_identity_matches_direct():
    # Verify Q = A M A^T equals the direct gradient Gram <g_k,g_l>, where
    # g_k = sum_i A_ki v_i. Random-ish small example.
    V = [[1.0, 0.5, -0.3], [0.2, 1.0, 0.1]]  # G=2 per-traj grads in R^3
    A = [[1.0, 0.0], [0.5, 0.5], [-1.0, 2.0]]  # K=3 axes over G=2
    G = len(V)
    M = [[sum(V[i][d] * V[j][d] for d in range(3)) for j in range(G)] for i in range(G)]
    Q = assemble_Q(A, M)
    # direct: g_k = sum_i A_ki V_i
    gk = [[sum(A[k][i] * V[i][d] for i in range(G)) for d in range(3)] for k in range(3)]
    Qdirect = [[sum(gk[k][d] * gk[l][d] for d in range(3)) for l in range(3)] for k in range(3)]
    for k in range(3):
        for l in range(3):
            assert approx(Q[k][l], Qdirect[k][l], 1e-9), (k, l, Q[k][l], Qdirect[k][l])
    print("ok: Q = A M A^T == direct <g_k,g_l>")


def test_beta_application_equals_mixture():
    # beta = A^T alpha; the update sum_i beta_i v_i must equal sum_k alpha_k g_k.
    V = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]  # G=3 in R^2
    A = [[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]   # K=2 over G=3
    G, Dim = 3, 2
    M = [[sum(V[i][d] * V[j][d] for d in range(Dim)) for j in range(G)] for i in range(G)]
    alpha, beta = mgda_beta(A, M)
    # sum_i beta_i v_i
    upd_beta = [sum(beta[i] * V[i][d] for i in range(G)) for d in range(Dim)]
    # sum_k alpha_k g_k, g_k = sum_i A_ki v_i
    gk = [[sum(A[k][i] * V[i][d] for i in range(G)) for d in range(Dim)] for k in range(2)]
    upd_alpha = [sum(alpha[k] * gk[k][d] for k in range(2)) for d in range(Dim)]
    for d in range(Dim):
        assert approx(upd_beta[d], upd_alpha[d], 1e-9), (d, upd_beta[d], upd_alpha[d])
    print("ok: sum_i beta_i v_i == sum_k alpha_k g_k  (beta feeds single backward)")


if __name__ == "__main__":
    test_orthogonal_unit_gradients_uniform()
    test_conflicting_interior()
    test_collinear_reduces_to_smaller_norm()
    test_three_axes_dominant_collinear()
    test_AMAt_identity_matches_direct()
    test_beta_application_equals_mixture()
    print("\nALL MGDA SOLVER TESTS PASSED")
