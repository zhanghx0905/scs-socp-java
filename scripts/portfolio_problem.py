"""
Generate testing data for Portfolio problem
"""
import argparse
import os

import scs
import numpy as np
import scipy.io as io
import scipy.sparse as sparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=1000, type=int)
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args()


args = parse_args()
LAMBA = 1.0
OUTPUT_DIR = f"datap/{args.size}/{args.seed}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(args.seed)

import numpy as np


def bad_case():
    n = args.size
    A = np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)
    eigvals = np.logspace(4, -3, num=n)
    # create a diagonal matrix with the eigenvalues
    D = np.diag(eigvals)
    V = np.dot(Q, np.dot(D, np.linalg.inv(Q)))
    assert np.all(np.linalg.eigvals(V) > 0)
    return V


def gen_data():
    mat = np.random.randn(int(1.2 * args.size), args.size)
    V = np.cov(mat.T)
    # V = bad_case()
    rbar = np.random.randn(args.size, 1)
    return V, rbar


def convert_data() -> tuple[sparse.csc_matrix, sparse.csc_matrix, sparse.csc_matrix]:
    V, rbar = gen_data()

    n1 = V.shape[1]
    L = sparse.csc_matrix(np.linalg.cholesky(V))
    Aq = sparse.hstack((np.zeros((n1, 1)), L, np.zeros((n1, 1))), format="csc")
    Aq = sparse.vstack((Aq, np.zeros((1, n1 + 2))), format="csc")
    Aq[n1, 0] = Aq[n1, n1 + 1] = 0.5
    # (n1 + 1) * (n1 + 2)

    cq = sparse.csc_matrix((n1 + 2, 1))
    cq[0] = 0.5
    cq[-1] = -0.5

    Al = sparse.hstack(
        [sparse.csc_matrix((n1, 1)), sparse.csc_matrix(np.ones((n1, 1)))], format="csc"
    )  # np.eye(n1 + 1, n1)
    Al = sparse.vstack([Al, np.zeros((1, 2))], format="csc")
    cl = np.array([[0], [-1]])

    Aus = []
    cus = []
    for i in range(n1):
        Au = np.zeros((n1 + 1, 2))
        Au[i, 0] = 1
        cu = np.zeros((2, 1))
        Aus.append(Au)
        cus.append(cu)

    A = sparse.hstack((Aq, Al, *Aus), format="csc").T
    c = -sparse.vstack((rbar, [LAMBA]), format="csc")
    b = sparse.vstack((cq, cl, *cus), format="csc")
    return A, b, c


def scs_solve(A: sparse.csc_matrix, b: sparse.csc_matrix, c: sparse.csc_matrix):
    A = sparse.csc_matrix(A)
    b = b.toarray().flatten()
    c = c.toarray().flatten()
    print(A.shape, b.shape, c.shape)
    m, n = A.shape
    qcones = np.array([n + 1, *[2 for _ in range(n)]])
    # Populate dicts with data to pass into SCS
    data = dict(P=None, A=A, b=b, c=c)
    cone = dict(q=qcones)

    # Initialize solver
    solver = scs.SCS(
        data,
        cone,
        use_indirect=True,
        normalize=False,
        acceleration_lookback=0,
        eps_abs=1e-6,
        eps_rel=1e-6,
    )
    # Solve!
    sol = solver.solve()
    x = sol["x"]
    return x


if __name__ == "__main__":
    A, b, c = convert_data()
    # scs_solve(A, b, c)
    io.savemat(
        f"{OUTPUT_DIR}/Abc.mat",
        {
            "A": sparse.csc_matrix(A),
            "b": sparse.csc_matrix(b),
            "c": sparse.csc_matrix(c),
        },
    )

    print(f"{OUTPUT_DIR} Data Generated")
