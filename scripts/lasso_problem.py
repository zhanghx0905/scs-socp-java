import argparse
import os

import numpy as np
import scs
from cvxpy import *
from scipy import io, sparse

from utils import read_java_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=1000, type=int)
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args()


args = parse_args()
# Generate problem data
np.random.seed(args.seed)

N = args.size  # Variables
M = args.size  # Measurements
LAMBA = 0.1
OUTPUT_DIR = f"datal/{args.size}/{args.seed}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def gen_data():
    Ad = sparse.random(M, N, density=0.5)  # Measurement matrix
    x_true = sparse.random(N, 1, density=0.1)  # True sparse vector
    x_true = np.array(x_true.todense()).squeeze()

    measurements = Ad @ x_true + 0.1 * np.random.randn(M)
    measurements = np.array(measurements).squeeze()

    return Ad, x_true, measurements


def convert_data(Ad, measurements):
    # Auxiliary data
    In = sparse.eye(N)
    Im = sparse.eye(M)
    On = sparse.csc_matrix((N, N))
    Onm = sparse.csc_matrix((N, M))

    # SCS data
    P = sparse.block_diag([On, sparse.eye(M), On], format="csc")
    q = np.zeros(2 * N + M)
    A = sparse.vstack(
        [
            # zero cone
            sparse.hstack([Ad, -Im, Onm.T]),
            # positive cones
            sparse.hstack([In, Onm, -In]),
            sparse.hstack([-In, Onm, -In]),
        ],  # m + 2n, 2n + m
        format="csc",
    )
    b = np.hstack([measurements, np.zeros(N), np.zeros(N)])
    c = np.hstack([np.zeros(N + M), LAMBA * np.ones(N)])
    return P, A, b, c


def scs_solve(x_true, P, A, b, c):
    data = dict(P=P, A=A, b=b, c=c)
    cone = dict(z=N, l=2 * M)
    # Setup workspace
    solver = scs.SCS(
        data,
        cone,
        use_indirect=True,
        normalize=False,
        acceleration_lookback=0,
    )
    sol = solver.solve()  # lambda = 0
    x = sol["x"][:N]
    print(f"Error : {np.linalg.norm(x_true - x) / np.linalg.norm(x_true)}")
    info = sol["info"]
    print(info)


def cvx_solve(Ad, measurements, x_true):
    x = Variable(N)
    objective = 0.5 * sum_squares(Ad @ x - measurements) + LAMBA * norm1(x)
    prob = Problem(Minimize(objective))
    prob.solve(solver=MOSEK, verbose=True)
    x_sol, obj = x.value, prob.value
    np.savez(f"{OUTPUT_DIR}/cvx.npz", sol=x_sol, obj=obj, allow_pickle=True)
    print(f"Error : {np.linalg.norm(x_true - x_sol) / np.linalg.norm(x_true)}")
    print(prob.solver_stats.solve_time, prob.solver_stats.setup_time)

def compare():
    result = np.load(f"{OUTPUT_DIR}/cvx.npz", allow_pickle=True)
    cvx_sol, cvx_obj = result["sol"], result["obj"]
    scs_sol, scs_obj = read_java_output(OUTPUT_DIR)
    scs_sol = scs_sol[: cvx_sol.shape[0]]
    valgap = np.abs((cvx_obj - scs_obj) / cvx_obj)
    solgap = np.linalg.norm(cvx_sol - scs_sol) / np.linalg.norm(cvx_sol)
    print(valgap, solgap)


if __name__ == "__main__":
    Ad, x_true, measurements = gen_data()
    P, A, b, c = convert_data(Ad, measurements)
    io.mmwrite(f"{OUTPUT_DIR}/P", sparse.csc_matrix(P))
    io.mmwrite(f"{OUTPUT_DIR}/A", sparse.csc_matrix(A))
    io.mmwrite(f"{OUTPUT_DIR}/b", sparse.csc_matrix(b))
    io.mmwrite(f"{OUTPUT_DIR}/c", sparse.csc_matrix(c))
    cvx_solve(Ad, measurements, x_true)
    scs_solve(x_true, P, A, b, c)
    # compare()
