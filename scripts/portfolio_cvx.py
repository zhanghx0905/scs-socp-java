import argparse
import os

import cvxpy as cp
import numpy as np
from utils import mm_matrix, read_java_output

MOSEK_ARGS = {
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6,
    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-6,
}



def test_cvx(datadir: str):  
    scs_sol, scs_obj = read_java_output(datadir)
        
    A = mm_matrix(f"{datadir}/A.mtx")
    b = mm_matrix(f"{datadir}/b.mtx").toarray()
    c = mm_matrix(f"{datadir}/c.mtx").toarray()
    m, n = A.shape
    # Define and solve the CVXPY problem.
    x = cp.Variable((n, 1))
    s = cp.Variable((m, 1))

    qcones = [n + 1, *[2 for _ in range(n)]]

    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.\
    cnt = 0
    soc_constraints = []
    for qcone in qcones:
        constraint = cp.SOC(s[cnt], s[cnt + 1 : cnt + qcone])
        cnt += qcone
        soc_constraints.append(constraint)


    prob = cp.Problem(cp.Minimize(c.T @ x), soc_constraints + [A @ x + s == b])
    prob.solve(solver=cp.MOSEK, verbose=True, mosek_params=MOSEK_ARGS)

    cvx_sol, cvx_obj = x.value.flatten(), prob.value
    valgap = np.abs((cvx_obj - scs_obj) / cvx_obj)
    solgap = np.linalg.norm(cvx_sol - scs_sol) / np.linalg.norm(cvx_sol)
    elapsed = prob.solver_stats.solve_time
    np.savez(f"{datadir}/cvx.npz", sol=x.value, obj=prob.value, allow_pickle=True)
    # result = np.load(f"{datadir}/cvx.npz", allow_pickle=True)
    return valgap, solgap, elapsed


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, required=True)
args = parser.parse_args()
datadir = args.datadir
result = []
for path in os.listdir(datadir):
    result.append(test_cvx(os.path.join(datadir, path)))
valgap, solgap, elapsed = np.mean(np.array(result), axis=0)

print(f"Avg Time: {elapsed} s")
print(f"Avg OptValGap =  {valgap}, Avg OptSolGap = {solgap}\n\n")
