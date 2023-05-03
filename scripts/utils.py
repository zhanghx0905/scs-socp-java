import scipy.sparse as sparse
import scipy.io as io
import numpy as np


def mm_matrix(path: str) -> sparse.csc_matrix:
    matrix = io.mmread(path)
    assert sparse.issparse(matrix)
    matrix = matrix.tocsc()
    assert matrix.indices.dtype == np.int32
    return matrix


def read_java_output(datadir: str):
    with open(f"{datadir}/scs-sol.csv", "r") as f:
        lines = f.read().splitlines()
    arr = []
    for line in lines[1:]:
        arr.append(float(line))
    sol = np.array(arr)
    with open(f"{datadir}/scs-obj.csv", "r") as f:
        lines = f.read().splitlines()
    obj = float(lines[1])
    return sol, obj
