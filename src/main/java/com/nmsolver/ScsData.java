package com.nmsolver;

import com.nmsolver.linalg.DCSCMatrix;
import com.nmsolver.linalg.DVector;

import lombok.AllArgsConstructor;
import lombok.NonNull;

@AllArgsConstructor
public class ScsData {
    final int m;
    final int n;
    final @NonNull DCSCMatrix A;    // m * n
    final DCSCMatrix P;    // n * n
    final @NonNull DVector b;    // m
    final @NonNull DVector c;    // n

    public ScsData(DCSCMatrix A, DVector b, DVector c) {
        this.A = A;
        this.b = b;
        this.c = c;
        this.m = A.nRow;
        this.n = A.nCol;
        this.P = null;
    }
}
