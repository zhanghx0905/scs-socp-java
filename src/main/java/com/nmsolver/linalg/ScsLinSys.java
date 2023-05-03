package com.nmsolver.linalg;

import lombok.NonNull;


public class ScsLinSys {
    // Sparse Indirect
    final int n, m;
    final DCSCMatrix A; // m * n
    final DCSCMatrix P; // n * n

    DVector diag_r; // n + m
    DVector diag_rx; // n
    DVector diag_ry; // m

    final public DCSCMatrix At;

    DVector p; /* cg iterate */
    DVector r; /* cg residual */
    DVector Gp;/* updated CG direction */
    DVector tmp;

    DVector M; /* Preconditioner */
    DVector z; /* z = M r (M is inverse preconditioner) */

    public ScsLinSys(
            @NonNull DCSCMatrix A,
            DCSCMatrix P,
            @NonNull DVector diag_r) {
        this.A = A;
        this.P = P;
        this.n = A.nCol;
        this.m = A.nRow;

        this.diag_r = diag_r;
        this.diag_rx = diag_r.slice(0, n);
        this.diag_ry = diag_r.slice(n, m);

        this.At = A.transpose();
        this.M = new DVector(this.n);
        this.z = new DVector(this.n);
        this.p = new DVector(this.n);
        this.r = new DVector(this.n);
        this.Gp = new DVector(this.n);
        this.tmp = new DVector(this.m);

        this.setPreconditioner();
        validate();
    }

    /* solves Mx = b, for x but stores result in b */
    /* s contains warm-start (if available) */
    /*
     * [x] = [R_x + P A' ]^{-1} [rx]
     * [y] [ A -R_y ] [ry]
     *
     * becomes:
     *
     * x = (R_x + P + A' R_y^{-1} A)^{-1} (rx + A' R_y^{-1} ry)
     * y = R_y^{-1} (Ax - ry)
     *
     */
    public void solveLinSys(DVector b, double tol, DVector s) {

        if (b.normInf() <= 1e-12) {
            b.setRange(0);
            return;
        }

        /* b = [rx; ry] */
        /* tmp = ry */
        tmp.setRange(0, b, n, m);
        /* tmp = R_y^{-1} * ry */
        tmp.idivide(diag_ry);
        /* b[:n] = rx + A' R_y^{-1} ry */
        DCSCMatrix.accumByATx(A, tmp, b);
        var max_iters = 10 * n;
        /*
         * solves (R_x + P + A' R_y^{-1} A)x = b,
         * solution stored in b
         */
        var x = b.slice(0, n);
        pcg(x, max_iters, tol, s);
        b.setRange(0, x, 0, n);

        var y = b.slice(n, m);
        /* b[n:] = -ry */
        y.imult(-1);
        /* b[n:] = Ax - ry */
        DCSCMatrix.accumByATx(At, x, y);
        /* b[n:] = R_y^{-1} (Ax - ry) = y */
        y.idivide(diag_ry);
        b.setRange(n, y, 0, m);
    }

    public void updateDiagR(DVector diag_r) {
        this.diag_r = diag_r;
        diag_rx.setRange(0, diag_r, 0, n);
        diag_ry.setRange(0, diag_r, n, m);
        setPreconditioner();
    }

    private void validate() {
        if (P != null) {
            if (P.nCol != A.nCol) {
                throw new IllegalArgumentException(
                        String.format(
                                "P dimension = %d, inconsistent with n = %d\n", P.nCol, A.nCol));
            }
            if (P.nRow != P.nCol) {
                throw new IllegalArgumentException("P is not square\n");
            }
            for (int j = 0; j < P.nCol; j++) { /* cols */
                for (int i = P.colPtr[j]; i < P.colPtr[j + 1]; i++) {
                    if (P.rowIdx[i] > j) { /* if row > */
                        throw new IllegalArgumentException("P is not upper triangular\n");
                    }
                }
            }
        }

    }

    /* y += R_x * x */
    private void accumByRxx(final DVector x, DVector y) {
        for (int i = 0; i < y.data.length; ++i) {
            y.data[i] += diag_rx.data[i] * x.data[i];
        }

    }

    /* set M = inv ( diag ( R_x + P + A' R_y^{-1} A ) ) */
    private void setPreconditioner() {
        /* M_ii = (R_x)_i + P_ii + a_i' (R_y)^-1 a_i */

        /* M_ii = (R_x)_i */
        M.setTo(diag_rx);
        /* M_ii += P_ii */
        if (P != null) {
            var P_diag = new DVector(this.P.diag());
            this.M.iadd(P_diag, 1);
        }
        /* M_ii += a_i' (R_y)^-1 a_i */
        for (int i = 0; i < n; ++i) {
            for (int k = A.colPtr[i]; k < A.colPtr[i + 1]; ++k) {
                var Axk = A.data[k];
                M.data[i] += Axk * Axk / diag_ry.data[A.rowIdx[k]];
            }
        }

        for (int i = 0; i < this.n; ++i) {
            M.data[i] = 1. / M.data[i];
        }

    }

    /* solves (R_x * I + P + A' R_y^{-1} A)x = b, s warm start, solution in b */
    private void pcg(DVector b, int max_its, double tol, DVector s) {
        if (s == null) {
            /* r = b */
            r.setTo(b);
            b.setRange(0);
        } else {
            /* r = Mat * s */
            matVec(s, r);
            /* r = Mat * s - b */
            r.iadd(b, -1);
            /* r = b - Mat * s */
            r.imult(-1);
            /* b = s */
            b.setTo(s);
        }

        if (r.normInf() < Math.max(1e-12, tol)) {
            return;
        }
        /* z = M r (M is inverse preconditioner) */
        DVector.mult(r, M, z);
        /* ztr = z'r */
        var ztr = DVector.dot(z, r);
        double ztr_prev;
        /* p = z */
        p.setTo(z);
        for (int i = 0; i < max_its; ++i) {
            /* Gp = Mat * p */
            matVec(p, Gp);
            /* alpha = z'r / p'G p */
            var alpha = ztr / DVector.dot(p, Gp);
            /* b += alpha * p */
            b.iadd(p, alpha);
            /* r -= alpha * G p */
            r.iadd(Gp, -alpha);

            if (r.normInf() < tol) {
                break;
            }

            /* z = M r (M is inverse preconditioner) */
            DVector.mult(r, M, z);
            ztr_prev = ztr;
            /* ztr = z'r */
            ztr = DVector.dot(z, r);
            /* p = beta * p, where beta = ztr / ztr_prev */
            p.imult(ztr / ztr_prev);
            /* p = z + beta * p */
            p.iadd(z, 1);
        }
    }

    /* y = (R_x + P + A' R_y^{-1} A) x */
    private void matVec(final DVector x, DVector y) {
        tmp.setRange(0);
        y.setRange(0);
        if (P != null) {
            DCSCMatrix.accumByPx(P, x, y);/* y = Px */
        }
        DCSCMatrix.accumByATx(At, x, tmp); /* tmp = Ax, use AT for better performance */
        tmp.idivide(diag_ry); /* z = R_y^{-1} A x */
        DCSCMatrix.accumByATx(A, tmp, y); /* y = Px + A' R_y^{-1} Ax */
        accumByRxx(x, y);/* y = (R_x + P + A' R_y^{-1} A) x */
    }

}
