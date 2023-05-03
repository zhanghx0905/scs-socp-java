package com.nmsolver;

import java.util.Arrays;
import java.util.Random;

import org.junit.Test;

import com.nmsolver.cones.ScsCone;
import com.nmsolver.cones.ScsConeWork;
import com.nmsolver.linalg.DCSCMatrix;
import com.nmsolver.linalg.DVector;

/*
 create data for problem:

 minimize 	    c'*x
 subject to 	Ax <=_K b

 where K is a product of zero, linear, and second-order cones. A is a sparse
 matrix in
 CSC format. A is factor * n by n with about sqrt(n) nonzeros per column.

 Construct data in such a way that the problem is primal and dual
 feasible and thus bounded.
 */
public class RandomTest {
    double p_z = 0.1; // Ratio of zero cones
    double p_l = 0.3; // Ratio of linear cones
    int n = 2000;
    int factor = 4; // size(A) = (n * factor, n)
    Random rand = new Random();
    long seed = 1;

    // [-1., 1.]
    double genRandDouble() {
        return (rand.nextDouble() - 0.5) * 2;
    }

    ScsData genRandomProbData(
            int nnz, int col_nnz,
            ScsCone k,
            ScsSolution opt) {
        final int m = n * factor;
        int[] A_i = new int[nnz],
                A_p = new int[n + 1];
        double[] A_x = new double[nnz];
        var A = new DCSCMatrix(m, n, A_p, A_i, A_x);
        var b = new DVector(m);
        var c = new DVector(n);

        var z = new DVector(m);
        for (int i = 0; i < m; ++i) {
            opt.y.data[i] = z.data[i] = genRandDouble();
        }
        var tmp_cone_work = new ScsConeWork(k, m);
        tmp_cone_work.projDualCone(opt.y, null, 0);

        for (int i = 0; i < m; ++i) {
            b.data[i] = opt.s.data[i] = opt.y.data[i] - z.data[i];
        }
        for (int i = 0; i < n; ++i) {
            opt.x.data[i] = genRandDouble();
        }
        /*
         * c = -A'*y
         * b = A*x + s
         */
        A.colPtr[0] = 0;
        for (int j = 0; j < n; ++j) {
            int r = 0;
            for (int i = 0; i < m && r < col_nnz; ++i) {
                int rn = m - i, rm = col_nnz - r;
                if (rand.nextInt(rn) < rm) {
                    A.data[r + j * col_nnz] = genRandDouble();
                    A.rowIdx[r + j * col_nnz] = i;
                    b.data[i] += A.data[r + j * col_nnz] * opt.x.data[j];
                    c.data[j] -= A.data[r + j * col_nnz] * opt.y.data[i];
                    r++;
                }
            }
            A.colPtr[j + 1] = (j + 1) * col_nnz;
        }
        return new ScsData(A, b, c);
    }

    @Test
    public void TestRandomSOCP() {
        rand.setSeed(seed);

        final int m = factor * n;
        int col_nnz = (int) Math.ceil(Math.sqrt(n));
        int nnz = n * col_nnz;

        int max_q = (int) Math.ceil(m / Math.log(m));
        int kz = (int) Math.floor(m * p_z);
        int kl = (int) Math.floor(m * p_l);
        int q_rows = m - kz - kl;
        int[] kq = new int[q_rows];
        int kq_idx = 0;
        while (q_rows > max_q) {
            int size = rand.nextInt(max_q) + 1;
            kq[kq_idx] = size;
            kq_idx++;
            q_rows -= size;
        }
        if (q_rows > 0) {
            kq[kq_idx] = q_rows;
            kq_idx++;
        }
        kq = Arrays.copyOf(kq, kq_idx);
        var k = new ScsCone(kz, kl, kq);
        var opt = new ScsSolution(m, n);
        var d = genRandomProbData(nnz, col_nnz, k, opt);
        var stgs = ScsSettings.defaulSettings();

        System.out.format(
                "true pri opt = %4f, true dua opt = %4f\n",
                DVector.dot(d.c, opt.x),
                -DVector.dot(d.b, opt.y));
        var sol = ScsSolver.scs(d, k, stgs);

        System.out.format(
                "true pri opt = %4f, true dua opt = %4f\n",
                DVector.dot(d.c, opt.x),
                -DVector.dot(d.b, opt.y));
        System.out.format(
                "test pri opt = %4f, test dua opt = %4f\n",
                sol.pobj,
                sol.dobj);
    }
}
