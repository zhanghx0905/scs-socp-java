package com.nmsolver;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.ejml.data.DMatrixRMaj;
import org.ejml.ops.MatrixIO;
import org.junit.Test;

import com.nmsolver.cones.ScsCone;

/*
 * Reformulating a Mean-Variance Portfolio
    Problem to a SOCP Dual Form
 */
public class PortfolioTest {
    String PATH = "datap/1000";
    boolean USE_MAT = true;

    public static ArrayList<String> listFiles(String path) {
        var ret = new ArrayList<String>();
        var dir = new File(path);
        if (dir.exists()) {
            for (var subf : dir.listFiles()) {
                ret.add(subf.getAbsolutePath());
            }
        }
        return ret;
    }

    @Test
    public void protfoiloTest() throws IOException {
        var paths = listFiles(PATH);
        if (paths.size() == 0) {
            return;
        }

        double avg_time = 0;
        int avg_iters = 0;
        for (var path : paths) {
            ScsData d = null;
            if (USE_MAT) {
                d = ScsIO.readMat(path + "/Abc.mat");
            } else {
                var A = ScsIO.readMtxCSC(path + "/A.mtx");
                var b = ScsIO.readMtxVector(path + "/b.mtx");
                var c = ScsIO.readMtxVector(path + "/c.mtx");
                d = new ScsData(A.nRow, A.nCol, A, null, b, c);
            }
            int n1 = d.A.nCol; // n1 = n+1
            // cone shapes: n+2, 2, 2, ..., 2
            // n + 2 + (n+1)*2=3n+4

            var q = new int[n1 + 1];
            q[0] = n1 + 1; // n + 2
            for (int i = 1; i < n1 + 1; ++i) { // 2 * (n + 1)
                q[i] = 2;
            }
            var k = new ScsCone(0, 0, q);

            var stgs = ScsSettings.defaulSettings();
            stgs.eps_abs = stgs.eps_rel = 1e-6;
            System.out.println(path);
            var sol = ScsSolver.scs(d, k, stgs);

            avg_time += sol.time_cost;
            avg_iters += sol.iterations;
            var pobj = sol.pobj;
            var dobj = sol.dobj;
            System.out.format(
                    "test pri opt = %4f, test dua opt = %4f\n",
                    pobj,
                    dobj);
            System.out.format("time cost %.4e\n", sol.time_cost);
            MatrixIO.saveDenseCSV(new DMatrixRMaj(sol.x.data), path + "/scs-sol.csv");
            MatrixIO.saveDenseCSV(new DMatrixRMaj(new double[] { pobj }), path + "/scs-obj.csv");
        }
        avg_time /= paths.size();
        avg_iters /= paths.size();
        System.out.format("avg time cost %.4e s, iters %d for %s\n", avg_time, avg_iters, PATH);
    }
}
