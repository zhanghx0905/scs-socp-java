package com.nmsolver;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.ejml.data.DMatrixRMaj;
import org.ejml.ops.MatrixIO;
import org.junit.Test;

import com.nmsolver.cones.ScsCone;
import com.nmsolver.linalg.DVector;

/*
 * Reformulating Lasso
    Problem to a SOCP Dual Form
 */
public class LassoTest {
    String PATH = "datal/500";

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
    public void lassoTest() throws IOException {
        var paths = listFiles(PATH);
        if (paths.size() == 0) {
            return;
        }

        double avg_time = 0;
        int avg_iters = 0;
        for (var path : paths) {
            var P = ScsIO.readMtxCSC(path + "/P.mtx");
            var A = ScsIO.readMtxCSC(path + "/A.mtx");
            var b = ScsIO.readMtxVector(path + "/b.mtx");
            var c = ScsIO.readMtxVector(path + "/c.mtx");

            int size = A.nRow / 3;
            var d = new ScsData(A.nRow, A.nCol, A, P, b, c);
            var k = new ScsCone(size, 2 * size, new int[] {});
            var stgs = ScsSettings.defaulSettings();
            // stgs.eps_abs = stgs.eps_rel = 1e-6;

            System.out.println(path);
            var sol = ScsSolver.scs(d, k, stgs);

            avg_time += sol.time_cost;
            avg_iters += sol.iterations;
            System.out.format(
                    "test pri opt = %4f, test dua opt = %4f\n",
                    sol.pobj,
                    sol.dobj);
            System.out.format("time cost %.4e\n", sol.time_cost);
            MatrixIO.saveDenseCSV(new DMatrixRMaj(sol.x.data), path + "/scs-sol.csv");
            MatrixIO.saveDenseCSV(new DMatrixRMaj(new double[] { sol.pobj }), path + "/scs-obj.csv");
        }
        avg_time /= paths.size();
        avg_iters /= paths.size();
        System.out.format("avg time cost %.4e s, iters %d for %s\n", avg_time, avg_iters, PATH);
    }
}
