package com.nmsolver;

import org.junit.Test;

import com.nmsolver.cones.ScsCone;
import com.nmsolver.linalg.DCSCMatrix;
import com.nmsolver.linalg.DVector;

import static org.junit.Assert.*;

public class ExampleTest {
    /*
     * A simple example taken from
     * https://www.cvxgrp.org/scs/examples/c.html#c-example
     */
    @Test
    public void TestSimpleProblem() {
        /* Set up the problem data */
        /* A and P must be in compressed sparse column format */
        double[] P_x = { 3., -1., 2. }; /* Upper triangular of P only */
        int[] P_i = { 0, 0, 1 };
        int[] P_p = { 0, 1, 3 };
        double[] A_x = { -1., 1., 1., 1. };
        int[] A_i = { 0, 1, 0, 2 };
        int[] A_p = { 0, 2, 4 };
        double[] b = { -1., 0.3, -0.5 };
        double[] c = { -1., -1. };
        /* data shapes */
        int n = 2; /* number of variables */
        int m = 3; /* number of constraints */

        int[] q = new int[0];
        var k = new ScsCone(1, 2, q);
        var P = new DCSCMatrix(n, n, P_p, P_i, P_x);
        var A = new DCSCMatrix(m, n, A_p, A_i, A_x);
        var d = new ScsData(m, n, A, P, new DVector(b), new DVector(c));
        var stgs = ScsSettings.defaulSettings();
        stgs.eps_abs = 1e-9;
        stgs.eps_rel = 1e-9;

        var sol = ScsSolver.scs(d, k, stgs);
        var trueX = new double[] { 0.3, -0.7 };
        var trueY = new double[] { 2.7, 2.1, 0 };

        assertArrayEquals(trueX, sol.x.data, 1E-6);
        assertArrayEquals(trueY, sol.y.data, 1E-6);
    }

}