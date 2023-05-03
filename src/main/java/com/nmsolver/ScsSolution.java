package com.nmsolver;

import com.nmsolver.linalg.DVector;

public class ScsSolution {
    public DVector x;
    public DVector y;
    public DVector s;
    
    public double time_cost;
    public int iterations;
    public double pobj; /* primal objective */
    public double dobj; /* dual objective */
    public double gap; /* pobj - dobj */

    ScsSolution(int m, int n) {
        this.s = new DVector(m);
        this.y = new DVector(m);
        this.x = new DVector(n);
    }
}
