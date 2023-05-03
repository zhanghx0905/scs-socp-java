package com.nmsolver;

import com.nmsolver.cones.ScsCone;

public class ScsSolver {

    public static ScsSolution scs(
            final ScsData d,
            final ScsCone k,
            final ScsSettings stgs) {
        /* scs_init */
        var w = new ScsWorkspace(d, k, stgs);
        if (stgs.verbose) {
            w.printHeader();
        }
        /* scs_solve */
        var starttime = System.currentTimeMillis();
        w.updateWork();

        int i = 0;
        for (; i < w.stgs.max_iters; ++i) {
            if (i >= ScsSettings.FEASIBLE_ITERS) {
                /* normalize v *after* applying any acceleration */
                /* the input to the DR step should be normalized */
                ScsWorkspace.normailizeV(w.v);
            }
            /* store v_prev = v, *after* normalizing */
            w.v_prev.setTo(w.v);
            w.projectLinSys(i);
            w.projectCones(i);
            w.computeRsk();
            if (i % ScsSettings.CONVERGED_INTERVAL == 0) {
                w.populateResidual(i);
                if (w.hasConverged() != 0) {
                    break;
                }
            }
            if (stgs.verbose && i % ScsSettings.PRINT_INTERVAL == 0) {
                w.populateResidual(i);
                w.updateTime(starttime);
                w.printSummary(i);
            }
            if (stgs.adaptive_scale && i == w.r.last_iter) {
                w.updateScale(i);
            }

            w.updateDualVars();

        }
        if (stgs.verbose) {
            w.populateResidual(i);
            w.updateTime(starttime);
            w.printSummary(i);
        }
        w.updateTime(starttime);
        w.finalize(i);
        if (stgs.verbose) {
            w.printFooter();
        }
        return w.xys;
    }
}
