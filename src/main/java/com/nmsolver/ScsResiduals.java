package com.nmsolver;

import com.nmsolver.linalg.DVector;

public class ScsResiduals {
    int last_iter;
    double xt_p_x; /* x' P x */
    double xt_p_x_tau; /* x'Px * tau^2 *not* divided out */
    double ctx;
    double ctx_tau; /* tau *not* divided out */
    double bty;
    double bty_tau; /* tau *not* divided out */
    double pobj; /* primal objective */
    double dobj; /* dual objective */
    double gap; /* pobj - dobj */
    double tau;
    double kap;
    double res_pri;
    double res_dual;
    // double res_infeas;
    // double res_unbdd_p;
    // double res_unbdd_a;
    DVector ax, ax_s, px, aty, ax_s_btau, px_aty_ctau;

    public ScsResiduals(int m, int n) {
        ax = new DVector(m);
        ax_s = new DVector(m);
        ax_s_btau = new DVector(m);
        px = new DVector(n);
        aty = new DVector(n);
        px_aty_ctau = new DVector(n);
    }

    void computeResiduals() {
        double nm_ax_s_btau = ax_s_btau.normInf();
        double nm_px_aty_ctau = px_aty_ctau.normInf();

        res_pri = ScsSettings.SAFEDIV_POS(nm_ax_s_btau, tau);
        res_dual = ScsSettings.SAFEDIV_POS(nm_px_aty_ctau, tau);

        /* Since we only consider SOC, bad cases can be ignored */
        // res_unbdd_a = Double.NaN;
        // res_unbdd_p = Double.NaN;
        // res_infeas = Double.NaN;
        // if (ctx_tau < 0) {
        // double nm_ax_s = ax_s.normInf();
        // double nm_px = px.normInf();
        // res_unbdd_a = ScsSettings.SAFEDIV_POS(nm_ax_s, -ctx_tau);
        // res_unbdd_p = ScsSettings.SAFEDIV_POS(nm_px, -ctx_tau);
        // }
        // if (bty_tau < 0) {
        // double nm_aty = aty.normInf();
        // res_infeas = ScsSettings.SAFEDIV_POS(nm_aty, -bty_tau);
        // }
    }
}
