package com.nmsolver;

import com.nmsolver.cones.ScsCone;
import com.nmsolver.cones.ScsConeWork;
import com.nmsolver.linalg.DCSCMatrix;
import com.nmsolver.linalg.DVector;
import com.nmsolver.linalg.ScsLinSys;

public class ScsWorkspace {
    DVector u, u_t;
    DVector v, v_prev;
    DVector rsk; /* rsk [ r; s; kappa ] */
    DVector h; /* h = [c; b] */
    DVector g; /* g = (I + M)^{-1} h */
    DVector diag_r; /* vector of R matrix diagonals (affects cone proj) */
    DVector lin_sys_warm_start; /* linear system warm-start (indirect only) */
    double norm_b, norm_c; // inf-norm of b and c
    ScsData d; /* Problem data deep copy NORMALIZED */
    ScsCone k; /* Problem cone deep copy */
    ScsConeWork cone_work;
    ScsSettings stgs; /* contains solver settings specified by user */
    ScsSolution xys; /* track x,y,s as alg progresses, tau *not* divided out */
    ScsResiduals r;
    ScsLinSys p; /* struct populated by linear system solver */

    int status_val; /* status */
    double elapsed; /* time elapsed */
    /* Scale updating workspace */
    double sum_log_scale_factor;
    int last_scale_update_iter, n_log_scale_factor;

    ScsWorkspace(
            final ScsData d,
            final ScsCone k,
            final ScsSettings stgs) {
        this.d = d;
        this.k = k;
        this.stgs = stgs;
        this.norm_b = d.b.normInf();
        this.norm_c = d.c.normInf();

        int l = d.m + d.n + 1;
        this.u = new DVector(l);
        this.u_t = new DVector(l);
        this.v = new DVector(l);
        this.v_prev = new DVector(l);
        this.rsk = new DVector(l);
        this.h = new DVector(l - 1);
        this.g = new DVector(l - 1);
        this.diag_r = new DVector(l);
        this.lin_sys_warm_start = new DVector(d.n);
        this.xys = new ScsSolution(d.m, d.n);
        this.r = new ScsResiduals(d.m, d.n);
        this.cone_work = new ScsConeWork(k, d.m);

        this.updateDiagR();
        this.p = new ScsLinSys(d.A, d.P, this.diag_r);
        this.status_val = ScsSettings.SCS_UNFINISHED;
        validate();

    }

    public void updateTime(long starttime) {
        elapsed = (System.currentTimeMillis() - starttime) / 1e3;
    }

    public void updateWork() {
        resetTracking();
        int l = d.m + d.n + 1;
        // cold_start_vars
        v.data[l - 1] = 1.;
        // h = [c; b]
        h.setRange(0, d.c, 0, d.n);
        h.setRange(d.n, d.b, 0, d.m);
        /* update work cache */
        updateWorkCache();
    }

    public void updateScale(int iter) {
        double nm_ax = r.ax.normInf(),
                nm_s = xys.s.normInf(),
                nm_px_aty_ctau = r.px_aty_ctau.normInf(),
                nm_px = r.px.normInf(),
                nm_aty = r.aty.normInf(),
                nm_ax_s_btau = r.ax_s_btau.normInf();

        int iters_since_last_update = iter - last_scale_update_iter;
        double denom_pri, denom_dual;

        /* ||Ax + s - b * tau|| */
        denom_pri = Math.max(nm_ax, nm_s);
        denom_pri = Math.max(denom_pri, norm_b * r.tau);
        double relative_res_pri = ScsSettings.SAFEDIV_POS(nm_ax_s_btau, denom_pri);

        /* ||Px + A'y + c * tau|| */
        denom_dual = Math.max(nm_px, nm_aty);
        denom_dual = Math.max(denom_dual, norm_c * r.tau);
        double relative_res_dual = ScsSettings.SAFEDIV_POS(nm_px_aty_ctau, denom_dual);

        sum_log_scale_factor += Math.log(relative_res_pri) - Math.log(relative_res_dual);
        n_log_scale_factor++;

        double factor = Math.sqrt(Math.exp(sum_log_scale_factor / (double) n_log_scale_factor));
        /* need at least RESCALING_MIN_ITERS since last update */
        if (iters_since_last_update < ScsSettings.RESCALING_MIN_ITERS) {
            return;
        }
        double new_scale = stgs.scale * factor;
        new_scale = Math.max(new_scale, ScsSettings.MIN_SCALE_VALUE);
        new_scale = Math.min(new_scale, ScsSettings.MAX_SCALE_VALUE);
        if (new_scale == stgs.scale) {
            return;
        }
        if (shouldUpdateR(factor)) {
            sum_log_scale_factor = 0;
            n_log_scale_factor = 0;
            last_scale_update_iter = iter;
            stgs.scale = new_scale;

            updateDiagR();
            p.updateDiagR(diag_r);
            updateWorkCache();
            /* update v, using fact that rsk, u, u_t vectors should be the same */
            /*
             * solve: R^+ (v^+ + u - 2u_t) = rsk = R(v + u - 2u_t)
             * => v^+ = R+^-1 rsk + 2u_t - u
             */
            for (int i = 0; i < d.n + d.m + 1; ++i) {
                v.data[i] = rsk.data[i] / diag_r.data[i] + 2 * u_t.data[i] - u.data[i];
            }
        }
    }

    /*
     * scs is homogeneous so scale the iterate to keep norm reasonable
     * maintain the iterates at L2 norm = ITERATE_NORM * sqrt(n+m+1)
     */
    public static void normailizeV(DVector v) {
        var vnorm = v.norm2();
        v.imult(Math.sqrt(v.data.length) * ScsSettings.ITERATE_NORM / vnorm);
    }

    public void projectLinSys(int iter) {
        int n = d.n, m = d.m, l = n + m + 1;
        u_t.setTo(v);
        for (int i = 0; i < l - 1; ++i) {
            u_t.data[i] *= ((i < n) ? 1 : -1) * diag_r.data[i];
        }
        double nm_ax_s_btau = r.ax_s_btau.normInf();
        double nm_px_aty_ctau = r.px_aty_ctau.normInf();
        /* warm_start = u[:n] + tau * g[:n] */
        lin_sys_warm_start.setRange(0, u, 0, n);
        lin_sys_warm_start.iadd(g, u.data[l - 1], 0, 0, n);
        /* use normalized residuals to compute tolerance */
        double tol = Math.min(nm_px_aty_ctau, nm_ax_s_btau);
        /* tol ~ O(1/k^(1+eps)) guarantees convergence */
        /*
         * use warm-start to calculate tolerance rather than u_t, since warm_start
         * should be approximately equal to the true solution
         */
        var nm_ws = lin_sys_warm_start.normInf() / Math.pow(iter + 1, ScsSettings.CG_RATE);
        tol = ScsSettings.CG_TOL_FACTOR * Math.min(tol, nm_ws);
        tol = Math.max(ScsSettings.CG_BEST_TOL, tol);
        p.solveLinSys(u_t, tol, lin_sys_warm_start);
        if (iter < ScsSettings.FEASIBLE_ITERS) {
            u_t.data[l - 1] = 1.;
        } else {
            u_t.data[l - 1] = rootPlus(u_t, v, v.data[l - 1]);
        }
        u_t.iadd(g, -u_t.data[l - 1], 0, 0, l - 1);
    }

    public void projectCones(int iter) {
        int n = d.n, l = d.n + d.m + 1;
        for (int i = 0; i < l; ++i) {
            u.data[i] = 2 * u_t.data[i] - v.data[i];
        }
        /* u = [x;y;tau] */
        cone_work.projDualCone(u, diag_r, n);
        if (iter < ScsSettings.FEASIBLE_ITERS) {
            u.data[l - 1] = 1.;
        } else {
            u.data[l - 1] = Math.max(u.data[l - 1], 0);
        }
    }

    public void updateDualVars() {
        int l = d.n + d.m + 1;
        for (int i = 0; i < l; ++i) {
            v.data[i] += stgs.alpha * (u.data[i] - u_t.data[i]);
        }
    }

    public void computeRsk() {
        int l = d.n + d.m + 1;
        for (int i = 0; i < l; ++i) {
            rsk.data[i] = (v.data[i] + u.data[i] - 2 * u_t.data[i]) * diag_r.data[i];
        }
    }

    public int hasConverged() {
        double eps_abs = stgs.eps_abs,
                eps_rel = stgs.eps_rel;
        if (r.tau > 0) {
            double abs_xt_p_x = Math.abs(r.xt_p_x);
            double abs_ctx = Math.abs(r.ctx);
            double abs_bty = Math.abs(r.bty);

            double nm_s = xys.s.normInf();
            double nm_px = r.px.normInf();
            double nm_aty = r.aty.normInf();
            double nm_ax = r.ax.normInf();

            double grl = Math.max(Math.max(abs_xt_p_x, abs_ctx), abs_bty);
            double prl = Math.max(Math.max(norm_b * r.tau, nm_s), nm_ax) / r.tau;
            double drl = Math.max(Math.max(norm_c * r.tau, nm_px), nm_aty) / r.tau;

            if ((r.res_pri < eps_abs + eps_rel * prl) &&
                    (r.res_dual < eps_abs + eps_rel * drl) &&
                    (r.gap < eps_abs + eps_rel * grl))
                status_val = ScsSettings.SCS_SOLVED;
        }

        /* no need for considering bad conditions for SOCP */
        // if ((r.res_unbdd_a < eps_infeas) &&
        // (r.res_unbdd_p < eps_infeas))
        // return ScsInfo.SCS_UNBOUNDED;

        // if (r.res_infeas < eps_infeas)
        // return ScsInfo.SCS_INFEASIBLE;

        return status_val;
    }

    public void finalize(int iter) {
        populateResidual(iter);
        double nm_s = xys.s.normInf();
        double nm_y = xys.y.normInf();
        double sty = DVector.dot(xys.s, xys.y);

        double comp_slack = Math.abs(sty);

        xys.time_cost = elapsed;
        xys.iterations = iter;
        xys.pobj = r.pobj;
        xys.dobj = r.dobj;
        xys.gap = r.gap;

        if (comp_slack > 1e-5 * Math.max(nm_s, nm_y)) {
            var warning = String.format("WARNING - large complementary slackness residual: %f\n", comp_slack);
            System.out.print(warning);
        }
        switch (status_val) {
            case ScsSettings.SCS_SOLVED:
                setSolved();
                break;
            case ScsSettings.SCS_UNFINISHED:
                setUnfinished();
                break;
            case ScsSettings.SCS_INFEASIBLE:
                setInfeasible();
                break;
            case ScsSettings.SCS_UNBOUNDED:
                setUnbounded();
                break;
            default:
                throw new RuntimeException("ERROR: should not be in this state.");
        }
    }

    public void populateResidual(int iter) {
        if (r.last_iter == iter) {
            return;
        }
        r.last_iter = iter;
        int m = d.m, n = d.n;
        xys.x.setRange(0, u, 0, n);
        xys.y.setRange(0, u, n, m);
        xys.s.setRange(0, rsk, n, m);

        r.tau = Math.abs(u.data[n + m]);
        r.kap = Math.abs(rsk.data[n + m]);
        /**************** PRIMAL *********************/
        r.ax.setRange(0);
        /* ax = Ax */
        DCSCMatrix.accumByAx(d.A, xys.x, r.ax);

        r.ax_s.setTo(r.ax);
        /* ax_s = Ax + s */
        r.ax_s.iadd(xys.s, 1);

        r.ax_s_btau.setTo(r.ax_s);
        /* ax_s_btau = Ax + s - b * tau */
        r.ax_s_btau.iadd(d.b, -r.tau);

        /**************** DUAL *********************/
        r.px.setRange(0);
        /* px = Px */
        if (d.P != null) {
            DCSCMatrix.accumByPx(d.P, xys.x, r.px);
            r.xt_p_x_tau = DVector.dot(r.px, xys.x);
        } else {
            r.xt_p_x_tau = 0;
        }
        r.aty.setRange(0);
        /* aty = A'y */
        DCSCMatrix.accumByATx(d.A, xys.y, r.aty);

        /* r->px_aty_ctau = Px + A'y + c * tau */
        r.px_aty_ctau.setTo(r.px);
        r.px_aty_ctau.iadd(r.aty, 1);
        r.px_aty_ctau.iadd(d.c, r.tau);

        /**************** OTHERS *****************/
        r.bty_tau = DVector.dot(xys.y, d.b);
        r.ctx_tau = DVector.dot(xys.x, d.c);

        r.bty = ScsSettings.SAFEDIV_POS(r.bty_tau, r.tau);
        r.ctx = ScsSettings.SAFEDIV_POS(r.ctx_tau, r.tau);
        r.xt_p_x = ScsSettings.SAFEDIV_POS(r.xt_p_x_tau, r.tau * r.tau);

        r.gap = Math.abs(r.xt_p_x + r.ctx + r.bty);
        r.pobj = r.xt_p_x / 2. + r.ctx;
        r.dobj = -r.xt_p_x / 2. - r.bty;

        r.computeResiduals();
    }

    private void updateDiagR() {
        diag_r.setRange(stgs.rho_x, 0, d.n);

        diag_r.setRange(1. / (1000. * stgs.scale), d.n, k.z);

        diag_r.setRange(1. / this.stgs.scale, d.n + k.z, d.m - k.z);

        diag_r.data[d.n + d.m] = ScsSettings.TAU_FACTOR;
    }

    private void validate() {
        if (d.m <= 0 || d.n <= 0) {
            throw new IllegalArgumentException(
                    String.format("m and n must both be greater than 0; m = %d, n = %d\n",
                            d.m, d.n));
        }
        stgs.validate();
    }

    /* Reset quantities specific to current solve */
    private void resetTracking() {
        last_scale_update_iter = 0;
        sum_log_scale_factor = 0.;
        n_log_scale_factor = 0;
        /* Need this to force residual calc if previous solve solved at iter 0 */
        r.last_iter = -1;
    }

    private void updateWorkCache() {
        /* g = (I + M)^{-1} h */
        this.g.setTo(this.h);
        for (int i = d.n; i < d.n + d.m; ++i) {
            g.data[i] = -g.data[i];
        }
        p.solveLinSys(g, ScsSettings.CG_BEST_TOL, null);
    }

    private boolean shouldUpdateR(double factor) {
        return (factor > Math.sqrt(10.) || factor < 1. / Math.sqrt(10.));
    }

    /* utility function that computes x'Ry */
    private double dotR(DVector x, DVector y) {
        double ip = 0;
        for (int i = 0; i < d.n + d.m; ++i) {
            ip += x.data[i] * y.data[i] * diag_r.data[i];
        }
        return ip;
    }

    private double rootPlus(DVector p, DVector mu, double eta) {
        double tau_scale = diag_r.data[d.m + d.n];
        double a = tau_scale + dotR(g, g);
        double b = dotR(mu, g) - 2 * dotR(p, g) - eta * tau_scale;
        double c = dotR(p, p) - dotR(p, mu);
        double rad = b * b - 4 * a * c;
        return (-b + Math.sqrt(Math.max(0, rad))) / (2 * a);
    }

    private void setSolved() {
        xys.x.imult(ScsSettings.SAFEDIV_POS(1., r.tau));
        xys.y.imult(ScsSettings.SAFEDIV_POS(1., r.tau));
        xys.s.imult(ScsSettings.SAFEDIV_POS(1., r.tau));
    }

    private void setInfeasible() {
        xys.y.imult(-1 / r.bty_tau);
        xys.x.imult(Double.NaN);
        xys.s.imult(Double.NaN);
    }

    private void setUnbounded() {
        xys.x.imult(-1 / r.ctx_tau);
        xys.s.imult(-1 / r.ctx_tau);
        xys.y.imult(Double.NaN);
    }

    private void setUnfinished() {
        if (r.tau > r.kap) {
            setSolved();
            status_val = ScsSettings.SCS_SOLVED_INACCURATE;
        } else if (r.bty_tau < r.ctx_tau) {
            setInfeasible();
            status_val = ScsSettings.SCS_INFEASIBLE_INACCURATE;
        } else {
            setUnbounded();
            status_val = ScsSettings.SCS_UNBOUNDED_INACCURATE;
        }
    }

    static final String[] HEADER = {
            " iter ", " pri res ", " dua res ", "   gap   ",
            "   obj   ", "  scale  ", " time (s)",
    };
    static final int LINE_LEN = 66;
    static final int HSPACE = 10;

    public void printHeader() {
        var sb = new StringBuilder();
        for (int i = 0; i < LINE_LEN; ++i) {
            sb.append("-");
        }
        sb.append("\n");
        sb.append(String.format(
                "problem:  variables n: %d, constraints m: %d\n" +
                        "\t  nnz(A): %d, nnz(P): %d\n",
                d.n, d.m, d.A.colPtr[d.A.nCol], (d.P == null) ? 0 : d.P.colPtr[d.P.nCol]));
        sb.append(k.toString());
        sb.append(String.format(
                "settings: eps_abs: %.1e, eps_rel: %.1e, eps_infeas: %.1e\n" +
                        "\t  alpha: %.2f, scale: %.2e, adaptive_scale: %b\n" +
                        "\t  max_iters: %d, rho_x: %.2e\n",
                stgs.eps_abs, stgs.eps_rel, stgs.eps_infeas, stgs.alpha,
                stgs.scale, stgs.adaptive_scale, stgs.max_iters, stgs.rho_x));

        for (int i = 0; i < LINE_LEN; ++i) {
            sb.append("-");
        }
        sb.append("\n");
        for (int i = 0; i < HEADER.length - 1; ++i) {
            sb.append(HEADER[i]);
            sb.append("|");
        }
        sb.append(HEADER[HEADER.length - 1]);
        sb.append("\n");
        for (int i = 0; i < LINE_LEN; ++i) {
            sb.append("-");
        }
        sb.append("\n");
        System.out.print(sb);
    }

    public void printSummary(int i) {
        var sb = new StringBuilder();
        sb.append(String.format("%" + HEADER[0].length() + "d|", i));
        var format = "%" + HSPACE + ".2e";
        sb.append(String.format(format, r.res_pri));
        sb.append(String.format(format, r.res_dual));
        sb.append(String.format(format, r.gap));
        sb.append(String.format(format, 0.5 * (r.pobj + r.dobj)));
        sb.append(String.format(format, stgs.scale));
        sb.append(String.format(format, elapsed));
        sb.append("\n");
        /* Verbose */
        if (ScsSettings.DEBUG) {
            sb.append(String.format("Norm u = %4f, ", u.norm2()));
            sb.append(String.format("Norm u_t = %4f, ", u_t.norm2()));
            sb.append(String.format("Norm v = %4f, ", v.norm2()));
            sb.append(String.format("Norm rsk = %4f, ", rsk.norm2()));
            sb.append(String.format("Norm x = %4f, ", xys.x.norm2()));
            sb.append(String.format("Norm y = %4f, ", xys.y.norm2()));
            sb.append(String.format("Norm y = %4f, ", xys.s.norm2()));
            sb.append(String.format("Norm |Ax + s| = %1.2e, ", r.ax_s.norm2()));
            sb.append(String.format("tau = %4f, ", u.data[d.n + d.m]));
            sb.append(String.format("kappa = %4f, ", rsk.data[d.n + d.m]));
            sb.append(String.format("ctx_tau = %1.2e, ", r.ctx_tau));
            sb.append(String.format("bty_tau = %1.2e\n", r.bty_tau));
        }
        System.out.print(sb);
    }

    public void printFooter() {
        var sb = new StringBuilder();
        for (int i = 0; i < LINE_LEN; ++i) {
            sb.append("-");
        }
        sb.append("\n");
        sb.append(String.format("status:  %s\n", ScsSettings.FLAG_TO_HINT(status_val)));
        sb.append(String.format("timings: total: %1.2es\n", elapsed));
        for (int i = 0; i < LINE_LEN; ++i) {
            sb.append("-");
        }
        sb.append("\n");
        sb.append(String.format("objective = %.6f\n", 0.5 * (r.pobj + r.dobj)));
        for (int i = 0; i < LINE_LEN; ++i) {
            sb.append("-");
        }
        sb.append("\n");
        System.out.print(sb);
    }
}
