package com.nmsolver;

import java.util.Map;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public class ScsSettings {
    /** Initial dual scaling factor (may be updated if adaptive_scale is on). */
    public double scale;
    /** Whether to adaptively update `scale`. */
    public boolean adaptive_scale;
    /** Primal constraint scaling factor. */
    public double rho_x;
    /** Maximum iterations to take. */
    public int max_iters;
    /** Absolute convergence tolerance. */
    public double eps_abs;
    /** Relative convergence tolerance. */
    public double eps_rel;
    /** Infeasible convergence tolerance. */
    public double eps_infeas;
    /** Douglas-Rachford relaxation parameter. */
    public double alpha;
    /** Whether to log progress to stdout. */
    public boolean verbose;

    public static ScsSettings defaulSettings() {
        return new ScsSettings(SCALE, ADAPTIVE_SCALE, RHO_X, MAX_ITERS, EPS_ABS, EPS_REL, EPS_INFEAS, ALPHA, VERBOSE);
    }

    void validate() {
        if (max_iters <= 0) {
            throw new IllegalArgumentException("max_iters must be positive\n");
        }
        if (eps_abs < 0) {
            throw new IllegalArgumentException("eps_abs tolerance must be positive\n");
        }
        if (eps_rel < 0) {
            throw new IllegalArgumentException("eps_rel tolerance must be positive\n");
        }
        if (eps_infeas < 0) {
            throw new IllegalArgumentException("eps_infeas tolerance must be positive\n");
        }
        if (alpha <= 0 || alpha >= 2) {
            throw new IllegalArgumentException("alpha must be in (0,2)\n");
        }
        if (rho_x <= 0) {
            throw new IllegalArgumentException("rho_x must be positive (1e-3 works well).\n");
        }
        if (scale <= 0) {
            throw new IllegalArgumentException("scale must be positive (1 works well).\n");
        }
    }

    /* DEFAULT SOLVER PARAMETERS AND SETTINGS -------------------------- */
    public static int MAX_ITERS = 100000;
    public static double EPS_REL = 1E-4;
    public static double EPS_ABS = 1E-4;
    public static double EPS_INFEAS = 1E-7;
    public static double ALPHA = 1.5;
    public static double RHO_X = 1E-6;
    public static double SCALE = 0.1;
    public static boolean VERBOSE = true;
    public static boolean ADAPTIVE_SCALE = true;

    /* (Dual) Scale updating parameters */
    public static double MAX_SCALE_VALUE = (1e6);
    public static double MIN_SCALE_VALUE = (1e-6);

    /* CG == Conjugate gradient */
    /* Linear system tolerances, only used with indirect */
    public static double CG_BEST_TOL = 1e-12;
    /*
     * This scales the current residuals to get the tolerance we solve the
     * linear system to at each iteration. Lower factors require more CG steps
     * but give better accuracy
     */
    public static double CG_TOL_FACTOR = 0.2;
    /* cg tol ~ O(1/k^(CG_RATE)) */
    public static double CG_RATE = 1.5;

    /* Factor which is scales tau in the linear system update */
    /* Larger factors prevent tau from moving as much */
    public static double TAU_FACTOR = 10.;

    /* Force SCS to treat the problem as (non-homogeneous) feasible for this many */
    /* iters. This acts like a warm-start that biases towards feasibility, which */
    /* is the most common use-case */
    public static int FEASIBLE_ITERS = 1;

    /* maintain the iterates at L2 norm = ITERATE_NORM * sqrt(n+m+1) */
    public static double ITERATE_NORM = 1.;

    /* print summary output every this num iterations */
    public static int PRINT_INTERVAL = 250;
    public static boolean DEBUG = false;
    static {
        if (DEBUG) {
            PRINT_INTERVAL = 1;
        }
    }
    /* check for convergence every this num iterations */
    public static int CONVERGED_INTERVAL = 25;
    /* how many iterations between heuristic residual rescaling */
    public static int RESCALING_MIN_ITERS = 100;

    public static double DIV_EPS_TOL = 1e-18;

    static double SAFEDIV_POS(double x, double y) {
        return x / Math.max(y, ScsSettings.DIV_EPS_TOL);
    }

    /* SCS returns one of the following integer exit flags: */
    public final static int SCS_INFEASIBLE_INACCURATE = -7;
    public final static int SCS_UNBOUNDED_INACCURATE = -6;
    public final static int SCS_SIGINT = -5;
    public final static int SCS_FAILED = -4;
    public final static int SCS_INDETERMINATE = -3;
    public final static int SCS_INFEASIBLE = -2; /* primal infeasible, dual unbounded */
    public final static int SCS_UNBOUNDED = -1; /* primal unbounded, dual infeasible */
    public final static int SCS_UNFINISHED = 0; /* never returned, used as placeholder */
    public final static int SCS_SOLVED = 1;
    public final static int SCS_SOLVED_INACCURATE = 2;
    public final static Map<Integer, String> FLAG_HINTS = Map.of(
            SCS_SOLVED, "Solved",
            SCS_SOLVED_INACCURATE, "Inaccurate solved");

    public static String FLAG_TO_HINT(int flag) {
        return FLAG_HINTS.get(flag);
    }

    /* Parallel Setings */
    public static boolean PARALLEL = true;
    public static int THREAD_COUNT = 8;
}
