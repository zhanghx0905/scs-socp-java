package com.nmsolver.cones;

import com.nmsolver.linalg.DVector;

public class ScsConeWork {
    final ScsCone k;
    final int m; // total length of cone
    DVector s;

    public ScsConeWork(ScsCone k, int m) {
        this.k = k;
        this.m = m;
        validate();
        this.s = new DVector(m);
    }

    private void validate() {
        var cone_dims = k.z + k.l;
        for (var qsize : k.q) {
            cone_dims += qsize;
        }
        if (cone_dims != m) {
            var error = String.format("cone dimensions %d not equal to num rows in A = m = %d\n",
                    cone_dims, m);
            throw new IllegalArgumentException(error);
        }
        if (k.z < 0) {
            throw new IllegalArgumentException("free cone dimension error\n");
        }
        if (k.l < 0) {
            throw new IllegalArgumentException("lp cone dimension error\n");
        }
        for (var qiter : k.q) {
            if (qiter <= 0) {
                throw new IllegalArgumentException("soc cone dimension must bigger than 0\n");
            }
        }
    }

    private void projSoc(DVector x, int start, int q) {
        if (q == 1) {
            x.data[start] = Math.max(0., x.data[start]);
            return;
        }
        double v1 = x.data[start];
        double s = x.norm2(start + 1, q - 1);
        double alpha = (s + v1) / 2.;
        if (s <= v1) {
            return;
        } else if (s <= -v1) {
            x.setRange(0, start, q);
        } else {
            x.data[start] = alpha;
            x.imultRange(alpha / s, start + 1, q - 1);
        }

    }

    private void projCone(DVector x, int start) {
        int count = start;
        /* project zero cones */
        if (k.z > 0) {
            x.setRange(0, count, k.z);
            count += k.z;
        }
        /* positive orthant cones */
        if (k.l > 0) {
            for (int i = count; i < count + k.l; ++i) {
                x.data[i] = Math.max(0., x.data[i]);
            }
            count += k.l;
        }

        /* project (s, y) onto SOC of size q */
        for (var q : k.q) {
            projSoc(x, count, q);
            count += q;
        }

    }

    /*
     * Outward facing cone projection routine, performs projection in-place.
     * If normalize > 0 then will use normalized (equilibrated) cones if applicable.
     * 
     * Moreau decomposition for R-norm projections:
     * 
     * `x + R^{-1} \Pi_{C^*}^{R^{-1}} ( - R x ) = \Pi_C^R ( x )`
     * 
     * where \Pi^R_C is the projection onto C under the R-norm:
     * 
     * `||x||_R = \sqrt{x ' R x}`.
     * 
     */
    public void projDualCone(DVector x, final DVector r_y, int start) {
        /* copy s = x */
        s.setRange(0, x, start, m);

        /* x -> - Rx */
        for (int i = 0, j = start; i < m; ++i, ++j) {
            if (r_y != null) {
                x.data[j] *= -r_y.data[j];
            } else {
                x.data[j] *= -1;
            }
        }

        /* project -x onto cone, x -> \Pi_{C^*}^{R^{-1}}(-x) under r_y metric */
        projCone(x, start);
        /* return x + R^{-1} \Pi_{C^*}^{R^{-1}} ( -x ) */
        for (int i = 0, j = start; i < m; ++i, ++j) {
            if (r_y != null) {
                x.data[j] = x.data[j] / r_y.data[j] + s.data[i];
            } else {
                x.data[j] += s.data[i];
            }
        }

    }
}
