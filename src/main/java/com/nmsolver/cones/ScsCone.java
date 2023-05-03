package com.nmsolver.cones;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ScsCone {
    /** Number of linear equality constraints (primal zero, dual free). */
    public int z;
    /** Number of positive orthant cones. */
    public int l;
    /** Array of second-order cone constraints, `len(q) = qsize`. */
    public int[] q;

    @Override
    public String toString() {
        var sb = new StringBuilder();
        sb.append("cones: ");
        if (z > 0) {
            sb.append(String.format("\t  z: primal zero / dual free vars: %d\n", z));
        }
        if (l > 0) {
            sb.append(String.format("\t  l: linear vars: %d\n", l));
        }
        if (q.length > 0) {
            int soc = 0;
            for (var qsize : q) {
                soc += qsize;
            }
            sb.append(String.format("\t  q: soc vars: %d, qsize: %d\n", soc, q.length));
        }
        return sb.toString();
    }
}
