package com.nmsolver.linalg;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;

/* A simple wrapper for double[]. Vector operations are simplified */
public class DVector {
    public double[] data = new double[0];

    private static void checkLengths(DVector u1, DVector u2) {
        // if (u1.data.length != u2.data.length) {
        // throw new IllegalArgumentException("Vectors have different lengths");
        // }
    }

    public DVector(double[] data) {
        this.data = data;
    }

    public DVector(int length) {
        this.data = new double[length];
    }

    public void setTo(DVector other) {
        checkLengths(this, other);
        System.arraycopy(other.data, 0, this.data, 0, data.length);
    }

    public void setRange(double value, int start, int length) {
        Arrays.fill(data, start, start + length, value);
    }

    public void setRange(double value) {
        Arrays.fill(data, value);
    }

    /* this[start0: start0 + length] = b[start1: start1 + length] */
    public void setRange(int start0, DVector b, int start1, int length) {
        System.arraycopy(b.data, start1, data, start0, length);
    }

    public DVector slice(int start, int length) {
        return new DVector(Arrays.copyOfRange(data, start, start + length));
    }

    private void iRangeOp(
            final DoubleBinaryOperator op,
            final double v,
            final int start,
            final int length) {
        for (int i = start; i < start + length; ++i) {
            data[i] = op.applyAsDouble(data[i], v);
        }
    }

    private void iRangeOp(
            final DoubleBinaryOperator op,
            final DVector b,
            final int start0,
            final int start1,
            final int length) {
        for (int i = 0, j = start0, k = start1; i < length; ++i, ++j, ++k) {
            data[j] = op.applyAsDouble(data[j], b.data[k]);
        }
    }


    /* x *= v */
    public void imult(final double v) {
        iRangeOp((x, y) -> x * y, v, 0, data.length);
    }

    /* x[start: start + legnth] *= v*/
    public void imultRange(final double v, final int start, final int length) {
        iRangeOp((x, y) -> x * y, v, start, length);
    }

    /* x /= b */
    public void idivide(final DVector b) {
        checkLengths(this, b);
        iRangeOp((x, y) -> x / y, b, 0, 0, data.length);
    }

    /* x += beta * b */
    public void iadd(final DVector b, final double beta) {
        checkLengths(this, b);
        iRangeOp((x, y) -> x + beta * y, b, 0, 0, data.length);
    }

    /* x[start0: start0+length] += beta * b[start1: start1+length] */
    public void iadd(final DVector b,
            final double beta,
            final int start0,
            final int start1,
            final int length) {
        iRangeOp((x, y) -> x + beta * y, b, start0, start1, length);
    }

    /* c = a * b, per element mult */
    public static void mult(final DVector a, final DVector b, DVector c) {
        checkLengths(a, b);
        checkLengths(a, c);
        for (int i = 0; i < a.data.length; ++i) {
            c.data[i] = a.data[i] * b.data[i];
        }
    }

    /* aT . b */
    public static double dot(final DVector a, final DVector b) {
        checkLengths(a, b);
        double ret = 0;
        for (int i = 0; i < a.data.length; ++i) {
            ret += a.data[i] * b.data[i];
        }
        return ret;
    }

    public double norm2(int start, int length) {
        double ret = 0;
        for (int i = start; i < start + length; ++i) {
            ret += data[i] * data[i];
        }
        return Math.sqrt(ret);
    }

    public double norm2() {
        return norm2(0, data.length);
    }

    public double normInf() {
        double ret = 0;
        for (var v : this.data) {
            ret = Math.max(ret, Math.abs(v));
        }
        return ret;
    }

    @Override
    public String toString() {
        return Arrays.toString(this.data);
    }
}
