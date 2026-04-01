/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.sampled;

import com.curioloop.numopt4j.quad.Integral;

/**
 * Builder for computing the cumulative integral of one-dimensional sampled data,
 * returning a {@code double[]} where element {@code i} is the integral from the
 * first sample up to sample {@code i} (so {@code result[0] == 0} always).
 *
 * <p>Set the data via {@link #samples(double[])} (equally spaced, dx=1),
 * {@link #samples(double[], double)} (equally spaced with given dx), or
 * {@link #samples(double[], double[])} (explicit x-coordinates, must be strictly increasing).
 * Then choose the integration rule and call {@link #integrate()}.</p>
 *
 * <p>Available rules (see {@link SampledRule}):</p>
 * <ul>
 *   <li>TRAPEZOIDAL — cumulative composite trapezoidal rule</li>
 *   <li>SIMPSON     — cumulative composite Simpson's rule with corrected first panel</li>
 * </ul>
 * <p>ROMBERG is not supported for cumulative integration.</p>
 */
public class CumulativeIntegral implements Integral<double[], Void> {

    private double[] x;
    private double[] y;
    private double dx = 1.0;
    private SampledRule rule = SampledRule.TRAPEZOIDAL;

    public CumulativeIntegral() {}

    /** Sets equally spaced sample values with spacing {@code dx = 1}. */
    public CumulativeIntegral samples(double[] y) {
        this.y = y;
        this.x = null;
        return this;
    }

    /** Sets equally spaced sample values with the given spacing. */
    public CumulativeIntegral samples(double[] y, double dx) {
        this.y = y;
        this.x = null;
        this.dx = dx;
        return this;
    }

    /** Sets sample values at explicit coordinates (must be strictly increasing). */
    public CumulativeIntegral samples(double[] x, double[] y) {
        this.x = x;
        this.y = y;
        return this;
    }

    /** Sets the integration rule. */
    public CumulativeIntegral rule(SampledRule rule) {
        if (rule == null) throw new IllegalArgumentException("rule must not be null");
        this.rule = rule;
        return this;
    }

    @Override
    public Void alloc() { return null; }

    @Override
    public double[] integrate(Void workspace) {
        if (rule == SampledRule.ROMBERG)
            throw new IllegalArgumentException("ROMBERG is not supported for cumulative integration");
        switch (rule) {
            case TRAPEZOIDAL: return x != null ? cumulativeTrapezoidalXY() : cumulativeTrapezoidalY();
            case SIMPSON:     return x != null ? cumulativeSimpsonXY()     : cumulativeSimpsonY();
            default: throw new IllegalStateException("Unknown rule: " + rule);
        }
    }

    // -----------------------------------------------------------------------
    // Cumulative trapezoidal
    // -----------------------------------------------------------------------

    private double[] cumulativeTrapezoidalY() {
        validateSamples(2, "cumulativeTrapezoidal");
        validateDx();
        double[] out = new double[y.length];
        for (int i = 1; i < y.length; i++)
            out[i] = out[i-1] + 0.5 * dx * (y[i-1] + y[i]);
        return out;
    }

    private double[] cumulativeTrapezoidalXY() {
        validateCoordinates(2, "cumulativeTrapezoidal");
        double[] out = new double[x.length];
        for (int i = 1; i < x.length; i++)
            out[i] = out[i-1] + 0.5 * (x[i] - x[i-1]) * (y[i-1] + y[i]);
        return out;
    }

    // -----------------------------------------------------------------------
    // Cumulative Simpson
    // -----------------------------------------------------------------------

    private double[] cumulativeSimpsonY() {
        validateSamples(2, "cumulativeSimpson");
        validateDx();
        int n = y.length;
        double[] out = new double[n];
        if (n == 2) { out[1] = 0.5 * dx * (y[0] + y[1]); return out; }
        out[1] = dx * (5.0*y[0] + 8.0*y[1] - y[2]) / 12.0;
        for (int i = 2; i < n; i++)
            out[i] = out[i-1] + dx * (-y[i-2] + 8.0*y[i-1] + 5.0*y[i]) / 12.0;
        return out;
    }

    private double[] cumulativeSimpsonXY() {
        validateCoordinates(2, "cumulativeSimpson");
        int n = x.length;
        double[] out = new double[n];
        if (n == 2) { out[1] = 0.5 * (x[1]-x[0]) * (y[0]+y[1]); return out; }
        out[1] = simpsonLower(x[0], x[1], x[2], y[0], y[1], y[2]);
        for (int i = 2; i < n; i++)
            out[i] = out[i-1] + simpsonUpper(x[i-2], x[i-1], x[i], y[i-2], y[i-1], y[i]);
        return out;
    }

    private static double simpsonLower(double x0, double x1, double x2,
                                       double y0, double y1, double y2) {
        double h0 = x1-x0, h1 = x2-x1, hph = h0+h1;
        return ((2*h0*h0 + 3*h0*h1) / (6*hph)) * y0
             + (h0*(h0 + 3*h1) / (6*h1)) * y1
             + (-(h0*h0*h0) / (6*h1*hph)) * y2;
    }

    private static double simpsonUpper(double x0, double x1, double x2,
                                       double y0, double y1, double y2) {
        double h0 = x1-x0, h1 = x2-x1, hph = h0+h1;
        return (-(h1*h1*h1) / (6*h0*hph)) * y0
             + ((h1*h1 + 3*h0*h1) / (6*h0)) * y1
             + ((2*h1*h1 + 3*h0*h1) / (6*hph)) * y2;
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    private void validateSamples(int min, String method) {
        if (y == null) throw new IllegalStateException("samples not set; call .samples() before .integrate()");
        if (y.length < min) throw new IllegalArgumentException(method + " requires at least " + min + " samples");
    }

    private void validateDx() {
        if (!(dx > 0.0)) throw new IllegalArgumentException("dx must be > 0");
    }

    private void validateCoordinates(int min, String method) {
        if (x == null || y == null) throw new IllegalStateException("samples not set; call .samples(x, y) before .integrate()");
        if (x.length != y.length) throw new IllegalArgumentException("x and y must have the same length");
        if (x.length < min) throw new IllegalArgumentException(method + " requires at least " + min + " samples");
        for (int i = 1; i < x.length; i++)
            if (!(x[i] > x[i-1])) throw new IllegalArgumentException("x must be strictly increasing");
    }
}
