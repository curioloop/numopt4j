/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.sampled;

import com.curioloop.numopt4j.quad.Integral;

/**
 * Builder for integrating one-dimensional sampled data, returning a scalar total.
 *
 * <p>Set the data via {@link #samples(double[])} (equally spaced, dx=1),
 * {@link #samples(double[], double)} (equally spaced with given dx), or
 * {@link #samples(double[], double[])} (explicit x-coordinates, must be strictly increasing).
 * Then choose the integration rule and call {@link #integrate()}.</p>
 *
 * <p>Available rules (see {@link SampledRule}):</p>
 * <ul>
 *   <li>TRAPEZOIDAL — composite trapezoidal rule, O(h²) accuracy</li>
 *   <li>SIMPSON     — composite Simpson's rule, O(h⁴) accuracy for odd sample counts;
 *                     uses a corrected end-panel for even counts</li>
 *   <li>ROMBERG     — Richardson-extrapolated trapezoidal rule;
 *                     requires sample count 2^k+1 (equally spaced only)</li>
 * </ul>
 */
public class SampledIntegral implements Integral<Double, Void> {

    private double[] x;      // null → equally spaced
    private double[] y;
    private double dx = 1.0;
    private SampledRule rule = SampledRule.TRAPEZOIDAL;

    public SampledIntegral() {}

    /** Sets equally spaced sample values with spacing {@code dx = 1}. */
    public SampledIntegral samples(double[] y) {
        this.y = y;
        this.x = null;
        return this;
    }

    /** Sets equally spaced sample values with the given spacing. */
    public SampledIntegral samples(double[] y, double dx) {
        this.y = y;
        this.x = null;
        this.dx = dx;
        return this;
    }

    /** Sets sample values at explicit coordinates (must be strictly increasing). */
    public SampledIntegral samples(double[] x, double[] y) {
        this.x = x;
        this.y = y;
        return this;
    }

    /** Sets the integration rule. */
    public SampledIntegral rule(SampledRule rule) {
        if (rule == null) throw new IllegalArgumentException("rule must not be null");
        this.rule = rule;
        return this;
    }

    @Override
    public Void alloc() { return null; }

    @Override
    public Double integrate(Void workspace) {
        switch (rule) {
            case TRAPEZOIDAL: return x != null ? trapezoidalXY() : trapezoidalY();
            case SIMPSON:     return x != null ? simpsonXY()     : simpsonY();
            case ROMBERG:     return rombergY();
            default: throw new IllegalStateException("Unknown rule: " + rule);
        }
    }

    // -----------------------------------------------------------------------
    // Trapezoidal
    // -----------------------------------------------------------------------

    private double trapezoidalY() {
        validateSamples(2, "trapezoidal");
        validateDx();
        double sum = 0.0;
        for (int i = 0; i < y.length - 1; i++) sum += y[i] + y[i + 1];
        return 0.5 * dx * sum;
    }

    private double trapezoidalXY() {
        validateCoordinates(2, "trapezoidal");
        double sum = 0.0;
        for (int i = 0; i < x.length - 1; i++)
            sum += 0.5 * (x[i + 1] - x[i]) * (y[i] + y[i + 1]);
        return sum;
    }

    // -----------------------------------------------------------------------
    // Simpson
    // -----------------------------------------------------------------------

    private double simpsonY() {
        validateSamples(3, "simpson");
        validateDx();
        int n = y.length;
        double integral = 0.0;
        for (int i = 1; i < n - 1; i += 2)
            integral += dx / 3.0 * (y[i - 1] + 4.0 * y[i] + y[i + 1]);
        if ((n & 1) == 0)
            integral += dx * (-y[n - 3] + 8.0 * y[n - 2] + 5.0 * y[n - 1]) / 12.0;
        return integral;
    }

    private double simpsonXY() {
        validateCoordinates(3, "simpson");
        int n = x.length;
        double integral = 0.0;
        for (int i = 1; i < n - 1; i += 2) {
            double h0 = x[i] - x[i-1], h1 = x[i+1] - x[i];
            double h0p3 = h0*h0*h0, h1p3 = h1*h1*h1, hph = h0 + h1;
            integral += ((2*h0p3 - h1p3 + 3*h1*h0*h0) / (6*h0*hph)) * y[i-1]
                      + ((h0p3 + h1p3 + 3*h0*h1*hph) / (6*h0*h1)) * y[i]
                      + ((-h0p3 + 2*h1p3 + 3*h0*h1*h1) / (6*h1*hph)) * y[i+1];
        }
        if ((n & 1) == 0) {
            double h0 = x[n-2]-x[n-3], h1 = x[n-1]-x[n-2];
            double h1p2 = h1*h1, h1p3 = h1p2*h1, hph = h0 + h1;
            integral += (-h1p3 / (6*h0*hph)) * y[n-3]
                      + ((h1p2 + 3*h0*h1) / (6*h0)) * y[n-2]
                      + ((2*h1p2 + 3*h0*h1) / (6*hph)) * y[n-1];
        }
        return integral;
    }

    // -----------------------------------------------------------------------
    // Romberg
    // -----------------------------------------------------------------------

    private double rombergY() {
        validateSamples(3, "romberg");
        validateDx();
        int intervals = y.length - 1;
        if ((intervals & (intervals - 1)) != 0)
            throw new IllegalArgumentException("Romberg requires sample count 2^k + 1");

        int levels = 0;
        for (int n = intervals; n > 1; n >>= 1) levels++;

        double[] prev = new double[levels + 1], curr = new double[levels + 1];
        double h = dx * intervals;
        prev[0] = 0.5 * h * (y[0] + y[y.length - 1]);

        int step = intervals;
        for (int level = 1; level <= levels; level++) {
            h *= 0.5; step >>= 1;
            double estimate = 0.0;
            for (int j = step; j < intervals; j += step << 1) estimate += y[j];
            curr[0] = h * estimate + 0.5 * prev[0];
            double factor = 4.0;
            for (int order = 1; order <= level; order++) {
                curr[order] = (factor * curr[order-1] - prev[order-1]) / (factor - 1.0);
                factor *= 4.0;
            }
            double[] swap = prev; prev = curr; curr = swap;
        }
        return prev[levels];
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
