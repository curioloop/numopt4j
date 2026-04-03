/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.sampled;
import java.util.Objects;

import com.curioloop.numopt4j.quad.Integral;

import java.util.function.DoubleUnaryOperator;

/**
 * Filon quadrature for highly oscillatory integrals of the form
 *   ∫_{a}^{b} f(x)·cos(t·x) dx   or   ∫_{a}^{b} f(x)·sin(t·x) dx.
 *
 * <p>Filon's method approximates f(x) by a piecewise quadratic interpolant
 * on 2n+1 equally spaced points and integrates the product with the
 * trigonometric factor exactly.  This makes it far more efficient than
 * standard quadrature when t is large (high-frequency oscillation), because
 * the oscillatory factor is handled analytically rather than by subdivision.</p>
 *
 * <p>The Filon coefficients α, β, γ are (Abramowitz &amp; Stegun §25.4.62–63):
 *   θ = t·h
 *   α = 1/θ + sin(2θ)/(2θ²) − 2·sin²(θ)/θ³
 *   β = 2·[(1 + cos²(θ))/θ² − sin(2θ)/θ³]
 *   γ = 4·[sin(θ)/θ³ − cos(θ)/θ²]
 * where h = (b−a)/(2n) is the half-step size.</p>
 *
 * <p>The integral estimate is:
 *   h · [α·(f(b)·trig1(t·b) − f(a)·trig1(t·a))·sign + β·C_{2n} + γ·C_{2n+1}]
 * where trig1 = sin (cosine kernel) or cos (sine kernel),
 * trig2 = cos (cosine kernel) or sin (sine kernel),
 * C_{2n} = Σ f(x_{2i})·trig2(t·x_{2i}) (even-indexed sum),
 * C_{2n+1} = Σ f(x_{2i−1})·trig2(t·x_{2i−1}) (odd-indexed sum),
 * and sign = +1 (cosine) or −1 (sine).</p>
 *
 * <p>Minimum required setters: {@code .function()}, {@code .bounds()},
 * {@code .frequency()}, {@code .intervals()}.</p>
 *
 * <p>References:</p>
 * <ul>
 *   <li>Abramowitz &amp; Stegun, §25.4.62–25.4.63</li>
 *   <li>QuantLib {@code FilonIntegral} (filonintegral.cpp)</li>
 * </ul>
 */
public class FilonIntegral implements Integral<Double, Void> {

    private DoubleUnaryOperator function;
    private double min = Double.NaN;
    private double max = Double.NaN;
    private double t = Double.NaN;
    private int intervals;   // must be even, >= 2
    private FilonOpts opts;

    FilonIntegral() {}

    /** Creates a builder pre-configured with the given kernel type. */
    public FilonIntegral(FilonOpts opts) {
        Objects.requireNonNull(opts, "opts must not be null");
        this.opts = opts;
    }

    public FilonIntegral function(DoubleUnaryOperator function) {
        this.function = function;
        return this;
    }

    /** Sets the integration interval. Both bounds must be finite with {@code min < max}. */
    public FilonIntegral bounds(double min, double max) {
        this.min = min;
        this.max = max;
        return this;
    }

    /** Sets the angular frequency t. Must be finite and positive. */
    public FilonIntegral frequency(double t) {
        if (!Double.isFinite(t) || t <= 0)
            throw new IllegalArgumentException("frequency t must be finite and positive");
        this.t = t;
        return this;
    }

    /**
     * Sets the number of sub-intervals (must be even and ≥ 2).
     * The integrand is evaluated at {@code intervals + 1} equally spaced points.
     * Higher values improve accuracy for smooth f(x); for highly oscillatory
     * integrands the accuracy is primarily controlled by the Filon coefficients.
     */
    public FilonIntegral intervals(int intervals) {
        if (intervals < 2)
            throw new IllegalArgumentException("intervals must be >= 2");
        if ((intervals & 1) != 0)
            throw new IllegalArgumentException("intervals must be even");
        this.intervals = intervals;
        return this;
    }

    /** Sets the kernel type (COS or SIN). */
    public FilonIntegral opts(FilonOpts opts) {
        Objects.requireNonNull(opts, "opts must not be null");
        this.opts = opts;
        return this;
    }

    @Override
    public Double integrate(Void workspace) {
        validate();
        final int n = intervals / 2;
        final double h = (max - min) / (2 * n);
        final double theta  = t * h;
        final double theta2 = theta * theta;
        final double theta3 = theta2 * theta;

        // Filon coefficients (A&S §25.4.62)
        final double sinT  = Math.sin(theta);
        final double cosT  = Math.cos(theta);
        final double sin2T = Math.sin(2 * theta);
        final double alpha = 1.0 / theta + sin2T / (2 * theta2) - 2 * sinT * sinT / theta3;
        final double beta  = 2.0 * ((1 + cosT * cosT) / theta2 - sin2T / theta3);
        final double gamma = 4.0 * (sinT / theta3 - cosT / theta2);

        // Evaluate f at all 2n+1 equally spaced points
        final double[] x = new double[2 * n + 1];
        final double[] v = new double[2 * n + 1];
        for (int i = 0; i <= 2 * n; i++) {
            x[i] = min + i * h;
            v[i] = function.applyAsDouble(x[i]);
        }

        // Even-indexed sum C_{2n} and odd-indexed sum C_{2n+1}
        // trig2 = cos (cosine kernel) or sin (sine kernel)
        double c2n  = v[0] * trig2(t * x[0])
                    - 0.5 * (v[2 * n] * trig2(t * x[2 * n]) + v[0] * trig2(t * x[0]));
        double c2n1 = 0.0;
        for (int i = 1; i <= n; i++) {
            c2n  += v[2 * i]     * trig2(t * x[2 * i]);
            c2n1 += v[2 * i - 1] * trig2(t * x[2 * i - 1]);
        }

        // trig1 = sin (cosine kernel) or cos (sine kernel)
        // sign  = +1 (cosine) or -1 (sine)
        final double sign = (opts == FilonOpts.COS) ? 1.0 : -1.0;
        return h * (alpha * (v[2 * n] * trig1(t * x[2 * n]) - v[0] * trig1(t * x[0])) * sign
                  + beta  * c2n
                  + gamma * c2n1);
    }

    // trig1: sin for COSINE kernel, cos for SINE kernel
    private double trig1(double u) {
        return opts == FilonOpts.COS ? Math.sin(u) : Math.cos(u);
    }

    // trig2: cos for COSINE kernel, sin for SINE kernel
    private double trig2(double u) {
        return opts == FilonOpts.COS ? Math.cos(u) : Math.sin(u);
    }

    private void validate() {
        if (function == null)
            throw new IllegalStateException("function is required. Call .function() before .integrate().");
        if (!Double.isFinite(min) || !Double.isFinite(max) || !(min < max))
            throw new IllegalArgumentException("bounds must be finite with min < max");
        if (!Double.isFinite(t) || t <= 0)
            throw new IllegalStateException("frequency t is required. Call .frequency() before .integrate().");
        if (intervals < 2)
            throw new IllegalStateException("intervals is required. Call .intervals(n) before .integrate().");
        if (opts == null)
            throw new IllegalStateException("opts is required. Call .opts(FilonOpts) before .integrate().");
    }
}
