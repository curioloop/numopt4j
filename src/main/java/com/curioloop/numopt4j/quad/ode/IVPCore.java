/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode;

import java.util.Objects;

/**
 * Abstract base class for all ODE initial-value-problem solvers.
 *
 * <p>Manages the common solver state (current time, current state, step size, counters)
 * and provides two shared utilities used by every concrete solver:
 * <ul>
 *   <li>{@link #selectInitialStep} — Hairer's empirical initial step-size estimate.</li>
 *   <li>{@link #normScaled} — RMS norm with per-component scaling.</li>
 * </ul>
 * </p>
 *
 * <p>Concrete subclasses implement {@link #step()} and {@link #interpolate(double, double[])}.</p>
 *
 * @param <W> workspace type, must extend {@link IVPPool}
 */
public abstract class IVPCore<W extends IVPPool> {

    /** System dimension. */
    protected final int n;

    /** Current time. */
    protected double t;

    /** Time at the start of the last accepted step ({@code NaN} before the first step). */
    protected double tOld;

    /** Integration end time. */
    protected double tBound;

    /** Integration direction: {@code +1.0} for forward, {@code -1.0} for backward. */
    protected double direction;

    /** Current state vector y(t), length n. */
    protected double[] y;

    /** Current derivative f(t, y), length n. */
    protected double[] f;

    /** Current absolute step size |h|. */
    protected double hAbs;

    /** Number of RHS function evaluations. */
    protected int nfev;

    /** Number of Jacobian evaluations (implicit solvers only). */
    protected int njev;

    /** Number of LU factorizations (implicit solvers only). */
    protected int nlu;

    /** Solver workspace (never null). */
    protected W ws;

    /**
     * Initialises common solver state.
     *
     * @param n       system dimension
     * @param t0      initial time
     * @param y0      initial state (copied)
     * @param tBound  end time
     * @param ws      workspace (must not be null)
     */
    protected IVPCore(int n, double t0, double[] y0, double tBound, W ws) {
        this.n         = n;
        this.t         = t0;
        this.tOld      = Double.NaN;
        this.tBound    = tBound;
        this.direction = tBound > t0 ? 1.0 : -1.0;
        this.y         = y0.clone();
        this.f         = new double[n];
        this.nfev      = 0;
        this.njev      = 0;
        this.nlu       = 0;
        this.ws        = Objects.requireNonNull(ws);
    }

    /**
     * Advances the solution by one adaptive step.
     *
     * @return {@code true} on success; {@code false} when the step size has become too small
     *         (step size below {@code 10 * ulp(t)})
     */
    protected abstract boolean step();

    /**
     * Interpolates the solution at time {@code t} using the dense-output coefficients
     * computed during the last accepted step.
     *
     * <p>Zero allocation — writes directly into {@code out}.</p>
     *
     * @param t   query time, must lie within the last accepted step interval
     * @param out output array of length n, written in-place
     */
    protected abstract void interpolate(double t, double[] out);

    // -----------------------------------------------------------------------
    // Initial step-size selection (scipy select_initial_step)
    // -----------------------------------------------------------------------

    /**
     * Estimates the initial step size using Hairer's empirical formula,
     * strictly following {@code select_initial_step} in scipy {@code common.py}.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>Compute scaled norms:
     *       {@code d₀ = ‖y₀ / scale‖},  {@code d₁ = ‖f₀ / scale‖},
     *       where {@code scale[i] = atol + |y₀[i]| * rtol}.</li>
     *   <li>Rough step: {@code h₀ = 0.01 * d₀/d₁}  (or {@code 1e-6} when either norm is tiny).</li>
     *   <li>Euler step: {@code y₁ = y₀ + h₀ * direction * f₀},  evaluate {@code f₁ = f(t₀+h₀, y₁)}.</li>
     *   <li>Second-derivative estimate: {@code d₂ = ‖(f₁-f₀)/scale‖ / h₀}.</li>
     *   <li>Refined step: {@code h₁ = (0.01 / max(d₁, d₂))^(1/(order+1))}.</li>
     *   <li>Return {@code min(100*h₀, h₁, intervalLength, maxStep)}.</li>
     * </ol>
     * </p>
     *
     * @param fun     RHS function
     * @param t0      initial time
     * @param y0      initial state, length n
     * @param f0      f(t₀, y₀), length n
     * @param order   method order (used for exponent {@code 1/(order+1)})
     * @param rtol    relative tolerance
     * @param atol    absolute tolerance
     * @param maxStep maximum allowed step size
     * @param y1Buf   scratch buffer of length n (reuse workspace, e.g. {@code ws.yTmp})
     * @param f1Buf   scratch buffer of length n (reuse workspace, e.g. {@code ws.fTmp})
     * @return estimated initial step size (positive)
     */
    protected double selectInitialStep(ODE.Equation fun, double t0, double[] y0,
                                       double[] f0, int order,
                                       double rtol, double[] atol, double maxStep,
                                       double[] y1Buf, double[] f1Buf) {
        int n = y0.length;
        if (n == 0) return Double.MAX_VALUE;

        double intervalLength = Math.abs(tBound - t0);
        if (intervalLength == 0.0) return 0.0;

        // scale[i] = atol[i] + |y0[i]| * rtol
        double d0 = 0, d1 = 0;
        for (int i = 0; i < n; i++) {
            double scale = atol(atol, i) + Math.abs(y0[i]) * rtol;
            d0 += (y0[i] / scale) * (y0[i] / scale);
            d1 += (f0[i] / scale) * (f0[i] / scale);
        }
        d0 = Math.sqrt(d0 / n);
        d1 = Math.sqrt(d1 / n);

        double h0;
        if (d0 < 1e-5 || d1 < 1e-5) {
            h0 = 1e-6;
        } else {
            h0 = 0.01 * d0 / d1;
        }
        h0 = Math.min(h0, intervalLength);

        // y1 = y0 + h0 * direction * f0
        for (int i = 0; i < n; i++) {
            y1Buf[i] = y0[i] + h0 * direction * f0[i];
        }
        fun.evaluate(t0 + h0 * direction, y1Buf, f1Buf);
        nfev++;

        // d2 = ‖(f1 - f0) / scale‖ / h0
        double d2 = 0;
        for (int i = 0; i < n; i++) {
            double scale = atol(atol, i) + Math.abs(y0[i]) * rtol;
            double diff = (f1Buf[i] - f0[i]) / scale;
            d2 += diff * diff;
        }
        d2 = Math.sqrt(d2 / n) / h0;

        double h1;
        if (d1 <= 1e-15 && d2 <= 1e-15) {
            h1 = Math.max(1e-6, h0 * 1e-3);
        } else {
            h1 = Math.pow(0.01 / Math.max(d1, d2), 1.0 / (order + 1));
        }

        return Math.min(Math.min(100 * h0, h1), Math.min(intervalLength, maxStep));
    }

    // -----------------------------------------------------------------------
    // Scaled RMS norm
    // -----------------------------------------------------------------------

    /**
     * Computes the scaled RMS norm:
     * <pre>
     *   ‖x / scale‖ = sqrt( (1/n) * Σᵢ (xᵢ / scaleᵢ)² )
     * </pre>
     *
     * <p>Used for error and Newton convergence checks throughout the implicit solvers.</p>
     *
     * @param x     input vector, length n
     * @param scale per-component scale, length n
     * @param n     vector length
     * @return scaled RMS norm
     */
    protected static double normScaled(double[] x, double[] scale, int n) {
        double sum = 0;
        for (int i = 0; i < n; i++) {
            double v = x[i] / scale[i];
            sum += v * v;
        }
        return Math.sqrt(sum / n);
    }

    /** Returns {@code atol[i]} for vector atol, or {@code atol[0]} for scalar atol. */
    protected static double atol(double[] atol, int i) {
        return atol.length == 1 ? atol[0] : atol[i];
    }
}
