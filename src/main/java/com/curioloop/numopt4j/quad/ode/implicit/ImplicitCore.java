/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode.implicit;

import com.curioloop.numopt4j.quad.ode.ODE;
import com.curioloop.numopt4j.quad.ode.IVPCore;

/**
 * Abstract base for implicit ODE solvers (BDF and Radau IIA).
 *
 * <p>Holds the fields and logic shared by both implicit methods:
 * <ul>
 *   <li>RHS function and optional analytic Jacobian.</li>
 *   <li>Tolerances, Newton convergence threshold, and step-size limit.</li>
 *   <li>Jacobian validity flags ({@link #jacCurrent}, {@link #luValid}).</li>
 *   <li>Adaptive forward-difference numerical Jacobian (scipy {@code num_jac}).</li>
 * </ul>
 * </p>
 *
 * @param <W> workspace type, must extend {@link ImplicitPool}
 */
abstract class ImplicitCore<W extends ImplicitPool> extends IVPCore<W> {

    // -----------------------------------------------------------------------
    // Numerical Jacobian constants (scipy num_jac adaptive forward difference)
    // -----------------------------------------------------------------------

    /** Machine epsilon ε ≈ 2.22e-16. */
    private static final double NJ_EPS          = Math.ulp(1.0);
    /** Initial perturbation factor: √ε ≈ 1.49e-8. */
    private static final double NJ_SQRT_EPS     = Math.sqrt(NJ_EPS);
    /** Reject threshold: ε^0.875 — step too small, increase factor. */
    private static final double NJ_DIFF_REJECT  = Math.pow(NJ_EPS, 0.875);
    /** Small-diff threshold: ε^0.75 — increase factor. */
    private static final double NJ_DIFF_SMALL   = Math.pow(NJ_EPS, 0.75);
    /** Large-diff threshold: ε^0.25 — decrease factor. */
    private static final double NJ_DIFF_BIG     = Math.pow(NJ_EPS, 0.25);
    /** Minimum allowed factor: 1000 * ε. */
    private static final double NJ_MIN_FACTOR   = 1e3 * NJ_EPS;
    /** Factor increase multiplier (10×). */
    private static final int    NJ_FACTOR_INCREASE = 10;
    /** Factor decrease multiplier (0.1×). */
    private static final double NJ_FACTOR_DECREASE = 0.1;

    // -----------------------------------------------------------------------
    // Shared implicit-solver fields
    // -----------------------------------------------------------------------

    /** RHS function (always non-null). Derived from {@link #ode} when user passes a full ODE. */
    protected final ODE.Equation fun;

    /** Full ODE interface with analytic Jacobian; {@code null} when user passes {@link ODE.Equation} only. */
    protected final ODE ode;

    /** Relative tolerance. */
    protected final double rtol;

    /** Absolute tolerance (scalar or per-component vector). */
    protected final double[] atol;

    /** Maximum allowed step size. */
    protected final double maxStep;

    /**
     * Newton convergence threshold:
     * <pre>
     *   newtonTol = max(10·ε/rtol, min(0.03, √rtol))
     * </pre>
     * Corresponds to scipy {@code newton_tol}.
     */
    protected final double newtonTol;

    /** Whether the current LU factorization is valid and can be reused. */
    protected boolean luValid;

    /** Whether the current Jacobian is up-to-date (not stale). */
    protected boolean jacCurrent;

    /**
     * Initialises shared implicit-solver state.
     *
     * @param n       system dimension
     * @param t0      initial time
     * @param y0      initial state (copied)
     * @param tBound  end time
     * @param ws      workspace
     * @param fun     RHS function
     * @param ode     full ODE interface (may be {@code null})
     * @param rtol    relative tolerance
     * @param atol    absolute tolerance
     * @param maxStep maximum step size
     */
    protected ImplicitCore(int n, double t0, double[] y0, double tBound, W ws,
                            ODE.Equation rhs, ODE ode,
                            double rtol, double[] atol, double maxStep) {
        super(n, t0, y0, tBound, ws);
        this.ode  = ode;
        this.fun  = rhs != null ? rhs : (t, y, dydt) -> ode.evaluate(t, y, dydt, null);
        this.rtol = rtol;
        this.atol = atol;
        this.maxStep   = maxStep;
        this.newtonTol = Math.max(10.0 * Math.ulp(1.0) / rtol, Math.min(0.03, Math.sqrt(rtol)));
        this.luValid    = false;
        this.jacCurrent = false;
    }

    // -----------------------------------------------------------------------
    // Jacobian computation
    // -----------------------------------------------------------------------

    /**
     * Returns a freshly initialised factor array for {@link #computeNumJac}.
     * All elements are set to {@code √ε} as recommended by scipy {@code num_jac}.
     *
     * @param n system dimension
     * @return factor array of length n, all elements = {@code √ε}
     */
    public static double[] numJacInitFactor(int n) {
        double[] f = new double[n];
        java.util.Arrays.fill(f, NJ_SQRT_EPS);
        return f;
    }

    /**
     * Computes the Jacobian and writes it to {@code ws.jacBuf}.
     *
     * <p>If an analytic Jacobian is available ({@code ode != null}), calls
     * {@link ODE#evaluate} with a non-null {@code jac} argument.
     * Otherwise, uses the adaptive forward-difference algorithm ({@link #computeNumJac}).</p>
     *
     * <p>Does NOT increment {@code nfev} for numerical differentiation (scipy convention:
     * Jacobian evaluations are counted separately in {@code njev}).</p>
     *
     * @param t current time
     * @param y current state, length n
     */
    protected void computeJacobian(double t, double[] y) {
        if (ode != null) {
            ode.evaluate(t, y, ws.fTmp, ws.jacBuf);
        } else {
            fun.evaluate(t, y, ws.fTmp);
            computeNumJac(fun, t, y, ws.fTmp, n, ws.jacBuf, ws.newtonBuf, ws.jacFactor);
        }
        njev++;
        jacCurrent = true;
        luValid    = false;
    }

    // -----------------------------------------------------------------------
    // Adaptive forward-difference numerical Jacobian (scipy num_jac)
    // -----------------------------------------------------------------------

    /**
     * Adaptive forward-difference Jacobian, strictly following scipy {@code num_jac}.
     *
     * <p>For each column {@code j}, perturbs {@code y[j]} by {@code hⱼ = factor[j] * yScale},
     * evaluates {@code f(t, y + hⱼ·eⱼ)}, and computes the finite-difference column:
     * <pre>
     *   J[:, j] ≈ (f(t, y + hⱼ·eⱼ) - f₀) / hⱼ
     * </pre>
     * The perturbation sign follows the sign of {@code f₀[j]} to improve accuracy.
     * The factor is then updated adaptively:
     * <ul>
     *   <li>If {@code maxDiff < DIFF_REJECT * scale}: retry with a larger factor (10×).</li>
     *   <li>If {@code maxDiff < DIFF_SMALL * scale}: increase factor (10×).</li>
     *   <li>If {@code maxDiff > DIFF_BIG * scale}: decrease factor (0.1×).</li>
     * </ul>
     * </p>
     *
     * <p>Perturbs {@code y} in-place and restores it; does NOT increment {@code nfev}.</p>
     *
     * @param fun    RHS function
     * @param t      current time
     * @param y      current state (perturbed in-place and restored)
     * @param f0     f(t, y) already evaluated, length n
     * @param n      system dimension
     * @param jacBuf output Jacobian buffer, n×n row-major
     * @param fTmp   scratch buffer, length n
     * @param factor per-component adaptive factor (updated in-place)
     */
    private static void computeNumJac(ODE.Equation fun, double t, double[] y, double[] f0,
                                      int n, double[] jacBuf, double[] fTmp, double[] factor) {
        for (int j = 0; j < n; j++) {
            // Perturbation direction follows sign of f0[j] (scipy convention)
            double fSign  = (f0[j] >= 0.0) ? 1.0 : -1.0;
            double yScale = fSign * Math.max(1.0, Math.abs(y[j]));
            double yj     = y[j];

            // Compute hj = (yj + factor[j]*yScale) - yj  (floating-point safe)
            double hj = (yj + factor[j] * yScale) - yj;
            while (hj == 0.0) {
                factor[j] *= NJ_FACTOR_INCREASE;
                hj = (yj + factor[j] * yScale) - yj;
            }

            // Evaluate f(t, y + hj*ej)
            y[j] = yj + hj;
            fun.evaluate(t, y, fTmp);
            y[j] = yj;

            // Fill Jacobian column j and find max |Δf|
            double invH    = 1.0 / hj;
            double maxDiff = 0.0;
            int    argMax  = 0;
            for (int i = 0; i < n; i++) {
                double d = (fTmp[i] - f0[i]) * invH;
                jacBuf[i * n + j] = d;
                double absDiff = Math.abs(fTmp[i] - f0[i]);
                if (absDiff > maxDiff) { maxDiff = absDiff; argMax = i; }
            }
            double scale = Math.max(Math.abs(f0[argMax]), Math.abs(fTmp[argMax]));

            // Retry with larger factor if diff is too small (may be in noise floor)
            if (maxDiff < NJ_DIFF_REJECT * scale) {
                double newFactor = factor[j] * NJ_FACTOR_INCREASE;
                double hNew = (yj + newFactor * yScale) - yj;
                if (hNew != 0.0) {
                    y[j] = yj + hNew;
                    fun.evaluate(t, y, fTmp);
                    y[j] = yj;

                    double invHNew    = 1.0 / hNew;
                    double maxDiffNew = 0.0;
                    int    argMaxNew  = 0;
                    for (int i = 0; i < n; i++) {
                        double absDiff = Math.abs(fTmp[i] - f0[i]);
                        if (absDiff > maxDiffNew) { maxDiffNew = absDiff; argMaxNew = i; }
                    }
                    double scaleNew = Math.max(Math.abs(f0[argMaxNew]), Math.abs(fTmp[argMaxNew]));

                    // Accept the retry if it gives a larger diff (better signal)
                    if (maxDiff * scaleNew < maxDiffNew * scale) {
                        factor[j] = newFactor;
                        for (int i = 0; i < n; i++) jacBuf[i * n + j] = (fTmp[i] - f0[i]) * invHNew;
                        maxDiff = maxDiffNew;
                        scale   = scaleNew;
                    }
                }
            }

            // Adapt factor for next call
            if      (maxDiff < NJ_DIFF_SMALL * scale) factor[j] *= NJ_FACTOR_INCREASE;
            else if (maxDiff > NJ_DIFF_BIG   * scale) factor[j] *= NJ_FACTOR_DECREASE;
            factor[j] = Math.max(factor[j], NJ_MIN_FACTOR);
        }
    }
}
