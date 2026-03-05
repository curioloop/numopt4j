/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * SLSQP (Sequential Least Squares Programming) line search implementation.
 *
 * This file implements two line search algorithms for the SLSQP optimizer:
 *
 * ── Inexact Line Search (Armijo Backtracking) ──────────────────────────────
 *
 * Given the augmented Lagrangian merit function 𝞥(𝐱;𝛒) and descent direction 𝐝,
 * the inexact search finds step length 𝛂 satisfying the Armijo condition:
 *
 *   𝞥(𝐱ᵏ + 𝛂𝐝; 𝛌, 𝛒) - 𝞥(𝐱ᵏ; 𝛌, 𝛒) < η · 𝛂 · 𝜵𝞥(𝐝; 𝐱ᵏ, 𝛒ᵏ)   (0 < η < 0.5)
 *
 * where the directional derivative is:
 *
 *   𝜵𝞥(𝐝; 𝐱ᵏ, 𝛒ᵏ) = 𝜵𝒇(𝐱ᵏ)ᵀ𝐝 - (1 - 𝛅) ∑ 𝛒ᵏⱼ ‖𝒄ⱼ(𝐱ᵏ)‖₁
 *
 * The step is reduced by quadratic interpolation: 𝛂_new = h3 / (2(h3 - h1))
 * and clamped to [𝛂_min, 𝛂_max] until the condition is met or 10 iterations pass.
 *
 * ── Exact Line Search (Brent's Method) ────────────────────────────────────
 *
 * The exact search minimizes the merit function 𝞿(𝛂) = 𝞥(𝐱ᵏ + 𝛂𝐝) over [𝛂_lo, 𝛂_hi]
 * using Brent's method: a combination of golden section search and successive
 * parabolic interpolation.
 *
 * The algorithm maintains a bracketing interval [a, b] and three key points
 * x (best), w (second best), v (previous w). At each step it chooses between:
 *
 *   Parabolic step:  d = p/q  where
 *     r = (x - w)(f(x) - f(v))
 *     q = (x - v)(f(x) - f(w))
 *     p = (x - v)q - (x - w)r,  q = 2(q - r)
 *
 *   Golden section:  d = c · e  where c = 1/φ² ≈ 0.38197,  e = b - x  or  a - x
 *
 * Convergence test: |x - m| ≤ tol₂ - ½(b - a)  where m = ½(a + b)
 *
 * The algorithm uses reverse communication: the caller evaluates 𝞿(𝛂) at the
 * returned point and passes the value back on the next call.
 *
 * Reference: Dieter Kraft, "A Software Package for Sequential Quadratic Programming",
 * DFVLR-FB 88-28, 1988.
 */
package com.curioloop.numopt4j.optim.slsqp;

import com.curioloop.numopt4j.linalg.blas.BLAS;

import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.FIND_CONV;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.FIND_INIT;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.FIND_NEXT;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.INF_BND;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.INV_PHI2;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.SQRT_EPS;

/**
 * Line search algorithms for the SLSQP optimization algorithm.
 * <p>
 * Two line search strategies are provided:
 * </p>
 * <ul>
 *   <li><b>Armijo backtracking</b> ({@link #armijo}): inexact search that reduces 𝛂 until
 *       the Armijo sufficient-decrease condition is satisfied:
 *       𝞥(𝐱ᵏ+𝛂𝐝;𝛌,𝛒) - 𝞥(𝐱ᵏ;𝛌,𝛒) &lt; η·𝛂·𝜵𝞥(𝐝;𝐱ᵏ,𝛒ᵏ)  where η = {@link SLSQPConstants#ARMIJO_ETA}.</li>
 *   <li><b>Brent's method</b> ({@link #findMin}): exact search that minimizes 𝞿(𝛂) = 𝞥(𝐱ᵏ+𝛂𝐝)
 *       over [𝛂_lo, 𝛂_hi] by combining golden section (ratio c = 1/φ² = {@link SLSQPConstants#INV_PHI2})
 *       with successive parabolic interpolation.</li>
 * </ul>
 *
 * @see SLSQPConstants#ARMIJO_ETA
 * @see SLSQPConstants#INV_PHI2
 * @see SLSQPWorkspace.FindWork
 */
public final class LineSearch {

    private LineSearch() {
        // Prevent instantiation
    }

    // ========================================================================
    // Armijo Backtracking Line Search
    // ========================================================================

    /**
     * Performs Armijo backtracking line search for the SLSQP merit function.
     * <p>
     * Finds step length 𝛂 satisfying the Armijo sufficient-decrease condition:
     * </p>
     * <pre>
     *   𝞥(𝐱ᵏ + 𝛂𝐝; 𝛌, 𝛒) - 𝞥(𝐱ᵏ; 𝛌, 𝛒) &lt; η · 𝛂 · 𝜵𝞥(𝐝; 𝐱ᵏ, 𝛒ᵏ)
     * </pre>
     * <p>
     * where η = {@link SLSQPConstants#ARMIJO_ETA} = 0.1 and the directional derivative is:
     * </p>
     * <pre>
     *   𝜵𝞥(𝐝; 𝐱ᵏ, 𝛒ᵏ) = 𝜵𝒇(𝐱ᵏ)ᵀ𝐝 - (1 - 𝛅) ∑ 𝛒ᵏⱼ ‖𝒄ⱼ(𝐱ᵏ)‖₁
     * </pre>
     * <p>
     * The caller passes h3 = η·𝛂·𝜵𝞥 (scaled directional derivative) and
     * t = 𝞥(𝐱ᵏ+𝛂𝐝) (current merit value). The condition simplifies to:
     * </p>
     * <pre>
     *   h1 = t - t0 ≤ h3 / 10
     * </pre>
     * <p>
     * When the condition is not met, 𝛂 is reduced by quadratic interpolation:
     * </p>
     * <pre>
     *   𝛂_new = h3 / (2 · (h3 - h1)),  clamped to [𝛂_min, 𝛂_max]
     * </pre>
     * <p>
     * The search direction is then updated: s = 𝛂_new · s, x = x0 + s,
     * and h3 is scaled: h3 = 𝛂_new · h3.
     * </p>
     *
     * @param n          problem dimension n
     * @param x          current position 𝐱 (output: updated to x0 + s after step)
     * @param xOff       offset into x array
     * @param x0         initial position 𝐱ᵏ at start of line search
     * @param x0Off      offset into x0 array
     * @param s          search direction 𝐝 (output: scaled by 𝛂_new on each call)
     * @param sOff       offset into s array
     * @param t          current merit value 𝞥(𝐱ᵏ+𝛂𝐝;𝛌,𝛒)
     * @param t0         initial merit value 𝞥(𝐱ᵏ;𝛌,𝛒) at start of line search
     * @param h3         on entry: η·𝛂·𝜵𝞥(𝐝;𝐱ᵏ,𝛒ᵏ) (scaled directional derivative);
     *                   on exit: updated to 𝛂_new · h3
     * @param alpha      current step length 𝛂
     * @param alphaMin   minimum step length (typically 0.1)
     * @param alphaMax   maximum step length (typically 1.0)
     * @param lineCount  current line search iteration count (stop if &gt; 10)
     * @param lower      lower bounds 𝒍 for projection (may be null if unbounded)
     * @param lowerOff   offset into lower array
     * @param upper      upper bounds 𝒖 for projection (may be null if unbounded)
     * @param upperOff   offset into upper array
     * @param result     output array of length 3:
     *                   [0] = new 𝛂,  [1] = new h3,  [2] = 1.0 to continue / 0.0 if done
     */
    public static void armijo(
            int n,
            double[] x, int xOff,
            double[] x0, int x0Off,
            double[] s, int sOff,
            double t, double t0,
            double h3,
            double alpha,
            double alphaMin, double alphaMax,
            int lineCount,
            double[] lower, int lowerOff,
            double[] upper, int upperOff,
            double[] result) {

        // Compute merit function decrease
        double h1 = t - t0;

        // Check Armijo condition: h1 <= h3/10 (since h3 < 0, this means sufficient decrease)
        // Also stop if max iterations reached
        if (h1 <= h3 / 10.0 || lineCount > 10) {
            // Armijo condition satisfied or max iterations reached
            result[0] = alpha;
            result[1] = h3;
            result[2] = 0.0;  // Done, exit line search
            return;
        }

        // Armijo condition not satisfied, reduce step
        // Compute new alpha using quadratic interpolation
        // α_new = h3 / (2 * (h3 - h1))
        double newAlpha = h3 / (2.0 * (h3 - h1));

        // Clamp to [alphaMin, alphaMax]
        newAlpha = Math.max(Math.min(newAlpha, alphaMax), alphaMin);

        // Scale search direction: s = newAlpha * s (accumulates)
        BLAS.dscal(n, newAlpha, s, sOff, 1);

        // Update position: x = x0 + s
        BLAS.dcopy(n, x0, x0Off, 1, x, xOff, 1);
        BLAS.daxpy(n, 1.0, s, sOff, 1, x, xOff, 1);

        // Project onto bounds
        if (lower != null || upper != null) {
            for (int i = 0; i < n; i++) {
                double lb = (lower != null && !Double.isNaN(lower[lowerOff + i])) 
                        ? lower[lowerOff + i] : -INF_BND;
                double ub = (upper != null && !Double.isNaN(upper[upperOff + i])) 
                        ? upper[upperOff + i] : INF_BND;
                
                if (lb > -INF_BND && x[xOff + i] < lb) {
                    x[xOff + i] = lb;
                } else if (ub < INF_BND && x[xOff + i] > ub) {
                    x[xOff + i] = ub;
                }
            }
        }

        // Update directional derivative
        double newH3 = h3 * newAlpha;

        result[0] = newAlpha;
        result[1] = newH3;
        result[2] = 1.0;  // Continue line search
    }

    // ========================================================================
    // Golden Section Search with Quadratic Interpolation (Brent's Method)
    // ========================================================================

    /**
     * Finds the minimum of a unimodal function over [𝛂_lo, 𝛂_hi] using Brent's method.
     * <p>
     * Brent's method combines golden section search (guaranteed convergence) with
     * successive parabolic interpolation (superlinear convergence when applicable).
     * The function uses reverse communication: the caller evaluates f at the returned
     * point and passes the value back on the next call.
     * </p>
     *
     * <h3>State Machine</h3>
     * <ul>
     *   <li>{@link SLSQPConstants#FIND_NOOP} → initialise interval, return first evaluation point x = a + c·(b-a)</li>
     *   <li>{@link SLSQPConstants#FIND_INIT} → store f(x) as fx = fv = fw, enter main loop</li>
     *   <li>{@link SLSQPConstants#FIND_NEXT} → update bracket [a,b] and best points x/w/v, choose next step</li>
     *   <li>{@link SLSQPConstants#FIND_CONV} → convergence reached, return best x</li>
     * </ul>
     *
     * <h3>Convergence Test</h3>
     * <pre>
     *   |x - m| ≤ tol₂ - ½(b - a)
     * </pre>
     * <p>
     * where m = ½(a + b), tol₁ = √ε·|x| + tol, tol₂ = 2·tol₁.
     * </p>
     *
     * <h3>Parabolic Interpolation</h3>
     * <p>
     * When |e| &gt; tol₁ (previous step was large enough), fit a parabola through
     * x (best), w (second best), v (previous w):
     * </p>
     * <pre>
     *   r = (x - w)(f(x) - f(v))
     *   q = (x - v)(f(x) - f(w))
     *   p = (x - v)·q - (x - w)·r,   q = 2(q - r)
     *   d = p / q   (parabolic step)
     * </pre>
     * <p>
     * The parabolic step is accepted only when it falls strictly inside [a, b]
     * and is smaller than half the step before last (|p| &lt; ½|q·r|).
     * Otherwise the golden section step is used:
     * </p>
     * <pre>
     *   d = c · e   where c = 1/φ² ≈ 0.38197,  e = b - x  or  a - x
     * </pre>
     * <p>
     * In both cases the step is forced to be at least tol₁ in magnitude.
     * </p>
     *
     * @param mode       int[1] mode array; on entry holds the current mode
     *                   ({@link SLSQPConstants#FIND_NOOP}, {@link SLSQPConstants#FIND_INIT},
     *                   {@link SLSQPConstants#FIND_NEXT}); on exit holds the next mode
     * @param work       {@link SLSQPWorkspace.FindWork} holding all search state (a, b, x, w, v, d, e, …)
     * @param f          function value at the evaluation point returned by the previous call
     *                   (ignored when mode = {@link SLSQPConstants#FIND_NOOP})
     * @param tol        desired length of the final interval of uncertainty
     * @param alphaLower lower bound 𝛂_lo of the search interval
     * @param alphaUpper upper bound 𝛂_hi of the search interval
     * @return next evaluation point (or best point when {@link SLSQPConstants#FIND_CONV})
     *
     * @see SLSQPConstants#INV_PHI2
     * @see SLSQPConstants#SQRT_EPS
     * @see SLSQPWorkspace.FindWork
     */
    public static double findMin(
            int[] mode,
            SLSQPWorkspace.FindWork work,
            double f,
            double tol,
            double alphaLower,
            double alphaUpper) {

        double c = INV_PHI2;  // Golden section ratio
        double ax = alphaLower;
        double bx = alphaUpper;

        switch (mode[0]) {
            case FIND_INIT:
                // Main loop starts - process initial function value
                work.fx = f;
                work.fv = work.fx;
                work.fw = work.fv;
                break;

            case FIND_NEXT:
                // Process function value at u, update interval
                work.fu = f;

                // Update a, b, v, w, and x based on new function value
                if (work.fu > work.fx) {
                    // New point is worse than current best
                    if (work.u < work.x) {
                        work.a = work.u;
                    }
                    if (work.u >= work.x) {
                        work.b = work.u;
                    }

                    // Update v and w if appropriate
                    if (work.fu <= work.fw || Math.abs(work.w - work.x) <= 0.0) {
                        work.v = work.w;
                        work.fv = work.fw;
                        work.w = work.u;
                        work.fw = work.fu;
                    } else if (work.fu <= work.fv || Math.abs(work.v - work.x) <= 0.0 ||
                               Math.abs(work.v - work.w) <= 0.0) {
                        work.v = work.u;
                        work.fv = work.fu;
                    }
                } else {
                    // New point is better than or equal to current best
                    if (work.u >= work.x) {
                        work.a = work.x;
                    }
                    if (work.u < work.x) {
                        work.b = work.x;
                    }

                    // Shift v <- w <- x <- u
                    work.v = work.w;
                    work.fv = work.fw;
                    work.w = work.x;
                    work.fw = work.fx;
                    work.x = work.u;
                    work.fx = work.fu;
                }
                break;

            default:
                // FIND_NOOP: Initialization
                work.a = ax;
                work.b = bx;
                work.e = 0.0;

                // Initial point using golden section from left: v = a + c*(b-a)
                work.v = work.a + c * (work.b - work.a);
                work.w = work.v;
                work.x = work.v;

                mode[0] = FIND_INIT;
                return work.x;
        }

        // Compute midpoint and tolerances
        work.m = 0.5 * (work.a + work.b);
        work.tol1 = SQRT_EPS * Math.abs(work.x) + tol;
        work.tol2 = 2.0 * work.tol1;

        // Test for convergence: |x - m| <= tol2 - 0.5*(b - a)
        if (Math.abs(work.x - work.m) <= work.tol2 - 0.5 * (work.b - work.a)) {
            // End of main loop - converged
            mode[0] = FIND_CONV;
            return work.x;
        }

        // Parabolic interpolation or golden-section step
        double rVal = 0.0;
        double qVal = 0.0;
        double pVal = 0.0;
        double dVal = work.d;
        double eVal = work.e;

        if (Math.abs(eVal) > work.tol1) {
            // Fit parabola
            double fx = work.fx;
            double fw = work.fw;
            double fv = work.fv;
            double xPt = work.x;
            double wPt = work.w;
            double vPt = work.v;

            rVal = (xPt - wPt) * (fx - fv);
            qVal = (xPt - vPt) * (fx - fw);
            pVal = (xPt - vPt) * qVal - (xPt - wPt) * rVal;
            qVal = 2.0 * (qVal - rVal);

            if (qVal > 0.0) {
                pVal = -pVal;
            }
            if (qVal < 0.0) {
                qVal = -qVal;
            }

            rVal = eVal;
            eVal = dVal;
        }

        // Store interpolation parameters
        work.r = rVal;
        work.q = qVal;
        work.p = pVal;

        // Decide whether to use parabolic step or golden section
        double aPt = work.a;
        double bPt = work.b;
        double xPt = work.x;

        if (Math.abs(pVal) >= 0.5 * Math.abs(qVal * rVal) ||
            pVal <= qVal * (aPt - xPt) ||
            pVal >= qVal * (bPt - xPt)) {
            // Golden-section step
            if (xPt >= work.m) {
                eVal = aPt - xPt;
            } else {
                eVal = bPt - xPt;
            }
            dVal = c * eVal;
        } else {
            // Parabolic interpolation step
            double uTemp = xPt + pVal / qVal;
            if (uTemp - aPt < work.tol2 || bPt - uTemp < work.tol2) {
                // Ensure not too close to bounds - use copysign like Go
                dVal = (work.m > xPt) ? work.tol1 : -work.tol1;
            } else {
                dVal = pVal / qVal;
            }
        }

        // Ensure step is not too small
        if (Math.abs(dVal) < work.tol1) {
            dVal = (dVal > 0.0) ? work.tol1 : -work.tol1;
        }

        // Store step information
        work.d = dVal;
        work.e = eVal;

        // Compute next evaluation point: u = x + d
        work.u = work.x + work.d;

        mode[0] = FIND_NEXT;
        return work.u;
    }

    // ========================================================================
    // Exact Line Search Integration
    // ========================================================================

    /**
     * Performs exact line search step using golden section and quadratic interpolation.
     * <p>
     * This method integrates {@link #findMin} into the SLSQP optimization loop.
     * It handles the reverse communication protocol and updates the position
     * x = x0 + alpha * s for merit function evaluation.
     * </p>
     *
     * <h3>Algorithm</h3>
     * <ul>
     *   <li>If mode != FIND_CONV: call findMin, set x = x0 + alpha * s</li>
     *   <li>If mode == FIND_CONV: scale s = alpha * s (final step)</li>
     * </ul>
     *
     * @param ws         workspace containing line search state (fw, lineMode)
     * @param t          current merit function value at the point returned by previous call
     * @param tol        desired length of interval of uncertainty
     * @param alphaLower lower bound of search interval (typically 0.1)
     * @param alphaUpper upper bound of search interval (typically 1.0)
     * @param n          problem dimension
     * @param x          current position (output: updated to x0 + alpha * s)
     * @param xOff       offset into x array
     * @param x0         initial position
     * @param x0Off      offset into x0 array
     * @param s          search direction (output: scaled to alpha * s when converged)
     * @param sOff       offset into s array
     * @return FindMode: FIND_CONV if converged, otherwise FIND_INIT or FIND_NEXT
     *
     * @see #findMin
     * @see SLSQPConstants#FIND_CONV
     *
     */
    public static int exactSearch(
            SLSQPWorkspace ws,
            double t,
            double tol,
            double alphaLower,
            double alphaUpper,
            int n,
            double[] x, int xOff,
            double[] x0, int x0Off,
            double[] s, int sOff) {

        int[] mode = new int[] { ws.getLineMode() };

        if (mode[0] != FIND_CONV) {
            // Call findMin to get the next step length
            double alpha = findMin(mode, ws.getFindWork(), t, tol, alphaLower, alphaUpper);
            ws.setAlpha(alpha);
            ws.setLineMode(mode[0]);

            // Update position: x = x0 + alpha * s
            BLAS.dcopy(n, x0, x0Off, 1, x, xOff, 1);
            BLAS.daxpy(n, alpha, s, sOff, 1, x, xOff, 1);
        } else {
            // Search converged: scale s to final step s = alpha * s
            BLAS.dscal(n, ws.getAlpha(), s, sOff, 1);
        }

        return mode[0];
    }
}
