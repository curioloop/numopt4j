/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * L-BFGS-B Line Search - Pure Java implementation.
 *
 * This module implements the line search step of the L-BFGS-B algorithm using
 * the Moré-Thuente algorithm to find a step length satisfying the strong Wolfe conditions:
 *
 *   Sufficient decrease (Armijo condition):
 *     f(xₖ + λₖdₖ) ≤ f(xₖ) + α·λₖ·gₖᵀdₖ
 *
 *   Curvature condition:
 *     |g(xₖ + λₖdₖ)ᵀdₖ| ≤ β·|gₖᵀdₖ|
 *
 * where:
 *   α = 10⁻³ (sufficient decrease parameter)
 *   β = 0.9  (curvature parameter)
 *
 * Reference: C implementation in lbfgsb/linesearch.c and lbfgsb/minpack.c
 * Reference: Go implementation in lbfgsb/linesearch.go and lbfgsb/minpack.go
 * Reference: J.J. Moré and D.J. Thuente, "Line Search Algorithms with Guaranteed
 *            Sufficient Decrease", ACM TOMS, Vol. 20, No. 3, 1994, pp. 286-307.
 *
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import static com.curioloop.numopt4j.linalg.blas.BLAS.*;
import com.curioloop.numopt4j.optim.Bound;

/**
 * Line search for the L-BFGS-B algorithm.
 *
 * <p>This class implements the Moré-Thuente line search algorithm to find a step
 * length λ that satisfies the strong Wolfe conditions. The algorithm maintains
 * an interval [stx, sty] that contains a minimizer and iteratively refines it.</p>
 *
 * @see LBFGSBWorkspace
 * @see CauchyPoint
 */
final class LineSearch implements LBFGSBConstants {

    private LineSearch() {}

    // ========================================================================
    // Minpack Constants (matching Go minpack.go)
    // ========================================================================

    /** Bisection factor */
    private static final double P5 = 0.5;

    /** Safeguard factor for step bounds */
    private static final double P66 = 0.66;

    /** Extrapolation lower factor */
    private static final double XTRAP_LOWER = 1.1;

    /** Extrapolation upper factor */
    private static final double XTRAP_UPPER = 4.0;

    /** Stage 1: Looking for sufficient decrease (Armijo) */
    private static final int STAGE_ARMIJO = 1;

    /** Stage 2: Sufficient decrease achieved, looking for curvature */
    private static final int STAGE_WOLFE = 2;

    // ========================================================================
    // Minpack Error/Warning Codes
    // ========================================================================

    private static final int ERR_OVER_LOWER = SEARCH_ERROR | 1;
    private static final int ERR_OVER_UPPER = SEARCH_ERROR | 2;
    private static final int ERR_NEG_INIT_G = SEARCH_ERROR | 3;
    private static final int ERR_NEG_ALPHA = SEARCH_ERROR | 4;
    private static final int ERR_NEG_BETA = SEARCH_ERROR | 5;
    private static final int ERR_NEG_EPS = SEARCH_ERROR | 6;
    private static final int ERR_LOWER = SEARCH_ERROR | 7;
    private static final int ERR_UPPER = SEARCH_ERROR | 8;

    private static final int WARN_ROUND_ERR = SEARCH_WARN | 9;
    private static final int WARN_REACH_EPS = SEARCH_WARN | 10;
    private static final int WARN_REACH_MAX = SEARCH_WARN | 11;
    private static final int WARN_REACH_MIN = SEARCH_WARN | 12;


    // ========================================================================
    // Public Methods
    // ========================================================================

    /**
     * Initializes line search parameters.
     *
     * <p>Computes:</p>
     * <ul>
     *   <li>‖d‖₂ = √(dᵀd) - Euclidean norm of search direction</li>
     *   <li>λₘₐₓ - Maximum step length respecting bounds</li>
     *   <li>λ₀ - Initial step length</li>
     * </ul>
     *
     * <p>Initial step selection:</p>
     * <ul>
     *   <li>First iteration (not boxed): λ₀ = min(1/‖d‖₂, λₘₐₓ)</li>
     *   <li>Otherwise: λ₀ = 1</li>
     * </ul>
     *
     * <p>Reference: C function init_line_search() in linesearch.c</p>
     * <p>Reference: Go function initLineSearch() in linesearch.go</p>
     *
     * @param n Problem dimension
     * @param x Current point xₖ
     * @param bounds Array of Bounds for each variable. Null means unbounded.
     * @param ws Workspace containing iteration state
     * @return Initial step length λ₀
     *
     */
    static double init(int n,
                       double[] x,
                       Bound[] bounds,
                       LBFGSBWorkspace ws) {
        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();

        // Compute d·d and ||d||_2
        // ‖d‖₂ = √(dᵀd) - Euclidean norm of search direction
        double dSqrt = ddot(n, buffer, dOffset, 1, buffer, dOffset, 1);
        double dNorm = Math.sqrt(dSqrt);
        ws.setDSqrt(dSqrt);
        ws.setDNorm(dNorm);

        // Determine the maximum step length λₘₐₓ
        double stepMax = SEARCH_NO_BND;

        if (ws.isConstrained()) {
            if (ws.getIter() == 0) {
                stepMax = 1.0;
            } else {
                // Compute maximum step that keeps x within bounds
                for (int i = 0; i < n; i++) {
                    Bound b = Bound.of(bounds, i, null);

                    if (b != null) {
                        double di = buffer[dOffset + i];

                        // Check lower bound: dᵢ < 0 and has lower bound
                        if (di < 0.0 && b.hasLower()) {
                            double span = b.getLower() - x[i];
                            if (span >= 0.0) {
                                stepMax = 0.0;  // Variable fixed at lower bound
                            } else if (di * stepMax < span) {
                                stepMax = span / di;  // Constrain step to bound
                            }
                        }
                        // Check upper bound: dᵢ > 0 and has upper bound
                        else if (di > 0.0 && b.hasUpper()) {
                            double span = b.getUpper() - x[i];
                            if (span <= 0.0) {
                                stepMax = 0.0;  // Variable fixed at upper bound
                            } else if (di * stepMax > span) {
                                stepMax = span / di;  // Constrain step to bound
                            }
                        }
                    }
                }
            }
        }

        // Initialize search tolerances for Wolfe conditions
        // Store in workspace line search context
        LBFGSBWorkspace.SearchCtx ctx = ws.getSearchCtx();
        ctx.stpMin = 0.0;
        ctx.stpMax = stepMax;

        // Set initial step length λ₀
        double stp;
        if (ws.getIter() == 0 && !ws.isBoxed()) {
            // First iteration and not all variables boxed: use scaled step
            stp = 1.0 / dNorm;
            if (stp > stepMax) {
                stp = stepMax;
            }
        } else {
            stp = 1.0;
        }

        // Initialize line search state
        ctx.numEval = 0;
        ctx.numBack = 0;
        ctx.searchTask = SEARCH_START;
        ctx.stp = stp;

        return stp;
    }

    /**
     * Performs line search along search direction.
     *
     * <p>Performs a line search along dₖ to find a step λₖ satisfying the strong
     * Wolfe conditions:</p>
     * <pre>
     *   Sufficient decrease (Armijo condition):
     *     f(xₖ + λₖdₖ) ≤ f(xₖ) + α·λₖ·gₖᵀdₖ
     *
     *   Curvature condition:
     *     |g(xₖ + λₖdₖ)ᵀdₖ| ≤ β·|gₖᵀdₖ|
     * </pre>
     *
     * <p>The new iterate is computed as:</p>
     * <pre>
     *   xₖ₊₁ = xₖ + λₖdₖ
     * </pre>
     *
     * <p>Or equivalently using stored values:</p>
     * <pre>
     *   xₖ₊₁ = λₖdₖ + t    (where t = xₖ before line search)
     * </pre>
     *
     * <p>Special case: if λₖ = 1, use the Cauchy point z directly:</p>
     * <pre>
     *   xₖ₊₁ = xᶜ
     * </pre>
     *
     * <p>Reference: C function perform_line_search() in linesearch.c</p>
     * <p>Reference: Go function performLineSearch() in linesearch.go</p>
     *
     * @param n Problem dimension
     * @param x Current point (modified: becomes xₖ₊₁)
     * @param f Function value at current point
     * @param g Gradient at current point
     * @param ws Workspace containing iteration state
     * @return Array of [info, done]: info is error code, done is 1 if search complete
     *
     */
    static int[] perform(int n, double[] x, double f, double[] g, LBFGSBWorkspace ws) {
        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();
        int tOffset = ws.getTOffset();
        int zOffset = ws.getZOffset();
        LBFGSBWorkspace.SearchCtx ctx = ws.getSearchCtx();

        // Compute directional derivative gd = gᵀd = gₖᵀdₖ
        double gd = ddot(n, g, 0, 1, buffer, dOffset, 1);
        ctx.gd = gd;

        // First evaluation: check descent direction
        if (ctx.numEval == 0) {
            ctx.gdOld = gd;
            if (gd >= 0.0) {
                // Line search is impossible when directional derivative ≥ 0
                return new int[] { ERR_DERIVATIVE, 0 };
            }
        }

        // Call scalar search
        scalarSearch(f, gd, ctx, ws.isConstrained());
        double stp = ctx.stp;
        int task = ctx.searchTask;

        // Check if done (converged, warning, or error)
        boolean done = (task & (SEARCH_CONV | SEARCH_WARN | SEARCH_ERROR)) != 0;

        int info = ERR_NONE;
        if (!done) {
            // Try another x: compute xₖ₊₁ = λₖdₖ + xₖ
            if (stp == 1.0) {
                dcopy(n, buffer, zOffset, 1, x, 0, 1);  // x = xᶜ (Cauchy point when λ = 1)
            } else {
                for (int i = 0; i < n; i++) {
                    x[i] = stp * buffer[dOffset + i] + buffer[tOffset + i];  // xₖ₊₁ = λₖdₖ + xₖ
                }
            }
        } else if ((task & SEARCH_ERROR) != 0) {
            info = ERR_LINE_SEARCH_TOL;
        }

        return new int[] { info, done ? 1 : 0 };
    }


    // ========================================================================
    // Moré-Thuente Scalar Search (dcsrch)
    // ========================================================================

    /**
     * ScalarSearch (dcsrch)
     *
     * <p>This subroutine finds a step λ that satisfies:</p>
     * <ul>
     *   <li>sufficient decrease condition: f(λ) ≤ f(0) + α·λ·f′(0)</li>
     *   <li>curvature condition: |f′(λ)| ≤ β·|f′(0)|</li>
     * </ul>
     *
     * <p>Each call of the subroutine updates an interval with endpoints stx and sty.</p>
     *
     * <p>The interval is initially chosen so that it contains a minimizer of the modified function:</p>
     * <pre>
     *   ψ(λ) = f(λ) - f(0) - α·λ·f′(0)
     * </pre>
     *
     * <p>If ψ(λ) ≤ 0 and f′(λ) ≥ 0 for some step, then the interval is chosen so that it contains a minimizer of f.</p>
     *
     * <p>If α is less than β and if, for example, the function is bounded below,
     * then there is always a step which satisfies both conditions.</p>
     *
     * <p>If no step can be found that satisfies both conditions, then the algorithm stops with a warning.
     * In this case stp only satisfies the sufficient decrease condition.</p>
     *
     * <h3>Parameters</h3>
     *
     * <p><b>f</b> — function value f(λ).
     * <ul>
     *   <li>On initial entry: f(0), the value of the function at λ = 0.</li>
     *   <li>On subsequent entries: f(λ), the value at the current step λ.</li>
     *   <li>On exit: f(λ) at the final step.</li>
     * </ul>
     *
     * <p><b>g</b> — directional derivative f′(λ) = ∇f(λ)ᵀd.
     * <ul>
     *   <li>On initial entry: f′(0), the derivative at λ = 0. Must be negative (descent direction).</li>
     *   <li>On subsequent entries: f′(λ), the derivative at the current step λ.</li>
     *   <li>On exit: f′(λ) at the final step.</li>
     * </ul>
     *
     * <p><b>ctx.stp</b> — current step estimate λ.
     * <ul>
     *   <li>On entry: current estimate of a satisfactory step. On initial entry, a positive initial estimate
     *       must be provided.</li>
     *   <li>On exit: current estimate λ if task = SearchFG. If task = SearchConv then λ satisfies
     *       the sufficient decrease and curvature conditions.</li>
     * </ul>
     *
     * @param f  Function value f(λ) at current step
     * @param g  Directional derivative f′(λ) = ∇f(λ)ᵀd at current step
     * @param ctx Search context containing λ (stp) and all interval state variables
     * @param constrained Whether the problem has constraints (affects step bounds)
     */
    private static void scalarSearch(double f, double g, LBFGSBWorkspace.SearchCtx ctx, boolean constrained) {
        // Get tolerance parameters
        double alpha = SEARCH_ALPHA;
        double beta = SEARCH_BETA;
        double eps = SEARCH_EPS;
        double stp = ctx.stp;
        int task = ctx.searchTask;
        double lower = ctx.stpMin;
        double upper = ctx.stpMax;

        // ========================================================================
        // Initialization block
        // ========================================================================
        if (task == SEARCH_START) {
            // Check the input arguments for errors
            if (stp < lower) {
                task = ERR_OVER_LOWER;
            } else if (stp > upper) {
                task = ERR_OVER_UPPER;
            } else if (g >= 0.0) {
                task = ERR_NEG_INIT_G;
            } else if (alpha < 0.0) {
                task = ERR_NEG_ALPHA;
            } else if (beta < 0.0) {
                task = ERR_NEG_BETA;
            } else if (eps < 0.0) {
                task = ERR_NEG_EPS;
            } else if (lower < 0.0) {
                task = ERR_LOWER;
            } else if (upper < lower) {
                task = ERR_UPPER;
            }

            // Exit if there are errors on input
            if ((task & SEARCH_ERROR) != 0) {
                ctx.stp = stp;
                ctx.searchTask = task;
                return;
            }

            // Initialize local variables
            ctx.bracket = false;
            ctx.stage = STAGE_ARMIJO;
            ctx.f0 = f;
            ctx.g0 = g;
            ctx.width0 = upper - lower;
            ctx.width1 = (upper - lower) / P5;

            // Initialize the points and their corresponding function and derivative values
            ctx.stx = 0.0;
            ctx.fx = f;
            ctx.gx = g;
            ctx.sty = 0.0;
            ctx.fy = f;
            ctx.gy = g;

            // Set initial bounds
            ctx.stpMin = 0.0;
            ctx.stpMax = stp + XTRAP_UPPER * stp;

            ctx.stp = stp;
            ctx.searchTask = SEARCH_FG;
            return;
        }

        // ========================================================================
        // Test for convergence or warnings
        // ========================================================================
        double gTest = alpha * ctx.g0;
        double fTest = ctx.f0 + stp * gTest;

        double stpMin = ctx.stpMin;
        double stpMax = ctx.stpMax;

        if (ctx.bracket && (stp <= stpMin || stp >= stpMax)) {
            task = WARN_ROUND_ERR;
        } else if (ctx.bracket && (stpMax - stpMin) <= eps * stpMax) {
            task = WARN_REACH_EPS;
        } else if (stp == upper && f <= fTest && g <= gTest) {
            task = WARN_REACH_MAX;
        } else if (stp == lower && (f > fTest || g >= gTest)) {
            task = WARN_REACH_MIN;
        } else if (f <= fTest && Math.abs(g) <= beta * (-ctx.g0)) {
            task = SEARCH_CONV;
        }

        if ((task & (SEARCH_WARN | SEARCH_CONV)) != 0) {
            ctx.stp = stp;
            ctx.searchTask = task;
            return;
        }

        // ========================================================================
        // Update search stage
        // ========================================================================
        if (ctx.stage == STAGE_ARMIJO && f <= fTest && g >= 0.0) {
            ctx.stage = STAGE_WOLFE;
        }

        // ========================================================================
        // A modified function is used to predict the step during the first stage
        // ========================================================================
        double stx = ctx.stx;
        double fx = ctx.fx;
        double gx = ctx.gx;
        double sty = ctx.sty;
        double fy = ctx.fy;
        double gy = ctx.gy;
        boolean bracket = ctx.bracket;

        if (ctx.stage == STAGE_ARMIJO && f <= fx && f > fTest) {
            // Define the modified function and derivative values
            double fm = f - stp * gTest;
            double fxm = fx - stx * gTest;
            double fym = fy - sty * gTest;
            double gm = g - gTest;
            double gxm = gx - gTest;
            double gym = gy - gTest;

            // Call scalar_step to update interval
            stp = scalarStep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, bracket, stpMin, stpMax, ctx);
            stx = ctx.stx;
            fxm = ctx.fx;
            gxm = ctx.gx;
            sty = ctx.sty;
            fym = ctx.fy;
            gym = ctx.gy;
            bracket = ctx.bracket;

            // Reset the function and derivative values for f
            fx = fxm + stx * gTest;
            fy = fym + sty * gTest;
            gx = gxm + gTest;
            gy = gym + gTest;
        } else {
            // Call scalar_step to update interval
            stp = scalarStep(stx, fx, gx, sty, fy, gy, stp, f, g, bracket, stpMin, stpMax, ctx);
            stx = ctx.stx;
            fx = ctx.fx;
            gx = ctx.gx;
            sty = ctx.sty;
            fy = ctx.fy;
            gy = ctx.gy;
            bracket = ctx.bracket;
        }

        // Update workspace
        ctx.stx = stx;
        ctx.fx = fx;
        ctx.gx = gx;
        ctx.sty = sty;
        ctx.fy = fy;
        ctx.gy = gy;

        // ========================================================================
        // Decide if a bisection step is needed
        // ========================================================================
        if (bracket) {
            if (Math.abs(sty - stx) >= P66 * ctx.width1) {
                stp = stx + P5 * (sty - stx);
            }
            ctx.width1 = ctx.width0;
            ctx.width0 = Math.abs(sty - stx);
        }

        // ========================================================================
        // Set the minimum and maximum steps allowed for stp
        // ========================================================================
        if (bracket) {
            stpMin = Math.min(stx, sty);
            stpMax = Math.max(stx, sty);
        } else {
            stpMin = stp + XTRAP_LOWER * (stp - stx);
            stpMax = stp + XTRAP_UPPER * (stp - stx);

            /*
             * Conservative safety for boxed/constrained problems:
             * The original dcstep/minpack behavior allows aggressive extrapolation
             * (multiplicative growth) which can produce step lengths far beyond
             * the maximum step that keeps the candidate inside bounds. When the
             * problem is constrained/boxed, cap the extrapolated stpMax by the
             * initial maximum step (stored in width0 at SEARCH_START as
             * initialUpper - initialLower). This is a small, documented deviation
             * from Fortran semantics that prevents wildly infeasible candidates
             * while preserving interpolation logic.
             */
            double initUpper = ctx.width0; // set at SEARCH_START as upper - lower
            if (constrained && initUpper > 0.0 && stpMax > initUpper) {
                // Cap extrapolated stpMax for boxed problems (important safety)
                stpMax = initUpper;
                if (stpMin > stpMax) stpMin = stpMax;
            }
        }
        ctx.stpMin = stpMin;
        ctx.stpMax = stpMax;

        // Force the step to be within the bounds
        stp = Math.min(Math.max(stp, stpMin), stpMax);

        // If further progress is not possible, let stp be the best point obtained so far
        if ((bracket && (stp <= stpMin || stp >= stpMax)) ||
            (bracket && stpMax - stpMin <= eps * stpMax)) {
            stp = stx;
        }

        ctx.stp = stp;
        ctx.searchTask = SEARCH_FG;
    }


    // ========================================================================
    // Moré-Thuente Scalar Step (dcstep)
    // ========================================================================

    /**
     * Subroutine scalarStep (dcstep)
     *
     * <p>This subroutine computes a safeguarded step for a search procedure and
     * updates an interval [stx, sty] that contains a step satisfying a sufficient
     * decrease and a curvature condition.</p>
     *
     * <p>The parameter stx contains the step with the least function value.
     * If bracket is set to true then a minimizer has been bracketed in an interval
     * with endpoints stx and sty. The subroutine assumes that if bracket is true then:</p>
     * <pre>
     *   min(stx, sty) &lt; stp &lt; max(stx, sty)
     * </pre>
     * <p>and that the derivative at stx is negative in the direction of the step.</p>
     *
     * <h3>Cubic Interpolation Formula</h3>
     * <p>The cubic minimizer stpc is computed from the cubic that interpolates
     * f and f′ at stx and stp. Given:</p>
     * <pre>
     *   θ = 3·(f(stx) - f(stp)) / (stp - stx) + f′(stx) + f′(stp)
     *   s = max(|θ|, |f′(stx)|, |f′(stp)|)
     *   γ = s·√((θ/s)² - (f′(stx)/s)·(f′(stp)/s))
     *   p = (γ - f′(stx)) + θ
     *   q = ((γ - f′(stx)) + γ) + f′(stp)
     *   r = p / q
     *   stpc = stx + r·(stp - stx)
     * </pre>
     * <p>The secant step stpq uses the secant approximation:</p>
     * <pre>
     *   stpq = stp + f′(stp) / (f′(stp) - f′(stx)) · (stx - stp)
     * </pre>
     *
     * <h3>Step Selection Strategy</h3>
     * <ul>
     *   <li><b>Case 1</b>: f(stp) &gt; f(stx) — higher function value, minimum is bracketed.
     *     If the cubic step stpc is closer to stx than the quadratic step stpq,
     *     take stpc; otherwise take the average (stpc + stpq) / 2.</li>
     *   <li><b>Case 2</b>: f(stp) ≤ f(stx) and f′(stp)·f′(stx) &lt; 0 — lower value, opposite-sign derivatives.
     *     Minimum is bracketed. If stpc is farther from stp than stpq, take stpc; otherwise take stpq.</li>
     *   <li><b>Case 3</b>: f(stp) ≤ f(stx), |f′(stp)| &lt; |f′(stx)|, same-sign derivatives — decreasing derivative.
     *     Cubic step only if it tends to infinity or minimum is beyond stp; otherwise use secant.
     *     If bracketed, clamp to stp + 0.66·(sty - stp) to prevent slow convergence.</li>
     *   <li><b>Case 4</b>: f(stp) ≤ f(stx), |f′(stp)| ≥ |f′(stx)|, same-sign derivatives — non-decreasing derivative.
     *     If bracketed, use cubic step stpc; otherwise step to stpMin or stpMax.</li>
     * </ul>
     *
     * <p>Reference: C function minpack_scalar_step() in minpack.c</p>
     * <p>Reference: Go function scalarStep() in minpack.go</p>
     *
     * @param stx    Best step obtained so far; endpoint of interval containing minimizer.
     *               On exit: updated best step.
     * @param fx     f(stx) — function value at stx.
     *               On exit: updated function value at stx.
     * @param dx     f′(stx) — derivative at stx. Must be negative in the direction of the step
     *               (i.e., dx and stp - stx must have opposite signs).
     *               On exit: updated derivative at stx.
     * @param sty    Second endpoint of interval containing minimizer.
     *               On exit: updated second endpoint.
     * @param fy     f(sty) — function value at sty.
     *               On exit: updated function value at sty.
     * @param dy     f′(sty) — derivative at sty.
     *               On exit: updated derivative at sty.
     * @param stp    Current step. If bracket is true, must satisfy min(stx,sty) &lt; stp &lt; max(stx,sty).
     *               On exit: new trial step.
     * @param fp     f(stp) — function value at stp.
     * @param dp     f′(stp) — derivative at stp.
     * @param bracket Whether a minimizer has been bracketed in [stx, sty].
     *                Initially must be false. On exit: updated bracketing flag.
     * @param stpMin Minimum allowed step bound.
     * @param stpMax Maximum allowed step bound.
     * @param ctx    Search context updated with new stx, fx, gx, sty, fy, gy, bracket.
     * @return New trial step stp.
     */
    private static double scalarStep(
            double stx, double fx, double dx,
            double sty, double fy, double dy,
            double stp, double fp, double dp,
            boolean bracket, double stpMin, double stpMax, LBFGSBWorkspace.SearchCtx ctx) {

        double gamma, p, q, r, s, sgnd;
        double stpc, stpf, stpq, theta;

        // Sign of dp * (dx / |dx|) - determines if derivatives have same sign
        sgnd = dp * (dx / Math.abs(dx));

        // ========================================================================
        // First case: A higher function value. The minimum is bracketed.
        // If the cubic step stpc is closer to stx than the quadratic step stpq,
        // the cubic step is taken; otherwise the average of the cubic and quadratic steps is taken.
        // ========================================================================
        if (fp > fx) {
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            s = Math.max(Math.max(Math.abs(theta), Math.abs(dx)), Math.abs(dp));
            gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp < stx) {
                gamma = -gamma;
            }
            p = (gamma - dx) + theta;
            q = ((gamma - dx) + gamma) + dp;
            r = p / q;
            stpc = stx + r * (stp - stx);
            stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
            if (Math.abs(stpc - stx) < Math.abs(stpq - stx)) {
                stpf = stpc;
            } else {
                stpf = stpc + (stpq - stpc) / 2.0;
            }
            bracket = true;
        }
        // ========================================================================
        // Second case: A lower function value and derivatives of opposite sign.
        // The minimum is bracketed.
        // If the cubic step stpc is farther from stp than the secant step stpq,
        // the cubic step is taken; otherwise the secant step is taken.
        // ========================================================================
        else if (sgnd < 0.0) {
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            s = Math.max(Math.max(Math.abs(theta), Math.abs(dx)), Math.abs(dp));
            gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp > stx) {
                gamma = -gamma;
            }
            p = (gamma - dp) + theta;
            q = ((gamma - dp) + gamma) + dx;
            r = p / q;
            stpc = stp + r * (stx - stp);
            stpq = stp + (dp / (dp - dx)) * (stx - stp);
            if (Math.abs(stpc - stp) > Math.abs(stpq - stp)) {
                stpf = stpc;
            } else {
                stpf = stpq;
            }
            bracket = true;
        }
        // ========================================================================
        // Third case: A lower function value, derivatives of the same sign,
        // and the magnitude of the derivative decreases.
        // The cubic step is computed only if either:
        //   - the cubic tends to infinity in the direction of the step
        //   - the minimum of the cubic is beyond stp.
        // Otherwise the cubic step is defined to be the secant step.
        // ========================================================================
        else if (Math.abs(dp) < Math.abs(dx)) {
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            s = Math.max(Math.max(Math.abs(theta), Math.abs(dx)), Math.abs(dp));
            // The case gamma = 0 only arises if the cubic does not tend
            // to infinity in the direction of the step.
            gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp > stx) {
                gamma = -gamma;
            }
            p = (gamma - dp) + theta;
            q = (gamma + (dx - dp)) + gamma;
            r = p / q;
            if (r < 0.0 && gamma != 0.0) {
                stpc = stp + r * (stx - stp);
            } else if (stp > stx) {
                stpc = stpMax;
            } else {
                stpc = stpMin;
            }
            stpq = stp + (dp / (dp - dx)) * (stx - stp);
            if (bracket) {
                // A minimizer has been bracketed.
                // If the cubic step is closer to stp than the secant step, the cubic step is taken;
                // otherwise the secant step is taken.
                if (Math.abs(stpc - stp) < Math.abs(stpq - stp)) {
                    stpf = stpc;
                } else {
                    stpf = stpq;
                }
                if (stp > stx) {
                    stpf = Math.min(stp + P66 * (sty - stp), stpf);
                } else {
                    stpf = Math.max(stp + P66 * (sty - stp), stpf);
                }
            } else {
                // A minimizer has not been bracketed.
                // If the cubic step is farther from stp than the secant step, the cubic step is taken;
                // otherwise the secant step is taken.
                if (Math.abs(stpc - stp) > Math.abs(stpq - stp)) {
                    stpf = stpc;
                } else {
                    stpf = stpq;
                }
                stpf = Math.min(stpMax, stpf);
                stpf = Math.max(stpMin, stpf);
            }
        }
        // ========================================================================
        // Fourth case: A lower function value, derivatives of the same sign,
        // and the magnitude of the derivative does not decrease.
        // If the minimum is not bracketed, the step is either stpMin or stpMax;
        // otherwise the cubic step is taken.
        // ========================================================================
        else {
            if (bracket) {
                theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
                s = Math.max(Math.max(Math.abs(theta), Math.abs(dy)), Math.abs(dp));
                gamma = s * Math.sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
                if (stp > sty) {
                    gamma = -gamma;
                }
                p = (gamma - dp) + theta;
                q = ((gamma - dp) + gamma) + dy;
                r = p / q;
                stpc = stp + r * (sty - stp);
                stpf = stpc;
            } else if (stp > stx) {
                stpf = stpMax;
            } else {
                stpf = stpMin;
            }
        }

        // ========================================================================
        // Update the interval which contains a minimizer.
        // ========================================================================
        if (fp > fx) {
            sty = stp;
            fy = fp;
            dy = dp;
        } else {
            if (sgnd < 0.0) {
                sty = stx;
                fy = fx;
                dy = dx;
            }
            stx = stp;
            fx = fp;
            dx = dp;
        }

        // Compute the new step
        stp = stpf;

        ctx.stx = stx;
        ctx.fx = fx;
        ctx.gx = dx;
        ctx.sty = sty;
        ctx.fy = fy;
        ctx.gy = dy;
        ctx.bracket = bracket;
        return stp;
    }
}
