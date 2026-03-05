/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * L-BFGS-B Core - Pure Java implementation.
 *
 * This class implements the core optimization loop and projection operations
 * for the L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds) algorithm.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Algorithm Overview
 * ════════════════════════════════════════════════════════════════════════════
 *
 * L-BFGS-B solves bound-constrained optimization problems:
 *
 *   minimize   f(x)
 *   subject to l ≤ x ≤ u
 *
 * At each iteration k, the algorithm minimizes a quadratic model of f:
 *
 *   mₖ(x) = fₖ + gₖᵀ(x - xₖ) + ½(x - xₖ)ᵀBₖ(x - xₖ)
 *
 * where Bₖ is the L-BFGS Hessian approximation in compact form:
 *
 *   Bₖ = θI - WMWᵀ
 *
 * with:
 *   θ  — scaling factor (updated each iteration)
 *   W  = [Yₖ, θSₖ]  — n × 2m matrix of curvature pairs
 *   M  — 2m × 2m indefinite matrix encoding BFGS corrections
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Main Iteration Steps
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   Step 1 — Generalized Cauchy Point (GCP):
 *     Minimize mₖ(x) along the projected gradient path to find xᶜ.
 *     Identifies the active set 𝒜(xᶜ) and free variable set 𝓕(xᶜ).
 *
 *   Step 2 — Subspace Minimization:
 *     Minimize mₖ(x) over the subspace of free variables at xᶜ:
 *       minimize   m̃ₖ(d̃) ≡ d̃ᵀr̃ᶜ + ½d̃ᵀB̃ₖd̃
 *       subject to lᵢ - xᶜᵢ ≤ d̃ᵢ ≤ uᵢ - xᶜᵢ   (i ∈ 𝓕)
 *     where B̃ₖ = ZₖᵀBₖZₖ is the reduced Hessian and r̃ᶜ = Zₖᵀ(gₖ + Bₖ(xᶜ - xₖ)).
 *
 *   Step 3 — Line Search:
 *     Find λₖ satisfying the strong Wolfe conditions along dₖ = x̂ - xₖ:
 *       f(xₖ + λₖdₖ) ≤ f(xₖ) + α·λₖ·gₖᵀdₖ   (sufficient decrease)
 *       |g(xₖ + λₖdₖ)ᵀdₖ| ≤ β·|gₖᵀdₖ|        (curvature condition)
 *
 *   Step 4 — BFGS Update:
 *     Update curvature pairs (sₖ, yₖ) = (xₖ₊₁ - xₖ, gₖ₊₁ - gₖ) and
 *     recompute the compact representation matrices W, M, T.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Convergence Criteria
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   (1) Projected gradient norm:
 *         ‖proj gₖ‖∞ ≤ pgtol
 *
 *   (2) Relative function value reduction:
 *         (fₖ - fₖ₊₁) / max(|fₖ|, |fₖ₊₁|, 1) ≤ factr × εₘₐ꜀ₕ
 *
 *   (3) Gradient descent threshold (step too small):
 *         ‖dₖ‖₂ ≤ pgtol × (1 + |fₖ|)
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Error Recovery
 * ════════════════════════════════════════════════════════════════════════════
 *
 * When numerical issues occur (singular triangular system, non-positive definite
 * Cholesky, or line search failure), the algorithm resets Bₖ = I (identity) and
 * restarts from the current xₖ. At most MAX_BFGS_RESETS resets are allowed.
 *
 * Reference: C implementation in lbfgsb/lbfgsb.c and lbfgsb/project.c
 * Reference: Go implementation in lbfgsb/driver.go, lbfgsb/optimize.go, lbfgsb/project.go
 *
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import static com.curioloop.numopt4j.linalg.blas.BLAS.*;
import com.curioloop.numopt4j.optim.Univariate;
import com.curioloop.numopt4j.optim.Bound;

/**
 * Core implementation of the L-BFGS-B optimization algorithm.
 *
 * <p>This class contains the main optimization loop and projection operations.
 * The projection operations are essential for bound-constrained optimization:</p>
 * <ul>
 *   <li>{@link #projectX} - Projects a point onto the feasible region</li>
 *   <li>{@link #projGradNorm} - Computes the infinity norm of the projected gradient</li>
 *   <li>{@link #projInitActive} - Initializes variable status and projects initial point</li>
 *   <li>{@link #optimize} - Main optimization loop</li>
 * </ul>
 *
 * <h2>Main Iteration Flow</h2>
 * <p>The optimization proceeds as follows:</p>
 * <ol>
 *   <li>INITIALIZATION: Project x₀ onto feasible region, compute f₀ and g₀</li>
 *   <li>MAIN LOOP:
 *     <ul>
 *       <li>Compute Generalized Cauchy Point (GCP)</li>
 *       <li>Subspace minimization</li>
 *       <li>Line search</li>
 *       <li>Convergence check</li>
 *       <li>BFGS update</li>
 *     </ul>
 *   </li>
 *   <li>TERMINATION: Return final x, f(x), and convergence status</li>
 * </ol>
 *
 * <h2>Projection Operator</h2>
 * <p>The projection operator P(x, l, u) maps a point x to the feasible region [l, u]:</p>
 * <pre>
 *   P(xᵢ, lᵢ, uᵢ) = lᵢ    if xᵢ &lt; lᵢ
 *   P(xᵢ, lᵢ, uᵢ) = uᵢ    if xᵢ &gt; uᵢ
 *   P(xᵢ, lᵢ, uᵢ) = xᵢ    otherwise
 * </pre>
 *
 * @see LBFGSBWorkspace
 */
final class LBFGSBCore implements LBFGSBConstants {

    // Private constructor - utility class
    private LBFGSBCore() {}

    // ========================================================================
    // Projection Operations (from project.c)
    // ========================================================================

    /**
     * Projects a point onto the feasible region defined by bounds.
     *
     * <p>This is a simple projection operation:</p>
     * <pre>
     *   P(xᵢ, lᵢ, uᵢ) = lᵢ    if xᵢ &lt; lᵢ
     *   P(xᵢ, lᵢ, uᵢ) = uᵢ    if xᵢ &gt; uᵢ
     *   P(xᵢ, lᵢ, uᵢ) = xᵢ    otherwise
     * </pre>
     *
     * <p>Reference: C function project_x() in project.c</p>
     *
     * @param n Problem dimension
     * @param x Point to project (modified in place)
     * @param bounds Array of Bounds for each variable. Null means unbounded.
     *
     * @see LBFGSBConstants#BOUND_NONE
     * @see LBFGSBConstants#BOUND_LOWER
     * @see LBFGSBConstants#BOUND_BOTH
     * @see LBFGSBConstants#BOUND_UPPER
     *
     */
    static void projectX(int n, double[] x, Bound[] bounds) {
        for (int i = 0; i < n; i++) {
            Bound b = Bound.of(bounds, i, null);
            if (b == null) continue; // no projection needed

            if (b.hasBoth()) {
                // Both bounds: xᵢ = min(uᵢ, max(lᵢ, xᵢ))
                if (x[i] < b.getLower()) {
                    x[i] = b.getLower();
                } else if (x[i] > b.getUpper()) {
                    x[i] = b.getUpper();
                }
            } else if (b.hasLower()) {
                // Only lower bound: xᵢ = max(lᵢ, xᵢ)
                if (x[i] < b.getLower()) {
                    x[i] = b.getLower();
                }
            } else if (b.hasUpper()) {
                // Only upper bound: xᵢ = min(uᵢ, xᵢ)
                if (x[i] > b.getUpper()) {
                    x[i] = b.getUpper();
                }
            }
        }
    }

    /**
     * Computes the infinity norm of the projected gradient.
     *
     * <p>For the next location xₖ₊₁ = xₖ - αₖBₖgₖ (where αₖBₖ &gt; 0), the gradient
     * projection P(gᵢ, lᵢ, uᵢ) limits the gradient to the feasible region:</p>
     * <pre>
     *   proj gᵢ = max(xᵢ - uᵢ, gᵢ)  if gᵢ &lt; 0
     *   proj gᵢ = min(xᵢ - lᵢ, gᵢ)  if gᵢ &gt; 0
     *   proj gᵢ = gᵢ                otherwise
     * </pre>
     *
     * <p>This function computes ‖proj g‖∞ = maxᵢ |proj gᵢ|</p>
     *
     * <p>The projected gradient norm is used as a convergence criterion:
     * when ‖proj g‖∞ &lt; ε, the current point is approximately optimal.</p>
     *
     * <p>Reference: C function proj_grad_norm() in project.c</p>
     * <p>Reference: Go function projGradNorm() in project.go</p>
     *
     * @param n Problem dimension
     * @param x Current point xₖ
     * @param g Gradient gₖ at current point
     * @param bounds Array of Bounds for each variable. Null means unbounded.
     * @return Infinity norm of the projected gradient ‖proj g‖∞
     *
     */
    static double projGradNorm(int n, double[] x, double[] g, Bound[] bounds) {
        /*
         * Bound hint values:
         *   bndNo    = 0 (BOUND_NONE)   - No bounds
         *   bndLower = 1 (BOUND_LOWER)  - Lower bound only
         *   bndBoth  = 2 (BOUND_BOTH)   - Both lower and upper bounds
         *   bndUpper = 3 (BOUND_UPPER)  - Upper bound only
         *
         * The condition bt >= BOUND_BOTH captures BOUND_BOTH (2) and BOUND_UPPER (3),
         * i.e., variables that have an upper bound.
         *
         * The condition bt <= BOUND_BOTH captures BOUND_LOWER (1) and BOUND_BOTH (2),
         * i.e., variables that have a lower bound.
         */

        double norm = 0.0; // ‖proj g‖∞

        for (int i = 0; i < n; i++) {
            double gi = g[i];
            double xi = x[i];
            Bound b = Bound.of(bounds, i, null);

            if (b != null) {
                if (gi < 0.0) {
                    // gᵢ < 0: check upper bound (bt >= BOUND_BOTH means has upper bound)
                    if (b.hasUpper()) {
                        // proj gᵢ = max(xᵢ - uᵢ, gᵢ)
                        double diff = xi - b.getUpper();
                        if (diff > gi) {
                            gi = diff;
                        }
                    }
                } else {
                    // gᵢ >= 0: check lower bound (bt <= BOUND_BOTH means has lower bound)
                    if (b.hasLower()) {
                        // proj gᵢ = min(xᵢ - lᵢ, gᵢ)
                        double diff = xi - b.getLower();
                        if (diff < gi) {
                            gi = diff;
                        }
                    }
                }
            }

            // Update infinity norm: ‖proj g‖∞ = maxᵢ |proj gᵢ|
            double absGi = Math.abs(gi);
            if (absGi > norm) {
                norm = absGi;
            }
        }

        return norm;
    }

    /**
     * Initializes the variable status array (where) and projects the initial point
     * to the feasible set if necessary.
     *
     * <p>Initial projection P(xᵢ, lᵢ, uᵢ) limits x to the feasible region:</p>
     * <pre>
     *   proj xᵢ = uᵢ    if xᵢ &gt; uᵢ
     *   proj xᵢ = lᵢ    if xᵢ &lt; lᵢ
     *   proj xᵢ = xᵢ    otherwise
     * </pre>
     *
     * <p>The function performs two passes:</p>
     * <ol>
     *   <li>Project x to feasible region and count variables at bounds</li>
     *   <li>Initialize where array and determine problem characteristics</li>
     * </ol>
     *
     * <p>Variable status values (where):</p>
     * <ul>
     *   <li>VAR_UNBOUND (-1): Variable has no bounds</li>
     *   <li>VAR_FREE (0): Variable is free to move within bounds</li>
     *   <li>VAR_FIXED (3): Variable is fixed (uᵢ - lᵢ ≤ 0)</li>
     * </ul>
     *
     * <p>Reference: C function proj_init_active() in project.c</p>
     * <p>Reference: Go function projInitActive() in project.go</p>
     *
     * @param n Problem dimension
     * @param x Current point (modified in place if projection needed)
     * @param bounds Array of Bounds for each variable. Null means unbounded.
     *
     */
    static boolean projInitActive(int n, double[] x,
                               Bound[] bounds, LBFGSBWorkspace ws) {
        int numBnd = 0;
        boolean projected = false;
        boolean constrained = false;
        boolean boxed = true;

        int[] iBuffer = ws.getIntBuffer();
        int whereOffset = ws.getWhereOffset();

        /*
         * First pass: Project x to feasible region
         *
         * For each variable with bounds:
         *   - If xᵢ ≤ lᵢ (and has lower bound): project to lᵢ
         *   - If xᵢ ≥ uᵢ (and has upper bound): project to uᵢ
         *
         * The condition bt <= BOUND_BOTH captures variables with lower bounds.
         * The condition bt >= BOUND_BOTH captures variables with upper bounds.
         */
        for (int i = 0; i < n; i++) {
            Bound b = Bound.of(bounds, i, null);
            if (b != null) {
                double xi = x[i];

                // Check lower bound (bt <= BOUND_BOTH means has lower bound)
                if (b.hasLower() && xi <= b.getLower()) {
                    if (xi < b.getLower()) {
                        projected = true;
                        x[i] = b.getLower(); // proj xᵢ = lᵢ
                    }
                    numBnd++;
                    continue; // Skip upper bound check if at lower bound
                }

                // Check upper bound (bt >= BOUND_BOTH means has upper bound)
                if (b.hasUpper() && xi >= b.getUpper()) {
                    if (xi > b.getUpper()) {
                        projected = true;
                        x[i] = b.getUpper(); // proj xᵢ = uᵢ
                    }
                    numBnd++;
                }
            }
        }

        /*
         * Second pass: Initialize where and determine problem characteristics
         *
         * Variable status assignment:
         *   - VAR_UNBOUND: No bounds on variable
         *   - VAR_FIXED:   Both bounds and uᵢ - lᵢ ≤ 0 (variable is fixed)
         *   - VAR_FREE:    Has bounds but free to move
         *
         * Problem characteristics:
         *   - boxed:       All variables have both bounds (BOUND_BOTH)
         *   - constrained: At least one variable has bounds
         */
        for (int i = 0; i < n; i++) {
            Bound b = Bound.of(bounds, i, Bound.unbounded());
            boxed &= b.hasBoth(); // Update boxed flag: true only if all variables have both bounds
            if (b.isUnbounded()) {
                iBuffer[whereOffset + i] = VAR_UNBOUND; // No bounds on this variable
            } else {
                constrained = true;
                if (b.hasBoth() && b.getUpper() - b.getLower() <= 0.0) {
                    iBuffer[whereOffset + i] = VAR_FIXED; // Variable is fixed: uᵢ - lᵢ ≤ 0
                } else {
                    iBuffer[whereOffset + i] = VAR_FREE; // Variable is free to move within bounds
                }
            }
        }

        // Set output flags
        ws.setConstrained(constrained);
        ws.setBoxed(boxed);
        return projected;
    }

    // ========================================================================
    // Core Optimization Loop (from lbfgsb.c)
    // ========================================================================

    /**
     * Result holder for optimization.
     */
    static final class OptimizeResult {
        /** Final function value */
        double f;
        /** Number of iterations performed */
        int iterations;
        /** Number of function evaluations */
        int evaluations;
        /** Termination status */
        int status;

        OptimizeResult(double f, int iterations, int evaluations, int status) {
            this.f = f;
            this.iterations = iterations;
            this.evaluations = evaluations;
            this.status = status;
        }
    }

    /**
     * Resets the BFGS approximation to identity matrix.
     *
     * <p>This is called when numerical issues occur during optimization:</p>
     * <ul>
     *   <li>Singular triangular system in Cauchy point computation</li>
     *   <li>Non-positive definite matrix in Cholesky factorization</li>
     *   <li>Line search failure with existing BFGS corrections</li>
     * </ul>
     *
     * <p>The reset clears the correction history and restarts with steepest descent.
     * Recovery is limited to MAX_BFGS_RESETS (5) per optimization run to prevent
     * infinite loops in pathological cases.</p>
     *
     * <p>Reference: C function reset_bfgs_to_identity() in lbfgsb.c</p>
     * <p>Reference: Go function iterBFGS.reset() in base.go</p>
     *
     * @param ws Workspace containing BFGS state
     * @return 0 on success, ERR_TOO_MANY_RESETS if reset limit exceeded
     *
     */
    private static int resetBfgsToIdentity(LBFGSBWorkspace ws) {
        // Increment reset counter
        ws.incrementResetCount();

        // Check if we've exceeded the maximum number of resets
        if (ws.getResetCount() > MAX_BFGS_RESETS) {
            return ERR_TOO_MANY_RESETS;
        }

        // Reset BFGS approximation to identity matrix
        // This is done by clearing the correction history
        ws.resetBfgs();

        return ERR_NONE;
    }

    /**
     * Main L-BFGS-B optimization loop.
     *
     * <p>This function implements the main L-BFGS-B iteration loop, matching the
     * C implementation in lbfgsb.c and Go implementation in driver.go.</p>
     *
     * <h3>Iteration Flow:</h3>
     * <ol>
     *   <li><b>INITIALIZATION</b>
     *     <ul>
     *       <li>Project x₀ onto feasible region [l, u]</li>
     *       <li>Compute f₀ = f(x₀) and g₀ = ∇f(x₀)</li>
     *       <li>Check initial convergence: ‖proj(g₀)‖∞ ≤ pgtol</li>
     *     </ul>
     *   </li>
     *   <li><b>MAIN LOOP</b> (for k = 0, 1, 2, ...)
     *     <ul>
     *       <li>Step 1: Search GCP (Generalized Cauchy Point)</li>
     *       <li>Step 2: Minimize subspace (if free &gt; 0 and col &gt; 0)</li>
     *       <li>Step 3: Line search</li>
     *       <li>Step 4: New iteration checks (limits)</li>
     *       <li>Step 5: Convergence check</li>
     *       <li>Step 6: BFGS update</li>
     *     </ul>
     *   </li>
     *   <li><b>TERMINATION</b>: Return final x, f(x), and convergence status</li>
     * </ol>
     *
     * <h3>Convergence Criteria:</h3>
     * <ol>
     *   <li>Projected gradient norm:
     *     <pre>‖proj gₖ‖∞ ≤ pgtol</pre>
     *   </li>
     *   <li>Relative function value reduction:
     *     <pre>(fₖ - fₖ₊₁) / max(|fₖ|, |fₖ₊₁|, 1) ≤ factr × εₘₐ꜀ₕ</pre>
     *   </li>
     *   <li>Gradient descent threshold (step too small):
     *     <pre>‖dₖ‖₂ ≤ pgtol × (1 + |fₖ|)</pre>
     *   </li>
     * </ol>
     *
     * <h3>Error Recovery:</h3>
     * <p>When numerical issues occur (info != 0), the algorithm attempts to
     * recover by resetting the BFGS approximation to identity (Bₖ = I).</p>
     *
     * <p>Reference: C function lbfgsb_optimize() in lbfgsb.c</p>
     * <p>Reference: Go function mainLoop() in driver.go</p>
     *
     * @param n Problem dimension
     * @param m Number of L-BFGS corrections
     * @param x Initial point (modified: becomes final solution)
     * @param objective Objective function
     * @param bounds Array of Bounds for each variable. Null means unbounded.
     * @param factr Function value tolerance factor for criterion (fₖ - fₖ₊₁)/max(|fₖ|,|fₖ₊₁|,1) ≤ factr × εₘₐ꜀ₕ
     * @param pgtol Projected gradient tolerance for criterion ‖proj gₖ‖∞ ≤ pgtol
     * @param maxIter Maximum iterations
     * @param maxEval Maximum function evaluations
     * @param maxTime Maximum time in microseconds (0 for no limit)
     * @param maxLsStp Maximum backtracking steps per line search (matches scipy maxls)
     * @param ws Workspace
     * @return Optimization result
     *
     */
    static OptimizeResult optimize(
            int n, int m,
            double[] x,
            Univariate objective,
            Bound[] bounds,
            double factr, double pgtol,
            int maxIter, int maxEval, long maxTime,
            int maxLsStp,
            LBFGSBWorkspace ws) {

        // Validate inputs
        if (x == null || x.length != n) {
            throw new IllegalArgumentException("x must have dimension " + n);
        }

        // Reset workspace for new optimization
        ws.reset();

        // Get workspace arrays
        double[] buffer = ws.getBuffer();
        int[] iBuffer = ws.getIntBuffer();

        int gOffset = ws.getGOffset();
        int zOffset = ws.getZOffset();
        int dOffset = ws.getDOffset();
        int tOffset = ws.getTOffset();
        int rOffset = ws.getROffset();
        int indexOffset = ws.getIndexOffset();

        // Use workspace buffer for gradient (avoid allocation in hot path)
        // g is stored at gOffset in the buffer, but we need a local reference for objective.evaluate()
        double[] g = new double[n];  // This allocation happens once per optimize() call, not per iteration

        // Track start time if time limit is set
        long startTimeNs = (maxTime > 0) ? System.nanoTime() : 0;

        // ====================================================================
        // INITIALIZATION
        // ====================================================================

        // Project initial point and initialize variable status
        projInitActive(n, x, bounds, ws);

        // Initialize free variable count (all variables start as potentially free)
        ws.setFree(n);
        ws.setActive(0);
        ws.setEnter(0);
        ws.setLeave(n);
        ws.setUpdated(false);
        ws.setUpdates(0);

        // Initialize index array: all variables are initially free
        for (int i = 0; i < n; i++) {
            iBuffer[indexOffset + i] = i;
        }

        // Calculate f₀ and g₀ (first function evaluation)
        double f;
        try {
            f = objective.evaluate(x, g);
            ws.incrementTotalEval();
        } catch (Exception e) {
            return new OptimizeResult(Double.NaN, 0, ws.getTotalEval(), HALT_EVAL_PANIC);
        }

        // Check for NaN or Infinity
        if (Double.isNaN(f) || Double.isInfinite(f)) {
            return new OptimizeResult(f, 0, ws.getTotalEval(), HALT_EVAL_PANIC);
        }

        ws.setF(f);

        // Copy gradient to workspace
        System.arraycopy(g, 0, buffer, gOffset, n);

        // Compute projected gradient norm
        double sbgNorm = projGradNorm(n, x, g, bounds);
        ws.setSbgNorm(sbgNorm);

        // Check initial convergence
        if (sbgNorm <= pgtol) {
            return new OptimizeResult(f, 0, ws.getTotalEval(), CONV_GRAD_PROG_NORM);
        }

        // ====================================================================
        // MAIN ITERATION LOOP
        // ====================================================================

        int info = ERR_NONE;  // Error info from subroutines
        int wrk = 0;          // Whether K matrix needs recomputation

        for (int iter = 0; iter < maxIter; iter++) {
            ws.setIter(iter);
            ws.setFOld(f);

            // ================================================================
            // Handle BFGS reset if needed
            // ================================================================
            if (info != ERR_NONE) {
                info = ERR_NONE;
                int resetStatus = resetBfgsToIdentity(ws);
                if (resetStatus != ERR_NONE) {
                    // Too many resets - give up
                    return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), STOP_ABNORMAL_SEARCH);
                }
                // Reset successful - continue with identity BFGS matrix
            }

            // ================================================================
            // Step 1: Search GCP (Generalized Cauchy Point)
            // ================================================================

            // Skip the search for GCP if unconstrained and col > 0
            if (!ws.isConstrained() && ws.getCol() > 0) {
                // Copy x to z directly
                dcopy(n, x, 0, 1, buffer, zOffset, 1);
                wrk = ws.isUpdated() ? 1 : 0;
                ws.setSeg(0);
            } else {
                // Compute the Generalized Cauchy Point (GCP)
                // CauchyPoint.compute now writes the result into the workspace z array
                int cauchyInfo = CauchyPoint.compute(n, m, x, g, bounds, ws);
                if (cauchyInfo != ERR_NONE) {
                    // Singular triangular system detected - try BFGS reset
                    info = cauchyInfo;
                    continue;
                }
                // Result is already stored in workspace buffer at zOffset

                // Count entering/leaving variables and build free variable index set
                wrk = CauchyPoint.freeVar(n, ws);
            }

            // ================================================================
            // Step 2: Minimize subspace
            // ================================================================

            ws.setWord(SOLUTION_UNKNOWN);

            // Only perform subspace minimization if:
            // - There are free variables (free > 0)
            // - There are BFGS corrections (col > 0)
            if (ws.getFree() > 0 && ws.getCol() > 0) {
                // Build K matrix if needed
                if (wrk != 0) {
                    int formkInfo = BfgsUpdater.formK(n, m, ws);
                    if (formkInfo != ERR_NONE) {
                        // Non-positive definite matrix - try BFGS reset
                        info = formkInfo;
                        continue;
                    }
                }

                // Compute reduced gradient (writes into workspace r)
                int rgInfo = SubspaceMinimizer.reduceGradient(n, m, x, g, ws);
                if (rgInfo != ERR_NONE) {
                    // Singular triangular matrix - try BFGS reset
                    info = rgInfo;
                    continue;
                }

                // Compute optimal direction (reads z/r from workspace and writes z into workspace)
                int odInfo = SubspaceMinimizer.optimalDirection(n, m, x, g, bounds, ws);
                if (odInfo != ERR_NONE) {
                    // Singular triangular matrix - try BFGS reset
                    info = odInfo;
                    continue;
                }
            }

            // ================================================================
            // Step 3: Line search
            // ================================================================

            // Compute search direction d = z - x
            dcopy(n, buffer, zOffset, 1, buffer, dOffset, 1);
            daxpy(n, -1.0, x, 0, 1, buffer, dOffset, 1);

            // Initialize line search
            LineSearch.init(n, x, bounds, ws);

            // Save original x, f, g
            dcopy(n, x, 0, 1, buffer, tOffset, 1);  // Save x to t
            ws.setFOld(f);                          // Save f
            dcopy(n, g, 0, 1, buffer, rOffset, 1);  // Save g to r

            // Line search loop
            boolean done = false;
            while (!done) {
                int[] lsResult = LineSearch.perform(n, x, f, g, ws);
                int lsInfo = lsResult[0];
                done = (lsResult[1] != 0);

                LBFGSBWorkspace.SearchCtx ctx = ws.getSearchCtx();
                if (lsInfo == ERR_NONE && ctx.numBack < maxLsStp) {
                    if (!done) {
                        // Need function evaluation at new point
                        try {
                            f = objective.evaluate(x, g);
                            ws.incrementTotalEval();
                            ctx.numEval++;
                            ctx.numBack = ctx.numEval - 1;
                        } catch (Exception e) {
                            // Restore previous iterate
                            dcopy(n, buffer, tOffset, 1, x, 0, 1);
                            f = ws.getFOld();
                            dcopy(n, buffer, rOffset, 1, g, 0, 1);
                            return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), HALT_EVAL_PANIC);
                        }

                        // Check for callback error
                        if (Double.isNaN(f) || Double.isInfinite(f)) {
                            // Restore previous iterate
                            dcopy(n, buffer, tOffset, 1, x, 0, 1);
                            f = ws.getFOld();
                            dcopy(n, buffer, rOffset, 1, g, 0, 1);
                            return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), HALT_EVAL_PANIC);
                        }
                    }
                    continue;
                }

                // Line search failed or too many backtracking steps
                if (ws.getCol() == 0) {
                    // No BFGS corrections - abnormal termination
                    dcopy(n, buffer, tOffset, 1, x, 0, 1);
                    f = ws.getFOld();
                    dcopy(n, buffer, rOffset, 1, g, 0, 1);
                    return new OptimizeResult(f, ws.getIter() + 1, ws.getTotalEval(), STOP_ABNORMAL_SEARCH);
                } else {
                    // Try BFGS reset
                    info = ERR_LINE_SEARCH_FAILED;
                }
                break;
            }

            if (!done) {
                // Restore the previous iterate
                dcopy(n, buffer, tOffset, 1, x, 0, 1);
                f = ws.getFOld();
                dcopy(n, buffer, rOffset, 1, g, 0, 1);

                if (info != ERR_NONE) {
                    continue;  // Will reset BFGS at start of next iteration
                }
            }

            ws.setF(f);

            // ================================================================
            // Step 4: New iteration checks
            // ================================================================

            ws.setIter(ws.getIter() + 1);

            // Check iteration limit
            if (ws.getIter() > maxIter) {
                return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), OVER_ITER_LIMIT);
            }

            // Check evaluation limit
            if (ws.getTotalEval() >= maxEval) {
                return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), OVER_EVAL_LIMIT);
            }

            // Check time limit
            if (maxTime > 0) {
                long elapsedNs = System.nanoTime() - startTimeNs;
                if (elapsedNs >= maxTime * 1000) {  // Convert microseconds to nanoseconds
                    return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), OVER_TIME_LIMIT);
                }
            }

            // Compute d_norm for gradient descent threshold check
            ws.setDNorm(dnrm2(n, buffer, dOffset, 1));

            // Check gradient descent threshold
            if (ws.getDNorm() <= pgtol * (1.0 + Math.abs(f))) {
                // If the step length is extremely small
                // we may be close to optimality — prefer reporting projected
                // gradient convergence when that is the case. This avoids spuriously
                // classifying tiny-step situations as OVER_GRAD_THRESH when the
                // projected gradient is already below pgtol (which the property
                // tests expect for simple quadratics).
                double sbg = projGradNorm(n, x, g, bounds);
                ws.setSbgNorm(sbg);
                if (sbg <= pgtol) {
                    return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), CONV_GRAD_PROG_NORM);
                }
                return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), OVER_GRAD_THRESH);
            }

            // ================================================================
            // Step 5: Convergence check
            // ================================================================

            // Compute projected gradient norm
            sbgNorm = projGradNorm(n, x, g, bounds);
            ws.setSbgNorm(sbgNorm);

            // Check projected gradient tolerance
            if (sbgNorm <= pgtol) {
                return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), CONV_GRAD_PROG_NORM);
            }

            // Check function value convergence
            if (ws.getIter() > 0) {
                double tolEps = EPS * factr;
                double change = Math.max(Math.abs(ws.getFOld()), Math.max(Math.abs(f), 1.0));
                double fDiff = ws.getFOld() - f;
                // Only treat as "enough accuracy" when there was a non-negative
                // improvement (f decreased). If f increased (fDiff < 0), do not
                // report function-value convergence — increases are not convergence
                // and can occur due to bound projections or extrapolated samples.
                if (fDiff >= 0.0 && fDiff <= tolEps * change) {
                    return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), CONV_ENOUGH_ACCURACY);
                }
            }

            // ================================================================
            // Step 6: BFGS update
            // ================================================================

            // Update BFGS matrices with new (s, y) pair
            BfgsUpdater.updateCorrection(n, m, g, ws);

            // Form T matrix if update was performed
            if (ws.isUpdated()) {
                int formtInfo = BfgsUpdater.formT(m, ws);
                if (formtInfo != ERR_NONE) {
                    // form_t failed - T matrix is not positive definite
                    // Try to recover by resetting BFGS approximation
                    info = formtInfo;
                    // Continue to next iteration which will reset BFGS
                }
            }

            // Copy gradient to workspace for next iteration
            System.arraycopy(g, 0, buffer, gOffset, n);
        }

        // Maximum iterations reached
        return new OptimizeResult(f, ws.getIter(), ws.getTotalEval(), OVER_ITER_LIMIT);
    }

}
