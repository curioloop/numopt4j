/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * SLSQP (Sequential Least Squares Programming) algorithm constants.
 *
 * This file collects all numerical constants, mode flags, and error codes
 * used throughout the SLSQP implementation, grouped into four categories:
 *
 * ── Numerical Constants ────────────────────────────────────────────────────
 *
 *   EPS      = machine epsilon (DBL_EPSILON)
 *   SQRT_EPS = √EPS  — used as the relative tolerance in Brent's convergence test
 *              tol₁ = √ε·|x| + tol,  tol₂ = 2·tol₁
 *   INF_BND  = 10²⁰  — sentinel for unbounded variables
 *
 * ── Line Search Constants ──────────────────────────────────────────────────
 *
 *   ARMIJO_ETA = η = 0.1  — Armijo sufficient-decrease parameter:
 *     𝞥(𝐱ᵏ+𝛂𝐝;𝛌,𝛒) - 𝞥(𝐱ᵏ;𝛌,𝛒) < η · 𝛂 · 𝜵𝞥(𝐝;𝐱ᵏ,𝛒ᵏ)
 *
 *   INV_PHI2 = 1/φ² ≈ 0.38197  — golden section ratio used in Brent's method:
 *     initial point x₀ = a + (1/φ²)·(b - a),  golden step d = (1/φ²)·e
 *
 * ── LSQ / LSEI Mode Constants ─────────────────────────────────────────────
 *
 *   MODE_OK / MODE_HAS_SOLUTION = 0  — QP sub-problem solved successfully
 *   MODE_CONS_INCOMPAT          = -2 — constraints inconsistent → augmented QP
 *   MODE_LSEI_SINGULAR          = -4 — equality constraint matrix C is singular
 *
 * ── Exact Line Search Modes (FindMode) ────────────────────────────────────
 *
 *   FIND_NOOP = 0  — initial state, triggers interval setup
 *   FIND_INIT = 1  — first evaluation point returned, awaiting f(x₀)
 *   FIND_NEXT = 2  — main loop, awaiting f(u) at next trial point
 *   FIND_CONV = 3  — convergence reached, best point is x
 */
package com.curioloop.numopt4j.optim.slsqp;

/**
 * Constants used in the SLSQP optimization algorithm.
 * <p>
 * This class defines numerical constants, line search parameters, LSQ mode constants,
 * exact line search modes, error codes, and optimizer status values.
 * </p>
 *
 * @see <a href="https://github.com/curioloop/numopt4j">numopt4j</a>
 */
public final class SLSQPConstants {

    private SLSQPConstants() {
        // Prevent instantiation
    }

    // ========================================================================
    // Numerical Constants
    // ========================================================================

    /**
     * Machine epsilon - the smallest positive number such that 1.0 + EPS != 1.0.
     * This is equivalent to DBL_EPSILON in C or Math.ulp(1.0) in Java.
     */
    public static final double EPS = Math.ulp(1.0);

    /**
     * Square root of machine epsilon.
     * Used for tolerance calculations in numerical algorithms.
     */
    public static final double SQRT_EPS = Math.sqrt(EPS);

    /**
     * Infinite bound value.
     * Used to represent unbounded variables in optimization problems.
     */
    public static final double INF_BND = 1e20;

    // ========================================================================
    // Line Search Constants
    // ========================================================================

    /**
     * Armijo sufficient-decrease parameter η = 0.1.
     * <p>
     * The inexact line search accepts step length 𝛂 when:
     * </p>
     * <pre>
     *   𝞥(𝐱ᵏ + 𝛂𝐝; 𝛌, 𝛒) - 𝞥(𝐱ᵏ; 𝛌, 𝛒) &lt; η · 𝛂 · 𝜵𝞥(𝐝; 𝐱ᵏ, 𝛒ᵏ)
     * </pre>
     * <p>
     * where 𝞥(𝐱;𝛒) is the augmented Lagrangian merit function and
     * 𝜵𝞥(𝐝;𝐱ᵏ,𝛒ᵏ) = 𝜵𝒇(𝐱ᵏ)ᵀ𝐝 - (1-𝛅)∑𝛒ᵏⱼ‖𝒄ⱼ(𝐱ᵏ)‖₁ is the directional derivative.
     * In the implementation h3 = η·𝛂·𝜵𝞥 is passed directly, so the condition
     * becomes h1 = t - t0 ≤ h3/10.
     * </p>
     */
    public static final double ARMIJO_ETA = 0.1;

    /**
     * Inverse of the golden ratio squared: 1/φ² where φ = (1 + √5)/2 ≈ 1.61803.
     * <p>
     * Value: 1/φ² = (3 - √5)/2 ≈ 0.38197.
     * </p>
     * <p>
     * Used in Brent's exact line search in two places:
     * </p>
     * <ul>
     *   <li>Initial evaluation point: x₀ = a + (1/φ²)·(b - a)</li>
     *   <li>Golden section step: d = (1/φ²)·e  where e = b - x or a - x</li>
     * </ul>
     * <p>
     * The golden section ratio guarantees that each step reduces the interval
     * by the same fraction regardless of where the minimum lies.
     * </p>
     */
    public static final double INV_PHI2;

    static {
        // Compute 1/φ² where φ = (1 + √5)/2 (golden ratio)
        double phi = (1.0 + Math.sqrt(5.0)) / 2.0;
        INV_PHI2 = 1.0 / (phi * phi);
    }

    // ========================================================================
    // LSQ Mode Constants
    // ========================================================================

    /**
     * LSQ mode: OK / successful.
     * The QP subproblem was solved successfully.
     */
    public static final int MODE_OK = 0;

    /**
     * LSQ mode: has solution.
     * Equivalent to MODE_OK, indicates a valid solution was found.
     */
    public static final int MODE_HAS_SOLUTION = 0;

    /**
     * LSQ mode: constraints incompatible.
     * The constraints in the QP subproblem are inconsistent.
     * This triggers augmented QP relaxation with slack variable.
     */
    public static final int MODE_CONS_INCOMPAT = -2;

    /**
     * LSQ mode: LSEI singular.
     * The equality constraint matrix C in LSEI is singular.
     */
    public static final int MODE_LSEI_SINGULAR = -4;

    // ========================================================================
    // Exact Line Search Modes (FindMode)
    // ========================================================================

    /**
     * Find mode: no operation / initialization.
     * Initial state before line search begins.
     */
    public static final int FIND_NOOP = 0;

    /**
     * Find mode: initialization.
     * Process initial function value and return next evaluation point.
     */
    public static final int FIND_INIT = 1;

    /**
     * Find mode: next iteration.
     * Process function value and return next evaluation point.
     */
    public static final int FIND_NEXT = 2;

    /**
     * Find mode: converged.
     * Line search has converged to the minimum.
     */
    public static final int FIND_CONV = 3;

    // ========================================================================
    // Error Codes
    // ========================================================================

    /**
     * Error code: no error.
     * Operation completed successfully.
     */
    public static final int ERR_NONE = 0;

    /**
     * Error code: constraints incompatible.
     * The constraints are inconsistent and cannot be satisfied.
     */
    public static final int ERR_CONS_INCOMPAT = -2;

    /**
     * Error code: LSI singular E matrix.
     * The matrix E in LSI (Least Squares with Inequality constraints) is singular.
     */
    public static final int ERR_LSI_SINGULAR_E = -3;

    /**
     * Error code: LSEI singular C matrix.
     * The equality constraint matrix C in LSEI is singular.
     */
    public static final int ERR_LSEI_SINGULAR_C = -4;

    /**
     * Error code: HFTI rank defect.
     * The matrix in HFTI (Householder Forward Triangulation with column Interchanges)
     * is rank deficient beyond the specified tolerance.
     */
    public static final int ERR_HFTI_RANK_DEFECT = -5;

    /**
     * Error code: NNLS maximum iterations exceeded.
     * The NNLS (Non-Negative Least Squares) solver exceeded the maximum
     * number of iterations without converging.
     */
    public static final int ERR_NNLS_MAX_ITER = 1;
}
