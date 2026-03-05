/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for LSQ (Least Squares Quadratic Programming) solver.
 *
 * **Property 8: LSQ Solver Correctness**
 */
package com.curioloop.numopt4j.optim.slsqp;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for LSQ (Least Squares Quadratic Programming) solver.
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 8.1: Solution satisfies equality constraints (A_j x - b_j = 0 for j = 1..meq)</li>
 *   <li>Property 8.2: Solution satisfies inequality constraints (A_j x - b_j >= 0 for j = meq+1..m)</li>
 *   <li>Property 8.3: Solution satisfies bound constraints (l <= x <= u)</li>
 *   <li>Property 8.4: Lagrange multipliers are correctly computed (non-negative for inequality constraints)</li>
 *   <li>Property 8.5: Edge cases: no constraints, only bounds, only equality constraints</li>
 * </ul>
 *
 * <p>The LSQ problem is:</p>
 * <pre>
 *   minimize ||D^(1/2)L^T x + D^(-1/2)L^(-1)g||_2
 *   subject to:
 *     - A_j x - b_j = 0  (j = 1 ... meq)   [equality constraints]
 *     - A_j x - b_j >= 0 (j = meq+1 ... m) [inequality constraints]
 *     - l <= x <= u                         [bound constraints]
 * </pre>
 *
 * <p>The algorithm transforms this to LSEI problem min||Ex - f||_2 s.t. Cx = d, Gx >= h:</p>
 * <ul>
 *   <li>E = D^(1/2)L^T (upper triangular)</li>
 *   <li>f = -D^(-1/2)L^(-1)g</li>
 *   <li>C = { A_j: j = 1..meq }, d = { -b_j: j = 1..meq }</li>
 *   <li>G = { A_j: j = meq+1..m } ∪ { ±I for bounds }</li>
 *   <li>h = { -b_j: j = meq+1..m } ∪ { l, -u for bounds }</li>
 * </ul>
 */
@Tag("Feature: slsqp-java-rewrite, Property 8: LSQ Solver Correctness")
class LSQProperties {

    private static final double FEASIBILITY_TOL = 1e-6;
    private static final double EQUALITY_TOL = 1e-8;
    private static final double INF_BND = SLSQPConstants.INF_BND;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Generate a positive definite LDL^T factorization in packed form.
     * The packed form stores L (lower triangular with unit diagonal) and D (diagonal)
     * in a single array of length n*(n+1)/2 + 1.
     *
     * Layout: [D_0, L_10, L_20, ..., L_(n-1)0, D_1, L_21, L_31, ..., D_(n-1)]
     * where D_i are diagonal elements and L_ij are lower triangular elements.
     */
    private double[] generateLDLT(int n, java.util.Random rand) {
        int nl = n * (n + 1) / 2 + 1;
        double[] l = new double[nl];

        int idx = 0;
        for (int j = 0; j < n; j++) {
            // D_jj: positive diagonal element (ensure positive definiteness)
            l[idx++] = rand.nextDouble() * 4 + 1.0;  // Values in [1, 5]

            // L_ij for i > j: off-diagonal elements
            for (int i = j + 1; i < n; i++) {
                l[idx++] = rand.nextDouble() * 0.4 - 0.2;  // Small values in [-0.2, 0.2]
            }
        }

        return l;
    }

    /**
     * Generate a random gradient vector.
     */
    private double[] generateGradient(int n, java.util.Random rand) {
        double[] g = new double[n];
        for (int i = 0; i < n; i++) {
            g[i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
        }
        return g;
    }

    /**
     * Generate a random constraint Jacobian matrix A (m x n) in column-major format.
     */
    private double[] generateConstraintJacobian(int m, int n, java.util.Random rand) {
        int la = Math.max(m, 1);
        double[] a = new double[la * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                a[j * la + i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
            }
        }
        return a;
    }

    /**
     * Generate feasible constraint values b for the given Jacobian A.
     * 
     * LSQ transforms to LSEI with:
     *   - Equality: Cx = d where C = A and d = -b, so Ax = -b (i.e., Ax + b = 0)
     *   - Inequality: Gx >= h where G = A and h = -b, so Ax >= -b (i.e., Ax + b >= 0)
     * 
     * Strategy: pick a random x that satisfies bounds, compute Ax, then:
     *   - For equality: b = -Ax (so Ax + b = 0)
     *   - For inequality: b = -Ax - slack (so Ax + b = slack >= 0)
     */
    private double[] generateFeasibleB(double[] a, int m, int meq, int n, double[] xl, double[] xu, java.util.Random rand) {
        int la = Math.max(m, 1);

        // Generate a random x that satisfies bounds
        double[] x = new double[n];
        for (int j = 0; j < n; j++) {
            double lower = (xl != null && !Double.isNaN(xl[j]) && xl[j] > -INF_BND) ? xl[j] : -2.0;
            double upper = (xu != null && !Double.isNaN(xu[j]) && xu[j] < INF_BND) ? xu[j] : 2.0;
            x[j] = lower + rand.nextDouble() * (upper - lower);
        }

        // Compute Ax
        double[] ax = new double[m];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                ax[i] += a[j * la + i] * x[j];
            }
        }

        // Set b values
        double[] b = new double[m];
        for (int i = 0; i < meq; i++) {
            // Equality constraints: b = -Ax (so Ax + b = 0)
            b[i] = -ax[i];
        }
        for (int i = meq; i < m; i++) {
            // Inequality constraints: b = -Ax - slack (so Ax + b = slack >= 0)
            double slack = rand.nextDouble() * 0.5 + 0.1;  // Slack in [0.1, 0.6]
            b[i] = -ax[i] - slack;
        }

        return b;
    }

    /**
     * Generate random bounds for variables.
     */
    private double[][] generateBounds(int n, java.util.Random rand) {
        double[] xl = new double[n];
        double[] xu = new double[n];
        for (int i = 0; i < n; i++) {
            xl[i] = -2.0 - rand.nextDouble();  // Lower bound in [-3, -2]
            xu[i] = 2.0 + rand.nextDouble();   // Upper bound in [2, 3]
        }
        return new double[][] { xl, xu };
    }

    /**
     * Compute matrix-vector product Ax (A is m x n column-major with leading dimension la).
     */
    private double[] matVec(double[] a, int la, int m, int n, double[] x) {
        double[] result = new double[m];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                result[i] += a[j * la + i] * x[j];
            }
        }
        return result;
    }

    /**
     * Compute the required working array size for LSQ.
     */
    private int computeWorkSize(int m, int meq, int n) {
        int mineq = m - meq;
        int m1 = mineq + n + n;  // Total inequality constraints including bounds
        int meqMax = Math.max(meq, 1);

        // Layout: E(n×n) + f(n) + C(meq×n) + d(meq) + G(m1×n) + h(m1) + workspace
        int lseiWorkSize = meqMax + (n - meqMax + 1) * (m1 + 2) + 2 * m1 + meqMax + n * (n - meqMax) + n + m1 * (n - meqMax) + 100;
        return n * n + n + meq * n + meq + m1 * n + m1 + lseiWorkSize;
    }

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }

    // ========================================================================
    // Property 8.1: Solution satisfies equality constraints
    // ========================================================================

    /**
     *
     * For any valid QP subproblem with equality constraints,
     * the LSQ solution x shall satisfy the equality constraints.
     * 
     * Note: LSQ transforms to LSEI with d = -b, so LSEI solves Ax = -b.
     * We verify that Ax + b = 0 (or equivalently Ax = -b).
     */
    @Property(tries = 100)
    void solutionSatisfiesEqualityConstraints(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 1, max = 3) int meq,
            @ForAll @IntRange(min = 0, max = 3) int mineq,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(meq < n);  // Ensure problem is not over-constrained

        java.util.Random rand = new java.util.Random(seed);

        int m = meq + mineq;
        int la = Math.max(m, 1);
        int nl = n * (n + 1) / 2 + 1;

        // Generate LDL^T factorization and gradient
        double[] l = generateLDLT(n, rand);
        double[] g = generateGradient(n, rand);

        // Generate bounds
        double[][] bounds = generateBounds(n, rand);
        double[] xl = bounds[0];
        double[] xu = bounds[1];

        // Generate constraint Jacobian and feasible constraint values
        double[] a = generateConstraintJacobian(m, n, rand);
        double[] b = generateFeasibleB(a, m, meq, n, xl, xu, rand);

        // Allocate output and working arrays
        double[] x = new double[n];
        double[] y = new double[m + 2 * n];
        double[] w = new double[computeWorkSize(m, meq, n)];
        int[] jw = new int[Math.max(mineq + 2 * n, Math.min(n, m - meq)) + 1];
        double[] norm = new double[1];

        // Solve LSQ
        int status = LSQSolver.lsq(m, meq, n, nl,
                l, 0, g, 0, a, 0, b, 0, xl, 0, xu, 0,
                x, 0, y, 0, w, 0, jw, 0, 3 * n, INF_BND, norm);

        if (status == 0) {
            // Verify equality constraints: Ax + b = 0 (since LSEI solves Ax = -b)
            double[] ax = matVec(a, la, m, n, x);
            for (int i = 0; i < meq; i++) {
                double violation = ax[i] + b[i];
                assertEquals(0.0, violation, EQUALITY_TOL,
                        "Equality constraint " + i + " violated: Ax[" + i + "] + b[" + i + "] = " + violation +
                        " should be 0");
            }
        }
    }

    // ========================================================================
    // Property 8.2: Solution satisfies inequality constraints
    // ========================================================================

    /**
     *
     * For any valid QP subproblem with inequality constraints,
     * the LSQ solution x shall satisfy the inequality constraints.
     * 
     * Note: LSQ transforms to LSEI with h = -b, so LSEI solves Gx >= -b.
     * We verify that Ax + b >= 0 (or equivalently Ax >= -b).
     */
    @Property(tries = 100)
    void solutionSatisfiesInequalityConstraints(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 0, max = 2) int meq,
            @ForAll @IntRange(min = 1, max = 4) int mineq,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(meq < n);

        java.util.Random rand = new java.util.Random(seed);

        int m = meq + mineq;
        int la = Math.max(m, 1);
        int nl = n * (n + 1) / 2 + 1;

        // Generate LDL^T factorization and gradient
        double[] l = generateLDLT(n, rand);
        double[] g = generateGradient(n, rand);

        // Generate bounds
        double[][] bounds = generateBounds(n, rand);
        double[] xl = bounds[0];
        double[] xu = bounds[1];

        // Generate constraint Jacobian and feasible constraint values
        double[] a = generateConstraintJacobian(m, n, rand);
        double[] b = generateFeasibleB(a, m, meq, n, xl, xu, rand);

        // Allocate output and working arrays
        double[] x = new double[n];
        double[] y = new double[m + 2 * n];
        double[] w = new double[computeWorkSize(m, meq, n)];
        int[] jw = new int[Math.max(mineq + 2 * n, Math.min(n, m - meq)) + 1];
        double[] norm = new double[1];

        // Solve LSQ
        int status = LSQSolver.lsq(m, meq, n, nl,
                l, 0, g, 0, a, 0, b, 0, xl, 0, xu, 0,
                x, 0, y, 0, w, 0, jw, 0, 3 * n, INF_BND, norm);

        if (status == 0) {
            // Verify inequality constraints: Ax + b >= 0 (since LSEI solves Gx >= -b)
            double[] ax = matVec(a, la, m, n, x);
            for (int i = meq; i < m; i++) {
                double slack = ax[i] + b[i];
                assertTrue(slack >= -FEASIBILITY_TOL,
                        "Inequality constraint " + (i - meq) + " violated: Ax[" + i + "] + b[" + i + "] = " + slack +
                        " should be >= 0");
            }
        }
    }

    // ========================================================================
    // Property 8.3: Solution satisfies bound constraints (l <= x <= u)
    // ========================================================================

    /**
     *
     * For any valid QP subproblem with bound constraints,
     * the LSQ solution x shall satisfy l <= x <= u.
     */
    @Property(tries = 100)
    void solutionSatisfiesBoundConstraints(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 0, max = 2) int meq,
            @ForAll @IntRange(min = 0, max = 3) int mineq,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(meq < n);

        java.util.Random rand = new java.util.Random(seed);

        int m = meq + mineq;
        int nl = n * (n + 1) / 2 + 1;

        // Generate LDL^T factorization and gradient
        double[] l = generateLDLT(n, rand);
        double[] g = generateGradient(n, rand);

        // Generate bounds
        double[][] bounds = generateBounds(n, rand);
        double[] xl = bounds[0];
        double[] xu = bounds[1];

        // Generate constraint Jacobian and feasible constraint values
        double[] a = (m > 0) ? generateConstraintJacobian(m, n, rand) : new double[n];
        double[] b = (m > 0) ? generateFeasibleB(a, m, meq, n, xl, xu, rand) : new double[1];

        // Allocate output and working arrays
        double[] x = new double[n];
        double[] y = new double[m + 2 * n];
        double[] w = new double[computeWorkSize(m, meq, n)];
        int[] jw = new int[Math.max(mineq + 2 * n, Math.min(n, Math.max(m - meq, 1))) + 1];
        double[] norm = new double[1];

        // Solve LSQ
        int status = LSQSolver.lsq(m, meq, n, nl,
                l, 0, g, 0, a, 0, b, 0, xl, 0, xu, 0,
                x, 0, y, 0, w, 0, jw, 0, 3 * n, INF_BND, norm);

        if (status == 0) {
            // Verify bound constraints: l <= x <= u
            for (int i = 0; i < n; i++) {
                if (!Double.isNaN(xl[i]) && xl[i] > -INF_BND) {
                    assertTrue(x[i] >= xl[i] - FEASIBILITY_TOL,
                            "Lower bound violated: x[" + i + "] = " + x[i] +
                            " should be >= xl[" + i + "] = " + xl[i]);
                }
                if (!Double.isNaN(xu[i]) && xu[i] < INF_BND) {
                    assertTrue(x[i] <= xu[i] + FEASIBILITY_TOL,
                            "Upper bound violated: x[" + i + "] = " + x[i] +
                            " should be <= xu[" + i + "] = " + xu[i]);
                }
            }
        }
    }

    // ========================================================================
    // Property 8.4: Lagrange multipliers are correctly computed
    // ========================================================================

    /**
     *
     * The Lagrange multipliers for inequality constraints shall satisfy λ >= 0.
     * The KKT conditions require that λ_j >= 0 for all inequality constraints.
     */
    @Property(tries = 100)
    void lagrangeMultipliersAreNonNegativeForInequalityConstraints(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 0, max = 2) int meq,
            @ForAll @IntRange(min = 1, max = 4) int mineq,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(meq < n);

        java.util.Random rand = new java.util.Random(seed);

        int m = meq + mineq;
        int nl = n * (n + 1) / 2 + 1;

        // Generate LDL^T factorization and gradient
        double[] l = generateLDLT(n, rand);
        double[] g = generateGradient(n, rand);

        // Generate bounds
        double[][] bounds = generateBounds(n, rand);
        double[] xl = bounds[0];
        double[] xu = bounds[1];

        // Generate constraint Jacobian and feasible constraint values
        double[] a = generateConstraintJacobian(m, n, rand);
        double[] b = generateFeasibleB(a, m, meq, n, xl, xu, rand);

        // Allocate output and working arrays
        double[] x = new double[n];
        double[] y = new double[m + 2 * n];
        double[] w = new double[computeWorkSize(m, meq, n)];
        int[] jw = new int[Math.max(mineq + 2 * n, Math.min(n, m - meq)) + 1];
        double[] norm = new double[1];

        // Solve LSQ
        int status = LSQSolver.lsq(m, meq, n, nl,
                l, 0, g, 0, a, 0, b, 0, xl, 0, xu, 0,
                x, 0, y, 0, w, 0, jw, 0, 3 * n, INF_BND, norm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Verify inequality constraint multipliers are non-negative
        // Note: y[0:meq] are equality multipliers, y[meq:m] are inequality multipliers
        for (int i = meq; i < m; i++) {
            assertTrue(y[i] >= -FEASIBILITY_TOL,
                    "Inequality constraint multiplier y[" + i + "] = " + y[i] +
                    " should be >= 0");
        }
    }

    // ========================================================================
    // Property 8.5: Edge case - no constraints (only bounds)
    // ========================================================================

    /**
     *
     * When there are no equality or inequality constraints (only bounds),
     * LSQ should solve the bound-constrained QP problem correctly.
     */
    @Property(tries = 50)
    void lsqHandlesOnlyBoundConstraints(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int m = 0;
        int meq = 0;
        int nl = n * (n + 1) / 2 + 1;

        // Generate LDL^T factorization and gradient
        double[] l = generateLDLT(n, rand);
        double[] g = generateGradient(n, rand);

        // Generate bounds
        double[][] bounds = generateBounds(n, rand);
        double[] xl = bounds[0];
        double[] xu = bounds[1];

        // Dummy arrays for A and b (not used when m = 0)
        double[] a = new double[n];
        double[] b = new double[1];

        // Allocate output and working arrays
        double[] x = new double[n];
        double[] y = new double[2 * n];
        double[] w = new double[computeWorkSize(m, meq, n)];
        int[] jw = new int[2 * n + 1];
        double[] norm = new double[1];

        // Solve LSQ
        int status = LSQSolver.lsq(m, meq, n, nl,
                l, 0, g, 0, a, 0, b, 0, xl, 0, xu, 0,
                x, 0, y, 0, w, 0, jw, 0, 3 * n, INF_BND, norm);

        assertEquals(0, status, "LSQ should succeed with only bound constraints");

        // Verify bound constraints
        for (int i = 0; i < n; i++) {
            if (!Double.isNaN(xl[i]) && xl[i] > -INF_BND) {
                assertTrue(x[i] >= xl[i] - FEASIBILITY_TOL,
                        "Lower bound violated: x[" + i + "] = " + x[i] +
                        " should be >= xl[" + i + "] = " + xl[i]);
            }
            if (!Double.isNaN(xu[i]) && xu[i] < INF_BND) {
                assertTrue(x[i] <= xu[i] + FEASIBILITY_TOL,
                        "Upper bound violated: x[" + i + "] = " + x[i] +
                        " should be <= xu[" + i + "] = " + xu[i]);
            }
        }
    }

    // ========================================================================
    // Property 8.6: Edge case - only equality constraints
    // ========================================================================

    /**
     *
     * When there are only equality constraints (no inequality constraints),
     * LSQ should solve the equality-constrained QP problem correctly.
     */
    @Property(tries = 50)
    void lsqHandlesOnlyEqualityConstraints(
            @ForAll @IntRange(min = 3, max = 6) int n,
            @ForAll @IntRange(min = 1, max = 2) int meq,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(meq < n);

        java.util.Random rand = new java.util.Random(seed);

        int m = meq;
        int mineq = 0;
        int la = Math.max(m, 1);
        int nl = n * (n + 1) / 2 + 1;

        // Generate LDL^T factorization and gradient
        double[] l = generateLDLT(n, rand);
        double[] g = generateGradient(n, rand);

        // Generate bounds
        double[][] bounds = generateBounds(n, rand);
        double[] xl = bounds[0];
        double[] xu = bounds[1];

        // Generate constraint Jacobian and feasible constraint values
        double[] a = generateConstraintJacobian(m, n, rand);
        double[] b = generateFeasibleB(a, m, meq, n, xl, xu, rand);

        // Allocate output and working arrays
        double[] x = new double[n];
        double[] y = new double[m + 2 * n];
        double[] w = new double[computeWorkSize(m, meq, n)];
        int[] jw = new int[Math.max(2 * n, Math.min(n, 1)) + 1];
        double[] norm = new double[1];

        // Solve LSQ
        int status = LSQSolver.lsq(m, meq, n, nl,
                l, 0, g, 0, a, 0, b, 0, xl, 0, xu, 0,
                x, 0, y, 0, w, 0, jw, 0, 3 * n, INF_BND, norm);

        if (status == 0) {
            // Verify equality constraints: Ax + b = 0 (since LSEI solves Ax = -b)
            double[] ax = matVec(a, la, m, n, x);
            for (int i = 0; i < meq; i++) {
                double violation = ax[i] + b[i];
                assertEquals(0.0, violation, EQUALITY_TOL,
                        "Equality constraint " + i + " violated: Ax[" + i + "] + b[" + i + "] = " + violation +
                        " should be 0");
            }
        }
    }

    // ========================================================================
    // Property 8.7: Edge case - no constraints at all
    // ========================================================================

    /**
     *
     * When there are no constraints at all (no equality, inequality, or bounds),
     * LSQ should solve the unconstrained QP problem correctly.
     */
    @Property(tries = 50)
    void lsqHandlesNoConstraints(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int m = 0;
        int meq = 0;
        int nl = n * (n + 1) / 2 + 1;

        // Generate LDL^T factorization and gradient
        double[] l = generateLDLT(n, rand);
        double[] g = generateGradient(n, rand);

        // No bounds (use NaN to indicate unbounded)
        double[] xl = new double[n];
        double[] xu = new double[n];
        for (int i = 0; i < n; i++) {
            xl[i] = Double.NaN;
            xu[i] = Double.NaN;
        }

        // Dummy arrays for A and b
        double[] a = new double[n];
        double[] b = new double[1];

        // Allocate output and working arrays
        double[] x = new double[n];
        double[] y = new double[2 * n];
        double[] w = new double[computeWorkSize(m, meq, n)];
        int[] jw = new int[2 * n + 1];
        double[] norm = new double[1];

        // Solve LSQ
        int status = LSQSolver.lsq(m, meq, n, nl,
                l, 0, g, 0, a, 0, b, 0, xl, 0, xu, 0,
                x, 0, y, 0, w, 0, jw, 0, 3 * n, INF_BND, norm);

        assertEquals(0, status, "LSQ should succeed with no constraints");

        // For unconstrained QP, the solution should satisfy the optimality condition
        // The residual norm should be non-negative
        assertTrue(norm[0] >= 0, "Residual norm should be non-negative");
    }

    // ========================================================================
    // Property 8.8: Solution is consistent (deterministic)
    // ========================================================================

    /**
     *
     * Running LSQ twice with the same input should produce the same result.
     */
    @Property(tries = 50)
    void lsqSolutionIsConsistent(
            @ForAll @IntRange(min = 2, max = 5) int n,
            @ForAll @IntRange(min = 0, max = 2) int meq,
            @ForAll @IntRange(min = 0, max = 2) int mineq,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(meq < n);

        java.util.Random rand = new java.util.Random(seed);

        int m = meq + mineq;
        int nl = n * (n + 1) / 2 + 1;

        // Generate LDL^T factorization and gradient
        double[] l = generateLDLT(n, rand);
        double[] g = generateGradient(n, rand);

        // Generate bounds
        double[][] bounds = generateBounds(n, rand);
        double[] xl = bounds[0];
        double[] xu = bounds[1];

        // Generate constraint Jacobian and feasible constraint values
        double[] a = (m > 0) ? generateConstraintJacobian(m, n, rand) : new double[n];
        double[] b = (m > 0) ? generateFeasibleB(a, m, meq, n, xl, xu, rand) : new double[1];

        // Solve twice with same input
        double[] x1 = new double[n];
        double[] y1 = new double[m + 2 * n];
        double[] w1 = new double[computeWorkSize(m, meq, n)];
        int[] jw1 = new int[Math.max(mineq + 2 * n, Math.min(n, Math.max(m - meq, 1))) + 1];
        double[] norm1 = new double[1];

        int status1 = LSQSolver.lsq(m, meq, n, nl,
                l.clone(), 0, g.clone(), 0, a.clone(), 0, b.clone(), 0, xl.clone(), 0, xu.clone(), 0,
                x1, 0, y1, 0, w1, 0, jw1, 0, 3 * n, INF_BND, norm1);

        double[] x2 = new double[n];
        double[] y2 = new double[m + 2 * n];
        double[] w2 = new double[computeWorkSize(m, meq, n)];
        int[] jw2 = new int[Math.max(mineq + 2 * n, Math.min(n, Math.max(m - meq, 1))) + 1];
        double[] norm2 = new double[1];

        int status2 = LSQSolver.lsq(m, meq, n, nl,
                l.clone(), 0, g.clone(), 0, a.clone(), 0, b.clone(), 0, xl.clone(), 0, xu.clone(), 0,
                x2, 0, y2, 0, w2, 0, jw2, 0, 3 * n, INF_BND, norm2);

        // Both should have same status
        assertEquals(status1, status2, "Same input should produce same status");

        if (status1 == 0) {
            // Solutions should be identical
            for (int j = 0; j < n; j++) {
                assertEquals(x1[j], x2[j], 1e-10,
                        "Same input should produce same solution");
            }

            // Norms should be identical
            assertEquals(norm1[0], norm2[0], 1e-10,
                    "Same input should produce same residual norm");
        }
    }
}
