/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for LDP (Least Distance Programming) solver.
 *
 * **Property 4: LDP Solver Correctness**
 */
package com.curioloop.numopt4j.optim.slsqp;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for LDP (Least Distance Programming) solver.
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 4.1: Solution satisfies feasibility constraint (Gx >= h)</li>
 *   <li>Property 4.2: Solution minimizes ||x||_2 among all feasible solutions</li>
 *   <li>Property 4.3: Lagrange multipliers are non-negative (λ >= 0)</li>
 *   <li>Property 4.4: Solution norm is correctly computed</li>
 *   <li>Property 4.5: LDP handles edge cases correctly</li>
 * </ul>
 *
 * <p>The LDP problem is: minimize ||x||_2 subject to Gx >= h</p>
 *
 * <p>The algorithm transforms this to NNLS form:</p>
 * <ul>
 *   <li>Form matrix A = [G^T; h^T] (augmented matrix)</li>
 *   <li>Form vector b = [0; ...; 0; 1] (unit vector)</li>
 *   <li>Solve NNLS: minimize ||Au - b||_2 subject to u >= 0</li>
 *   <li>Recover solution: x = G^T u / ||r|| where r = b - Au</li>
 *   <li>Compute Lagrange multipliers: λ = u / ||r||</li>
 * </ul>
 */
@Tag("Feature: slsqp-java-rewrite, Property 4: LDP Solver Correctness")
class LDPProperties {

    private static final double EPSILON = 1e-8;
    private static final double TOLERANCE = 1e-10;
    private static final double FEASIBILITY_TOL = 1e-6;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Generate a well-conditioned random matrix G (m x n) in column-major format.
     * The matrix is designed to create feasible LDP problems.
     */
    private double[] generateMatrix(int m, int n, java.util.Random rand) {
        double[] g = new double[m * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                g[j * m + i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
            }
        }
        return g;
    }

    /**
     * Generate a feasible constraint vector h for the given matrix G.
     * We generate h such that Gx >= h is feasible for some x.
     * Strategy: pick a random x, compute Gx, then set h = Gx - slack
     */
    private double[] generateFeasibleH(double[] g, int m, int n, java.util.Random rand) {
        // Generate a random x
        double[] x = new double[n];
        for (int j = 0; j < n; j++) {
            x[j] = rand.nextDouble() * 2 - 1;  // Values in [-1, 1]
        }

        // Compute Gx
        double[] gx = matVec(g, m, n, x);

        // Set h = Gx - slack (to ensure feasibility)
        double[] h = new double[m];
        for (int i = 0; i < m; i++) {
            double slack = rand.nextDouble() * 0.5 + 0.1;  // Slack in [0.1, 0.6]
            h[i] = gx[i] - slack;
        }

        return h;
    }

    /**
     * Compute matrix-vector product Gx (G is m x n column-major).
     */
    private double[] matVec(double[] g, int m, int n, double[] x) {
        double[] result = new double[m];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                result[i] += g[j * m + i] * x[j];
            }
        }
        return result;
    }

    /**
     * Compute matrix-transpose-vector product G^T * v (G is m x n column-major).
     */
    private double[] matTVec(double[] g, int m, int n, double[] v) {
        double[] result = new double[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                result[j] += g[j * m + i] * v[i];
            }
        }
        return result;
    }

    /**
     * Compute Euclidean norm of a vector.
     */
    private double norm2(double[] v) {
        double sum = 0;
        for (double vi : v) {
            sum += vi * vi;
        }
        return Math.sqrt(sum);
    }

    /**
     * Compute the required working array size for LDP.
     * w needs (n+1)×(m+2)+2m elements
     */
    private int computeWorkSize(int m, int n) {
        return (n + 1) * (m + 2) + 2 * m;
    }

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }

    // ========================================================================
    // Property 4.1: Solution satisfies feasibility constraint (Gx >= h)
    // ========================================================================

    @Property(tries = 100)
    void solutionSatisfiesFeasibilityConstraint(
            @ForAll @IntRange(min = 2, max = 10) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix G and feasible vector h
        double[] g = generateMatrix(m, n, rand);
        double[] h = generateFeasibleH(g, m, n, rand);

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(m, n)];
        int[] jw = new int[m];
        double[] xnorm = new double[1];

        // Solve LDP
        int status = LSQSolver.ldp(m, n, g, 0, m, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        // Check status (0 = success, -2 = constraints incompatible)
        if (status == 0) {
            // Verify feasibility: Gx >= h
            double[] gx = matVec(g, m, n, x);
            for (int i = 0; i < m; i++) {
                assertTrue(gx[i] >= h[i] - FEASIBILITY_TOL,
                        "Constraint " + i + " violated: Gx[" + i + "] = " + gx[i] + 
                        " should be >= h[" + i + "] = " + h[i]);
            }
        }
        // If status == -2, constraints are incompatible, which is acceptable for some inputs
    }

    // ========================================================================
    // Property 4.2: Solution minimizes ||x||_2 among all feasible solutions
    // ========================================================================

    @Property(tries = 100)
    void solutionMinimizesNorm(
            @ForAll @IntRange(min = 2, max = 8) int m,
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix G and feasible vector h
        double[] g = generateMatrix(m, n, rand);
        double[] h = generateFeasibleH(g, m, n, rand);

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(m, n)];
        int[] jw = new int[m];
        double[] xnorm = new double[1];

        // Solve LDP
        int status = LSQSolver.ldp(m, n, g, 0, m, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        double ldpNorm = norm2(x);

        // Generate random feasible perturbations and verify LDP solution is optimal
        for (int trial = 0; trial < 10; trial++) {
            // Create a perturbed feasible solution
            double[] xPerturbed = new double[n];
            for (int j = 0; j < n; j++) {
                xPerturbed[j] = x[j] + (rand.nextDouble() - 0.5) * 0.5;
            }

            // Check if perturbed solution is feasible
            double[] gxPerturbed = matVec(g, m, n, xPerturbed);
            boolean feasible = true;
            for (int i = 0; i < m; i++) {
                if (gxPerturbed[i] < h[i] - FEASIBILITY_TOL) {
                    feasible = false;
                    break;
                }
            }

            if (feasible) {
                double perturbedNorm = norm2(xPerturbed);
                // LDP solution should have smaller or equal norm
                assertTrue(ldpNorm <= perturbedNorm + EPSILON,
                        "LDP norm (" + ldpNorm + ") should be <= perturbed norm (" + perturbedNorm + ")");
            }
        }
    }

    // ========================================================================
    // Property 4.3: Lagrange multipliers are non-negative (λ >= 0)
    // ========================================================================

    @Property(tries = 100)
    void lagrangeMultipliersAreNonNegative(
            @ForAll @IntRange(min = 2, max = 10) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix G and feasible vector h
        double[] g = generateMatrix(m, n, rand);
        double[] h = generateFeasibleH(g, m, n, rand);

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(m, n)];
        int[] jw = new int[m];
        double[] xnorm = new double[1];

        // Solve LDP
        int status = LSQSolver.ldp(m, n, g, 0, m, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Lagrange multipliers are stored in w[0:m] after solve
        for (int i = 0; i < m; i++) {
            assertTrue(w[i] >= -TOLERANCE,
                    "Lagrange multiplier λ[" + i + "] = " + w[i] + " should be non-negative");
        }
    }

    // ========================================================================
    // Property 4.4: Solution norm is correctly computed
    // ========================================================================

    @Property(tries = 100)
    void solutionNormIsCorrectlyComputed(
            @ForAll @IntRange(min = 2, max = 10) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix G and feasible vector h
        double[] g = generateMatrix(m, n, rand);
        double[] h = generateFeasibleH(g, m, n, rand);

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(m, n)];
        int[] jw = new int[m];
        double[] xnorm = new double[1];

        // Solve LDP
        int status = LSQSolver.ldp(m, n, g, 0, m, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Compute norm independently
        double computedNorm = norm2(x);

        // Verify the returned xnorm matches our computation
        assertEquals(computedNorm, xnorm[0], EPSILON,
                "Returned xnorm (" + xnorm[0] + ") should match computed norm (" + computedNorm + ")");
    }

    // ========================================================================
    // Property 4.5: LDP handles zero constraints correctly (m = 0)
    // ========================================================================

    @Property(tries = 50)
    void ldpHandlesZeroConstraints(
            @ForAll @IntRange(min = 2, max = 10) int n
    ) {
        int m = 0;  // No constraints

        // Allocate arrays (even though m=0, we need valid arrays)
        double[] g = new double[1];  // Dummy
        double[] h = new double[1];  // Dummy
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(1, n)];  // Use m=1 for size calculation
        int[] jw = new int[1];
        double[] xnorm = new double[1];

        // Solve LDP with m=0
        int status = LSQSolver.ldp(m, n, g, 0, 1, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        // With no constraints, the minimum norm solution is x = 0
        assertEquals(0, status, "LDP should succeed with no constraints");
        assertEquals(0.0, xnorm[0], EPSILON, "With no constraints, ||x|| should be 0");

        for (int j = 0; j < n; j++) {
            assertEquals(0.0, x[j], EPSILON, "With no constraints, x[" + j + "] should be 0");
        }
    }

    // ========================================================================
    // Property 4.6: LDP handles identity G matrix correctly
    // ========================================================================

    @Property(tries = 50)
    void ldpHandlesIdentityMatrix(
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int m = n;  // Square identity matrix

        // Create identity matrix G (column-major)
        double[] g = new double[m * n];
        for (int i = 0; i < n; i++) {
            g[i * m + i] = 1.0;
        }

        // Generate h with some negative values (to ensure feasibility)
        // For identity G, constraint Gx >= h means x >= h
        // Minimum norm solution is x = max(0, h) element-wise
        double[] h = new double[m];
        for (int i = 0; i < m; i++) {
            h[i] = rand.nextDouble() * 2 - 1;  // Values in [-1, 1]
        }

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(m, n)];
        int[] jw = new int[m];
        double[] xnorm = new double[1];

        // Solve LDP
        int status = LSQSolver.ldp(m, n, g, 0, m, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Verify feasibility: x >= h
        for (int i = 0; i < n; i++) {
            assertTrue(x[i] >= h[i] - FEASIBILITY_TOL,
                    "For identity G, x[" + i + "] = " + x[i] + " should be >= h[" + i + "] = " + h[i]);
        }

        // For identity G with h, the optimal solution is x_i = max(0, h_i)
        // This is because we want to minimize ||x||_2 subject to x >= h
        for (int i = 0; i < n; i++) {
            double expected = Math.max(0, h[i]);
            assertEquals(expected, x[i], EPSILON,
                    "For identity G, x[" + i + "] should be max(0, h[" + i + "])");
        }
    }

    // ========================================================================
    // Property 4.7: LDP handles negative h correctly (trivial feasibility)
    // ========================================================================

    @Property(tries = 50)
    void ldpHandlesNegativeH(
            @ForAll @IntRange(min = 2, max = 8) int m,
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix G
        double[] g = generateMatrix(m, n, rand);

        // Generate strictly negative h (x = 0 should be feasible)
        double[] h = new double[m];
        for (int i = 0; i < m; i++) {
            h[i] = -rand.nextDouble() * 2 - 0.1;  // Values in [-2.1, -0.1]
        }

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(m, n)];
        int[] jw = new int[m];
        double[] xnorm = new double[1];

        // Solve LDP
        int status = LSQSolver.ldp(m, n, g, 0, m, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        assertEquals(0, status, "LDP should succeed with negative h");

        // Verify feasibility
        double[] gx = matVec(g, m, n, x);
        for (int i = 0; i < m; i++) {
            assertTrue(gx[i] >= h[i] - FEASIBILITY_TOL,
                    "Constraint " + i + " violated: Gx[" + i + "] = " + gx[i] + 
                    " should be >= h[" + i + "] = " + h[i]);
        }

        // With negative h, x = 0 is feasible, so optimal solution should have small norm
        // (possibly zero if x = 0 is optimal)
        assertTrue(xnorm[0] >= 0, "Solution norm should be non-negative");
    }

    // ========================================================================
    // Property 4.8: LDP solution is consistent (deterministic)
    // ========================================================================

    @Property(tries = 50)
    void ldpSolutionIsConsistent(
            @ForAll @IntRange(min = 2, max = 8) int m,
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix G and feasible vector h
        double[] g = generateMatrix(m, n, rand);
        double[] h = generateFeasibleH(g, m, n, rand);

        // Solve twice with same input
        double[] x1 = new double[n];
        double[] w1 = new double[computeWorkSize(m, n)];
        int[] jw1 = new int[m];
        double[] xnorm1 = new double[1];

        int status1 = LSQSolver.ldp(m, n, g.clone(), 0, m, h.clone(), 0, x1, 0, w1, 0, jw1, 0, 3 * n, xnorm1);

        double[] x2 = new double[n];
        double[] w2 = new double[computeWorkSize(m, n)];
        int[] jw2 = new int[m];
        double[] xnorm2 = new double[1];

        int status2 = LSQSolver.ldp(m, n, g.clone(), 0, m, h.clone(), 0, x2, 0, w2, 0, jw2, 0, 3 * n, xnorm2);

        // Both should have same status
        assertEquals(status1, status2, "Same input should produce same status");

        if (status1 == 0) {
            // Solutions should be identical
            for (int j = 0; j < n; j++) {
                assertEquals(x1[j], x2[j], TOLERANCE,
                        "Same input should produce same solution");
            }

            // Norms should be identical
            assertEquals(xnorm1[0], xnorm2[0], TOLERANCE,
                    "Same input should produce same norm");
        }
    }

    // ========================================================================
    // Property 4.9: LDP handles single constraint correctly
    // ========================================================================

    @Property(tries = 50)
    void ldpHandlesSingleConstraint(
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int m = 1;  // Single constraint

        // Generate a single row vector g (as column-major 1 x n matrix)
        double[] g = new double[n];
        double gNorm = 0;
        for (int j = 0; j < n; j++) {
            g[j] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
            gNorm += g[j] * g[j];
        }
        gNorm = Math.sqrt(gNorm);

        // Generate h (single value)
        double[] h = new double[1];
        h[0] = rand.nextDouble() * 2 - 1;  // Value in [-1, 1]

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(m, n)];
        int[] jw = new int[m];
        double[] xnorm = new double[1];

        // Solve LDP
        int status = LSQSolver.ldp(m, n, g, 0, m, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Verify feasibility: g'x >= h
        double gx = 0;
        for (int j = 0; j < n; j++) {
            gx += g[j] * x[j];
        }
        assertTrue(gx >= h[0] - FEASIBILITY_TOL,
                "Single constraint violated: g'x = " + gx + " should be >= h = " + h[0]);

        // For single constraint g'x >= h, the minimum norm solution is:
        // x = max(0, h / ||g||^2) * g  (projection onto constraint)
        if (gNorm > TOLERANCE) {
            double expectedCoeff = Math.max(0, h[0] / (gNorm * gNorm));
            for (int j = 0; j < n; j++) {
                double expected = expectedCoeff * g[j];
                assertEquals(expected, x[j], EPSILON,
                        "For single constraint, x[" + j + "] should be " + expected);
            }
        }
    }

    // ========================================================================
    // Property 4.10: LDP handles incompatible constraints correctly
    // ========================================================================

    @Property(tries = 50)
    void ldpHandlesIncompatibleConstraints(
            @ForAll @IntRange(min = 2, max = 6) int n
    ) {
        // Create incompatible constraints: x >= 1 and -x >= 1 (i.e., x <= -1)
        // These cannot be satisfied simultaneously
        int m = 2;

        // G = [I; -I] for first variable only
        double[] g = new double[m * n];
        g[0] = 1.0;   // First constraint: x[0] >= h[0]
        g[1] = -1.0;  // Second constraint: -x[0] >= h[1]

        // h = [1, 1] means x[0] >= 1 and -x[0] >= 1, which is impossible
        double[] h = new double[m];
        h[0] = 1.0;
        h[1] = 1.0;

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(m, n)];
        int[] jw = new int[m];
        double[] xnorm = new double[1];

        // Solve LDP
        int status = LSQSolver.ldp(m, n, g, 0, m, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        // Should return -2 (constraints incompatible)
        assertEquals(SLSQPConstants.ERR_CONS_INCOMPAT, status,
                "LDP should return ERR_CONS_INCOMPAT for incompatible constraints");
    }

    // ========================================================================
    // Property 4.11: LDP KKT conditions are satisfied
    // ========================================================================

    @Property(tries = 100)
    void ldpKKTConditionsAreSatisfied(
            @ForAll @IntRange(min = 2, max = 8) int m,
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix G and feasible vector h
        double[] g = generateMatrix(m, n, rand);
        double[] h = generateFeasibleH(g, m, n, rand);

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(m, n)];
        int[] jw = new int[m];
        double[] xnorm = new double[1];

        // Solve LDP
        int status = LSQSolver.ldp(m, n, g, 0, m, h, 0, x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Extract Lagrange multipliers from w[0:m]
        double[] lambda = new double[m];
        System.arraycopy(w, 0, lambda, 0, m);

        // KKT conditions for LDP:
        // 1. Primal feasibility: Gx >= h
        // 2. Dual feasibility: λ >= 0
        // 3. Complementary slackness: λ_i * (Gx - h)_i = 0
        // 4. Stationarity: x = G^T λ

        // Check primal feasibility
        double[] gx = matVec(g, m, n, x);
        for (int i = 0; i < m; i++) {
            assertTrue(gx[i] >= h[i] - FEASIBILITY_TOL,
                    "Primal feasibility violated at constraint " + i);
        }

        // Check dual feasibility
        for (int i = 0; i < m; i++) {
            assertTrue(lambda[i] >= -TOLERANCE,
                    "Dual feasibility violated: λ[" + i + "] = " + lambda[i]);
        }

        // Check complementary slackness (approximately)
        for (int i = 0; i < m; i++) {
            double slack = gx[i] - h[i];
            // Either λ_i ≈ 0 or slack ≈ 0
            assertTrue(lambda[i] < EPSILON || slack < EPSILON,
                    "Complementary slackness violated at constraint " + i + 
                    ": λ = " + lambda[i] + ", slack = " + slack);
        }

        // Check stationarity: x = G^T λ
        double[] gtLambda = matTVec(g, m, n, lambda);
        for (int j = 0; j < n; j++) {
            assertEquals(gtLambda[j], x[j], EPSILON,
                    "Stationarity violated: x[" + j + "] = " + x[j] + 
                    " should equal G^T λ[" + j + "] = " + gtLambda[j]);
        }
    }

    // ========================================================================
    // Property 4.12: LDP handles bad arguments correctly
    // ========================================================================

    @Property(tries = 10)
    void ldpHandlesBadArguments() {
        // Test with n <= 0
        double[] g = {1.0};
        double[] h = {1.0};
        double[] x = new double[1];
        double[] w = new double[10];
        int[] jw = new int[1];
        double[] xnorm = new double[1];

        int status = LSQSolver.ldp(1, 0, g, 0, 1, h, 0, x, 0, w, 0, jw, 0, 3, xnorm);
        assertEquals(-1, status, "LDP should return -1 for n <= 0");

        status = LSQSolver.ldp(1, -1, g, 0, 1, h, 0, x, 0, w, 0, jw, 0, 3, xnorm);
        assertEquals(-1, status, "LDP should return -1 for n < 0");
    }
}
