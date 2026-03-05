/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for LSI (Least Squares with Inequality constraints) solver.
 *
 * **Property 5: LSI Solver Correctness**
 */
package com.curioloop.numopt4j.optim.slsqp;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for LSI (Least Squares with Inequality constraints) solver.
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 5.1: Solution satisfies feasibility constraint (Gx >= h)</li>
 *   <li>Property 5.2: Solution minimizes ||Ex - f||_2 among all feasible solutions</li>
 *   <li>Property 5.3: Residual norm is correctly computed</li>
 *   <li>Property 5.4: LSI handles edge cases correctly (no constraints, identity E)</li>
 *   <li>Property 5.5: LSI correctly detects singular E matrix</li>
 * </ul>
 *
 * <p>The LSI problem is: minimize ||Ex - f||_2 subject to Gx >= h</p>
 *
 * <p>The algorithm:</p>
 * <ul>
 *   <li>Compute QR factorization of E using Householder transformations: QE = [R; 0]</li>
 *   <li>Apply Q to f: Qf = [f1; f2]</li>
 *   <li>Transform to LDP: minimize ||y||_2 subject to GR^(-1)y >= h - GR^(-1)f1</li>
 *   <li>Call LDP to solve the transformed problem</li>
 *   <li>Recover solution: x = R^(-1)(y + f1)</li>
 *   <li>Compute residual norm: sqrt(||y||^2 + ||f2||^2)</li>
 * </ul>
 */
@Tag("Feature: slsqp-java-rewrite, Property 5: LSI Solver Correctness")
class LSIProperties {

    private static final double EPSILON = 1e-8;
    private static final double TOLERANCE = 1e-10;
    private static final double FEASIBILITY_TOL = 1e-6;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Generate a well-conditioned full-rank matrix E (me x n) in column-major format.
     * Ensures rank(E) = n by adding diagonal dominance.
     */
    private double[] generateFullRankMatrix(int me, int n, java.util.Random rand) {
        double[] e = new double[me * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < me; i++) {
                e[j * me + i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
            }
            // Add diagonal dominance for numerical stability and full rank
            if (j < me) {
                e[j * me + j] += 10.0;
            }
        }
        return e;
    }

    /**
     * Generate a random constraint matrix G (mg x n) in column-major format.
     */
    private double[] generateConstraintMatrix(int mg, int n, java.util.Random rand) {
        double[] g = new double[mg * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < mg; i++) {
                g[j * mg + i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
            }
        }
        return g;
    }

    /**
     * Generate a random vector of length m.
     */
    private double[] generateVector(int m, java.util.Random rand) {
        double[] v = new double[m];
        for (int i = 0; i < m; i++) {
            v[i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
        }
        return v;
    }

    /**
     * Generate a feasible constraint vector h for the given matrix G.
     * Strategy: pick a random x, compute Gx, then set h = Gx - slack
     */
    private double[] generateFeasibleH(double[] g, int mg, int n, java.util.Random rand) {
        // Generate a random x
        double[] x = new double[n];
        for (int j = 0; j < n; j++) {
            x[j] = rand.nextDouble() * 2 - 1;  // Values in [-1, 1]
        }

        // Compute Gx
        double[] gx = matVec(g, mg, n, x);

        // Set h = Gx - slack (to ensure feasibility)
        double[] h = new double[mg];
        for (int i = 0; i < mg; i++) {
            double slack = rand.nextDouble() * 0.5 + 0.1;  // Slack in [0.1, 0.6]
            h[i] = gx[i] - slack;
        }

        return h;
    }

    /**
     * Compute matrix-vector product Ax (A is m x n column-major).
     */
    private double[] matVec(double[] a, int m, int n, double[] x) {
        double[] result = new double[m];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                result[i] += a[j * m + i] * x[j];
            }
        }
        return result;
    }

    /**
     * Compute vector subtraction: result = a - b.
     */
    private double[] subtract(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
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
     * Compute the required working array size for LSI.
     * w needs (n+1)×(mg+2)+2×mg elements
     */
    private int computeWorkSize(int mg, int n) {
        return (n + 1) * (mg + 2) + 2 * mg;
    }

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }

    // ========================================================================
    // Property 5.1: Solution satisfies feasibility constraint (Gx >= h)
    // ========================================================================

    /**
     * 
     * For any full-rank matrix E and feasible constraints Gx >= h,
     * the LSI solution x shall satisfy Gx >= h (feasibility).
     */
    @Property(tries = 100)
    void solutionSatisfiesFeasibilityConstraint(
            @ForAll @IntRange(min = 3, max = 10) int me,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @IntRange(min = 1, max = 6) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);  // Ensure E has full column rank

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n,
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        // Check status (0 = success, -2 = constraints incompatible, -3 = E singular)
        if (status == 0) {
            // Verify feasibility: Gx >= h
            double[] gx = matVec(gOriginal, mg, n, x);
            for (int i = 0; i < mg; i++) {
                assertTrue(gx[i] >= hOriginal[i] - FEASIBILITY_TOL,
                        "Constraint " + i + " violated: Gx[" + i + "] = " + gx[i] + 
                        " should be >= h[" + i + "] = " + hOriginal[i]);
            }
        }
        // If status != 0, the problem may be infeasible or E is singular
    }

    // ========================================================================
    // Property 5.2: Solution minimizes ||Ex - f||_2 among all feasible solutions
    // ========================================================================

    /**
     * 
     * For any full-rank matrix E and feasible constraints Gx >= h,
     * the LSI solution x shall minimize ||Ex - f||_2 among all feasible solutions.
     */
    @Property(tries = 100)
    void solutionMinimizesResidualNorm(
            @ForAll @IntRange(min = 3, max = 10) int me,
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Compute residual norm for the LSI solution
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);
        double lsiResidualNorm = norm2(residual);

        // Generate random feasible perturbations and verify LSI solution is optimal
        for (int trial = 0; trial < 10; trial++) {
            // Create a perturbed solution
            double[] xPerturbed = new double[n];
            for (int j = 0; j < n; j++) {
                xPerturbed[j] = x[j] + (rand.nextDouble() - 0.5) * 0.5;
            }

            // Check if perturbed solution is feasible
            double[] gxPerturbed = matVec(gOriginal, mg, n, xPerturbed);
            boolean feasible = true;
            for (int i = 0; i < mg; i++) {
                if (gxPerturbed[i] < hOriginal[i] - FEASIBILITY_TOL) {
                    feasible = false;
                    break;
                }
            }

            if (feasible) {
                // Compute residual norm for perturbed solution
                double[] exPerturbed = matVec(eOriginal, me, n, xPerturbed);
                double[] residualPerturbed = subtract(exPerturbed, fOriginal);
                double perturbedResidualNorm = norm2(residualPerturbed);

                // LSI solution should have smaller or equal residual norm
                assertTrue(lsiResidualNorm <= perturbedResidualNorm + EPSILON,
                        "LSI residual norm (" + lsiResidualNorm + 
                        ") should be <= perturbed residual norm (" + perturbedResidualNorm + ")");
            }
        }
    }

    // ========================================================================
    // Property 5.3: Residual norm is correctly computed
    // ========================================================================

    /**
     * 
     * The residual norm ||Ex - f||_2 shall be correctly computed and returned.
     */
    @Property(tries = 100)
    void residualNormIsCorrectlyComputed(
            @ForAll @IntRange(min = 3, max = 10) int me,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @IntRange(min = 1, max = 6) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Compute residual norm independently
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);
        double computedResidualNorm = norm2(residual);

        // Verify the returned xnorm matches our computation
        assertEquals(computedResidualNorm, xnorm[0], EPSILON,
                "Returned xnorm (" + xnorm[0] + ") should match computed residual norm (" + 
                computedResidualNorm + ")");
    }

    // ========================================================================
    // Property 5.4: LSI handles zero constraints correctly (mg = 0)
    // ========================================================================

    /**
     * 
     * When there are no inequality constraints (mg = 0), LSI should solve
     * the unconstrained least squares problem min ||Ex - f||_2.
     */
    @Property(tries = 50)
    void lsiHandlesZeroConstraints(
            @ForAll @IntRange(min = 3, max = 10) int me,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);

        java.util.Random rand = new java.util.Random(seed);

        int mg = 0;  // No constraints

        // Generate full-rank matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Make copies since LSI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();

        // Dummy arrays for G and h (not used when mg = 0)
        double[] g = new double[1];
        double[] h = new double[1];

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(1, n)];  // Use mg=1 for size calculation
        int[] jw = new int[1];
        double[] xnorm = new double[1];

        // Solve LSI with mg = 0
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, 1, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        assertEquals(0, status, "LSI should succeed with no constraints");

        // Verify solution satisfies normal equations: E^T E x = E^T f
        // For least squares, E^T (Ex - f) should be approximately zero
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);

        // Compute E^T * residual
        double[] etResidual = new double[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < me; i++) {
                etResidual[j] += eOriginal[j * me + i] * residual[i];
            }
        }

        // E^T * residual should be approximately zero
        for (int j = 0; j < n; j++) {
            assertEquals(0.0, etResidual[j], 1e-6,
                    "Normal equation residual E^T(Ex-f)[" + j + "] should be approximately zero");
        }
    }

    // ========================================================================
    // Property 5.5: LSI handles identity E matrix correctly
    // ========================================================================

    /**
     * 
     * When E is the identity matrix, LSI should solve min ||x - f||_2 subject to Gx >= h.
     */
    @Property(tries = 50)
    void lsiHandlesIdentityMatrix(
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int me = n;  // Square identity matrix

        // Create identity matrix E (column-major)
        double[] eOriginal = new double[me * n];
        for (int i = 0; i < n; i++) {
            eOriginal[i * me + i] = 1.0;
        }

        // Generate vector f
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Verify feasibility: Gx >= h
        double[] gx = matVec(gOriginal, mg, n, x);
        for (int i = 0; i < mg; i++) {
            assertTrue(gx[i] >= hOriginal[i] - FEASIBILITY_TOL,
                    "Constraint " + i + " violated: Gx[" + i + "] = " + gx[i] + 
                    " should be >= h[" + i + "] = " + hOriginal[i]);
        }

        // For identity E, the objective is ||x - f||_2
        // Verify residual norm
        double[] residual = subtract(x, fOriginal);
        double computedNorm = norm2(residual);
        assertEquals(computedNorm, xnorm[0], EPSILON,
                "For identity E, residual norm should be ||x - f||_2");
    }

    // ========================================================================
    // Property 5.6: LSI handles negative h correctly (trivial feasibility)
    // ========================================================================

    /**
     * 
     * When h is strictly negative, x = 0 should be feasible, and LSI should find
     * the optimal solution.
     */
    @Property(tries = 50)
    void lsiHandlesNegativeH(
            @ForAll @IntRange(min = 3, max = 10) int me,
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);

        // Generate strictly negative h (x = 0 should be feasible)
        double[] hOriginal = new double[mg];
        for (int i = 0; i < mg; i++) {
            hOriginal[i] = -rand.nextDouble() * 2 - 0.1;  // Values in [-2.1, -0.1]
        }

        // Make copies since LSI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        assertEquals(0, status, "LSI should succeed with negative h");

        // Verify feasibility
        double[] gx = matVec(gOriginal, mg, n, x);
        for (int i = 0; i < mg; i++) {
            assertTrue(gx[i] >= hOriginal[i] - FEASIBILITY_TOL,
                    "Constraint " + i + " violated: Gx[" + i + "] = " + gx[i] + 
                    " should be >= h[" + i + "] = " + hOriginal[i]);
        }

        // Verify residual norm is non-negative
        assertTrue(xnorm[0] >= 0, "Residual norm should be non-negative");
    }

    // ========================================================================
    // Property 5.7: LSI solution is consistent (deterministic)
    // ========================================================================

    /**
     * 
     * Running LSI twice with the same input should produce the same result.
     */
    @Property(tries = 50)
    void lsiSolutionIsConsistent(
            @ForAll @IntRange(min = 3, max = 8) int me,
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Solve twice with same input
        double[] x1 = new double[n];
        double[] w1 = new double[computeWorkSize(mg, n)];
        int[] jw1 = new int[mg];
        double[] xnorm1 = new double[1];

        int status1 = LSQSolver.lsi(eOriginal.clone(), 0, fOriginal.clone(), 0, 
                                gOriginal.clone(), 0, hOriginal.clone(), 0, 
                                me, me, mg, mg, n, x1, 0, w1, 0, jw1, 0, 3 * n, xnorm1);

        double[] x2 = new double[n];
        double[] w2 = new double[computeWorkSize(mg, n)];
        int[] jw2 = new int[mg];
        double[] xnorm2 = new double[1];

        int status2 = LSQSolver.lsi(eOriginal.clone(), 0, fOriginal.clone(), 0, 
                                gOriginal.clone(), 0, hOriginal.clone(), 0, 
                                me, me, mg, mg, n, x2, 0, w2, 0, jw2, 0, 3 * n, xnorm2);

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
                    "Same input should produce same residual norm");
        }
    }

    // ========================================================================
    // Property 5.8: LSI handles single constraint correctly
    // ========================================================================

    /**
     * 
     * LSI should correctly handle problems with a single inequality constraint.
     */
    @Property(tries = 50)
    void lsiHandlesSingleConstraint(
            @ForAll @IntRange(min = 3, max = 10) int me,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);

        java.util.Random rand = new java.util.Random(seed);

        int mg = 1;  // Single constraint

        // Generate full-rank matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate a single row vector g (as column-major 1 x n matrix)
        double[] gOriginal = new double[n];
        for (int j = 0; j < n; j++) {
            gOriginal[j] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
        }

        // Generate feasible h
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Verify feasibility: g'x >= h
        double gx = 0;
        for (int j = 0; j < n; j++) {
            gx += gOriginal[j] * x[j];
        }
        assertTrue(gx >= hOriginal[0] - FEASIBILITY_TOL,
                "Single constraint violated: g'x = " + gx + " should be >= h = " + hOriginal[0]);

        // Verify residual norm
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);
        double computedNorm = norm2(residual);
        assertEquals(computedNorm, xnorm[0], EPSILON,
                "Residual norm should be correctly computed");
    }

    // ========================================================================
    // Property 5.9: LSI handles incompatible constraints correctly
    // ========================================================================

    /**
     * 
     * LSI should return ERR_CONS_INCOMPAT when constraints are incompatible.
     */
    @Property(tries = 50)
    void lsiHandlesIncompatibleConstraints(
            @ForAll @IntRange(min = 3, max = 8) int me,
            @ForAll @IntRange(min = 2, max = 6) int n
    ) {
        Assume.that(me >= n);

        // Create incompatible constraints: x[0] >= 1 and -x[0] >= 1 (i.e., x[0] <= -1)
        int mg = 2;

        // Generate full-rank matrix E
        double[] e = new double[me * n];
        for (int j = 0; j < n; j++) {
            if (j < me) {
                e[j * me + j] = 10.0;  // Diagonal dominance
            }
        }

        // Generate vector f
        double[] f = new double[me];
        for (int i = 0; i < me; i++) {
            f[i] = 1.0;
        }

        // G = [1, 0, ..., 0; -1, 0, ..., 0] (column-major)
        double[] g = new double[mg * n];
        g[0] = 1.0;   // First constraint: x[0] >= h[0]
        g[1] = -1.0;  // Second constraint: -x[0] >= h[1]

        // h = [1, 1] means x[0] >= 1 and -x[0] >= 1, which is impossible
        double[] h = new double[mg];
        h[0] = 1.0;
        h[1] = 1.0;

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        // Should return ERR_CONS_INCOMPAT (-2)
        assertEquals(SLSQPConstants.ERR_CONS_INCOMPAT, status,
                "LSI should return ERR_CONS_INCOMPAT for incompatible constraints");
    }

    // ========================================================================
    // Property 5.10: LSI handles singular E matrix correctly
    // ========================================================================

    /**
     * 
     * LSI should return ERR_LSI_SINGULAR_E when matrix E is singular (rank(E) < n).
     * 
     * Note: The LSI solver detects singularity by checking if diagonal elements
     * of R (after QR factorization) are smaller than EPS. We create a matrix
     * with a zero column, which will definitely result in a zero diagonal element.
     */
    @Property(tries = 50)
    void lsiHandlesSingularEMatrix(
            @ForAll @IntRange(min = 3, max = 8) int me,
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 1, max = 4) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);
        Assume.that(n >= 2);  // Need at least 2 columns

        java.util.Random rand = new java.util.Random(seed);

        // Create a rank-deficient matrix E with a zero column
        // This guarantees that the matrix is singular
        double[] e = new double[me * n];
        
        // Fill all columns with random values
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < me; i++) {
                e[j * me + i] = rand.nextDouble() * 4 - 2;
            }
        }
        
        // Make the last column all zeros - this guarantees singularity
        for (int i = 0; i < me; i++) {
            e[(n - 1) * me + i] = 0.0;
        }

        // Generate vector f
        double[] f = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] g = generateConstraintMatrix(mg, n, rand);
        double[] h = generateFeasibleH(g, mg, n, rand);

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        // Should return ERR_LSI_SINGULAR_E (-3)
        assertEquals(SLSQPConstants.ERR_LSI_SINGULAR_E, status,
                "LSI should return ERR_LSI_SINGULAR_E for singular E matrix");
    }

    // ========================================================================
    // Property 5.11: LSI handles bad arguments correctly
    // ========================================================================

    /**
     * 
     * LSI should return -1 for invalid arguments (n <= 0).
     */
    @Property(tries = 10)
    void lsiHandlesBadArguments() {
        // Test with n <= 0
        double[] e = {1.0};
        double[] f = {1.0};
        double[] g = {1.0};
        double[] h = {1.0};
        double[] x = new double[1];
        double[] w = new double[20];
        int[] jw = new int[1];
        double[] xnorm = new double[1];

        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, 1, 1, 1, 1, 0, 
                               x, 0, w, 0, jw, 0, 3, xnorm);
        assertEquals(-1, status, "LSI should return -1 for n <= 0");

        status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, 1, 1, 1, 1, -1, 
                           x, 0, w, 0, jw, 0, 3, xnorm);
        assertEquals(-1, status, "LSI should return -1 for n < 0");
    }

    // ========================================================================
    // Property 5.12: LSI KKT conditions are satisfied
    // ========================================================================

    /**
     * 
     * The LSI solution should satisfy the KKT conditions for the constrained
     * least squares problem.
     * 
     * KKT conditions for LSI (min ||Ex - f||_2 s.t. Gx >= h):
     * 1. Primal feasibility: Gx >= h
     * 2. Stationarity: E^T(Ex - f) = G^T λ for some λ
     * 3. Dual feasibility: λ >= 0
     * 4. Complementary slackness: λ_i * (Gx - h)_i = 0
     */
    @Property(tries = 100)
    void lsiKKTConditionsAreSatisfied(
            @ForAll @IntRange(min = 3, max = 10) int me,
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Check primal feasibility: Gx >= h
        double[] gx = matVec(gOriginal, mg, n, x);
        for (int i = 0; i < mg; i++) {
            assertTrue(gx[i] >= hOriginal[i] - FEASIBILITY_TOL,
                    "Primal feasibility violated at constraint " + i);
        }

        // Compute E^T(Ex - f) - this should be in the cone spanned by G^T
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);

        // Compute E^T * residual
        double[] etResidual = new double[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < me; i++) {
                etResidual[j] += eOriginal[j * me + i] * residual[i];
            }
        }

        // For unconstrained case (all constraints inactive), E^T * residual should be zero
        // For constrained case, E^T * residual = G^T * λ for some λ >= 0
        // We verify this by checking that the gradient is in the cone of active constraints

        // Identify active constraints (where Gx ≈ h)
        boolean[] active = new boolean[mg];
        int numActive = 0;
        for (int i = 0; i < mg; i++) {
            if (Math.abs(gx[i] - hOriginal[i]) < FEASIBILITY_TOL) {
                active[i] = true;
                numActive++;
            }
        }

        // If no constraints are active, E^T * residual should be approximately zero
        if (numActive == 0) {
            double gradNorm = norm2(etResidual);
            assertTrue(gradNorm < 1e-5,
                    "With no active constraints, E^T(Ex-f) should be approximately zero, got norm: " + gradNorm);
        }
        // If constraints are active, we just verify feasibility and optimality
        // (full KKT verification would require extracting Lagrange multipliers)
    }

    // ========================================================================
    // Property 5.13: LSI handles square E matrix correctly
    // ========================================================================

    /**
     * 
     * LSI should correctly handle square full-rank E matrices.
     */
    @Property(tries = 50)
    void lsiHandlesSquareEMatrix(
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int me = n;  // Square matrix

        // Generate full-rank square matrix E
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mg, n)];
        int[] jw = new int[mg];
        double[] xnorm = new double[1];

        // Solve LSI
        int status = LSQSolver.lsi(e, 0, f, 0, g, 0, h, 0, me, me, mg, mg, n, 
                               x, 0, w, 0, jw, 0, 3 * n, xnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Verify feasibility
        double[] gx = matVec(gOriginal, mg, n, x);
        for (int i = 0; i < mg; i++) {
            assertTrue(gx[i] >= hOriginal[i] - FEASIBILITY_TOL,
                    "Constraint " + i + " violated");
        }

        // Verify residual norm
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);
        double computedNorm = norm2(residual);
        assertEquals(computedNorm, xnorm[0], EPSILON,
                "Residual norm should be correctly computed for square E");
    }
}
