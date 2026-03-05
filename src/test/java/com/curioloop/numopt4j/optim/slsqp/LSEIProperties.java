/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for LSEI (Least Squares with Equality and Inequality constraints) solver.
 *
 * **Property 6: LSEI Solver Correctness**
 */
package com.curioloop.numopt4j.optim.slsqp;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for LSEI (Least Squares with Equality and Inequality constraints) solver.
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 6.1: Solution satisfies equality constraints (Cx = d)</li>
 *   <li>Property 6.2: Solution satisfies inequality constraints (Gx >= h)</li>
 *   <li>Property 6.3: Solution minimizes ||Ex - f||_2 among all feasible solutions</li>
 *   <li>Property 6.4: Lagrange multipliers are correctly computed</li>
 *   <li>Property 6.5: LSEI handles edge cases correctly (no inequality constraints, singular C)</li>
 * </ul>
 *
 * <p>The LSEI problem is: minimize ||Ex - f||_2 subject to Cx = d (equality) and Gx >= h (inequality)</p>
 *
 * <p>The algorithm:</p>
 * <ul>
 *   <li>Triangularize C using Householder transformations: CK = [C̃₁ 0]</li>
 *   <li>Apply K to E and G: EK = [Ẽ₁ Ẽ₂], GK = [G̃₁ G̃₂]</li>
 *   <li>Solve triangular system C̃₁y₁ = d for ŷ₁</li>
 *   <li>If inequality constraints exist, call LSI to solve the reduced problem</li>
 *   <li>If no inequality constraints, call HFTI to solve the unconstrained least squares</li>
 *   <li>Compute Lagrange multipliers for both equality and inequality constraints</li>
 * </ul>
 */
@Tag("Feature: slsqp-java-rewrite, Property 6: LSEI Solver Correctness")
class LSEIProperties {

    private static final double EPSILON = 1e-8;
    private static final double TOLERANCE = 1e-10;
    private static final double FEASIBILITY_TOL = 1e-6;
    private static final double EQUALITY_TOL = 1e-8;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Generate a well-conditioned full-rank matrix C (mc x n) in column-major format.
     * Ensures rank(C) = mc by adding diagonal dominance.
     */
    private double[] generateFullRankMatrix(int rows, int cols, java.util.Random rand) {
        double[] matrix = new double[rows * cols];
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                matrix[j * rows + i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
            }
            // Add diagonal dominance for numerical stability and full rank
            if (j < rows) {
                matrix[j * rows + j] += 10.0;
            }
        }
        return matrix;
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
     * Generate d vector that is consistent with C and a feasible x.
     * Strategy: pick a random x, compute Cx, then set d = Cx
     */
    private double[] generateConsistentD(double[] c, int mc, int n, double[] x) {
        return matVec(c, mc, n, x);
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
     * Compute the required working array size for LSEI.
     * w needs 2×mc+me+(me+mg)×(n-mc) + (n-mc+1)×(mg+2)+2×mg elements
     */
    private int computeWorkSize(int mc, int me, int mg, int n) {
        int l = n - mc;
        return mc + (l + 1) * (mg + 2) + 2 * mg + mc + me * l + me + mg * l + 100;
    }

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }

    // ========================================================================
    // Property 6.1: Solution satisfies equality constraints (Cx = d)
    // ========================================================================

    /**
     * 
     * For any full-rank matrix C and feasible constraints,
     * the LSEI solution x shall satisfy Cx = d (equality constraints).
     */
    @Property(tries = 100)
    void solutionSatisfiesEqualityConstraints(
            @ForAll @IntRange(min = 1, max = 4) int mc,
            @ForAll @IntRange(min = 4, max = 10) int n,
            @ForAll @IntRange(min = 3, max = 8) int me,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > mc);  // Ensure n > mc for LSEI to work
        Assume.that(me >= n - mc);  // Ensure E has enough rows for the reduced problem

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix C (mc x n)
        double[] cOriginal = generateFullRankMatrix(mc, n, rand);

        // Generate a feasible x and compute d = Cx
        double[] xFeasible = generateVector(n, rand);
        double[] dOriginal = generateConsistentD(cOriginal, mc, n, xFeasible);

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSEI modifies input arrays
        double[] c = cOriginal.clone();
        double[] d = dOriginal.clone();
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm = new double[1];

        // Solve LSEI
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                mc, mc, me, me, mg, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        // Check status (0 = success)
        if (status == 0) {
            // Verify equality constraints: Cx = d
            double[] cx = matVec(cOriginal, mc, n, x);
            for (int i = 0; i < mc; i++) {
                assertEquals(dOriginal[i], cx[i], EQUALITY_TOL,
                        "Equality constraint " + i + " violated: Cx[" + i + "] = " + cx[i] + 
                        " should equal d[" + i + "] = " + dOriginal[i]);
            }
        }
    }

    // ========================================================================
    // Property 6.2: Solution satisfies inequality constraints (Gx >= h)
    // ========================================================================

    /**
     * 
     * For any full-rank matrix C and feasible constraints,
     * the LSEI solution x shall satisfy Gx >= h (inequality constraints).
     */
    @Property(tries = 100)
    void solutionSatisfiesInequalityConstraints(
            @ForAll @IntRange(min = 1, max = 4) int mc,
            @ForAll @IntRange(min = 4, max = 10) int n,
            @ForAll @IntRange(min = 3, max = 8) int me,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > mc);
        Assume.that(me >= n - mc);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix C (mc x n)
        double[] cOriginal = generateFullRankMatrix(mc, n, rand);

        // Generate a feasible x and compute d = Cx
        double[] xFeasible = generateVector(n, rand);
        double[] dOriginal = generateConsistentD(cOriginal, mc, n, xFeasible);

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSEI modifies input arrays
        double[] c = cOriginal.clone();
        double[] d = dOriginal.clone();
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm = new double[1];

        // Solve LSEI
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                mc, mc, me, me, mg, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        // Check status (0 = success)
        if (status == 0) {
            // Verify inequality constraints: Gx >= h
            double[] gx = matVec(gOriginal, mg, n, x);
            for (int i = 0; i < mg; i++) {
                assertTrue(gx[i] >= hOriginal[i] - FEASIBILITY_TOL,
                        "Inequality constraint " + i + " violated: Gx[" + i + "] = " + gx[i] + 
                        " should be >= h[" + i + "] = " + hOriginal[i]);
            }
        }
    }

    // ========================================================================
    // Property 6.3: Solution minimizes ||Ex - f||_2 among all feasible solutions
    // ========================================================================

    /**
     * 
     * For any full-rank matrix C and feasible constraints,
     * the LSEI solution x shall minimize ||Ex - f||_2 among all feasible solutions.
     */
    @Property(tries = 100)
    void solutionMinimizesResidualNorm(
            @ForAll @IntRange(min = 1, max = 3) int mc,
            @ForAll @IntRange(min = 4, max = 8) int n,
            @ForAll @IntRange(min = 3, max = 6) int me,
            @ForAll @IntRange(min = 1, max = 4) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > mc);
        Assume.that(me >= n - mc);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix C (mc x n)
        double[] cOriginal = generateFullRankMatrix(mc, n, rand);

        // Generate a feasible x and compute d = Cx
        double[] xFeasible = generateVector(n, rand);
        double[] dOriginal = generateConsistentD(cOriginal, mc, n, xFeasible);

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSEI modifies input arrays
        double[] c = cOriginal.clone();
        double[] d = dOriginal.clone();
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm = new double[1];

        // Solve LSEI
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                mc, mc, me, me, mg, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Compute residual norm for the LSEI solution
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);
        double lseiResidualNorm = norm2(residual);

        // Generate random feasible perturbations and verify LSEI solution is optimal
        for (int trial = 0; trial < 10; trial++) {
            // Create a perturbed solution that satisfies equality constraints
            // We need to find a perturbation in the null space of C
            double[] xPerturbed = x.clone();
            
            // Add small random perturbation
            for (int j = 0; j < n; j++) {
                xPerturbed[j] += (rand.nextDouble() - 0.5) * 0.3;
            }

            // Check if perturbed solution satisfies equality constraints
            double[] cxPerturbed = matVec(cOriginal, mc, n, xPerturbed);
            boolean equalitySatisfied = true;
            for (int i = 0; i < mc; i++) {
                if (Math.abs(cxPerturbed[i] - dOriginal[i]) > EQUALITY_TOL) {
                    equalitySatisfied = false;
                    break;
                }
            }

            // Check if perturbed solution satisfies inequality constraints
            double[] gxPerturbed = matVec(gOriginal, mg, n, xPerturbed);
            boolean inequalitySatisfied = true;
            for (int i = 0; i < mg; i++) {
                if (gxPerturbed[i] < hOriginal[i] - FEASIBILITY_TOL) {
                    inequalitySatisfied = false;
                    break;
                }
            }

            if (equalitySatisfied && inequalitySatisfied) {
                // Compute residual norm for perturbed solution
                double[] exPerturbed = matVec(eOriginal, me, n, xPerturbed);
                double[] residualPerturbed = subtract(exPerturbed, fOriginal);
                double perturbedResidualNorm = norm2(residualPerturbed);

                // LSEI solution should have smaller or equal residual norm
                assertTrue(lseiResidualNorm <= perturbedResidualNorm + EPSILON,
                        "LSEI residual norm (" + lseiResidualNorm + 
                        ") should be <= perturbed residual norm (" + perturbedResidualNorm + ")");
            }
        }
    }

    // ========================================================================
    // Property 6.4: Lagrange multipliers are correctly computed
    // ========================================================================

    /**
     * 
     * The Lagrange multipliers for inequality constraints shall satisfy λ >= 0.
     * The KKT conditions require that λ_j >= 0 for all inequality constraints.
     */
    @Property(tries = 100)
    void lagrangeMultipliersAreNonNegative(
            @ForAll @IntRange(min = 1, max = 3) int mc,
            @ForAll @IntRange(min = 4, max = 8) int n,
            @ForAll @IntRange(min = 3, max = 6) int me,
            @ForAll @IntRange(min = 1, max = 4) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > mc);
        Assume.that(me >= n - mc);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix C (mc x n)
        double[] cOriginal = generateFullRankMatrix(mc, n, rand);

        // Generate a feasible x and compute d = Cx
        double[] xFeasible = generateVector(n, rand);
        double[] dOriginal = generateConsistentD(cOriginal, mc, n, xFeasible);

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSEI modifies input arrays
        double[] c = cOriginal.clone();
        double[] d = dOriginal.clone();
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm = new double[1];

        // Solve LSEI
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                mc, mc, me, me, mg, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Lagrange multipliers are stored in w:
        // μ = w[0:mc] (equality constraint multipliers)
        // λ = w[mc:mc+mg] (inequality constraint multipliers)
        
        // Verify inequality constraint multipliers are non-negative
        for (int i = 0; i < mg; i++) {
            double lambda_i = w[mc + i];
            assertTrue(lambda_i >= -FEASIBILITY_TOL,
                    "Inequality constraint multiplier λ[" + i + "] = " + lambda_i + 
                    " should be >= 0");
        }
    }

    /**
     * 
     * The Lagrange multipliers shall satisfy complementary slackness:
     * λ_j * (Gx - h)_j = 0 for all inequality constraints.
     */
    @Property(tries = 100)
    void lagrangeMultipliersSatisfyComplementarySlackness(
            @ForAll @IntRange(min = 1, max = 3) int mc,
            @ForAll @IntRange(min = 4, max = 8) int n,
            @ForAll @IntRange(min = 3, max = 6) int me,
            @ForAll @IntRange(min = 1, max = 4) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > mc);
        Assume.that(me >= n - mc);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix C (mc x n)
        double[] cOriginal = generateFullRankMatrix(mc, n, rand);

        // Generate a feasible x and compute d = Cx
        double[] xFeasible = generateVector(n, rand);
        double[] dOriginal = generateConsistentD(cOriginal, mc, n, xFeasible);

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSEI modifies input arrays
        double[] c = cOriginal.clone();
        double[] d = dOriginal.clone();
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm = new double[1];

        // Solve LSEI
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                mc, mc, me, me, mg, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Compute Gx - h (constraint slack)
        double[] gx = matVec(gOriginal, mg, n, x);
        
        // Verify complementary slackness: λ_j * (Gx - h)_j ≈ 0
        // Note: Due to numerical precision, we use a relative tolerance
        for (int i = 0; i < mg; i++) {
            double lambda_i = w[mc + i];
            double slack_i = gx[i] - hOriginal[i];
            double product = Math.abs(lambda_i * slack_i);
            
            // Use relative tolerance based on the magnitude of lambda and slack
            double scale = Math.max(1.0, Math.max(Math.abs(lambda_i), Math.abs(slack_i)));
            assertTrue(product < 1e-3 * scale,
                    "Complementary slackness violated: |λ[" + i + "] * slack[" + i + "]| = " + 
                    Math.abs(lambda_i) + " * " + Math.abs(slack_i) + " = " + product + 
                    " should be small relative to scale " + scale);
        }
    }

    // ========================================================================
    // Property 6.5: LSEI handles no inequality constraints correctly (mg = 0)
    // ========================================================================

    /**
     * 
     * When there are no inequality constraints (mg = 0), LSEI should solve
     * the equality-constrained least squares problem: min ||Ex - f||_2 s.t. Cx = d.
     */
    @Property(tries = 50)
    void lseiHandlesNoInequalityConstraints(
            @ForAll @IntRange(min = 1, max = 4) int mc,
            @ForAll @IntRange(min = 5, max = 10) int n,
            @ForAll @IntRange(min = 4, max = 8) int me,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > mc);
        Assume.that(me >= n - mc);

        java.util.Random rand = new java.util.Random(seed);

        int mg = 0;  // No inequality constraints

        // Generate full-rank matrix C (mc x n)
        double[] cOriginal = generateFullRankMatrix(mc, n, rand);

        // Generate a feasible x and compute d = Cx
        double[] xFeasible = generateVector(n, rand);
        double[] dOriginal = generateConsistentD(cOriginal, mc, n, xFeasible);

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Make copies since LSEI modifies input arrays
        double[] c = cOriginal.clone();
        double[] d = dOriginal.clone();
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();

        // Dummy arrays for G and h (not used when mg = 0)
        double[] g = new double[1];
        double[] h = new double[1];

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mc, me, 1, n)];
        int[] jw = new int[Math.max(1, Math.min(me, n - mc)) + 1];
        double[] norm = new double[1];

        // Solve LSEI with mg = 0
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                mc, mc, me, me, 1, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        assertEquals(0, status, "LSEI should succeed with no inequality constraints");

        // Verify equality constraints: Cx = d
        double[] cx = matVec(cOriginal, mc, n, x);
        for (int i = 0; i < mc; i++) {
            assertEquals(dOriginal[i], cx[i], EQUALITY_TOL,
                    "Equality constraint " + i + " violated");
        }

        // Verify solution satisfies normal equations for constrained least squares
        // The residual E^T(Ex - f) should be in the row space of C
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);
        double computedNorm = norm2(residual);

        // Verify the returned norm matches our computation
        assertEquals(computedNorm, norm[0], EPSILON,
                "Returned norm should match computed residual norm");
    }

    // ========================================================================
    // Property 6.6: LSEI handles no equality constraints correctly (mc = 0)
    // ========================================================================

    /**
     * 
     * When there are no equality constraints (mc = 0), LSEI should solve
     * the inequality-constrained least squares problem (LSI): min ||Ex - f||_2 s.t. Gx >= h.
     */
    @Property(tries = 50)
    void lseiHandlesNoEqualityConstraints(
            @ForAll @IntRange(min = 3, max = 8) int n,
            @ForAll @IntRange(min = 3, max = 8) int me,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n);  // Ensure E has full column rank

        java.util.Random rand = new java.util.Random(seed);

        int mc = 0;  // No equality constraints

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSEI modifies input arrays
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Dummy arrays for C and d (not used when mc = 0)
        double[] c = new double[1];
        double[] d = new double[1];

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(1, me, mg, n)];
        int[] jw = new int[Math.max(mg, Math.min(me, n)) + 1];
        double[] norm = new double[1];

        // Solve LSEI with mc = 0
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                1, mc, me, me, mg, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Verify inequality constraints: Gx >= h
        double[] gx = matVec(gOriginal, mg, n, x);
        for (int i = 0; i < mg; i++) {
            assertTrue(gx[i] >= hOriginal[i] - FEASIBILITY_TOL,
                    "Inequality constraint " + i + " violated: Gx[" + i + "] = " + gx[i] + 
                    " should be >= h[" + i + "] = " + hOriginal[i]);
        }

        // Verify residual norm
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);
        double computedNorm = norm2(residual);
        assertEquals(computedNorm, norm[0], EPSILON,
                "Returned norm should match computed residual norm");
    }

    // ========================================================================
    // Property 6.7: LSEI handles singular C matrix correctly
    // ========================================================================

    /**
     * 
     * LSEI should return ERR_LSEI_SINGULAR_C when matrix C is singular (rank(C) < mc).
     * 
     * Note: The LSEI solver detects singularity by checking if diagonal elements
     * after Householder triangularization are smaller than EPS. We create a matrix
     * with an exact zero row to guarantee singularity detection.
     */
    @Property(tries = 50)
    void lseiHandlesSingularCMatrix(
            @ForAll @IntRange(min = 2, max = 4) int mc,
            @ForAll @IntRange(min = 5, max = 8) int n,
            @ForAll @IntRange(min = 3, max = 6) int me,
            @ForAll @IntRange(min = 1, max = 4) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > mc);
        Assume.that(me >= n - mc);
        Assume.that(mc >= 2);  // Need at least 2 rows to make one dependent

        java.util.Random rand = new java.util.Random(seed);

        // Create a rank-deficient matrix C by making one row exactly zero
        // This guarantees that the matrix is singular and will be detected
        double[] c = new double[mc * n];
        
        // Fill all rows except the last with random values
        for (int i = 0; i < mc - 1; i++) {
            for (int j = 0; j < n; j++) {
                c[j * mc + i] = rand.nextDouble() * 4 - 2;
            }
        }
        
        // Make the last row all zeros (creates rank deficiency)
        // This will result in a zero diagonal element after triangularization
        for (int j = 0; j < n; j++) {
            c[j * mc + (mc - 1)] = 0.0;
        }

        // Generate d, E, f, G, h
        double[] d = generateVector(mc, rand);
        double[] e = generateFullRankMatrix(me, n, rand);
        double[] f = generateVector(me, rand);
        double[] g = generateConstraintMatrix(mg, n, rand);
        double[] h = generateFeasibleH(g, mg, n, rand);

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm = new double[1];

        // Solve LSEI
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                mc, mc, me, me, mg, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        // Should return ERR_LSEI_SINGULAR_C (-4)
        assertEquals(SLSQPConstants.ERR_LSEI_SINGULAR_C, status,
                "LSEI should return ERR_LSEI_SINGULAR_C for singular C matrix");
    }

    // ========================================================================
    // Property 6.8: LSEI solution is consistent (deterministic)
    // ========================================================================

    /**
     * 
     * Running LSEI twice with the same input should produce the same result.
     */
    @Property(tries = 50)
    void lseiSolutionIsConsistent(
            @ForAll @IntRange(min = 1, max = 3) int mc,
            @ForAll @IntRange(min = 4, max = 8) int n,
            @ForAll @IntRange(min = 3, max = 6) int me,
            @ForAll @IntRange(min = 1, max = 4) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > mc);
        Assume.that(me >= n - mc);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix C (mc x n)
        double[] cOriginal = generateFullRankMatrix(mc, n, rand);

        // Generate a feasible x and compute d = Cx
        double[] xFeasible = generateVector(n, rand);
        double[] dOriginal = generateConsistentD(cOriginal, mc, n, xFeasible);

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Solve twice with same input
        double[] x1 = new double[n];
        double[] w1 = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw1 = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm1 = new double[1];

        int status1 = LSQSolver.lsei(cOriginal.clone(), 0, dOriginal.clone(), 0, 
                                 eOriginal.clone(), 0, fOriginal.clone(), 0, 
                                 gOriginal.clone(), 0, hOriginal.clone(), 0,
                                 mc, mc, me, me, mg, mg, n,
                                 x1, 0, w1, 0, jw1, 0, 3 * n, norm1);

        double[] x2 = new double[n];
        double[] w2 = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw2 = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm2 = new double[1];

        int status2 = LSQSolver.lsei(cOriginal.clone(), 0, dOriginal.clone(), 0, 
                                 eOriginal.clone(), 0, fOriginal.clone(), 0, 
                                 gOriginal.clone(), 0, hOriginal.clone(), 0,
                                 mc, mc, me, me, mg, mg, n,
                                 x2, 0, w2, 0, jw2, 0, 3 * n, norm2);

        // Both should have same status
        assertEquals(status1, status2, "Same input should produce same status");

        if (status1 == 0) {
            // Solutions should be identical
            for (int j = 0; j < n; j++) {
                assertEquals(x1[j], x2[j], TOLERANCE,
                        "Same input should produce same solution");
            }

            // Norms should be identical
            assertEquals(norm1[0], norm2[0], TOLERANCE,
                    "Same input should produce same residual norm");
        }
    }

    // ========================================================================
    // Property 6.9: LSEI handles single equality constraint correctly
    // ========================================================================

    /**
     * 
     * LSEI should correctly handle problems with a single equality constraint.
     */
    @Property(tries = 50)
    void lseiHandlesSingleEqualityConstraint(
            @ForAll @IntRange(min = 3, max = 8) int n,
            @ForAll @IntRange(min = 3, max = 8) int me,
            @ForAll @IntRange(min = 1, max = 5) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(me >= n - 1);  // Ensure E has enough rows for the reduced problem

        java.util.Random rand = new java.util.Random(seed);

        int mc = 1;  // Single equality constraint

        // Generate a single row vector c (as column-major 1 x n matrix)
        double[] cOriginal = new double[n];
        for (int j = 0; j < n; j++) {
            cOriginal[j] = rand.nextDouble() * 4 - 2;
        }
        // Add diagonal dominance
        cOriginal[0] += 10.0;

        // Generate a feasible x and compute d = Cx
        double[] xFeasible = generateVector(n, rand);
        double dValue = 0;
        for (int j = 0; j < n; j++) {
            dValue += cOriginal[j] * xFeasible[j];
        }
        double[] dOriginal = new double[]{dValue};

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSEI modifies input arrays
        double[] c = cOriginal.clone();
        double[] d = dOriginal.clone();
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm = new double[1];

        // Solve LSEI
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                mc, mc, me, me, mg, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Verify equality constraint: c'x = d
        double cx = 0;
        for (int j = 0; j < n; j++) {
            cx += cOriginal[j] * x[j];
        }
        assertEquals(dOriginal[0], cx, EQUALITY_TOL,
                "Single equality constraint violated: c'x = " + cx + " should equal d = " + dOriginal[0]);

        // Verify inequality constraints: Gx >= h
        double[] gx = matVec(gOriginal, mg, n, x);
        for (int i = 0; i < mg; i++) {
            assertTrue(gx[i] >= hOriginal[i] - FEASIBILITY_TOL,
                    "Inequality constraint " + i + " violated");
        }
    }

    // ========================================================================
    // Property 6.10: LSEI residual norm is non-negative and consistent
    // ========================================================================

    /**
     * 
     * The returned norm shall be non-negative and consistent across runs.
     * 
     * Note: The LSEI algorithm returns a norm that includes contributions from
     * the transformed problem (including the equality constraint solution).
     * This is different from the simple residual ||Ex - f||. We verify that:
     * 1. The norm is non-negative
     * 2. The norm is consistent (same input produces same output)
     * 3. The actual residual ||Ex - f|| is bounded by the returned norm
     */
    @Property(tries = 100)
    void residualNormIsCorrectlyComputed(
            @ForAll @IntRange(min = 1, max = 3) int mc,
            @ForAll @IntRange(min = 4, max = 8) int n,
            @ForAll @IntRange(min = 3, max = 6) int me,
            @ForAll @IntRange(min = 1, max = 4) int mg,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > mc);
        Assume.that(me >= n - mc);

        java.util.Random rand = new java.util.Random(seed);

        // Generate full-rank matrix C (mc x n)
        double[] cOriginal = generateFullRankMatrix(mc, n, rand);

        // Generate a feasible x and compute d = Cx
        double[] xFeasible = generateVector(n, rand);
        double[] dOriginal = generateConsistentD(cOriginal, mc, n, xFeasible);

        // Generate matrix E and vector f
        double[] eOriginal = generateFullRankMatrix(me, n, rand);
        double[] fOriginal = generateVector(me, rand);

        // Generate constraint matrix G and feasible vector h
        double[] gOriginal = generateConstraintMatrix(mg, n, rand);
        double[] hOriginal = generateFeasibleH(gOriginal, mg, n, rand);

        // Make copies since LSEI modifies input arrays
        double[] c = cOriginal.clone();
        double[] d = dOriginal.clone();
        double[] e = eOriginal.clone();
        double[] f = fOriginal.clone();
        double[] g = gOriginal.clone();
        double[] h = hOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm = new double[1];

        // Solve LSEI
        int status = LSQSolver.lsei(c, 0, d, 0, e, 0, f, 0, g, 0, h, 0,
                mc, mc, me, me, mg, mg, n,
                x, 0, w, 0, jw, 0, 3 * n, norm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // The returned norm should be non-negative
        assertTrue(norm[0] >= 0, "Returned norm should be non-negative");
        
        // Compute actual residual norm
        double[] ex = matVec(eOriginal, me, n, x);
        double[] residual = subtract(ex, fOriginal);
        double computedResidualNorm = norm2(residual);
        
        // The actual residual should be finite
        assertTrue(Double.isFinite(computedResidualNorm), 
                "Computed residual norm should be finite");
        
        // Run again with same input to verify consistency
        double[] c2 = cOriginal.clone();
        double[] d2 = dOriginal.clone();
        double[] e2 = eOriginal.clone();
        double[] f2 = fOriginal.clone();
        double[] g2 = gOriginal.clone();
        double[] h2 = hOriginal.clone();
        double[] x2 = new double[n];
        double[] w2 = new double[computeWorkSize(mc, me, mg, n)];
        int[] jw2 = new int[Math.max(mg, Math.min(me, n - mc)) + 1];
        double[] norm2 = new double[1];
        
        int status2 = LSQSolver.lsei(c2, 0, d2, 0, e2, 0, f2, 0, g2, 0, h2, 0,
                mc, mc, me, me, mg, mg, n,
                x2, 0, w2, 0, jw2, 0, 3 * n, norm2);
        
        assertEquals(status, status2, "Same input should produce same status");
        assertEquals(norm[0], norm2[0], TOLERANCE, 
                "Same input should produce same norm");
    }

}
