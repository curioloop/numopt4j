/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for LineSearch (Armijo backtracking and Golden Section search).
 *
 * **Property 9: Line Search Armijo Condition**
 */
package com.curioloop.numopt4j.optim.slsqp;

import com.curioloop.numopt4j.optim.slsqp.LineSearch;
import com.curioloop.numopt4j.optim.slsqp.SLSQPConstants;
import com.curioloop.numopt4j.optim.slsqp.SLSQPWorkspace;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for LineSearch (Armijo backtracking and Golden Section search).
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 9.1: Armijo condition - when h1 <= h3/10, line search terminates (continue flag = 0)</li>
 *   <li>Property 9.2: Armijo step reduction - when h1 > h3/10, step is reduced using quadratic interpolation</li>
 *   <li>Property 9.3: Step length bounds - new alpha is clamped to [alphaMin, alphaMax]</li>
 *   <li>Property 9.4: Golden section convergence - findMin converges to a local minimum</li>
 *   <li>Property 9.5: Golden section bracketing - the interval [a, b] shrinks at each iteration</li>
 *   <li>Property 9.6: Exact search integration - exactSearch correctly updates position x = x0 + alpha * s</li>
 * </ul>
 *
 * <p>The Armijo condition ensures sufficient decrease:</p>
 * <pre>
 * φ(x + αs) - φ(x) ≤ η · α · ∇φ(s)
 * </pre>
 * <p>where φ is the merit function with L1 penalty and η = 0.1.</p>
 *
 * <p>The exact line search uses Brent's method which combines:</p>
 * <ul>
 *   <li>Golden section search for guaranteed convergence</li>
 *   <li>Quadratic interpolation for faster convergence when applicable</li>
 * </ul>
 */
@Tag("Feature: slsqp-java-rewrite, Property 9: Line Search Armijo Condition")
class LineSearchProperties {

    private static final double TOLERANCE = 1e-10;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Generate a random vector of length n.
     */
    private double[] generateVector(int n, java.util.Random rand) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
        }
        return v;
    }

    /**
     * Generate a descent direction (negative gradient direction).
     */
    private double[] generateDescentDirection(int n, java.util.Random rand) {
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            // Generate a direction with negative components (descent)
            s[i] = -(rand.nextDouble() * 0.5 + 0.1);  // Values in [-0.6, -0.1]
        }
        return s;
    }

    /**
     * Compute dot product of two vectors.
     */
    private double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * Simple quadratic function: f(x) = sum(x_i^2)
     */
    private double quadraticFunction(double[] x) {
        double sum = 0;
        for (double xi : x) {
            sum += xi * xi;
        }
        return sum;
    }

    /**
     * Gradient of quadratic function: g(x) = 2*x
     */
    private double[] quadraticGradient(double[] x) {
        double[] g = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            g[i] = 2 * x[i];
        }
        return g;
    }

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }

    // ========================================================================
    // Property 9.1: Armijo condition - when h1 <= h3/10, line search terminates
    // ========================================================================

    /**
     *
     * When the Armijo condition is satisfied (h1 <= h3/10), the line search
     * should terminate with continue flag = 0.
     */
    @Property(tries = 100)
    void armijoConditionSatisfiedTerminatesSearch(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate initial position and search direction
        double[] x0 = generateVector(n, rand);
        double[] s = generateDescentDirection(n, rand);
        double[] x = x0.clone();

        // Compute initial merit function value
        double t0 = quadraticFunction(x0);

        // Compute directional derivative h3 = grad(f)^T * s (should be negative for descent)
        double[] grad = quadraticGradient(x0);
        double h3 = dot(grad, s);

        // Ensure h3 is negative (descent direction)
        Assume.that(h3 < -TOLERANCE);

        // Take a step: x = x0 + alpha * s
        double alpha = 0.5;
        for (int i = 0; i < n; i++) {
            x[i] = x0[i] + alpha * s[i];
        }

        // Compute new merit function value
        double t = quadraticFunction(x);

        // Compute h1 = t - t0 (should be negative for sufficient decrease)
        double h1 = t - t0;

        // If Armijo condition is satisfied (h1 <= h3/10), search should terminate
        // Note: h3 < 0, so h3/10 < 0, and h1 <= h3/10 means sufficient decrease
        if (h1 <= h3 / 10.0) {
            double[] result = new double[3];
            LineSearch.armijo(n, x, 0, x0, 0, s, 0, t, t0, h3, alpha,
                    0.1, 1.0, 1, null, 0, null, 0, result);

            // Continue flag should be 0 (done)
            assertEquals(0.0, result[2], TOLERANCE,
                    "Armijo condition satisfied: search should terminate (continue flag = 0)");
        }
    }

    // ========================================================================
    // Property 9.2: Armijo step reduction - when h1 > h3/10, step is reduced
    // ========================================================================

    /**
     *
     * When the Armijo condition is NOT satisfied (h1 > h3/10), the step
     * should be reduced using quadratic interpolation.
     */
    @Property(tries = 100)
    void armijoStepReductionWhenConditionNotSatisfied(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate initial position
        double[] x0 = generateVector(n, rand);

        // Generate a direction that will NOT satisfy Armijo condition initially
        // Use a large step that overshoots
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = rand.nextDouble() * 2 + 1;  // Large positive step
        }

        double[] x = x0.clone();

        // Compute initial merit function value
        double t0 = quadraticFunction(x0);

        // Compute directional derivative h3 (make it negative artificially for test)
        double h3 = -1.0;

        // Take a large step that will NOT satisfy Armijo
        double alpha = 1.0;
        for (int i = 0; i < n; i++) {
            x[i] = x0[i] + alpha * s[i];
        }

        // Compute new merit function value (will be larger due to large step)
        double t = quadraticFunction(x);
        double h1 = t - t0;

        // Ensure Armijo condition is NOT satisfied
        Assume.that(h1 > h3 / 10.0);

        double[] result = new double[3];
        LineSearch.armijo(n, x, 0, x0, 0, s, 0, t, t0, h3, alpha,
                0.1, 1.0, 1, null, 0, null, 0, result);

        // Continue flag should be 1 (continue searching)
        assertEquals(1.0, result[2], TOLERANCE,
                "Armijo condition not satisfied: search should continue (continue flag = 1)");

        // New alpha should be reduced
        assertTrue(result[0] < alpha,
                "Step should be reduced when Armijo condition not satisfied");
    }

    // ========================================================================
    // Property 9.3: Step length bounds - new alpha is clamped to [alphaMin, alphaMax]
    // ========================================================================

    /**
     *
     * The new step length alpha should always be clamped to [alphaMin, alphaMax].
     */
    @Property(tries = 100)
    void stepLengthIsBounded(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @DoubleRange(min = 0.05, max = 0.2) double alphaMin,
            @ForAll @DoubleRange(min = 0.8, max = 1.0) double alphaMax,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(alphaMin < alphaMax);

        java.util.Random rand = new java.util.Random(seed);

        // Generate initial position
        double[] x0 = generateVector(n, rand);
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = rand.nextDouble() * 2 + 1;
        }
        double[] x = x0.clone();

        double t0 = quadraticFunction(x0);
        double h3 = -1.0;
        double alpha = 1.0;

        for (int i = 0; i < n; i++) {
            x[i] = x0[i] + alpha * s[i];
        }

        double t = quadraticFunction(x);
        double h1 = t - t0;

        // Ensure Armijo condition is NOT satisfied to trigger step reduction
        Assume.that(h1 > h3 / 10.0);

        double[] result = new double[3];
        LineSearch.armijo(n, x, 0, x0, 0, s, 0, t, t0, h3, alpha,
                alphaMin, alphaMax, 1, null, 0, null, 0, result);

        // If search continues, new alpha should be within bounds
        if (result[2] > 0.5) {  // Continue flag = 1
            assertTrue(result[0] >= alphaMin - TOLERANCE,
                    "New alpha (" + result[0] + ") should be >= alphaMin (" + alphaMin + ")");
            assertTrue(result[0] <= alphaMax + TOLERANCE,
                    "New alpha (" + result[0] + ") should be <= alphaMax (" + alphaMax + ")");
        }
    }

    // ========================================================================
    // Property 9.4: Golden section convergence - findMin converges to a local minimum
    // ========================================================================

    /**
     *
     * The golden section search (findMin) should converge to a local minimum
     * of a unimodal function.
     */
    @Property(tries = 50)
    void goldenSectionConvergesToMinimum(
            @ForAll @DoubleRange(min = 0.0, max = 0.3) double alphaLower,
            @ForAll @DoubleRange(min = 0.7, max = 1.0) double alphaUpper,
            @ForAll @DoubleRange(min = 0.3, max = 0.7) double trueMin,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(alphaLower < trueMin && trueMin < alphaUpper);

        // Create a simple unimodal function: f(alpha) = (alpha - trueMin)^2
        // This has a minimum at alpha = trueMin

        SLSQPWorkspace.FindWork work = new SLSQPWorkspace.FindWork();
        int[] mode = new int[] { SLSQPConstants.FIND_NOOP };

        double tol = 1e-6;
        double alpha = 0;
        int maxIter = 100;
        int iter = 0;

        while (mode[0] != SLSQPConstants.FIND_CONV && iter < maxIter) {
            // Evaluate function at current alpha
            double f = (alpha - trueMin) * (alpha - trueMin);

            // Get next evaluation point
            alpha = LineSearch.findMin(mode, work, f, tol, alphaLower, alphaUpper);
            iter++;
        }

        // Should converge
        assertEquals(SLSQPConstants.FIND_CONV, mode[0],
                "Golden section search should converge");

        // Final alpha should be close to true minimum
        assertEquals(trueMin, alpha, 0.01,
                "Golden section should find minimum at " + trueMin + ", got " + alpha);
    }

    // ========================================================================
    // Property 9.5: Golden section bracketing - interval [a, b] shrinks
    // ========================================================================

    /**
     *
     * During golden section search, the bracketing interval [a, b] should
     * shrink at each iteration (or stay the same).
     */
    @Property(tries = 50)
    void goldenSectionIntervalShrinks(
            @ForAll @DoubleRange(min = 0.0, max = 0.2) double alphaLower,
            @ForAll @DoubleRange(min = 0.8, max = 1.0) double alphaUpper,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(alphaLower < alphaUpper);

        java.util.Random rand = new java.util.Random(seed);
        double trueMin = alphaLower + rand.nextDouble() * (alphaUpper - alphaLower);

        SLSQPWorkspace.FindWork work = new SLSQPWorkspace.FindWork();
        int[] mode = new int[] { SLSQPConstants.FIND_NOOP };

        double tol = 1e-6;
        double alpha = 0;
        int maxIter = 50;
        int iter = 0;

        double prevIntervalSize = alphaUpper - alphaLower;

        while (mode[0] != SLSQPConstants.FIND_CONV && iter < maxIter) {
            double f = (alpha - trueMin) * (alpha - trueMin);
            alpha = LineSearch.findMin(mode, work, f, tol, alphaLower, alphaUpper);

            // After initialization, check interval size
            if (mode[0] == SLSQPConstants.FIND_NEXT || mode[0] == SLSQPConstants.FIND_CONV) {
                double currentIntervalSize = work.b - work.a;

                // Interval should not grow
                assertTrue(currentIntervalSize <= prevIntervalSize + TOLERANCE,
                        "Interval should not grow: prev=" + prevIntervalSize + 
                        ", current=" + currentIntervalSize);

                prevIntervalSize = currentIntervalSize;
            }

            iter++;
        }
    }

    // ========================================================================
    // Property 9.6: Exact search integration - x = x0 + alpha * s
    // ========================================================================

    /**
     *
     * The exactSearch method should correctly update position x = x0 + alpha * s.
     */
    @Property(tries = 50)
    void exactSearchUpdatesPositionCorrectly(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Create workspace
        SLSQPWorkspace ws = new SLSQPWorkspace(); ws.ensure(n, 0, 0);

        // Generate initial position and search direction
        double[] x0 = generateVector(n, rand);
        double[] s = generateDescentDirection(n, rand);
        double[] x = new double[n];

        // Compute initial merit function value
        double t0 = quadraticFunction(x0);

        // Initialize line search
        ws.setLineMode(SLSQPConstants.FIND_NOOP);

        double alphaLower = 0.1;
        double alphaUpper = 1.0;
        double tol = 1e-6;

        // Run one iteration of exact search
        LineSearch.exactSearch(ws, t0, tol, alphaLower, alphaUpper,
                n, x, 0, x0, 0, s, 0);

        // Get the alpha value
        double alpha = ws.getAlpha();

        // Verify x = x0 + alpha * s
        for (int i = 0; i < n; i++) {
            double expected = x0[i] + alpha * s[i];
            assertEquals(expected, x[i], TOLERANCE,
                    "x[" + i + "] should equal x0[" + i + "] + alpha * s[" + i + "]");
        }

    }

    // ========================================================================
    // Property 9.7: Armijo terminates after max iterations
    // ========================================================================

    /**
     *
     * The Armijo line search should terminate after maximum iterations (lineCount > 10).
     */
    @Property(tries = 50)
    void armijoTerminatesAfterMaxIterations(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] x0 = generateVector(n, rand);
        double[] s = generateVector(n, rand);
        double[] x = x0.clone();

        double t0 = 1.0;
        double t = 2.0;  // Worse than t0
        double h3 = -0.1;  // Negative directional derivative
        double alpha = 1.0;

        // With lineCount > 10, should terminate regardless of Armijo condition
        double[] result = new double[3];
        LineSearch.armijo(n, x, 0, x0, 0, s, 0, t, t0, h3, alpha,
                0.1, 1.0, 11, null, 0, null, 0, result);

        // Should terminate (continue flag = 0)
        assertEquals(0.0, result[2], TOLERANCE,
                "Armijo should terminate after max iterations (lineCount > 10)");
    }

    // ========================================================================
    // Property 9.8: Armijo respects bounds
    // ========================================================================

    /**
     *
     * When bounds are provided, the Armijo line search should project
     * the updated position onto the bounds.
     */
    @Property(tries = 100)
    void armijoRespectsVariableBounds(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate bounds
        double[] lower = new double[n];
        double[] upper = new double[n];
        for (int i = 0; i < n; i++) {
            lower[i] = -1.0;
            upper[i] = 1.0;
        }

        // Generate initial position within bounds
        double[] x0 = new double[n];
        for (int i = 0; i < n; i++) {
            x0[i] = rand.nextDouble() * 1.5 - 0.75;  // Values in [-0.75, 0.75]
        }

        // Generate a large search direction that would violate bounds
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
        }

        double[] x = x0.clone();

        double t0 = 1.0;
        double t = 2.0;  // Worse than t0 to trigger step reduction
        double h3 = -0.1;
        double alpha = 1.0;

        double[] result = new double[3];
        LineSearch.armijo(n, x, 0, x0, 0, s, 0, t, t0, h3, alpha,
                0.1, 1.0, 1, lower, 0, upper, 0, result);

        // If search continues, verify bounds are respected
        if (result[2] > 0.5) {  // Continue flag = 1
            for (int i = 0; i < n; i++) {
                assertTrue(x[i] >= lower[i] - TOLERANCE,
                        "x[" + i + "] = " + x[i] + " should be >= lower[" + i + "] = " + lower[i]);
                assertTrue(x[i] <= upper[i] + TOLERANCE,
                        "x[" + i + "] = " + x[i] + " should be <= upper[" + i + "] = " + upper[i]);
            }
        }
    }

    // ========================================================================
    // Property 9.9: Golden section with quadratic function
    // ========================================================================

    /**
     *
     * Golden section search should find the minimum of a quadratic function
     * along a line.
     */
    @Property(tries = 50)
    void goldenSectionFindsQuadraticMinimum(
            @ForAll @IntRange(min = 2, max = 5) int n,
            @ForAll @DoubleRange(min = 0.1, max = 2.0) double alphaOptimal,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate a descent direction, then construct x0 so the line minimum is
        // guaranteed to occur at the generated alphaOptimal.
        double[] s = generateDescentDirection(n, rand);
        double[] x0 = new double[n];
        for (int i = 0; i < n; i++) {
            x0[i] = -alphaOptimal * s[i];
        }

        // For quadratic f(x) = ||x||^2 with x0 = -alphaOptimal * s,
        // the minimum along x0 + alpha*s occurs exactly at alpha = alphaOptimal.

        double alphaLower = 0.0;
        double alphaUpper = Math.max(2.0, alphaOptimal * 1.5);

        SLSQPWorkspace.FindWork work = new SLSQPWorkspace.FindWork();
        int[] mode = new int[] { SLSQPConstants.FIND_NOOP };

        double tol = 1e-6;
        double alpha = 0;
        int maxIter = 100;
        int iter = 0;

        while (mode[0] != SLSQPConstants.FIND_CONV && iter < maxIter) {
            // Compute x = x0 + alpha * s
            double[] x = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = x0[i] + alpha * s[i];
            }

            // Evaluate quadratic function
            double f = quadraticFunction(x);

            // Get next evaluation point
            alpha = LineSearch.findMin(mode, work, f, tol, alphaLower, alphaUpper);
            iter++;
        }

        // Should converge
        assertEquals(SLSQPConstants.FIND_CONV, mode[0],
                "Golden section search should converge");

        // Final alpha should be close to optimal
        assertEquals(alphaOptimal, alpha, 0.05,
                "Golden section should find optimal alpha=" + alphaOptimal + ", got " + alpha);
    }

    // ========================================================================
    // Property 9.10: Exact search convergence
    // ========================================================================

    /**
     *
     * The exactSearch method should eventually converge (return FIND_CONV).
     */
    @Property(tries = 50)
    void exactSearchEventuallyConverges(
            @ForAll @IntRange(min = 2, max = 5) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        SLSQPWorkspace ws = new SLSQPWorkspace(); ws.ensure(n, 0, 0);

        double[] x0 = generateVector(n, rand);
        double[] s = generateDescentDirection(n, rand);
        double[] x = new double[n];

        double alphaLower = 0.1;
        double alphaUpper = 1.0;
        double tol = 1e-6;

        ws.setLineMode(SLSQPConstants.FIND_NOOP);

        int maxIter = 100;
        int iter = 0;

        while (ws.getLineMode() != SLSQPConstants.FIND_CONV && iter < maxIter) {
            // Compute merit function at current x
            double t = quadraticFunction(x);

            LineSearch.exactSearch(ws, t, tol, alphaLower, alphaUpper,
                    n, x, 0, x0, 0, s, 0);
            iter++;
        }

        // Should converge within max iterations
        assertEquals(SLSQPConstants.FIND_CONV, ws.getLineMode(),
                "Exact search should converge within " + maxIter + " iterations");

    }

    // ========================================================================
    // Property 9.11: Deterministic behavior
    // ========================================================================

    /**
     *
     * Running line search twice with the same input should produce the same result.
     */
    @Property(tries = 50)
    void lineSearchIsDeterministic(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] x0 = generateVector(n, rand);
        double[] s = generateVector(n, rand);

        double t0 = 1.0;
        double t = 1.5;
        double h3 = -0.5;
        double alpha = 1.0;

        // First run
        double[] x1 = x0.clone();
        double[] s1 = s.clone();
        double[] result1 = new double[3];
        LineSearch.armijo(n, x1, 0, x0.clone(), 0, s1, 0, t, t0, h3, alpha,
                0.1, 1.0, 1, null, 0, null, 0, result1);

        // Second run with same input
        double[] x2 = x0.clone();
        double[] s2 = s.clone();
        double[] result2 = new double[3];
        LineSearch.armijo(n, x2, 0, x0.clone(), 0, s2, 0, t, t0, h3, alpha,
                0.1, 1.0, 1, null, 0, null, 0, result2);

        // Results should be identical
        assertEquals(result1[0], result2[0], TOLERANCE, "Alpha should be identical");
        assertEquals(result1[1], result2[1], TOLERANCE, "h3 should be identical");
        assertEquals(result1[2], result2[2], TOLERANCE, "Continue flag should be identical");

        // Updated x should be identical
        for (int i = 0; i < n; i++) {
            assertEquals(x1[i], x2[i], TOLERANCE, "x[" + i + "] should be identical");
        }
    }
}
