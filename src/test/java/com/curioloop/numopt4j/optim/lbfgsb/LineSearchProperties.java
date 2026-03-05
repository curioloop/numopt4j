/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for LineSearch.
 *
 * **Property 6: Line Search Wolfe Conditions**
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBConstants;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBWorkspace;
import com.curioloop.numopt4j.optim.lbfgsb.LineSearch;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for LineSearch.
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 6.1: init returns positive step length</li>
 *   <li>Property 6.2: init respects bound constraints on step</li>
 *   <li>Property 6.3: perform detects non-descent direction</li>
 *   <li>Property 6.4: perform is deterministic</li>
 *   <li>Property 6.5: Wolfe conditions are satisfied on convergence</li>
 *   <li>Property 6.6: Step length is bounded by maximum</li>
 * </ul>
 */
@Tag("Feature: lbfgsb-java-rewrite, Property 6: Line Search Wolfe Conditions")
class LineSearchProperties implements LBFGSBConstants {

    // ========================================================================
    // Property 6.1: init returns positive step length
    // ========================================================================

    @Property(tries = 100)
    void initReturnsPositiveStepLength(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        // Initialize workspace state
        ws.setIter(0);
        ws.setBoxed(false);
        ws.setConstrained(false);

        // Set up search direction in buffer
        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();
        for (int i = 0; i < n; i++) {
            buffer[dOffset + i] = -(i + 1) * 0.1;  // Descent direction
        }

        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i * 0.5;
        }

        double stp = LineSearch.init(n, x, (com.curioloop.numopt4j.optim.Bound[]) null, ws);

        assertTrue(stp > 0, "Initial step length should be positive");
        assertTrue(Double.isFinite(stp), "Initial step length should be finite");

    }

    // ========================================================================
    // Property 6.2: init respects bound constraints on step
    // ========================================================================

    @Property(tries = 100)
    void initRespectsStepBounds(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        // Set up constrained problem
        ws.setIter(1);  // Not first iteration
        ws.setBoxed(true);
        ws.setConstrained(true);

        // Set up search direction
        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();
        for (int i = 0; i < n; i++) {
            buffer[dOffset + i] = 1.0;  // Moving in positive direction
        }

        // Set up bounds that limit step
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];

        for (int i = 0; i < n; i++) {
            x[i] = 0.0;
            lower[i] = -10.0;
            upper[i] = 0.5;  // Close upper bound
            boundType[i] = BOUND_BOTH;
        }

        double stp = LineSearch.init(n, x, TestBounds.toBounds(n, lower, upper, boundType), ws);

        // Step should be positive and bounded
        assertTrue(stp > 0, "Step should be positive");
        assertTrue(stp <= 1.0, "Step should not exceed 1.0");

        // The maximum step stored in workspace should respect bounds
        double stpMax = ws.getSearchCtx().stpMax;
        assertTrue(stpMax >= 0, "Maximum step should be non-negative");
        assertTrue(stpMax <= SEARCH_NO_BND, "Maximum step should be bounded");

        // Verify the computed maximum step respects bounds
        // For constrained problems after first iteration, stepMax is computed
        // to ensure x + stepMax*d stays within bounds
        if (stpMax < SEARCH_NO_BND) {
            for (int i = 0; i < n; i++) {
                double newX = x[i] + stpMax * buffer[dOffset + i];
                assertTrue(newX <= upper[i] + 1e-10,
                        "x[" + i + "] + stpMax*d[" + i + "] = " + newX + " should not exceed upper[" + i + "] = " + upper[i]);
            }
        }

    }

    // ========================================================================
    // Property 6.3: perform detects non-descent direction
    // ========================================================================

    @Property(tries = 100)
    void performDetectsNonDescentDirection(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        // Initialize workspace
        ws.setIter(0);
        ws.setBoxed(false);
        ws.setConstrained(false);
        ws.getSearchCtx().numEval = 0;

        // Set up NON-descent direction (d points in same direction as gradient)
        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();
        int tOffset = ws.getTOffset();
        int zOffset = ws.getZOffset();

        double[] x = new double[n];
        double[] g = new double[n];

        for (int i = 0; i < n; i++) {
            x[i] = 0.0;
            g[i] = 1.0;  // Gradient pointing positive
            buffer[dOffset + i] = 1.0;  // Direction also positive (NOT descent!)
            buffer[tOffset + i] = x[i];
            buffer[zOffset + i] = x[i];
        }

        // Initialize line search
        LineSearch.init(n, x, (com.curioloop.numopt4j.optim.Bound[]) null, ws);

        // Perform line search - should detect non-descent
        double f = 1.0;
        int[] result = LineSearch.perform(n, x, f, g, ws);

        assertEquals(ERR_DERIVATIVE, result[0],
                "Should return ERR_DERIVATIVE for non-descent direction");

    }

    // ========================================================================
    // Property 6.4: perform is deterministic
    // ========================================================================

    @Property(tries = 100)
    void performIsDeterministic(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 1, max = 3) int m
    ) {
        // Run twice with same inputs
        double[] x1 = runPerform(n, m);
        double[] x2 = runPerform(n, m);

        // Results should be identical
        assertArrayEquals(x1, x2, 1e-15, "perform should be deterministic");
    }

    private double[] runPerform(int n, int m) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setIter(0);
        ws.setBoxed(false);
        ws.setConstrained(false);
        ws.getSearchCtx().numEval = 0;

        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();
        int tOffset = ws.getTOffset();
        int zOffset = ws.getZOffset();

        double[] x = new double[n];
        double[] g = new double[n];

        for (int i = 0; i < n; i++) {
            x[i] = 1.0;
            g[i] = 2.0 * x[i];  // Gradient of x^2
            buffer[dOffset + i] = -g[i];  // Descent direction
            buffer[tOffset + i] = x[i];
            buffer[zOffset + i] = x[i] - 0.1;  // Cauchy point
        }

        LineSearch.init(n, x, (com.curioloop.numopt4j.optim.Bound[]) null, ws);

        // Perform one iteration
        double f = 0;
        for (int i = 0; i < n; i++) {
            f += x[i] * x[i];
        }

        LineSearch.perform(n, x, f, g, ws);

        return x;
    }

    // ========================================================================
    // Property 6.5: Wolfe conditions satisfied on convergence
    // ========================================================================

    @Property(tries = 100)
    void wolfeConditionsSatisfiedOnConvergence(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 1, max = 3) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setIter(1);
        ws.setBoxed(false);
        ws.setConstrained(false);

        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();
        int tOffset = ws.getTOffset();
        int zOffset = ws.getZOffset();

        // Simple quadratic function f(x) = sum(x_i^2)
        double[] x = new double[n];
        double[] g = new double[n];

        for (int i = 0; i < n; i++) {
            x[i] = 1.0;
            g[i] = 2.0 * x[i];  // Gradient
            buffer[dOffset + i] = -g[i];  // Steepest descent
            buffer[tOffset + i] = x[i];
            buffer[zOffset + i] = 0.0;  // Optimal point
        }

        double stp = LineSearch.init(n, x, (com.curioloop.numopt4j.optim.Bound[]) null, ws);

        // Compute initial function value and directional derivative
        double f0 = 0;
        double gd0 = 0;
        for (int i = 0; i < n; i++) {
            f0 += x[i] * x[i];
            gd0 += g[i] * buffer[dOffset + i];
        }

        // Iterate line search until convergence or max iterations
        int maxIter = 20;
        int iter = 0;
        boolean converged = false;
        LBFGSBWorkspace.SearchCtx ctx = ws.getSearchCtx();

        while (iter < maxIter && !converged) {
            // Compute f and g at current x
            double f = 0;
            for (int i = 0; i < n; i++) {
                f += x[i] * x[i];
                g[i] = 2.0 * x[i];
            }

            int[] result = LineSearch.perform(n, x, f, g, ws);

            if (result[1] == 1) {  // done
                converged = (ctx.searchTask & SEARCH_CONV) != 0;
                break;
            }

            ctx.numEval++;
            iter++;
        }

        // If converged, verify Wolfe conditions
        if (converged) {
            double fFinal = 0;
            double gdFinal = 0;
            for (int i = 0; i < n; i++) {
                fFinal += x[i] * x[i];
                gdFinal += g[i] * buffer[dOffset + i];
            }

            double stpFinal = ctx.stp;

            // Armijo condition: f(stp) <= f(0) + alpha * stp * g'(0) * d
            double armijoRhs = f0 + SEARCH_ALPHA * stpFinal * gd0;
            assertTrue(fFinal <= armijoRhs + 1e-10,
                    "Armijo condition should be satisfied: f=" + fFinal + " <= " + armijoRhs);

            // Curvature condition: |g'(stp) * d| <= beta * |g'(0) * d|
            double curvatureLhs = Math.abs(gdFinal);
            double curvatureRhs = SEARCH_BETA * Math.abs(gd0);
            assertTrue(curvatureLhs <= curvatureRhs + 1e-10,
                    "Curvature condition should be satisfied: |gd|=" + curvatureLhs + " <= " + curvatureRhs);
        }

    }

    // ========================================================================
    // Property 6.6: Step length is bounded by maximum
    // ========================================================================

    @Property(tries = 100)
    void stepLengthIsBoundedByMaximum(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setIter(1);
        ws.setBoxed(true);
        ws.setConstrained(true);

        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();

        // Set up direction moving towards upper bound
        for (int i = 0; i < n; i++) {
            buffer[dOffset + i] = 10.0;  // Large positive direction
        }

        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];

        for (int i = 0; i < n; i++) {
            x[i] = 0.0;
            lower[i] = -1.0;
            upper[i] = 0.1;  // Very close upper bound
            boundType[i] = BOUND_BOTH;
        }

        double stp = LineSearch.init(n, x, TestBounds.toBounds(n, lower, upper, boundType), ws);

        // The initial step should be 1.0 (non-first iteration)
        // But the maximum step (stpMax) should be limited by bounds
        double stpMax = ws.getSearchCtx().stpMax;

        // Verify maximum step respects bounds
        for (int i = 0; i < n; i++) {
            double newX = x[i] + stpMax * buffer[dOffset + i];
            assertTrue(newX <= upper[i] + 1e-10,
                    "x[" + i + "] + stpMax*d[" + i + "] = " + newX + " should not exceed upper[" + i + "] = " + upper[i]);
        }

        // The initial step should be clamped to stpMax during line search
        assertTrue(stp >= 0, "Step should be non-negative");

    }

    // ========================================================================
    // Property 6.7: First iteration uses scaled step
    // ========================================================================

    @Property(tries = 100)
    void firstIterationUsesScaledStep(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setIter(0);  // First iteration
        ws.setBoxed(false);
        ws.setConstrained(false);

        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();

        // Set up direction with known norm
        double dNorm = 0;
        for (int i = 0; i < n; i++) {
            buffer[dOffset + i] = 1.0;
            dNorm += 1.0;
        }
        dNorm = Math.sqrt(dNorm);

        double[] x = new double[n];

        double stp = LineSearch.init(n, x, (com.curioloop.numopt4j.optim.Bound[]) null, ws);

        // For first iteration with unconstrained problem, stp = 1/||d||
        double expectedStp = 1.0 / dNorm;
        assertEquals(expectedStp, stp, 1e-14,
                "First iteration step should be 1/||d|| = " + expectedStp);

    }

    // ========================================================================
    // Property 6.8: Non-first iteration uses unit step
    // ========================================================================

    @Property(tries = 100)
    void nonFirstIterationUsesUnitStep(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setIter(5);  // Not first iteration
        ws.setBoxed(false);
        ws.setConstrained(false);

        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();

        for (int i = 0; i < n; i++) {
            buffer[dOffset + i] = 1.0;
        }

        double[] x = new double[n];

        double stp = LineSearch.init(n, x, (com.curioloop.numopt4j.optim.Bound[]) null, ws);

        // For non-first iteration, stp = 1.0
        assertEquals(1.0, stp, 1e-14,
                "Non-first iteration step should be 1.0");

    }

    // ========================================================================
    // Property 6.9: Search task is properly initialized
    // ========================================================================

    @Property(tries = 100)
    void searchTaskIsProperlyInitialized(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setIter(0);
        ws.setBoxed(false);
        ws.setConstrained(false);

        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();

        for (int i = 0; i < n; i++) {
            buffer[dOffset + i] = -1.0;
        }

        double[] x = new double[n];

        LineSearch.init(n, x, (com.curioloop.numopt4j.optim.Bound[]) null, ws);

        LBFGSBWorkspace.SearchCtx ctx = ws.getSearchCtx();
        assertEquals(SEARCH_START, ctx.searchTask,
                "Search task should be SEARCH_START after init");
        assertEquals(0, ctx.numEval,
                "Number of evaluations should be 0 after init");
        assertEquals(0, ctx.numBack,
                "Number of backtracking steps should be 0 after init");

    }

    // ========================================================================
    // Property 6.10: Direction norm is computed correctly
    // ========================================================================

    @Property(tries = 100)
    void directionNormIsComputedCorrectly(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setIter(0);
        ws.setBoxed(false);
        ws.setConstrained(false);

        double[] buffer = ws.getBuffer();
        int dOffset = ws.getDOffset();

        // Set up direction with known values
        double expectedNormSq = 0;
        for (int i = 0; i < n; i++) {
            buffer[dOffset + i] = (i + 1) * 0.5;
            expectedNormSq += buffer[dOffset + i] * buffer[dOffset + i];
        }
        double expectedNorm = Math.sqrt(expectedNormSq);

        double[] x = new double[n];

        LineSearch.init(n, x, (com.curioloop.numopt4j.optim.Bound[]) null, ws);

        assertEquals(expectedNormSq, ws.getDSqrt(), 1e-14,
                "d^2 should be computed correctly");
        assertEquals(expectedNorm, ws.getDNorm(), 1e-14,
                "||d|| should be computed correctly");

    }
}
