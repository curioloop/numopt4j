/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for SubspaceMinimizer.
 *
 * **Property 5: Subspace Minimization Correctness**
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBConstants;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBWorkspace;
import com.curioloop.numopt4j.optim.lbfgsb.SubspaceMinimizer;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for SubspaceMinimizer.
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 5.1: reduceGradient produces finite results</li>
 *   <li>Property 5.2: optimalDirection produces points within bounds</li>
 *   <li>Property 5.3: optimalDirection is deterministic</li>
 *   <li>Property 5.4: Unconstrained case sets r = -g</li>
 *   <li>Property 5.5: Solution status is correctly set</li>
 * </ul>
 */
@Tag("Feature: lbfgsb-java-rewrite, Property 5: Subspace Minimization Correctness")
class SubspaceMinimizationProperties implements LBFGSBConstants {

    // ========================================================================
    // Property 5.1: reduceGradient produces finite results
    // ========================================================================

    @Property(tries = 100)
    void reduceGradientProducesFiniteResults(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        // Initialize workspace state
        ws.setTheta(1.0);
        ws.setCol(0);  // No BFGS corrections yet
        ws.setHead(0);
        ws.setConstrained(true);

        // Set up free variables
        int[] iBuffer = ws.getIntBuffer();
        int indexOffset = ws.getIndexOffset();
        int whereOffset = ws.getWhereOffset();

        // All variables are free
        for (int i = 0; i < n; i++) {
            iBuffer[indexOffset + i] = i;
            iBuffer[whereOffset + i] = VAR_FREE;
        }
        ws.setFree(n);

        // Create test data
        double[] x = new double[n];
        double[] g = new double[n];
        double[] z = new double[n];
        double[] r = new double[n];

        for (int i = 0; i < n; i++) {
            x[i] = i * 0.1;
            g[i] = (i + 1) * 0.5;
            z[i] = x[i] + 0.01;  // Cauchy point slightly different from x
        }

        // Copy Cauchy point z into workspace and call reduceGradient
        double[] buffer = ws.getBuffer();
        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();
        System.arraycopy(z, 0, buffer, zOffset, n);

        int result = SubspaceMinimizer.reduceGradient(n, m, x, g, ws);

        // Verify result
        assertEquals(ERR_NONE, result, "reduceGradient should succeed");

        // Copy reduced gradient out of workspace and verify results are finite
        System.arraycopy(buffer, rOffset, r, 0, ws.getFree());
        for (int i = 0; i < ws.getFree(); i++) {
            assertTrue(Double.isFinite(r[i]),
                    "Reduced gradient element " + i + " should be finite");
        }

    }

    // ========================================================================
    // Property 5.2: optimalDirection produces points within bounds
    // ========================================================================

    @Property(tries = 100)
    void optimalDirectionProducesPointsWithinBounds(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        // Initialize workspace state
        ws.setTheta(1.0);
        ws.setCol(0);  // No BFGS corrections
        ws.setHead(0);
        ws.setConstrained(true);

        // Set up free variables
        int[] iBuffer = ws.getIntBuffer();
        int indexOffset = ws.getIndexOffset();
        int whereOffset = ws.getWhereOffset();

        for (int i = 0; i < n; i++) {
            iBuffer[indexOffset + i] = i;
            iBuffer[whereOffset + i] = VAR_FREE;
        }
        ws.setFree(n);

        // Create test data with bounds
        double[] x = new double[n];
        double[] g = new double[n];
        double[] z = new double[n];
        double[] r = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];

        for (int i = 0; i < n; i++) {
            lower[i] = -10.0;
            upper[i] = 10.0;
            boundType[i] = BOUND_BOTH;
            x[i] = 0.0;
            g[i] = (i % 2 == 0) ? 1.0 : -1.0;
            z[i] = x[i];  // Start at x
            r[i] = -g[i];  // Simple reduced gradient
        }

        // Copy initial z and r into workspace then call optimalDirection
        double[] buffer = ws.getBuffer();
        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();
        System.arraycopy(z, 0, buffer, zOffset, n);
        System.arraycopy(r, 0, buffer, rOffset, n);

        int result = SubspaceMinimizer.optimalDirection(n, m, x, g, TestBounds.toBounds(n, lower, upper, boundType), ws);

        // Verify result
        assertEquals(ERR_NONE, result, "optimalDirection should succeed");

        // Copy resulting z out of workspace and verify all values are within bounds
        System.arraycopy(buffer, zOffset, z, 0, n);
        for (int i = 0; i < n; i++) {
            assertTrue(z[i] >= lower[i] - 1e-10,
                    "z[" + i + "] = " + z[i] + " should be >= lower[" + i + "] = " + lower[i]);
            assertTrue(z[i] <= upper[i] + 1e-10,
                    "z[" + i + "] = " + z[i] + " should be <= upper[" + i + "] = " + upper[i]);
        }

    }

    // ========================================================================
    // Property 5.3: optimalDirection is deterministic
    // ========================================================================

    @Property(tries = 100)
    void optimalDirectionIsDeterministic(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 1, max = 3) int m
    ) {
        // Run twice with same inputs
        double[] z1 = runOptimalDirection(n, m);
        double[] z2 = runOptimalDirection(n, m);

        // Results should be identical
        assertArrayEquals(z1, z2, 1e-15, "optimalDirection should be deterministic");
    }

    private double[] runOptimalDirection(int n, int m) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setTheta(1.0);
        ws.setCol(0);
        ws.setHead(0);
        ws.setConstrained(true);

        int[] iBuffer = ws.getIntBuffer();
        int indexOffset = ws.getIndexOffset();
        int whereOffset = ws.getWhereOffset();

        for (int i = 0; i < n; i++) {
            iBuffer[indexOffset + i] = i;
            iBuffer[whereOffset + i] = VAR_FREE;
        }
        ws.setFree(n);

        double[] x = new double[n];
        double[] g = new double[n];
        double[] z = new double[n];
        double[] r = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];

        for (int i = 0; i < n; i++) {
            lower[i] = -5.0;
            upper[i] = 5.0;
            boundType[i] = BOUND_BOTH;
            x[i] = i * 0.1;
            g[i] = (i + 1) * 0.2;
            z[i] = x[i];
            r[i] = -g[i];
        }

        // Copy initial z/r into workspace and run optimalDirection, then return z from workspace
        double[] buffer = ws.getBuffer();
        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();
        System.arraycopy(z, 0, buffer, zOffset, n);
        System.arraycopy(r, 0, buffer, rOffset, n);

        SubspaceMinimizer.optimalDirection(n, m, x, g, TestBounds.toBounds(n, lower, upper, boundType), ws);

        // Copy result z out
        System.arraycopy(buffer, zOffset, z, 0, n);

        return z;
    }

    // ========================================================================
    // Property 5.4: Unconstrained case sets r = -g
    // ========================================================================

    @Property(tries = 100)
    void unconstrainedCaseSetsRToNegativeG(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        // Set up unconstrained case with col > 0
        ws.setTheta(1.0);
        ws.setCol(1);  // At least one correction
        ws.setHead(0);
        ws.setConstrained(false);  // Unconstrained!

        // Initialize S'Y matrix diagonal (needed for bmv)
        double[] buffer = ws.getBuffer();
        int syOffset = ws.getSyOffset();
        buffer[syOffset] = 1.0;  // sy[0,0] = 1.0

        // Initialize Cholesky factor
        int wtOffset = ws.getWtOffset();
        buffer[wtOffset] = 1.0;  // wt[0,0] = 1.0

        double[] x = new double[n];
        double[] g = new double[n];
        double[] z = new double[n];
        double[] r = new double[n];

        for (int i = 0; i < n; i++) {
            x[i] = i * 0.1;
            g[i] = (i + 1) * 0.5;
            z[i] = x[i];
        }

        // Copy z into workspace and call reduceGradient
        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();
        // buffer already declared above when initializing sy/wt
        System.arraycopy(z, 0, buffer, zOffset, n);

        int result = SubspaceMinimizer.reduceGradient(n, m, x, g, ws);

        assertEquals(ERR_NONE, result, "reduceGradient should succeed");

        // Copy reduced gradient out and verify r == -g
        System.arraycopy(buffer, rOffset, r, 0, n);
        for (int i = 0; i < n; i++) {
            assertEquals(-g[i], r[i], 1e-15,
                    "r[" + i + "] should equal -g[" + i + "] for unconstrained case");
        }

    }

    // ========================================================================
    // Property 5.5: Solution status is correctly set
    // ========================================================================

    @Property(tries = 100)
    void solutionStatusIsCorrectlySet(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 1, max = 3) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setTheta(1.0);
        ws.setCol(0);
        ws.setHead(0);
        ws.setConstrained(true);
        ws.setWord(SOLUTION_UNKNOWN);

        int[] iBuffer = ws.getIntBuffer();
        int indexOffset = ws.getIndexOffset();
        int whereOffset = ws.getWhereOffset();

        for (int i = 0; i < n; i++) {
            iBuffer[indexOffset + i] = i;
            iBuffer[whereOffset + i] = VAR_FREE;
        }
        ws.setFree(n);

        double[] x = new double[n];
        double[] g = new double[n];
        double[] z = new double[n];
        double[] r = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];

        for (int i = 0; i < n; i++) {
            lower[i] = -1.0;
            upper[i] = 1.0;
            boundType[i] = BOUND_BOTH;
            x[i] = 0.0;
            g[i] = 0.1;  // Small gradient
            z[i] = x[i];
            r[i] = -g[i];
        }

        // Copy initial z/r into workspace and run optimalDirection
        double[] buffer = ws.getBuffer();
        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();
        System.arraycopy(z, 0, buffer, zOffset, n);
        System.arraycopy(r, 0, buffer, rOffset, n);

        SubspaceMinimizer.optimalDirection(n, m, x, g, TestBounds.toBounds(n, lower, upper, boundType), ws);

        // Solution status should be set (not UNKNOWN)
        int word = ws.getWord();
        assertTrue(word == SOLUTION_WITHIN_BOX || word == SOLUTION_BEYOND_BOX,
                "Solution status should be set after optimalDirection");

    }

    // ========================================================================
    // Property 5.6: No free variables returns immediately
    // ========================================================================

    @Property(tries = 100)
    void noFreeVariablesReturnsImmediately(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 1, max = 3) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setTheta(1.0);
        ws.setCol(0);
        ws.setHead(0);
        ws.setConstrained(true);
        ws.setFree(0);  // No free variables!

        double[] x = new double[n];
        double[] g = new double[n];
        double[] z = new double[n];
        double[] r = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];

        for (int i = 0; i < n; i++) {
            lower[i] = -1.0;
            upper[i] = 1.0;
            boundType[i] = BOUND_BOTH;
            x[i] = 0.5;
            g[i] = 1.0;
            z[i] = 0.5;
            r[i] = -1.0;
        }

        // Save original z values
        double[] zOriginal = z.clone();

        // Copy initial z/r into workspace and call workspace-only optimalDirection
        double[] buffer = ws.getBuffer();
        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();
        System.arraycopy(z, 0, buffer, zOffset, n);
        System.arraycopy(r, 0, buffer, rOffset, n);

        int result = SubspaceMinimizer.optimalDirection(n, m, x, g,
                TestBounds.toBounds(n, lower, upper, boundType), ws);

        assertEquals(ERR_NONE, result, "optimalDirection should succeed with no free variables");

        // Copy z back out and verify unchanged
        System.arraycopy(buffer, zOffset, z, 0, n);
        assertArrayEquals(zOriginal, z, 1e-15, "z should be unchanged when no free variables");

    }

    // ========================================================================
    // Property 5.7: reduceGradient with col=0 computes simple formula
    // ========================================================================

    @Property(tries = 100)
    void reduceGradientWithNoCorrectionsComputesSimpleFormula(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 1, max = 5) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        double theta = 2.0;
        ws.setTheta(theta);
        ws.setCol(0);  // No BFGS corrections
        ws.setHead(0);
        ws.setConstrained(true);

        int[] iBuffer = ws.getIntBuffer();
        int indexOffset = ws.getIndexOffset();
        int whereOffset = ws.getWhereOffset();

        for (int i = 0; i < n; i++) {
            iBuffer[indexOffset + i] = i;
            iBuffer[whereOffset + i] = VAR_FREE;
        }
        ws.setFree(n);

        double[] x = new double[n];
        double[] g = new double[n];
        double[] z = new double[n];
        double[] r = new double[n];

        for (int i = 0; i < n; i++) {
            x[i] = i * 0.1;
            g[i] = (i + 1) * 0.5;
            z[i] = x[i] + 0.02;  // Cauchy point
        }

        // Copy z into workspace, run reduceGradient, then copy reduced gradient out
        double[] buffer = ws.getBuffer();
        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();
        System.arraycopy(z, 0, buffer, zOffset, n);

        int result = SubspaceMinimizer.reduceGradient(n, m, x, g, ws);
        assertEquals(ERR_NONE, result, "reduceGradient should succeed");

        // With col=0, r[i] = -theta*(z[k] - x[k]) - g[k]
        System.arraycopy(buffer, rOffset, r, 0, ws.getFree());
        for (int i = 0; i < n; i++) {
            int k = iBuffer[indexOffset + i];
            double expected = -theta * (z[k] - x[k]) - g[k];
            assertEquals(expected, r[i], 1e-14,
                    "r[" + i + "] should equal -theta*(z[k]-x[k]) - g[k]");
        }

    }

    // ========================================================================
    // Property 5.8: Unbounded variables can move freely
    // ========================================================================

    @Property(tries = 100)
    void unboundedVariablesCanMoveFreely(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 1, max = 3) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);

        ws.setTheta(1.0);
        ws.setCol(0);
        ws.setHead(0);
        ws.setConstrained(true);

        int[] iBuffer = ws.getIntBuffer();
        int indexOffset = ws.getIndexOffset();
        int whereOffset = ws.getWhereOffset();

        for (int i = 0; i < n; i++) {
            iBuffer[indexOffset + i] = i;
            iBuffer[whereOffset + i] = VAR_UNBOUND;  // All unbounded
        }
        ws.setFree(n);

        double[] x = new double[n];
        double[] g = new double[n];
        double[] z = new double[n];
        double[] r = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];

        for (int i = 0; i < n; i++) {
            lower[i] = Double.NEGATIVE_INFINITY;
            upper[i] = Double.POSITIVE_INFINITY;
            boundType[i] = BOUND_NONE;  // No bounds
            x[i] = 0.0;
            g[i] = 1.0;
            z[i] = x[i];
            r[i] = -g[i];  // Direction = -gradient
        }

        // Copy initial z/r into workspace and call optimalDirection (workspace-only API)
        double[] buffer = ws.getBuffer();
        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();
        System.arraycopy(z, 0, buffer, zOffset, n);
        System.arraycopy(r, 0, buffer, rOffset, n);

        int result = SubspaceMinimizer.optimalDirection(n, m, x, g,
                TestBounds.toBounds(n, lower, upper, boundType), ws);

        assertEquals(ERR_NONE, result, "optimalDirection should succeed");

        // Copy resulting z out of workspace and verify finite
        System.arraycopy(buffer, zOffset, z, 0, n);
        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(z[i]), "z[" + i + "] should be finite");
        }

    }
}
