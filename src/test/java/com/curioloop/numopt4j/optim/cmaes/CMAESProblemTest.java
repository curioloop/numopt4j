/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CMAESProblem fluent API, parameter validation, and workspace management.
 */
class CMAESProblemTest {

    // ── Factory method ────────────────────────────────────────────────────

    @Test
    void minimizerCmaesReturnsCorrectType() {
        CMAESProblem p = Minimizer.cmaes();
        assertNotNull(p);
        assertInstanceOf(CMAESProblem.class, p);
    }

    // ── Default parameter values ──────────────────────────────────────────

    @Test
    void defaultParametersAreCorrect() {
        CMAESProblem p = new CMAESProblem();
        assertEquals(0.3, p.sigma0(), 1e-15);
        assertEquals(0, p.lambdaConfig());
        assertEquals(1000, p.maxIterations());
        assertNull(p.restartMode());
        assertEquals(Double.NEGATIVE_INFINITY, p.stopFitness());
        assertEquals(1e-11, p.parameterTolerance(), 1e-20);
        assertEquals(1e-12, p.functionTolerance(), 1e-20);
        assertEquals(1e3, p.sigmaUpperBound(), 1e-10);
        assertEquals(UpdateMode.ACTIVE_CMA, p.updateMode());
    }

    @Test
    void effectiveLambdaAutoComputed() {
        // n=10: lambda = 4 + floor(3*ln(10)) = 4 + floor(6.907) = 4 + 6 = 10
        CMAESProblem p = new CMAESProblem().initialPoint(new double[10]);
        assertEquals(10, p.effectiveLambda());

        // n=2: lambda = 4 + floor(3*ln(2)) = 4 + floor(2.079) = 4 + 2 = 6
        p = new CMAESProblem().initialPoint(new double[2]);
        assertEquals(6, p.effectiveLambda());
    }

    @Test
    void effectiveLambdaUsesExplicitValue() {
        CMAESProblem p = new CMAESProblem().initialPoint(new double[10]).populationSize(20);
        assertEquals(20, p.effectiveLambda());
    }

    // ── Strategy parameter computation ────────────────────────────────────

    @Test
    void strategyParametersComputedCorrectly() {
        int n = 10;
        CMAESWorkspace ws = new CMAESWorkspace();
        ws.ensure(n, 10, false);
        CMAESCore.initParams(ws, 1000);

        // mu = floor(lambda/2) = 5
        assertEquals(5, ws.mu);

        // weights sum to 1
        double sumW = 0;
        for (int i = 0; i < ws.mu; i++) sumW += ws.lVec[ws.WEIGHTS + i];
        assertEquals(1.0, sumW, 1e-12);

        // weights are positive and decreasing
        for (int i = 0; i < ws.mu - 1; i++) {
            assertTrue(ws.lVec[ws.WEIGHTS + i] > 0, "weight[" + i + "] should be positive");
            assertTrue(ws.lVec[ws.WEIGHTS + i] > ws.lVec[ws.WEIGHTS + i + 1],
                "weight[" + i + "] should be > weight[" + (i+1) + "]");
        }

        // mueff > 1
        assertTrue(ws.mueff > 1.0);

        // chiN ≈ sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
        double expectedChiN = Math.sqrt(n) * (1.0 - 1.0/(4*n) + 1.0/(21.0*n*n));
        assertEquals(expectedChiN, ws.chiN, 1e-12);

        // cc, cs, damps are positive
        assertTrue(ws.cc > 0 && ws.cc < 1);
        assertTrue(ws.cs > 0 && ws.cs < 1);
        assertTrue(ws.damps > 0);

        // ccov1, ccovmu are positive and sum <= 1
        assertTrue(ws.ccov1 > 0);
        assertTrue(ws.ccovmu >= 0);
        assertTrue(ws.ccov1 + ws.ccovmu <= 1.0 + 1e-12);
    }

    // ── Fluent API validation ─────────────────────────────────────────────

    @Test
    void solveWithoutObjectiveThrowsIllegalState() {
        CMAESProblem p = new CMAESProblem().initialPoint(1.0, 2.0);
        assertThrows(IllegalStateException.class, p::solve);
    }

    @Test
    void solveWithoutInitialPointThrowsIllegalState() {
        CMAESProblem p = new CMAESProblem().objective((x, n) -> 0.0);
        assertThrows(IllegalStateException.class, p::solve);
    }

    @Test
    void solveWithNaNInitialPointThrowsIllegalArgument() {
        CMAESProblem p = new CMAESProblem()
            .objective((x, n) -> 0.0)
            .initialPoint(1.0, Double.NaN);
        assertThrows(IllegalArgumentException.class, p::solve);
    }

    @Test
    void solveWithInfInitialPointThrowsIllegalArgument() {
        CMAESProblem p = new CMAESProblem()
            .objective((x, n) -> 0.0)
            .initialPoint(1.0, Double.POSITIVE_INFINITY);
        assertThrows(IllegalArgumentException.class, p::solve);
    }

    @Test
    void negativeSigmaThrowsIllegalArgument() {
        assertThrows(IllegalArgumentException.class,
            () -> new CMAESProblem().sigma(-1.0));
    }

    @Test
    void zeroSigmaThrowsIllegalArgument() {
        assertThrows(IllegalArgumentException.class,
            () -> new CMAESProblem().sigma(0.0));
    }

    // ── Workspace dimension mismatch ──────────────────────────────────────

    @Test
    void workspaceDimensionMismatchAutoResizes() {
        CMAESProblem p = new CMAESProblem()
            .objective((x, n) -> { double s = 0; for (double v : x) s += v*v; return s; })
            .initialPoint(new double[5])
            .populationSize(10)
            .maxEvaluations(100);

        // Workspace with wrong n — should auto-resize instead of throwing
        CMAESWorkspace wrongWs = new CMAESWorkspace();
        wrongWs.ensure(3, 10, false);
        assertDoesNotThrow(() -> p.solve(wrongWs));
        assertEquals(5, wrongWs.n);
    }

    @Test
    void workspaceLambdaMismatchAutoResizes() {
        CMAESProblem p = new CMAESProblem()
            .objective((x, n) -> { double s = 0; for (double v : x) s += v*v; return s; })
            .initialPoint(new double[5])
            .populationSize(10)
            .maxEvaluations(100);

        // Workspace with wrong lambda — should auto-resize instead of throwing
        CMAESWorkspace wrongWs = new CMAESWorkspace();
        wrongWs.ensure(5, 20, false);
        assertDoesNotThrow(() -> p.solve(wrongWs));
        assertEquals(10, wrongWs.lambda);
    }

    @Test
    void compatibleWorkspaceIsAccepted() {
        CMAESProblem p = new CMAESProblem()
            .objective((x, n) -> { double s = 0; for (double v : x) s += v*v; return s; })
            .initialPoint(new double[]{1.0, 1.0, 1.0})
            .populationSize(10)
            .maxEvaluations(100);

        CMAESWorkspace ws = new CMAESWorkspace();
        ws.ensure(3, 10, false);
        assertDoesNotThrow(() -> p.solve(ws));
    }

    // ── workspace() ──────────────────────────────────────────────────────────

    @Test
    void workspaceReturnsCorrectDimensions() {
        int n = 5;
        CMAESProblem p = new CMAESProblem()
            .objective((x, dim) -> 0.0)
            .initialPoint(new double[n])
            .populationSize(12);

        CMAESWorkspace ws = new CMAESWorkspace();
        ws.ensure(n, 12, p.updateMode().separable);
        assertNotNull(ws);
        assertEquals(n, ws.n);
        assertEquals(12, ws.lambda);
    }

    @Test
    void workspaceWithAutoLambdaComputesCorrectly() {
        int n = 10;
        CMAESProblem p = new CMAESProblem()
            .objective((x, dim) -> 0.0)
            .initialPoint(new double[n]);

        CMAESWorkspace ws = new CMAESWorkspace();
        ws.ensure(n, p.effectiveLambda(), p.updateMode().separable);
        assertEquals(n, ws.n);
        assertEquals(p.effectiveLambda(), ws.lambda);
    }

    // ── Simple convergence smoke test ─────────────────────────────────────

    @Test
    void sphereFunctionConverges() {
        Optimization result = Minimizer.cmaes()
            .objective((x, n) -> { double s = 0; for (double v : x) s += v*v; return s; })
            .initialPoint(1.0, 1.0, 1.0)
            .maxEvaluations(5000)
            .solve();

        assertNotNull(result);
        assertNotNull(result.getSolution());
        assertTrue(result.getCost() < 1e-6,
            "Expected cost < 1e-6, got " + result.getCost());
    }
}
