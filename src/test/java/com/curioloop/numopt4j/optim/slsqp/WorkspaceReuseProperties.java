/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for Workspace Reuse.
 *
 * **Property 12: Workspace Reuse Correctness**
 */
package com.curioloop.numopt4j.optim.slsqp;

import com.curioloop.numopt4j.optim.Optimization;
import com.curioloop.numopt4j.optim.Univariate;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;
import com.curioloop.numopt4j.optim.TestTemplates;

/**
 * Property-based tests for Workspace Reuse.
 */
@Tag("Feature: slsqp-java-rewrite, Property 12: Workspace Reuse Correctness")
class WorkspaceReuseProperties {

    private static final double EPSILON = 1e-10;
    private static final double TOLERANCE = 1e-6;

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }

    private double[] generateInitialPoint(int n, java.util.Random rand) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rand.nextDouble() * 4 - 2;
        }
        return x;
    }

    /** Build a simple quadratic SLSQPProblem (no constraints). */
    private SLSQPProblem quadraticProblem(int n, double[] x0) {
        return new SLSQPProblem()
                .objective((double[] x) -> {
                    double f = 0;
                    for (double v : x) f += v * v;
                    return f;
                })
                .maxIterations(300)
                .functionTolerance(1e-12)
                .nnlsIterations(100)
                .initialPoint(x0);
    }

    // ========================================================================
    // Property 12.1: Workspace can be reused for multiple optimizations
    // ========================================================================

    @Property(tries = 50)
    void workspaceCanBeReusedForMultipleOptimizations(
            @ForAll @IntRange(min = 2, max = 5) int n,
            @ForAll @IntRange(min = 2, max = 5) int numOptimizations,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        SLSQPProblem problem = quadraticProblem(n, new double[n]);
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, 0, 0);

        for (int i = 0; i < numOptimizations; i++) {
            double[] x = new double[n];
            for (int j = 0; j < n; j++) x[j] = rand.nextDouble() * 2 - 1;
            problem.initialPoint(x);
            Optimization result = problem.solve(workspace);
            assertNotNull(result, "Optimization " + i + " should return a result");
            assertNotNull(result.getStatus(), "Optimization " + i + " should have a status");
        }
    }

    @Property(tries = 50)
    void multipleOptimizationsAllFindMinimum(
            @ForAll @IntRange(min = 2, max = 4) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        SLSQPProblem problem = quadraticProblem(n, new double[n]);
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, 0, 0);

        for (int i = 0; i < 3; i++) {
            problem.initialPoint(generateInitialPoint(n, rand));
            Optimization result = problem.solve(workspace);
            if (result.isSuccessful()) {
                assertTrue(result.getCost() < TOLERANCE,
                        "Optimization " + i + " should find minimum near 0, got: " + result.getCost());
            }
        }
    }

    // ========================================================================
    // Property 12.2: Workspace reset clears all state
    // ========================================================================

    @Property(tries = 100)
    void resetClearsIterationCounter(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 0, max = 5) int meq,
            @ForAll @IntRange(min = 0, max = 5) int mineq
    ) {
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, meq, mineq);
        workspace.setIter(100);
        workspace.setMode(5);
        workspace.setAlpha(0.5);
        workspace.setF0(123.456);
        workspace.setBadQP(true);
        workspace.setLineMode(SLSQPConstants.FIND_NEXT);

        workspace.reset();

        assertEquals(0, workspace.getIter(), "Iteration counter should be 0 after reset");
        assertEquals(SLSQPConstants.MODE_OK, workspace.getMode(), "Mode should be MODE_OK after reset");
        assertEquals(1.0, workspace.getAlpha(), EPSILON, "Alpha should be 1.0 after reset");
        assertEquals(0.0, workspace.getF0(), EPSILON, "F0 should be 0.0 after reset");
        assertFalse(workspace.isBadQP(), "BadQP should be false after reset");
        assertEquals(SLSQPConstants.FIND_NOOP, workspace.getLineMode(), "LineMode should be FIND_NOOP after reset");
    }

    @Property(tries = 100)
    void resetClearsFindWorkState(
            @ForAll @IntRange(min = 2, max = 10) int n
    ) {
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, 0, 0);
        SLSQPWorkspace.FindWork fw = workspace.getFindWork();
        fw.a = 1.0; fw.b = 2.0; fw.x = 1.5; fw.fx = 100.0;

        workspace.reset();

        assertEquals(0.0, fw.a, EPSILON, "FindWork.a should be 0 after reset");
        assertEquals(0.0, fw.b, EPSILON, "FindWork.b should be 0 after reset");
        assertEquals(0.0, fw.x, EPSILON, "FindWork.x should be 0 after reset");
        assertEquals(0.0, fw.fx, EPSILON, "FindWork.fx should be 0 after reset");
    }

    @Property(tries = 100)
    void resetDoesNotReallocateMemory(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 0, max = 5) int meq,
            @ForAll @IntRange(min = 0, max = 5) int mineq
    ) {
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, meq, mineq);
        double[] bufferBefore = workspace.getBuffer();
        int[] iBufferBefore = workspace.getIBuffer();

        workspace.reset();

        assertSame(bufferBefore, workspace.getBuffer(), "Buffer should not be reallocated after reset");
        assertSame(iBufferBefore, workspace.getIBuffer(), "IBuffer should not be reallocated after reset");
    }

    // ========================================================================
    // Property 12.3: Workspace compatibility check
    // ========================================================================

    @Property(tries = 100)
    void workspaceIsCompatibleWithSameDimensions(
            @ForAll @IntRange(min = 1, max = 20) int n,
            @ForAll @IntRange(min = 0, max = 10) int meq,
            @ForAll @IntRange(min = 0, max = 10) int mineq
    ) {
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, meq, mineq);
        assertTrue(workspace.isCompatible(n, meq, mineq));
    }

    @Property(tries = 100)
    void workspaceIsIncompatibleWithDifferentN(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 0, max = 5) int meq,
            @ForAll @IntRange(min = 0, max = 5) int mineq,
            @ForAll @IntRange(min = 1, max = 5) int delta
    ) {
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, meq, mineq);
        assertFalse(workspace.isCompatible(n + delta, meq, mineq));
        assertFalse(workspace.isCompatible(n - 1, meq, mineq));
    }

    @Property(tries = 100)
    void workspaceIsIncompatibleWithDifferentMeq(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 1, max = 5) int meq,
            @ForAll @IntRange(min = 0, max = 5) int mineq
    ) {
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, meq, mineq);
        assertFalse(workspace.isCompatible(n, meq + 1, mineq));
        assertFalse(workspace.isCompatible(n, meq - 1, mineq));
    }

    @Property(tries = 100)
    void workspaceIsIncompatibleWithDifferentMineq(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 0, max = 5) int meq,
            @ForAll @IntRange(min = 1, max = 5) int mineq
    ) {
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, meq, mineq);
        assertFalse(workspace.isCompatible(n, meq, mineq + 1));
        assertFalse(workspace.isCompatible(n, meq, mineq - 1));
    }

    @Property(tries = 50)
    void problemRejectsIncompatibleWorkspace(
            @ForAll @IntRange(min = 2, max = 5) int n,
            @ForAll @IntRange(min = 1, max = 3) int delta
    ) {
        SLSQPProblem problem = quadraticProblem(n, new double[n]);
        SLSQPWorkspace workspace = new SLSQPWorkspace(n + delta, 0, 0);

        assertThrows(IllegalArgumentException.class, () -> problem.solve(workspace),
                "Problem should reject incompatible workspace");
    }

    // ========================================================================
    // Property 12.6: Consistent results
    // ========================================================================

    @Property(tries = 50)
    void sameOptimizationProducesConsistentResults(
            @ForAll @IntRange(min = 2, max = 4) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double[] initialPoint = generateInitialPoint(n, rand);

        SLSQPProblem problem = quadraticProblem(n, initialPoint.clone());
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, 0, 0);

        Optimization result1 = problem.solve(workspace);
        problem.initialPoint(initialPoint.clone());
        Optimization result2 = problem.solve(workspace);

        assertEquals(result1.getCost(), result2.getCost(), TOLERANCE);
        assertEquals(result1.getStatus(), result2.getStatus());
    }

    @Property(tries = 50)
    void optimizationIsDeterministic(
            @ForAll @IntRange(min = 2, max = 4) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double[] initialPoint = generateInitialPoint(n, rand);

        SLSQPProblem p1 = quadraticProblem(n, initialPoint.clone());
        Optimization result1 = p1.solve(new SLSQPWorkspace(n, 0, 0));

        SLSQPProblem p2 = quadraticProblem(n, initialPoint.clone());
        Optimization result2 = p2.solve(new SLSQPWorkspace(n, 0, 0));

        assertEquals(result1.getCost(), result2.getCost(), EPSILON);
        assertEquals(result1.getStatus(), result2.getStatus());
    }

    // ========================================================================
    // Property 12.7: Workspace reuse vs fresh workspace
    // ========================================================================

    @Property(tries = 50)
    void workspaceReuseProducesSameResultsAsFreshWorkspace(
            @ForAll @IntRange(min = 2, max = 4) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double[] initialPoint = generateInitialPoint(n, rand);

        SLSQPProblem problem = quadraticProblem(n, initialPoint.clone());
        Optimization resultFresh = problem.solve(new SLSQPWorkspace(n, 0, 0));

        SLSQPWorkspace reusedWs = new SLSQPWorkspace(n, 0, 0);
        // "dirty" the workspace
        problem.initialPoint(generateInitialPoint(n, new java.util.Random(seed + 1)));
        problem.solve(reusedWs);
        // now run with original initial point
        problem.initialPoint(initialPoint.clone());
        Optimization resultReused = problem.solve(reusedWs);

        assertEquals(resultFresh.getCost(), resultReused.getCost(), TOLERANCE);
        assertEquals(resultFresh.getStatus(), resultReused.getStatus());
    }

    @Property(tries = 30)
    void workspaceRemainsCorrectAfterManyReuses(
            @ForAll @IntRange(min = 2, max = 3) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        SLSQPProblem problem = quadraticProblem(n, new double[n]);
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, 0, 0);

        for (int i = 0; i < 10; i++) {
            problem.initialPoint(generateInitialPoint(n, rand));
            Optimization result = problem.solve(workspace);
            if (result.isSuccessful()) {
                assertTrue(result.getCost() < TOLERANCE,
                        "After " + (i + 1) + " reuses, should still find minimum");
            }
        }
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    @Property(tries = 50)
    void workspaceAllocationValidatesInputs() {
        assertThrows(IllegalArgumentException.class, () -> new SLSQPWorkspace(0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new SLSQPWorkspace(-1, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new SLSQPWorkspace(5, -1, 0));
        assertThrows(IllegalArgumentException.class, () -> new SLSQPWorkspace(5, 0, -1));
    }

    @Property(tries = 100)
    void workspaceDimensionsAreCorrectlyStored(
            @ForAll @IntRange(min = 1, max = 20) int n,
            @ForAll @IntRange(min = 0, max = 10) int meq,
            @ForAll @IntRange(min = 0, max = 10) int mineq
    ) {
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, meq, mineq);
        assertEquals(n, workspace.getN());
        assertEquals(meq, workspace.getMeq());
        assertEquals(mineq, workspace.getMineq());
        assertEquals(meq + mineq, workspace.getM());
        assertEquals(n + 1, workspace.getN1());
        assertEquals(Math.max(1, meq + mineq), workspace.getLa());
    }

    @Property(tries = 100)
    void workspaceBuffersAreProperlyAllocated(
            @ForAll @IntRange(min = 1, max = 10) int n,
            @ForAll @IntRange(min = 0, max = 5) int meq,
            @ForAll @IntRange(min = 0, max = 5) int mineq
    ) {
        SLSQPWorkspace workspace = new SLSQPWorkspace(n, meq, mineq);
        assertNotNull(workspace.getBuffer());
        assertNotNull(workspace.getIBuffer());
        assertTrue(workspace.getBuffer().length > 0);
        assertTrue(workspace.getIBuffer().length > 0);
        assertTrue(workspace.getLOffset() >= 0);
        assertTrue(workspace.getX0Offset() >= workspace.getLOffset());
        assertTrue(workspace.getGOffset() >= workspace.getX0Offset());
        assertTrue(workspace.getWOffset() < workspace.getBuffer().length);
    }

    @Property(tries = 30)
    void workspaceReuseWithConstraintsWorks(
            @ForAll @IntRange(min = 2, max = 4) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        Univariate eqConstraint = TestTemplates.sumConstraint(1.0);

        SLSQPProblem problem = new SLSQPProblem()
                .objective((double[] x) -> { double f = 0; for (double v : x) f += v * v; return f; })
                .equalityConstraints(eqConstraint)
                .maxIterations(300)
                .functionTolerance(1e-12)
                .nnlsIterations(100)
                .initialPoint(new double[n]);

        SLSQPWorkspace workspace = new SLSQPWorkspace(n, 1, 0);

        for (int i = 0; i < 3; i++) {
            double[] x = new double[n];
            for (int j = 0; j < n; j++) x[j] = 1.0 / n;
            x[0] += rand.nextDouble() * 0.1 - 0.05;
            x[n - 1] -= rand.nextDouble() * 0.1 - 0.05;
            problem.initialPoint(x);
            assertNotNull(problem.solve(workspace), "Optimization " + i + " should return a result");
        }
    }

    @Property(tries = 30)
    void workspaceReuseWithInequalityConstraintsWorks(
            @ForAll @IntRange(min = 2, max = 4) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        Univariate ineqConstraint = TestTemplates.inequalityAtIndex(0, -0.5);

        SLSQPProblem problem = new SLSQPProblem()
                .objective((double[] x) -> { double f = 0; for (double v : x) f += v * v; return f; })
                .inequalityConstraints(ineqConstraint)
                .maxIterations(300)
                .functionTolerance(1e-12)
                .nnlsIterations(100)
                .initialPoint(new double[n]);

        SLSQPWorkspace workspace = new SLSQPWorkspace(n, 0, 1);

        for (int i = 0; i < 3; i++) {
            double[] x = new double[n];
            x[0] = 0.5 + rand.nextDouble() * 0.5;
            for (int j = 1; j < n; j++) x[j] = rand.nextDouble() * 0.5 - 0.25;
            problem.initialPoint(x);
            assertNotNull(problem.solve(workspace), "Optimization " + i + " should return a result");
        }
    }
}
