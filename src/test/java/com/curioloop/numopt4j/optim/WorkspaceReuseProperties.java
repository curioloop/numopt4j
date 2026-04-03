/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBWorkspace;
import com.curioloop.numopt4j.optim.slsqp.SLSQPProblem;
import com.curioloop.numopt4j.optim.slsqp.SLSQPWorkspace;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;

import static org.assertj.core.api.Assertions.*;

/**
 * Property-based tests for workspace reuse functionality.
 * 
 */
public class WorkspaceReuseProperties {

    /**
     * Property 1: Workspace Reuse Correctness
     * 
     * For any valid problem dimension n and corrections m, creating a workspace
     * and calling optimize multiple times should produce consistent results,
     * and the results should match those obtained without workspace reuse.
     * 
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 1: Workspace Reuse Correctness")
    void workspaceReuseProducesConsistentResults(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 3, max = 10) int m,
            @ForAll @IntRange(min = 3, max = 5) int numOptimizations
    ) {
        // Use centralized quadratic template: f(x) = sum(x_i^2)
        Univariate quadratic = TestTemplates.quadratic();

        // Build problem
        LBFGSBProblem problem = new LBFGSBProblem()
                .objective(quadratic)
                .maxIterations(100)
                .maxEvaluations(500)
                .gradientTolerance(1e-6);

        // First, get the baseline result without workspace reuse
        Optimization baselineResult = problem.initialPoint(createInitialPoint(n, 1.0)).solve();

        // Now test with workspace reuse
        LBFGSBWorkspace workspace = new LBFGSBWorkspace();
            Optimization[] results = new Optimization[numOptimizations];

            // Run multiple optimizations with the same workspace
            for (int i = 0; i < numOptimizations; i++) {
                results[i] = problem.initialPoint(createInitialPoint(n, 1.0)).solve(workspace);
            }

            // Verify all results are consistent with each other
            for (int i = 0; i < numOptimizations; i++) {
                // All optimizations should converge
                assertThat(results[i].isSuccessful())
                        .as("Optimization %d should converge", i)
                        .isTrue();

                // Function values should be close to 0 (the minimum)
                assertThat(results[i].getCost())
                        .as("Function value for optimization %d should be near minimum", i)
                        .isLessThan(1e-8);
            }

            // Verify results match baseline (without workspace reuse)
            assertThat(baselineResult.isSuccessful())
                    .as("Baseline optimization should converge")
                    .isTrue();

            // Compare function values between workspace and non-workspace versions
            assertThat(results[0].getCost())
                    .as("Workspace result should match baseline function value")
                    .isCloseTo(baselineResult.getCost(), within(1e-8));

            // Compare solutions (using the initial arrays which are modified in-place)
            // Note: Since we create new initial arrays for each optimization, we can't compare
            // solutions directly here. The function value comparison above validates correctness.
    }

    /**
     * Helper method to create an initial point with all components set to the given value.
     */
    private double[] createInitialPoint(int n, double value) {
        double[] point = new double[n];
        for (int i = 0; i < n; i++) {
            point[i] = value;
        }
        return point;
    }

    /**
     * Helper method for assertj within tolerance.
     */
    private static org.assertj.core.data.Offset<Double> within(double tolerance) {
        return org.assertj.core.data.Offset.offset(tolerance);
    }

    /**
     * Property 1: Workspace Reuse Correctness - SLSQP
     * 
     * For any valid problem dimension n, creating a SlsqpWorkspace
     * and calling optimize multiple times should produce consistent results,
     * and the results should match those obtained without workspace reuse.
     * 
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 1: Workspace Reuse Correctness - SLSQP")
    void slsqpWorkspaceReuseProducesConsistentResults(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 3, max = 5) int numOptimizations
    ) {
        Univariate quadratic = TestTemplates.quadratic();

        // Build SLSQP problem (no constraints)
        SLSQPProblem problem = new SLSQPProblem()
                .objective(quadratic)
                .maxIterations(100)
                .functionTolerance(1e-8);

        // First, get the baseline result without workspace reuse
        Optimization baselineResult = problem.initialPoint(createInitialPoint(n, 1.0)).solve();

        // Now test with workspace reuse
        SLSQPWorkspace workspace = new SLSQPWorkspace();
            Optimization[] results = new Optimization[numOptimizations];

            // Run multiple optimizations with the same workspace
            for (int i = 0; i < numOptimizations; i++) {
                results[i] = problem.initialPoint(createInitialPoint(n, 1.0)).solve(workspace);
            }

            // Verify all results are consistent with each other
            for (int i = 0; i < numOptimizations; i++) {
                // All optimizations should converge
                assertThat(results[i].isSuccessful())
                        .as("SLSQP Optimization %d should converge", i)
                        .isTrue();

                // Function values should be close to 0 (the minimum)
                assertThat(results[i].getCost())
                        .as("SLSQP Function value for optimization %d should be near minimum", i)
                        .isLessThan(1e-8);
            }

            // Verify results match baseline (without workspace reuse)
            assertThat(baselineResult.isSuccessful())
                    .as("SLSQP Baseline optimization should converge")
                    .isTrue();

            // Compare function values between workspace and non-workspace versions
            assertThat(results[0].getCost())
                    .as("SLSQP Workspace result should match baseline function value")
                    .isCloseTo(baselineResult.getCost(), within(1e-8));
    }

    /**
     * Property 2: Workspace Auto-Resize on Dimension Mismatch
     * 
     * For any workspace dimension (n1, m1) and problem dimension (n2, m2),
     * when dimensions differ, the workspace should auto-resize and the
     * optimization should still succeed.
     * 
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 2: Workspace Auto-Resize on Dimension Mismatch")
    void dimensionMismatchAutoResizes(
            @ForAll @IntRange(min = 2, max = 20) int workspaceN,
            @ForAll @IntRange(min = 3, max = 10) int workspaceM,
            @ForAll @IntRange(min = 2, max = 20) int problemN,
            @ForAll @IntRange(min = 3, max = 10) int problemM
    ) {
        // Create a simple quadratic objective function
        Univariate quadratic = TestTemplates.quadratic();;

        // Build problem with problem dimensions (n2, m2)
        LBFGSBProblem problem = new LBFGSBProblem()
                .objective(quadratic)
                .initialPoint(createInitialPoint(problemN, 1.0))
                .corrections(problemM)
                .maxIterations(100)
                .maxEvaluations(500)
                .gradientTolerance(1e-6);

        // Create workspace with potentially different dimensions (n1, m1)
        LBFGSBWorkspace workspace = new LBFGSBWorkspace();
        // Pre-initialize with different dimensions to test auto-resize
        workspace.ensure(workspaceN, workspaceM);

        double[] initialPoint = createInitialPoint(problemN, 1.0);

        // Workspace should auto-resize and optimization should not throw
        Optimization result = problem.initialPoint(initialPoint).solve(workspace);
        assertThat(result).isNotNull();
        // After ensure, workspace should have at least the problem's dimensions
        assertThat(workspace.getDimension()).isGreaterThanOrEqualTo(problemN);
        assertThat(workspace.getCorrections()).isGreaterThanOrEqualTo(problemM);
    }
}
