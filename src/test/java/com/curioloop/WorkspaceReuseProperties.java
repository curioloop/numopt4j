/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;

import static org.assertj.core.api.Assertions.*;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Property-based tests for workspace reuse functionality.
 * 
 * **Validates: Requirements 1.1, 2.6**
 */
public class WorkspaceReuseProperties {

    /**
     * Property 1: Workspace Reuse Correctness
     * 
     * For any valid problem dimension n and corrections m, creating a workspace
     * and calling optimize multiple times should produce consistent results,
     * and the results should match those obtained without workspace reuse.
     * 
     * **Validates: Requirements 1.1, 2.6**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 1: Workspace Reuse Correctness")
    void workspaceReuseProducesConsistentResults(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 3, max = 10) int m,
            @ForAll @IntRange(min = 3, max = 5) int numOptimizations
    ) {
        // Create a simple quadratic objective function: f(x) = sum(x_i^2)
        // This has a known minimum at x = 0 with f(0) = 0
        ObjectiveFunction quadratic = (x, gradient) -> {
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                f += x[i] * x[i];
            }
            if (gradient != null) {
                for (int i = 0; i < x.length; i++) {
                    gradient[i] = 2.0 * x[i];
                }
            }
            return f;
        };

        // Build optimizer
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
                .dimension(n)
                .corrections(m)
                .objective(quadratic)
                .termination(Termination.builder()
                        .maxIterations(100)
                        .maxEvaluations(500)
                        .gradientTolerance(1e-6)
                        .build())
                .build();

        // First, get the baseline result without workspace reuse
        double[] baselineInitial = createInitialPoint(n, 1.0);
        OptimizationResult baselineResult = optimizer.optimize(baselineInitial);

        // Now test with workspace reuse
        try (LbfgsbWorkspace workspace = LbfgsbWorkspace.allocate(n, m)) {
            OptimizationResult[] results = new OptimizationResult[numOptimizations];

            // Run multiple optimizations with the same workspace
            for (int i = 0; i < numOptimizations; i++) {
                double[] initial = createInitialPoint(n, 1.0);
                results[i] = optimizer.optimize(initial, workspace);
            }

            // Verify all results are consistent with each other
            for (int i = 0; i < numOptimizations; i++) {
                // All optimizations should converge
                assertThat(results[i].isConverged())
                        .as("Optimization %d should converge", i)
                        .isTrue();

                // Function values should be close to 0 (the minimum)
                assertThat(results[i].getFunctionValue())
                        .as("Function value for optimization %d should be near minimum", i)
                        .isLessThan(1e-8);
            }

            // Verify results match baseline (without workspace reuse)
            assertThat(baselineResult.isConverged())
                    .as("Baseline optimization should converge")
                    .isTrue();

            // Compare function values between workspace and non-workspace versions
            assertThat(results[0].getFunctionValue())
                    .as("Workspace result should match baseline function value")
                    .isCloseTo(baselineResult.getFunctionValue(), within(1e-8));

            // Compare solutions (using the initial arrays which are modified in-place)
            // Note: Since we create new initial arrays for each optimization, we can't compare
            // solutions directly here. The function value comparison above validates correctness.
        }
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
     * **Validates: Requirements 1.1, 2.6**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 1: Workspace Reuse Correctness - SLSQP")
    void slsqpWorkspaceReuseProducesConsistentResults(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 3, max = 5) int numOptimizations
    ) {
        // Create a simple quadratic objective function: f(x) = sum(x_i^2)
        ObjectiveFunction quadratic = (x, gradient) -> {
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                f += x[i] * x[i];
            }
            if (gradient != null) {
                for (int i = 0; i < x.length; i++) {
                    gradient[i] = 2.0 * x[i];
                }
            }
            return f;
        };

        // Build SLSQP optimizer (no constraints)
        SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(quadratic)
                .termination(Termination.builder()
                        .maxIterations(100)
                        .accuracy(1e-8)
                        .build())
                .build();

        // First, get the baseline result without workspace reuse
        double[] baselineInitial = createInitialPoint(n, 1.0);
        OptimizationResult baselineResult = optimizer.optimize(baselineInitial);

        // Now test with workspace reuse
        try (SlsqpWorkspace workspace = SlsqpWorkspace.allocate(n, 0, 0)) {
            OptimizationResult[] results = new OptimizationResult[numOptimizations];

            // Run multiple optimizations with the same workspace
            for (int i = 0; i < numOptimizations; i++) {
                double[] initial = createInitialPoint(n, 1.0);
                results[i] = optimizer.optimize(initial, workspace);
            }

            // Verify all results are consistent with each other
            for (int i = 0; i < numOptimizations; i++) {
                // All optimizations should converge
                assertThat(results[i].isConverged())
                        .as("SLSQP Optimization %d should converge", i)
                        .isTrue();

                // Function values should be close to 0 (the minimum)
                assertThat(results[i].getFunctionValue())
                        .as("SLSQP Function value for optimization %d should be near minimum", i)
                        .isLessThan(1e-8);
            }

            // Verify results match baseline (without workspace reuse)
            assertThat(baselineResult.isConverged())
                    .as("SLSQP Baseline optimization should converge")
                    .isTrue();

            // Compare function values between workspace and non-workspace versions
            assertThat(results[0].getFunctionValue())
                    .as("SLSQP Workspace result should match baseline function value")
                    .isCloseTo(baselineResult.getFunctionValue(), within(1e-8));
        }
    }

    /**
     * Property 2: Dimension Mismatch Error Handling
     * 
     * For any workspace dimension (n1, m1) and problem dimension (n2, m2),
     * when n1 ≠ n2 or m1 ≠ m2, the optimize method should throw an
     * IllegalArgumentException indicating dimension mismatch.
     * 
     * **Validates: Requirements 1.3**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 2: Dimension Mismatch Error Handling")
    void dimensionMismatchThrowsException(
            @ForAll @IntRange(min = 2, max = 20) int workspaceN,
            @ForAll @IntRange(min = 3, max = 10) int workspaceM,
            @ForAll @IntRange(min = 2, max = 20) int problemN,
            @ForAll @IntRange(min = 3, max = 10) int problemM
    ) {
        // Skip cases where dimensions match (no mismatch to test)
        Assume.that(workspaceN != problemN || workspaceM != problemM);

        // Create a simple quadratic objective function
        ObjectiveFunction quadratic = (x, gradient) -> {
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                f += x[i] * x[i];
            }
            if (gradient != null) {
                for (int i = 0; i < x.length; i++) {
                    gradient[i] = 2.0 * x[i];
                }
            }
            return f;
        };

        // Build optimizer with problem dimensions (n2, m2)
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
                .dimension(problemN)
                .corrections(problemM)
                .objective(quadratic)
                .termination(Termination.builder()
                        .maxIterations(100)
                        .maxEvaluations(500)
                        .gradientTolerance(1e-6)
                        .build())
                .build();

        // Create workspace with different dimensions (n1, m1)
        try (LbfgsbWorkspace workspace = LbfgsbWorkspace.allocate(workspaceN, workspaceM)) {
            double[] initialPoint = createInitialPoint(problemN, 1.0);

            // Attempting to optimize with mismatched workspace should throw IllegalArgumentException
            assertThatThrownBy(() -> optimizer.optimize(initialPoint, workspace))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("do not match");
        }
    }
}
