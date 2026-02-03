/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;

import static org.assertj.core.api.Assertions.*;

/**
 * Property-based tests for large constraint stability.
 * 
 * This test verifies that SLSQP optimization with more than 16 constraints
 * completes without crashing due to JNI local reference table overflow.
 * 
 * The JNI bridge uses PushLocalFrame/PopLocalFrame when the number of
 * constraints of a single type (equality or inequality) exceeds 16.
 * 
 * **Validates: Requirements 6.1**
 */
public class LargeConstraintStabilityProperties {

    /**
     * Property 5: Large Constraint Stability
     * 
     * For any SLSQP optimization problem with constraint count m > 16,
     * the optimization process should complete normally without crashing
     * due to local reference table overflow.
     * 
     * This test verifies:
     * 1. SLSQP optimization with > 16 inequality constraints completes without JNI crash
     * 2. The optimizer returns a valid result (converged or valid status)
     * 3. No OutOfMemoryError or JNI-related crashes occur
     * 
     * **Validates: Requirements 6.1**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 5: Large Constraint Stability")
    void largeConstraintCountDoesNotCauseReferenceTableOverflow(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 17, max = 50) int numConstraints
    ) {
        // Create a simple quadratic objective function: f(x) = sum(x_i^2)
        // This has a known minimum at x = 0 with f(0) = 0
        Evaluation quadratic = (x, gradient) -> {
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

        // Build SLSQP optimizer with many inequality constraints
        SlsqpOptimizer.Builder builder = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(quadratic)
                .termination(Termination.builder()
                        .maxIterations(100)
                        .accuracy(1e-6)
                        .build());

        // Add many inequality constraints: x_i >= -10 (always satisfied for reasonable x)
        // These are simple constraints that should be easily satisfied
        for (int i = 0; i < numConstraints; i++) {
            final int idx = i % n;  // Cycle through dimensions
            final double bound = -10.0 - i;  // Different bounds for variety
            
            Evaluation constraint = (x, gradient) -> {
                // Constraint: x[idx] + bound >= 0, i.e., x[idx] >= -bound
                if (gradient != null) {
                    for (int j = 0; j < x.length; j++) {
                        gradient[j] = (j == idx) ? 1.0 : 0.0;
                    }
                }
                return x[idx] + bound;
            };
            builder.inequalityConstraints(constraint);
        }

        SlsqpOptimizer optimizer = builder.build();

        // Verify the optimizer has the expected number of constraints
        assertThat(optimizer.getInequalityConstraintCount())
                .as("Optimizer should have %d inequality constraints", numConstraints)
                .isEqualTo(numConstraints);

        // Create initial point
        double[] initialPoint = createInitialPoint(n, 1.0);

        // Run optimization - this should NOT crash due to local reference table overflow
        // The key test is that this completes without JNI errors
        OptimizationResult result = optimizer.optimize(initialPoint);

        // Verify we got a valid result (not crashed)
        assertThat(result)
                .as("Optimization should return a valid result")
                .isNotNull();

        assertThat(result.getStatus())
                .as("Optimization status should be valid")
                .isNotNull();

        // Verify the solution array (modified in-place) has correct dimension
        assertThat(initialPoint.length)
                .as("Solution should have correct dimension")
                .isEqualTo(n);

        // For this simple problem, we expect convergence or at least a reasonable result
        // The main goal is to verify no crash occurs
        if (result.isConverged()) {
            // If converged, verify the solution is reasonable
            assertThat(result.getFunctionValue())
                    .as("Function value should be non-negative for quadratic")
                    .isGreaterThanOrEqualTo(0.0);
        }
    }

    /**
     * Property 5: Large Constraint Stability - Boundary Case
     * 
     * Verifies stability at the boundary (exactly 17 constraints, just above threshold).
     * 
     * **Validates: Requirements 6.1**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 5: Large Constraint Stability - Boundary Case")
    void boundaryConstraintCountDoesNotCauseReferenceTableOverflow(
            @ForAll @IntRange(min = 2, max = 10) int n
    ) {
        // Test with exactly 17 constraints (just above the threshold of 16)
        final int numConstraints = 17;

        // Create a simple quadratic objective function
        Evaluation quadratic = (x, gradient) -> {
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

        // Build SLSQP optimizer with exactly 17 constraints
        SlsqpOptimizer.Builder builder = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(quadratic)
                .termination(Termination.builder()
                        .maxIterations(100)
                        .accuracy(1e-6)
                        .build());

        // Add exactly 17 inequality constraints
        for (int i = 0; i < numConstraints; i++) {
            final int idx = i % n;
            final double bound = -10.0 - i;
            
            Evaluation constraint = (x, gradient) -> {
                if (gradient != null) {
                    for (int j = 0; j < x.length; j++) {
                        gradient[j] = (j == idx) ? 1.0 : 0.0;
                    }
                }
                return x[idx] + bound;
            };
            builder.inequalityConstraints(constraint);
        }

        SlsqpOptimizer optimizer = builder.build();

        // Verify exactly 17 constraints
        assertThat(optimizer.getInequalityConstraintCount())
                .as("Optimizer should have exactly 17 inequality constraints")
                .isEqualTo(17);

        // Create initial point
        double[] initialPoint = createInitialPoint(n, 1.0);

        // Run optimization - should NOT crash
        OptimizationResult result = optimizer.optimize(initialPoint);

        // Verify we got a valid result
        assertThat(result)
                .as("Optimization should return a valid result")
                .isNotNull();

        assertThat(result.getStatus())
                .as("Optimization status should be valid")
                .isNotNull();
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
}
