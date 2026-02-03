/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;

import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.*;

/**
 * Property-based tests for callback exception handling.
 * 
 * **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
 */
public class CallbackExceptionProperties {

    /**
     * Custom exception for testing callback exception handling.
     */
    public static class TestCallbackException extends RuntimeException {
        private final int evaluationNumber;
        
        public TestCallbackException(int evaluationNumber) {
            super("Test exception at evaluation " + evaluationNumber);
            this.evaluationNumber = evaluationNumber;
        }
        
        public int getEvaluationNumber() {
            return evaluationNumber;
        }
    }

    /**
     * Property 4: Callback Exception Handling
     * 
     * For any callback function that throws an exception at iteration k,
     * the optimizer should terminate immediately after detecting the exception,
     * return STATUS_CALLBACK_ERROR status, and the original Java exception
     * should be preserved.
     * 
     * This test verifies:
     * 1. When objective function callback throws an exception, optimizer terminates
     * 2. STATUS_CALLBACK_ERROR is returned OR the original exception is thrown
     * 3. The exception is thrown after a limited number of additional evaluations
     * 
     * **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 4: Callback Exception Handling")
    void objectiveExceptionTerminatesOptimization(
            @ForAll @IntRange(min = 5, max = 20) int n,
            @ForAll @IntRange(min = 3, max = 7) int m,
            @ForAll @IntRange(min = 1, max = 3) int throwAfterK
    ) {
        AtomicInteger evaluationCount = new AtomicInteger(0);
        
        // Create an objective function that throws an exception after k evaluations
        // Use Rosenbrock function which requires many iterations to converge
        Evaluation throwingObjective = (x, gradient) -> {
            int currentEval = evaluationCount.incrementAndGet();
            
            // Throw exception after k evaluations
            if (currentEval > throwAfterK) {
                throw new TestCallbackException(currentEval);
            }
            
            // Rosenbrock function: f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
            // This function is harder to optimize and requires many iterations
            double f = 0.0;
            for (int i = 0; i < x.length - 1; i++) {
                double t1 = x[i + 1] - x[i] * x[i];
                double t2 = 1.0 - x[i];
                f += 100.0 * t1 * t1 + t2 * t2;
            }
            if (gradient != null) {
                gradient[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
                for (int i = 1; i < x.length - 1; i++) {
                    gradient[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                                - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
                                - 2.0 * (1.0 - x[i]);
                }
                gradient[x.length - 1] = 200.0 * (x[x.length - 1] - x[x.length - 2] * x[x.length - 2]);
            }
            return f;
        };

        // Build optimizer with tight tolerance to ensure it won't converge quickly
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
                .dimension(n)
                .corrections(m)
                .objective(throwingObjective)
                .termination(Termination.builder()
                        .maxIterations(1000)
                        .maxEvaluations(5000)
                        .gradientTolerance(1e-12)  // Very tight tolerance
                        .build())
                .build();

        // Create initial point far from minimum to ensure multiple evaluations
        double[] initialPoint = createInitialPoint(n, -2.0);
        
        // Run optimization - should either throw exception or return CALLBACK_ERROR
        // According to requirement 4.4, the original Java exception should be preserved
        try {
            OptimizationResult result = optimizer.optimize(initialPoint);
            
            // If no exception thrown, verify CALLBACK_ERROR status
            assertThat(result.getStatus())
                    .as("Optimizer should return CALLBACK_ERROR when callback throws exception")
                    .isEqualTo(OptimizationStatus.CALLBACK_ERROR);
        } catch (TestCallbackException e) {
            // Exception was preserved and propagated - this is valid behavior per requirement 4.4
            assertThat(e.getEvaluationNumber())
                    .as("Exception should be thrown at evaluation %d", throwAfterK + 1)
                    .isEqualTo(throwAfterK + 1);
        }
        
        // Verify the optimizer terminated shortly after the exception
        // The optimizer should not continue many iterations after the exception
        int totalEvaluations = evaluationCount.get();
        assertThat(totalEvaluations)
                .as("Optimizer should terminate shortly after exception (threw at %d, total %d)", 
                    throwAfterK + 1, totalEvaluations)
                .isLessThanOrEqualTo(throwAfterK + 5); // Allow a small buffer for in-flight evaluations
    }

    /**
     * Property 4: Callback Exception Handling - SLSQP Objective Exception
     * 
     * Verifies that SLSQP optimizer also handles objective function exceptions correctly.
     * The exception should be preserved and either thrown or STATUS_CALLBACK_ERROR returned.
     * 
     * **Validates: Requirements 4.1, 4.3, 4.4**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 4: Callback Exception Handling - SLSQP Objective")
    void slsqpObjectiveExceptionTerminatesOptimization(
            @ForAll @IntRange(min = 5, max = 15) int n,
            @ForAll @IntRange(min = 1, max = 3) int throwAfterK
    ) {
        AtomicInteger evaluationCount = new AtomicInteger(0);
        
        // Create an objective function that throws an exception after k evaluations
        // Use Rosenbrock function which requires many iterations
        Evaluation throwingObjective = (x, gradient) -> {
            int currentEval = evaluationCount.incrementAndGet();
            
            // Throw exception after k evaluations
            if (currentEval > throwAfterK) {
                throw new TestCallbackException(currentEval);
            }
            
            // Rosenbrock function
            double f = 0.0;
            for (int i = 0; i < x.length - 1; i++) {
                double t1 = x[i + 1] - x[i] * x[i];
                double t2 = 1.0 - x[i];
                f += 100.0 * t1 * t1 + t2 * t2;
            }
            if (gradient != null) {
                gradient[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
                for (int i = 1; i < x.length - 1; i++) {
                    gradient[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                                - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
                                - 2.0 * (1.0 - x[i]);
                }
                gradient[x.length - 1] = 200.0 * (x[x.length - 1] - x[x.length - 2] * x[x.length - 2]);
            }
            return f;
        };

        // Build SLSQP optimizer without constraints
        SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(throwingObjective)
                .termination(Termination.builder()
                        .maxIterations(1000)
                        .accuracy(1e-12)  // Very tight tolerance
                        .build())
                .build();

        // Create initial point far from minimum
        double[] initialPoint = createInitialPoint(n, -2.0);
        
        // Run optimization - should either throw exception or return CALLBACK_ERROR
        // According to requirement 4.4, the original Java exception should be preserved
        try {
            OptimizationResult result = optimizer.optimize(initialPoint);
            
            // If no exception thrown, verify CALLBACK_ERROR status
            assertThat(result.getStatus())
                    .as("SLSQP optimizer should return CALLBACK_ERROR when objective throws exception")
                    .isEqualTo(OptimizationStatus.CALLBACK_ERROR);
        } catch (TestCallbackException e) {
            // Exception was preserved and propagated - this is valid behavior per requirement 4.4
            assertThat(e.getEvaluationNumber())
                    .as("Exception should be thrown at evaluation %d", throwAfterK + 1)
                    .isEqualTo(throwAfterK + 1);
        }
        
        // Verify the optimizer terminated shortly after the exception
        int totalEvaluations = evaluationCount.get();
        assertThat(totalEvaluations)
                .as("SLSQP optimizer should terminate shortly after exception (threw at %d, total %d)", 
                    throwAfterK + 1, totalEvaluations)
                .isLessThanOrEqualTo(throwAfterK + 5);
    }

    /**
     * Property 4: Callback Exception Handling - SLSQP Constraint Exception
     * 
     * Verifies that SLSQP optimizer handles constraint function exceptions correctly.
     * The exception should be preserved and either thrown or STATUS_CALLBACK_ERROR returned.
     * 
     * **Validates: Requirements 4.2, 4.3, 4.4**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 4: Callback Exception Handling - SLSQP Constraint")
    void slsqpConstraintExceptionTerminatesOptimization(
            @ForAll @IntRange(min = 5, max = 15) int n,
            @ForAll @IntRange(min = 1, max = 3) int throwAfterK
    ) {
        AtomicInteger constraintEvalCount = new AtomicInteger(0);
        
        // Normal objective function - Rosenbrock
        Evaluation objective = (x, gradient) -> {
            double f = 0.0;
            for (int i = 0; i < x.length - 1; i++) {
                double t1 = x[i + 1] - x[i] * x[i];
                double t2 = 1.0 - x[i];
                f += 100.0 * t1 * t1 + t2 * t2;
            }
            if (gradient != null) {
                gradient[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
                for (int i = 1; i < x.length - 1; i++) {
                    gradient[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                                - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
                                - 2.0 * (1.0 - x[i]);
                }
                gradient[x.length - 1] = 200.0 * (x[x.length - 1] - x[x.length - 2] * x[x.length - 2]);
            }
            return f;
        };

        // Constraint function that throws an exception after k evaluations
        Evaluation throwingConstraint = (x, gradient) -> {
            int currentEval = constraintEvalCount.incrementAndGet();
            
            // Throw exception after k evaluations
            if (currentEval > throwAfterK) {
                throw new TestCallbackException(currentEval);
            }
            
            // Simple constraint: sum(x) >= 0
            double sum = 0.0;
            for (int i = 0; i < x.length; i++) {
                sum += x[i];
            }
            if (gradient != null) {
                for (int i = 0; i < x.length; i++) {
                    gradient[i] = 1.0;
                }
            }
            return sum;
        };

        // Build SLSQP optimizer with inequality constraint
        SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(objective)
                .inequalityConstraints(throwingConstraint)
                .termination(Termination.builder()
                        .maxIterations(1000)
                        .accuracy(1e-12)
                        .build())
                .build();

        // Create initial point
        double[] initialPoint = createInitialPoint(n, -2.0);
        
        // Run optimization - should either throw exception or return CALLBACK_ERROR
        try {
            OptimizationResult result = optimizer.optimize(initialPoint);
            
            // If no exception thrown, verify CALLBACK_ERROR status
            assertThat(result.getStatus())
                    .as("SLSQP optimizer should return CALLBACK_ERROR when constraint throws exception")
                    .isEqualTo(OptimizationStatus.CALLBACK_ERROR);
        } catch (TestCallbackException e) {
            // Exception was preserved and propagated - this is valid behavior per requirement 4.4
            assertThat(e.getEvaluationNumber())
                    .as("Exception should be thrown at evaluation %d", throwAfterK + 1)
                    .isEqualTo(throwAfterK + 1);
        }
        
        // Verify the optimizer terminated shortly after the exception
        int totalConstraintEvals = constraintEvalCount.get();
        assertThat(totalConstraintEvals)
                .as("SLSQP optimizer should terminate shortly after constraint exception (threw at %d, total %d)", 
                    throwAfterK + 1, totalConstraintEvals)
                .isLessThanOrEqualTo(throwAfterK + 5);
    }

    /**
     * Property 4: Callback Exception Handling - Equality Constraint Exception
     * 
     * Verifies that SLSQP optimizer handles equality constraint function exceptions correctly.
     * The exception should be preserved and either thrown or STATUS_CALLBACK_ERROR returned.
     * 
     * **Validates: Requirements 4.2, 4.3, 4.4**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 4: Callback Exception Handling - SLSQP Equality Constraint")
    void slsqpEqualityConstraintExceptionTerminatesOptimization(
            @ForAll @IntRange(min = 5, max = 15) int n,
            @ForAll @IntRange(min = 1, max = 3) int throwAfterK
    ) {
        AtomicInteger constraintEvalCount = new AtomicInteger(0);
        
        // Normal objective function - Rosenbrock
        Evaluation objective = (x, gradient) -> {
            double f = 0.0;
            for (int i = 0; i < x.length - 1; i++) {
                double t1 = x[i + 1] - x[i] * x[i];
                double t2 = 1.0 - x[i];
                f += 100.0 * t1 * t1 + t2 * t2;
            }
            if (gradient != null) {
                gradient[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
                for (int i = 1; i < x.length - 1; i++) {
                    gradient[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                                - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
                                - 2.0 * (1.0 - x[i]);
                }
                gradient[x.length - 1] = 200.0 * (x[x.length - 1] - x[x.length - 2] * x[x.length - 2]);
            }
            return f;
        };

        // Equality constraint function that throws an exception after k evaluations
        Evaluation throwingEqConstraint = (x, gradient) -> {
            int currentEval = constraintEvalCount.incrementAndGet();
            
            // Throw exception after k evaluations
            if (currentEval > throwAfterK) {
                throw new TestCallbackException(currentEval);
            }
            
            // Simple equality constraint: sum(x) - 1 = 0
            double sum = 0.0;
            for (int i = 0; i < x.length; i++) {
                sum += x[i];
            }
            if (gradient != null) {
                for (int i = 0; i < x.length; i++) {
                    gradient[i] = 1.0;
                }
            }
            return sum - 1.0;
        };

        // Build SLSQP optimizer with equality constraint
        SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(objective)
                .equalityConstraints(throwingEqConstraint)
                .termination(Termination.builder()
                        .maxIterations(1000)
                        .accuracy(1e-12)
                        .build())
                .build();

        // Create initial point
        double[] initialPoint = createInitialPoint(n, -2.0);
        
        // Run optimization - should either throw exception or return CALLBACK_ERROR
        try {
            OptimizationResult result = optimizer.optimize(initialPoint);
            
            // If no exception thrown, verify CALLBACK_ERROR status
            assertThat(result.getStatus())
                    .as("SLSQP optimizer should return CALLBACK_ERROR when equality constraint throws exception")
                    .isEqualTo(OptimizationStatus.CALLBACK_ERROR);
        } catch (TestCallbackException e) {
            // Exception was preserved and propagated - this is valid behavior per requirement 4.4
            assertThat(e.getEvaluationNumber())
                    .as("Exception should be thrown at evaluation %d", throwAfterK + 1)
                    .isEqualTo(throwAfterK + 1);
        }
        
        // Verify the optimizer terminated shortly after the exception
        int totalConstraintEvals = constraintEvalCount.get();
        assertThat(totalConstraintEvals)
                .as("SLSQP optimizer should terminate shortly after equality constraint exception (threw at %d, total %d)", 
                    throwAfterK + 1, totalConstraintEvals)
                .isLessThanOrEqualTo(throwAfterK + 5);
    }

    /**
     * Property 4: Callback Exception Handling - With Workspace Reuse
     * 
     * Verifies that exception handling works correctly when using workspace reuse.
     * The exception should be preserved and either thrown or STATUS_CALLBACK_ERROR returned.
     * 
     * **Validates: Requirements 4.1, 4.3, 4.4**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 4: Callback Exception Handling - Workspace Reuse")
    void exceptionHandlingWithWorkspaceReuse(
            @ForAll @IntRange(min = 5, max = 20) int n,
            @ForAll @IntRange(min = 3, max = 7) int m,
            @ForAll @IntRange(min = 1, max = 3) int throwAfterK
    ) {
        AtomicInteger evaluationCount = new AtomicInteger(0);
        
        // Create an objective function that throws an exception after k evaluations
        // Use Rosenbrock function which requires many iterations
        Evaluation throwingObjective = (x, gradient) -> {
            int currentEval = evaluationCount.incrementAndGet();
            
            // Throw exception after k evaluations
            if (currentEval > throwAfterK) {
                throw new TestCallbackException(currentEval);
            }
            
            // Rosenbrock function
            double f = 0.0;
            for (int i = 0; i < x.length - 1; i++) {
                double t1 = x[i + 1] - x[i] * x[i];
                double t2 = 1.0 - x[i];
                f += 100.0 * t1 * t1 + t2 * t2;
            }
            if (gradient != null) {
                gradient[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
                for (int i = 1; i < x.length - 1; i++) {
                    gradient[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                                - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
                                - 2.0 * (1.0 - x[i]);
                }
                gradient[x.length - 1] = 200.0 * (x[x.length - 1] - x[x.length - 2] * x[x.length - 2]);
            }
            return f;
        };

        // Build optimizer with tight tolerance
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
                .dimension(n)
                .corrections(m)
                .objective(throwingObjective)
                .termination(Termination.builder()
                        .maxIterations(1000)
                        .maxEvaluations(5000)
                        .gradientTolerance(1e-12)
                        .build())
                .build();

        // Test with workspace reuse
        try (LbfgsbWorkspace workspace = LbfgsbWorkspace.allocate(n, m)) {
            double[] initialPoint = createInitialPoint(n, -2.0);
            
            // Run optimization - should either throw exception or return CALLBACK_ERROR
            try {
                OptimizationResult result = optimizer.optimize(initialPoint, workspace);
                
                // If no exception thrown, verify CALLBACK_ERROR status
                assertThat(result.getStatus())
                        .as("Optimizer with workspace should return CALLBACK_ERROR when callback throws exception")
                        .isEqualTo(OptimizationStatus.CALLBACK_ERROR);
            } catch (TestCallbackException e) {
                // Exception was preserved and propagated - this is valid behavior per requirement 4.4
                assertThat(e.getEvaluationNumber())
                        .as("Exception should be thrown at evaluation %d", throwAfterK + 1)
                        .isEqualTo(throwAfterK + 1);
            }
            
            // Verify the optimizer terminated shortly after the exception
            int totalEvaluations = evaluationCount.get();
            assertThat(totalEvaluations)
                    .as("Optimizer with workspace should terminate shortly after exception (threw at %d, total %d)", 
                        throwAfterK + 1, totalEvaluations)
                    .isLessThanOrEqualTo(throwAfterK + 5);
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
}
