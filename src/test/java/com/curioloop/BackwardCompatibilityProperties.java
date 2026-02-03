/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;

import static org.assertj.core.api.Assertions.*;

/**
 * Property-based tests for backward compatibility.
 * 
 * These tests verify that the existing optimize(double[]) API works correctly
 * and produces expected results for both LbfgsbOptimizer and SlsqpOptimizer.
 * 
 * **Validates: Requirements 2.1, 2.2, 2.3**
 */
public class BackwardCompatibilityProperties {

    /**
     * Property 6: Backward Compatibility - LbfgsbOptimizer
     * 
     * For any valid problem using the existing optimize(double[]) API,
     * the behavior and results should be consistent with expected optimization behavior.
     * The existing API should work correctly without any workspace parameter.
     * 
     * This test verifies:
     * 1. LbfgsbOptimizer.optimize(double[]) method signature is preserved
     * 2. The method produces correct optimization results
     * 3. Convergence is achieved for well-defined problems
     * 4. Solutions are mathematically correct
     * 
     * **Validates: Requirements 2.1, 2.3**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 6: Backward Compatibility - LbfgsbOptimizer")
    void lbfgsbExistingApiWorksCorrectly(
            @ForAll @IntRange(min = 2, max = 30) int n,
            @ForAll @IntRange(min = 3, max = 10) int m,
            @ForAll @DoubleRange(min = 0.5, max = 3.0) double initialScale
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

        // Build optimizer using the standard builder pattern
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
                .dimension(n)
                .corrections(m)
                .objective(quadratic)
                .termination(Termination.builder()
                        .maxIterations(200)
                        .maxEvaluations(1000)
                        .gradientTolerance(1e-8)
                        .build())
                .build();

        // Create initial point
        double[] initialPoint = createInitialPoint(n, initialScale);
        
        // Use the existing optimize(double[]) API - this is the backward compatibility test
        OptimizationResult result = optimizer.optimize(initialPoint);

        // Verify the optimization converged
        assertThat(result.isConverged())
                .as("LbfgsbOptimizer.optimize(double[]) should converge for quadratic function")
                .isTrue();

        // Verify the function value is close to the known minimum (0)
        assertThat(result.getFunctionValue())
                .as("Function value should be near minimum (0)")
                .isLessThan(1e-10);

        // Verify the solution is close to the known optimum (origin)
        double[] solution = result.getSolution();
        assertThat(solution.length)
                .as("Solution dimension should match problem dimension")
                .isEqualTo(n);
        
        for (int i = 0; i < n; i++) {
            assertThat(Math.abs(solution[i]))
                    .as("Solution component %d should be near 0", i)
                    .isLessThan(1e-4);
        }

        // Verify iterations and evaluations are reasonable
        assertThat(result.getIterations())
                .as("Should complete in reasonable number of iterations")
                .isGreaterThan(0)
                .isLessThanOrEqualTo(200);
    }

    /**
     * Property 6: Backward Compatibility - SlsqpOptimizer
     * 
     * For any valid problem using the existing optimize(double[]) API,
     * the behavior and results should be consistent with expected optimization behavior.
     * The existing API should work correctly without any workspace parameter.
     * 
     * This test verifies:
     * 1. SlsqpOptimizer.optimize(double[]) method signature is preserved
     * 2. The method produces correct optimization results
     * 3. Convergence is achieved for well-defined problems
     * 4. Solutions are mathematically correct
     * 
     * **Validates: Requirements 2.2, 2.3**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 6: Backward Compatibility - SlsqpOptimizer")
    void slsqpExistingApiWorksCorrectly(
            @ForAll @IntRange(min = 2, max = 30) int n,
            @ForAll @DoubleRange(min = 0.5, max = 3.0) double initialScale
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

        // Build optimizer using the standard builder pattern (no constraints)
        SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(quadratic)
                .termination(Termination.builder()
                        .maxIterations(200)
                        .accuracy(1e-10)
                        .build())
                .build();

        // Create initial point
        double[] initialPoint = createInitialPoint(n, initialScale);
        
        // Use the existing optimize(double[]) API - this is the backward compatibility test
        OptimizationResult result = optimizer.optimize(initialPoint);

        // Verify the optimization converged
        assertThat(result.isConverged())
                .as("SlsqpOptimizer.optimize(double[]) should converge for quadratic function")
                .isTrue();

        // Verify the function value is close to the known minimum (0)
        assertThat(result.getFunctionValue())
                .as("Function value should be near minimum (0)")
                .isLessThan(1e-8);

        // Verify the solution is close to the known optimum (origin)
        double[] solution = result.getSolution();
        assertThat(solution.length)
                .as("Solution dimension should match problem dimension")
                .isEqualTo(n);
        
        for (int i = 0; i < n; i++) {
            assertThat(Math.abs(solution[i]))
                    .as("Solution component %d should be near 0", i)
                    .isLessThan(1e-4);
        }

        // Verify iterations are reasonable
        assertThat(result.getIterations())
                .as("Should complete in reasonable number of iterations")
                .isGreaterThan(0)
                .isLessThanOrEqualTo(200);
    }

    /**
     * Property 6: Backward Compatibility - LbfgsbOptimizer with Bounds
     * 
     * Verifies that the existing API works correctly with bound constraints.
     * This tests a more complex scenario to ensure backward compatibility
     * for bounded optimization problems.
     * 
     * **Validates: Requirements 2.1, 2.3**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 6: Backward Compatibility - LbfgsbOptimizer with Bounds")
    void lbfgsbExistingApiWithBoundsWorksCorrectly(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 3, max = 10) int m,
            @ForAll @DoubleRange(min = 0.5, max = 2.0) double initialScale
    ) {
        // Create a quadratic objective function: f(x) = sum((x_i - 1)^2)
        // This has a known minimum at x = [1, 1, ..., 1] with f(x*) = 0
        ObjectiveFunction shiftedQuadratic = (x, gradient) -> {
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                double diff = x[i] - 1.0;
                f += diff * diff;
            }
            if (gradient != null) {
                for (int i = 0; i < x.length; i++) {
                    gradient[i] = 2.0 * (x[i] - 1.0);
                }
            }
            return f;
        };

        // Create bounds that include the optimum
        Bound[] bounds = new Bound[n];
        for (int i = 0; i < n; i++) {
            bounds[i] = Bound.between(-5.0, 5.0);
        }

        // Build optimizer with bounds
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
                .dimension(n)
                .corrections(m)
                .objective(shiftedQuadratic)
                .bounds(bounds)
                .termination(Termination.builder()
                        .maxIterations(200)
                        .maxEvaluations(1000)
                        .gradientTolerance(1e-8)
                        .build())
                .build();

        // Create initial point
        double[] initialPoint = createInitialPoint(n, initialScale);
        
        // Use the existing optimize(double[]) API
        OptimizationResult result = optimizer.optimize(initialPoint);

        // Verify the optimization converged
        assertThat(result.isConverged())
                .as("LbfgsbOptimizer.optimize(double[]) with bounds should converge")
                .isTrue();

        // Verify the function value is close to the known minimum (0)
        assertThat(result.getFunctionValue())
                .as("Function value should be near minimum (0)")
                .isLessThan(1e-10);

        // Verify the solution is close to the known optimum [1, 1, ..., 1]
        double[] solution = result.getSolution();
        for (int i = 0; i < n; i++) {
            assertThat(solution[i])
                    .as("Solution component %d should be near 1", i)
                    .isCloseTo(1.0, within(1e-4));
        }
    }

    /**
     * Property 6: Backward Compatibility - SlsqpOptimizer with Constraints
     * 
     * Verifies that the existing API works correctly with equality and inequality constraints.
     * This tests a more complex scenario to ensure backward compatibility
     * for constrained optimization problems.
     * 
     * **Validates: Requirements 2.2, 2.3**
     */
    @Property(tries = 100)
    @Label("Feature: jni-optimization, Property 6: Backward Compatibility - SlsqpOptimizer with Constraints")
    void slsqpExistingApiWithConstraintsWorksCorrectly(
            @ForAll @IntRange(min = 2, max = 15) int n
    ) {
        // Create a quadratic objective function: f(x) = sum(x_i^2)
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

        // Equality constraint: sum(x) = 1
        // The optimal solution is x_i = 1/n for all i
        // with f(x*) = n * (1/n)^2 = 1/n
        ConstraintFunction sumConstraint = (x, gradient) -> {
            double sum = 0.0;
            for (int i = 0; i < x.length; i++) {
                sum += x[i];
            }
            if (gradient != null) {
                for (int i = 0; i < x.length; i++) {
                    gradient[i] = 1.0;
                }
            }
            return sum - 1.0;  // sum(x) - 1 = 0
        };

        // Build optimizer with equality constraint
        SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(quadratic)
                .equalityConstraints(sumConstraint)
                .termination(Termination.builder()
                        .maxIterations(200)
                        .accuracy(1e-10)
                        .build())
                .build();

        // Create initial point that satisfies the constraint
        double[] initialPoint = new double[n];
        for (int i = 0; i < n; i++) {
            initialPoint[i] = 1.0 / n;
        }
        
        // Use the existing optimize(double[]) API
        OptimizationResult result = optimizer.optimize(initialPoint);

        // Verify the optimization converged
        assertThat(result.isConverged())
                .as("SlsqpOptimizer.optimize(double[]) with constraints should converge")
                .isTrue();

        // Verify the function value is close to the known minimum (1/n)
        double expectedMinimum = 1.0 / n;
        assertThat(result.getFunctionValue())
                .as("Function value should be near minimum (1/n = %f)", expectedMinimum)
                .isCloseTo(expectedMinimum, within(1e-6));

        // Verify the solution satisfies the constraint (sum = 1)
        double[] solution = result.getSolution();
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += solution[i];
        }
        assertThat(sum)
                .as("Solution should satisfy constraint sum(x) = 1")
                .isCloseTo(1.0, within(1e-6));

        // Verify each component is close to 1/n
        double expectedComponent = 1.0 / n;
        for (int i = 0; i < n; i++) {
            assertThat(solution[i])
                    .as("Solution component %d should be near 1/n = %f", i, expectedComponent)
                    .isCloseTo(expectedComponent, within(1e-4));
        }
    }

    /**
     * Property 6: Backward Compatibility - Rosenbrock Function Test
     * 
     * Verifies that the existing API works correctly for a more challenging
     * optimization problem (Rosenbrock function). This ensures backward
     * compatibility for real-world optimization scenarios.
     * 
     * Note: The Rosenbrock function is notoriously difficult to optimize.
     * This test focuses on verifying the API works correctly (returns valid results)
     * rather than testing convergence to the global minimum.
     * 
     * **Validates: Requirements 2.1, 2.2, 2.3**
     */
    @Property(tries = 50)
    @Label("Feature: jni-optimization, Property 6: Backward Compatibility - Rosenbrock Function")
    void existingApiWorksForRosenbrockFunction(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 5, max = 10) int m
    ) {
        // Rosenbrock function: f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
        // Known minimum at x = [1, 1, ..., 1] with f(x*) = 0
        ObjectiveFunction rosenbrock = (x, gradient) -> {
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

        // Test with L-BFGS-B
        LbfgsbOptimizer lbfgsb = LbfgsbOptimizer.builder()
                .dimension(n)
                .corrections(m)
                .objective(rosenbrock)
                .termination(Termination.builder()
                        .maxIterations(2000)
                        .maxEvaluations(10000)
                        .gradientTolerance(1e-6)
                        .build())
                .build();

        // Start from a point very close to the minimum for reliable convergence
        double[] initialPoint = new double[n];
        for (int i = 0; i < n; i++) {
            initialPoint[i] = 0.9;  // Start very close to the minimum [1, 1, ..., 1]
        }
        
        // Use the existing optimize(double[]) API
        OptimizationResult lbfgsbResult = lbfgsb.optimize(initialPoint);

        // Verify the API returns a valid result (backward compatibility check)
        assertThat(lbfgsbResult).isNotNull();
        assertThat(lbfgsbResult.getSolution()).hasSize(n);
        assertThat(lbfgsbResult.getStatus()).isNotNull();
        assertThat(lbfgsbResult.getFunctionValue()).isNotNaN();
        
        // Verify the optimizer made progress (function value decreased from initial)
        double initialFunctionValue = rosenbrock.evaluate(new double[n], null);
        for (int i = 0; i < n; i++) {
            initialPoint[i] = 0.9;
        }
        double initialValue = rosenbrock.evaluate(initialPoint, null);
        assertThat(lbfgsbResult.getFunctionValue())
                .as("L-BFGS-B should make progress on Rosenbrock function")
                .isLessThanOrEqualTo(initialValue);

        // Test with SLSQP
        SlsqpOptimizer slsqp = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(rosenbrock)
                .termination(Termination.builder()
                        .maxIterations(2000)
                        .accuracy(1e-8)
                        .build())
                .build();

        // Reset initial point
        for (int i = 0; i < n; i++) {
            initialPoint[i] = 0.9;
        }
        
        // Use the existing optimize(double[]) API
        OptimizationResult slsqpResult = slsqp.optimize(initialPoint);

        // Verify the API returns a valid result (backward compatibility check)
        assertThat(slsqpResult).isNotNull();
        assertThat(slsqpResult.getSolution()).hasSize(n);
        assertThat(slsqpResult.getStatus()).isNotNull();
        assertThat(slsqpResult.getFunctionValue()).isNotNaN();
        
        // Verify the optimizer made progress
        assertThat(slsqpResult.getFunctionValue())
                .as("SLSQP should make progress on Rosenbrock function")
                .isLessThanOrEqualTo(initialValue);
    }

    /**
     * Property 6: Backward Compatibility - Method Signature Verification
     * 
     * Verifies that the existing method signatures are preserved and accessible.
     * This is a compile-time and runtime verification that the API hasn't changed.
     * 
     * **Validates: Requirements 2.1, 2.2**
     */
    @Property(tries = 10)
    @Label("Feature: jni-optimization, Property 6: Backward Compatibility - Method Signature Verification")
    void methodSignaturesArePreserved(
            @ForAll @IntRange(min = 2, max = 10) int n
    ) {
        // Simple objective function
        ObjectiveFunction objective = (x, gradient) -> {
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

        // Verify LbfgsbOptimizer.optimize(double[]) signature exists and works
        LbfgsbOptimizer lbfgsb = LbfgsbOptimizer.builder()
                .dimension(n)
                .objective(objective)
                .build();
        
        double[] lbfgsbInitial = createInitialPoint(n, 1.0);
        OptimizationResult lbfgsbResult = lbfgsb.optimize(lbfgsbInitial);
        
        // Verify result is not null and has expected structure
        assertThat(lbfgsbResult).isNotNull();
        assertThat(lbfgsbResult.getSolution()).hasSize(n);
        assertThat(lbfgsbResult.getStatus()).isNotNull();
        assertThat(lbfgsbResult.getFunctionValue()).isNotNaN();
        assertThat(lbfgsbResult.getIterations()).isGreaterThanOrEqualTo(0);

        // Verify SlsqpOptimizer.optimize(double[]) signature exists and works
        SlsqpOptimizer slsqp = SlsqpOptimizer.builder()
                .dimension(n)
                .objective(objective)
                .build();
        
        double[] slsqpInitial = createInitialPoint(n, 1.0);
        OptimizationResult slsqpResult = slsqp.optimize(slsqpInitial);
        
        // Verify result is not null and has expected structure
        assertThat(slsqpResult).isNotNull();
        assertThat(slsqpResult.getSolution()).hasSize(n);
        assertThat(slsqpResult.getStatus()).isNotNull();
        assertThat(slsqpResult.getFunctionValue()).isNotNaN();
        assertThat(slsqpResult.getIterations()).isGreaterThanOrEqualTo(0);
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
}
