/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests demonstrating the improved API usability.
 */
public class ApiUsabilityTest {

    @Test
    @DisplayName("L-BFGS-B: Simple minimize with function only (numerical gradient)")
    void testLbfgsbSimpleMinimize() {
        // Minimize x^2 + y^2, minimum at (0, 0) using builder with ToDoubleFunction
        OptimizationResult result = LbfgsbOptimizer.builder()
            .dimension(2)
            .objective(x -> x[0] * x[0] + x[1] * x[1], NumericalGradient.CENTRAL)
            .build()
            .optimize(new double[]{1.0, 1.0});
        
        assertThat(result.isConverged()).isTrue();
        assertThat(result.getFunctionValue()).isCloseTo(0.0, within(1e-6));
    }
    
    @Test
    @DisplayName("L-BFGS-B: Minimize with analytical gradient")
    void testLbfgsbMinimizeWithGradient() {
        // Quadratic function with analytical gradient: f(x) = (x-2)^2 + (y-3)^2
        // Minimum at (2, 3)
        ObjectiveFunction quadratic = (x, g) -> {
            double f = Math.pow(x[0] - 2, 2) + Math.pow(x[1] - 3, 2);
            if (g != null) {
                g[0] = 2 * (x[0] - 2);
                g[1] = 2 * (x[1] - 3);
            }
            return f;
        };
        
        double[] x = {0.0, 0.0};
        OptimizationResult result = LbfgsbOptimizer.minimize(quadratic, x);
        
        assertThat(result.isConverged()).isTrue();
        assertThat(x[0]).isCloseTo(2.0, within(1e-6));
        assertThat(x[1]).isCloseTo(3.0, within(1e-6));
    }
    
    @Test
    @DisplayName("L-BFGS-B: Builder with function-only objective")
    void testLbfgsbBuilderWithFunctionOnly() {
        // Using builder with ToDoubleFunction (no gradient needed)
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
            .dimension(2)
            .objective(x -> x[0] * x[0] + x[1] * x[1], NumericalGradient.CENTRAL)
            .termination(Termination.builder().maxIterations(50).accuracy(1e-4).build())
            .build();
        
        OptimizationResult result = optimizer.optimize(new double[]{1.0, 1.0});
        
        assertThat(result.isConverged()).isTrue();
        assertThat(result.getFunctionValue()).isCloseTo(0.0, within(1e-3));
    }
    
    @Test
    @DisplayName("L-BFGS-B: Builder with bounds")
    void testLbfgsbBuilderWithBounds() {
        // Minimize (x-5)^2 + (y-5)^2 with bounds [0, 3]
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
            .dimension(2)
            .objective(x -> Math.pow(x[0] - 5, 2) + Math.pow(x[1] - 5, 2), NumericalGradient.CENTRAL)
            .bounds(Bound.between(0.0, 3.0))  // single bound for all
            .build();
        
        double[] x = {1.0, 1.0};
        OptimizationResult result = optimizer.optimize(x);
        
        assertThat(result.isConverged()).isTrue();
        assertThat(x[0]).isCloseTo(3.0, within(1e-6));
        assertThat(x[1]).isCloseTo(3.0, within(1e-6));
    }
    
    @Test
    @DisplayName("L-BFGS-B: Builder with single bound for all variables")
    void testLbfgsbBuilderWithSingleBound() {
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
            .dimension(3)
            .objective(x -> x[0] * x[0] + x[1] * x[1] + x[2] * x[2], NumericalGradient.CENTRAL)
            .bounds(Bound.atLeast(0))  // x >= 0 for all variables
            .build();
        
        double[] x = {1.0, 1.0, 1.0};
        OptimizationResult result = optimizer.optimize(x);
        
        assertThat(result.isConverged()).isTrue();
        for (double v : x) {
            assertThat(v).isGreaterThanOrEqualTo(-1e-10);
        }
    }
    
    @Test
    @DisplayName("Bound: Factory methods")
    void testBoundFactoryMethods() {
        // Test factory methods
        assertThat(Bound.between(-1, 1).getLower()).isEqualTo(-1);
        assertThat(Bound.between(-1, 1).getUpper()).isEqualTo(1);
        
        assertThat(Bound.atLeast(0).getLower()).isEqualTo(0);
        assertThat(Bound.atLeast(0).hasUpper()).isFalse();
        
        assertThat(Bound.atMost(10).getUpper()).isEqualTo(10);
        assertThat(Bound.atMost(10).hasLower()).isFalse();
        
        assertThat(Bound.exactly(5).isFixed()).isTrue();
        assertThat(Bound.exactly(5).getLower()).isEqualTo(5);
    }
    
    @Test
    @DisplayName("Termination: Builder configuration")
    void testTerminationBuilder() {
        Termination term = Termination.builder()
            .maxIterations(500)
            .maxEvaluations(5000)
            .accuracy(1e-8)
            .gradientTolerance(1e-6)
            .build();
        
        assertThat(term.getMaxIterations()).isEqualTo(500);
        assertThat(term.getMaxEvaluations()).isEqualTo(5000);
        assertThat(term.getAccuracy()).isEqualTo(1e-8);
        assertThat(term.getGradientTolerance()).isEqualTo(1e-6);
    }
    
    @Test
    @DisplayName("SLSQP: Simple minimize")
    void testSlsqpSimpleMinimize() {
        ObjectiveFunction objective = (x, g) -> {
            double f = x[0] * x[0] + x[1] * x[1];
            if (g != null) {
                g[0] = 2 * x[0];
                g[1] = 2 * x[1];
            }
            return f;
        };
        
        OptimizationResult result = SlsqpOptimizer.minimize(objective, new double[]{1.0, 1.0});
        
        assertThat(result.isConverged()).isTrue();
        assertThat(result.getFunctionValue()).isCloseTo(0.0, within(1e-6));
    }
    
    @Test
    @DisplayName("SLSQP: Builder with multiple inequality constraints using varargs")
    void testSlsqpWithMultipleConstraints() {
        // Minimize x^2 + y^2 subject to x >= 1 and y >= 1
        ObjectiveFunction objective = (x, g) -> {
            double f = x[0] * x[0] + x[1] * x[1];
            if (g != null) {
                g[0] = 2 * x[0];
                g[1] = 2 * x[1];
            }
            return f;
        };
        
        // x - 1 >= 0 (i.e., x >= 1)
        ConstraintFunction c1 = (x, g) -> {
            if (g != null) { g[0] = 1; g[1] = 0; }
            return x[0] - 1;
        };
        
        // y - 1 >= 0 (i.e., y >= 1)
        ConstraintFunction c2 = (x, g) -> {
            if (g != null) { g[0] = 0; g[1] = 1; }
            return x[1] - 1;
        };
        
        // Using varargs method in builder
        SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
            .dimension(2)
            .objective(objective)
            .inequalityConstraints(c1, c2)  // varargs!
            .build();
        
        double[] x = {0.0, 0.0};
        OptimizationResult result = optimizer.optimize(x);
        
        assertThat(result.isConverged()).isTrue();
        assertThat(x[0]).isCloseTo(1.0, within(1e-6));
        assertThat(x[1]).isCloseTo(1.0, within(1e-6));
    }

    @Test
    @DisplayName("L-BFGS-B: maxComputations time limit")
    void testLbfgsbMaxComputations() {
        // Simple quadratic function with sleep to simulate computation time
        ObjectiveFunction slowObjective = (x, g) -> {
            try {
                Thread.sleep(1);  // Sleep 1ms per evaluation
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            double f = x[0] * x[0] + x[1] * x[1];
            if (g != null) {
                g[0] = 2 * x[0];
                g[1] = 2 * x[1];
            }
            return f;
        };
        
        // Set a 5000 microseconds (5ms) time limit - should terminate after a few evaluations
        LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
            .dimension(2)
            .objective(slowObjective)
            .termination(Termination.builder()
                .maxIterations(1000)
                .maxComputations(5000)  // 5000 microseconds = 5ms
                .build())
            .build();
        
        double[] x = {10.0, 10.0};  // Start far from optimum
        OptimizationResult result = optimizer.optimize(x);
        
        // Should terminate due to time limit
        assertThat(result.getStatus()).isEqualTo(OptimizationStatus.MAX_COMPUTATIONS_REACHED);
    }
    
    @Test
    @DisplayName("Termination: maxComputations and nnlsIterations configuration")
    void testTerminationNewFields() {
        Termination term = Termination.builder()
            .maxComputations(5000000)  // 5 seconds in microseconds
            .nnlsIterations(100)
            .build();
        
        assertThat(term.getMaxComputations()).isEqualTo(5000000);
        assertThat(term.getNnlsIterations()).isEqualTo(100);
    }
    
    @Test
    @DisplayName("SLSQP: maxComputations time limit")
    void testSlsqpMaxComputations() {
        // Simple quadratic function with sleep to simulate computation time
        ObjectiveFunction slowObjective = (x, g) -> {
            try {
                Thread.sleep(2);  // Sleep 2ms per evaluation
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            double f = x[0] * x[0] + x[1] * x[1];
            if (g != null) {
                g[0] = 2 * x[0];
                g[1] = 2 * x[1];
            }
            return f;
        };
        
        // Set a 3000 microseconds (3ms) time limit - should trigger after 1-2 evaluations
        SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
            .dimension(2)
            .objective(slowObjective)
            .termination(Termination.builder()
                .maxIterations(1000)
                .maxComputations(3000)  // 3000 microseconds = 3ms
                .build())
            .build();
        
        double[] x = {100.0, 100.0};  // Start very far from optimum
        OptimizationResult result = optimizer.optimize(x);
        
        // Should terminate due to time limit
        assertThat(result.getStatus()).isEqualTo(OptimizationStatus.MAX_COMPUTATIONS_REACHED);
    }
}