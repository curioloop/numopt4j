/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.numopt4j.optim.slsqp.SLSQPProblem;
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
        // Minimize x^2 + y^2, minimum at (0, 0) using NumericalGradient.wrap()
        OptimizationResult result = LBFGSBProblem.create()
            .objective(NumericalGradient.CENTRAL.wrap(x -> x[0] * x[0] + x[1] * x[1]))
            .initialPoint(1.0, 1.0)
            .solve();
        
        assertThat(result.isConverged()).isTrue();
        assertThat(result.getObjectiveValue()).isCloseTo(0.0, within(1e-6));
    }
    
    @Test
    @DisplayName("L-BFGS-B: Minimize with analytical gradient")
    void testLbfgsbMinimizeWithGradient() {
        // Quadratic function with analytical gradient: f(x) = (x-2)^2 + (y-3)^2
        // Minimum at (2, 3)
        Univariate quadratic = TestTemplates.quadraticWithTarget(new double[]{2, 3});
        
        OptimizationResult result = LBFGSBProblem.create()
            .objective(quadratic)
            .initialPoint(0.0, 0.0)
            .solve();
        double[] x = result.getSolution();
        
        assertThat(result.isConverged()).isTrue();
        assertThat(x[0]).isCloseTo(2.0, within(1e-6));
        assertThat(x[1]).isCloseTo(3.0, within(1e-6));
    }
    
    @Test
    @DisplayName("L-BFGS-B: Builder with function-only objective")
    void testLbfgsbBuilderWithFunctionOnly() {
        // Using LBFGSBProblem with NumericalGradient.wrap()
        OptimizationResult result = LBFGSBProblem.create()
            .objective(NumericalGradient.CENTRAL.wrap(x -> x[0] * x[0] + x[1] * x[1]))
            .initialPoint(1.0, 1.0)
            .maxIterations(50)
            .gradientTolerance(1e-4)
            .solve();
        
        assertThat(result.isConverged()).isTrue();
        assertThat(result.getObjectiveValue()).isCloseTo(0.0, within(1e-3));
    }
    
    @Test
    @DisplayName("L-BFGS-B: Builder with bounds")
    void testLbfgsbBuilderWithBounds() {
        // Minimize (x-5)^2 + (y-5)^2 with bounds [0, 3]
        OptimizationResult result = LBFGSBProblem.create()
            .objective(NumericalGradient.CENTRAL.wrap(x -> Math.pow(x[0] - 5, 2) + Math.pow(x[1] - 5, 2)))
            .initialPoint(1.0, 1.0)
            .bounds(Bound.between(0.0, 3.0), Bound.between(0.0, 3.0))
            .solve();
        double[] x = result.getSolution();
        
        assertThat(result.isConverged()).isTrue();
        assertThat(x[0]).isCloseTo(3.0, within(1e-6));
        assertThat(x[1]).isCloseTo(3.0, within(1e-6));
    }
    
    @Test
    @DisplayName("L-BFGS-B: Builder with single bound for all variables")
    void testLbfgsbBuilderWithSingleBound() {
        OptimizationResult result = LBFGSBProblem.create()
            .objective(NumericalGradient.CENTRAL.wrap(x -> x[0] * x[0] + x[1] * x[1] + x[2] * x[2]))
            .initialPoint(1.0, 1.0, 1.0)
            .bounds(Bound.atLeast(0), Bound.atLeast(0), Bound.atLeast(0))  // x >= 0 for all variables
            .solve();
        double[] x = result.getSolution();
        
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
    @DisplayName("SLSQP: Simple minimize")
    void testSlsqpSimpleMinimize() {
        Univariate objective = TestTemplates.quadratic();
        
        OptimizationResult result = SLSQPProblem.create()
            .objective(objective)
            .initialPoint(1.0, 1.0)
            .solve();
        
        assertThat(result.isConverged()).isTrue();
        assertThat(result.getObjectiveValue()).isCloseTo(0.0, within(1e-6));
    }
    
    @Test
    @DisplayName("SLSQP: Builder with multiple inequality constraints using varargs")
    void testSlsqpWithMultipleConstraints() {
        // Minimize x^2 + y^2 subject to x >= 1 and y >= 1
        Univariate objective = TestTemplates.quadratic();
        
        // x - 1 >= 0 (i.e., x >= 1)
        Univariate c1 = TestTemplates.inequalityAtIndex(0, -1.0);

        // y - 1 >= 0 (i.e., y >= 1)
        Univariate c2 = TestTemplates.inequalityAtIndex(1, -1.0);
        
        // Using SLSQPProblem with varargs constraints
        OptimizationResult result = SLSQPProblem.create()
            .objective(objective)
            .initialPoint(0.0, 0.0)
            .inequalityConstraints(c1, c2)
            .solve();
        double[] x = result.getSolution();
        
        assertThat(result.isConverged()).isTrue();
        assertThat(x[0]).isCloseTo(1.0, within(1e-6));
        assertThat(x[1]).isCloseTo(1.0, within(1e-6));
    }

    @Test
    @DisplayName("L-BFGS-B: maxComputations time limit")
    void testLbfgsbMaxComputations() {
        Univariate slowObjective = TestTemplates.slowQuadratic(1);

        OptimizationResult result = LBFGSBProblem.create()
            .objective(slowObjective)
            .initialPoint(10.0, 10.0)
            .maxIterations(1000)
            .maxEvaluations(5000)
            .solve();
        
        assertThat(result.getStatus().isConverged() || 
                   result.getStatus() == OptimizationStatus.MAX_COMPUTATIONS_REACHED).isTrue();
    }
    

    
    @Test
    @DisplayName("SLSQP: maxComputations time limit")
    void testSlsqpMaxComputations() {
        // Simple quadratic function with sleep to simulate computation time
        Univariate slowObjective = TestTemplates.slowQuadratic(2);
        
        // Set a 3ms time limit - should trigger after 1-2 evaluations
        OptimizationResult result = SLSQPProblem.create()
            .objective(slowObjective)
            .initialPoint(100.0, 100.0)
            .maxIterations(1000)
            .maxComputations(java.time.Duration.ofMillis(3))
            .solve();
        
        // Should terminate due to time limit
        assertThat(result.getStatus()).isEqualTo(OptimizationStatus.MAX_COMPUTATIONS_REACHED);
    }
}
