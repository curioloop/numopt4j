/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.function.ToDoubleFunction;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for NumericalGradient methods.
 */
public class NumericalGradientTest {

    // Test function: f(x) = x[0]^2 + 2*x[1]^2 + 3*x[2]^2
    // Analytical gradient: g = [2*x[0], 4*x[1], 6*x[2]]
    private static final ToDoubleFunction<double[]> QUADRATIC = x -> 
        x[0] * x[0] + 2 * x[1] * x[1] + 3 * x[2] * x[2];
    
    private static double[] analyticalGradient(double[] x) {
        return new double[]{2 * x[0], 4 * x[1], 6 * x[2]};
    }

    @ParameterizedTest
    @EnumSource(NumericalGradient.class)
    @DisplayName("All gradient methods should approximate analytical gradient")
    void testAllMethodsApproximateGradient(NumericalGradient method) {
        double[] x = {1.0, 2.0, 3.0};
        double[] g = new double[3];
        
        Evaluation wrapped = method.wrap(QUADRATIC);
        double f = wrapped.evaluate(x, g);
        
        double[] expected = analyticalGradient(x);
        
        // Function value should be correct
        assertThat(f).isCloseTo(1 + 8 + 27, within(1e-10));
        
        // Gradient should be close to analytical (tolerance depends on method)
        double tol = (method == NumericalGradient.FIVE_POINT) ? 1e-10 : 1e-6;
        for (int i = 0; i < 3; i++) {
            assertThat(g[i]).isCloseTo(expected[i], within(tol));
        }
    }

    @Test
    @DisplayName("FORWARD: gradient approximation")
    void testForwardDifference() {
        double[] x = {1.0, -2.0, 0.5};
        double[] g = new double[3];
        
        Evaluation wrapped = NumericalGradient.FORWARD.wrap(QUADRATIC);
        wrapped.evaluate(x, g);
        
        double[] expected = analyticalGradient(x);
        for (int i = 0; i < 3; i++) {
            assertThat(g[i]).isCloseTo(expected[i], within(1e-5));
        }
    }

    @Test
    @DisplayName("BACKWARD: gradient approximation")
    void testBackwardDifference() {
        double[] x = {1.0, -2.0, 0.5};
        double[] g = new double[3];
        
        Evaluation wrapped = NumericalGradient.BACKWARD.wrap(QUADRATIC);
        wrapped.evaluate(x, g);
        
        double[] expected = analyticalGradient(x);
        for (int i = 0; i < 3; i++) {
            assertThat(g[i]).isCloseTo(expected[i], within(1e-5));
        }
    }

    @Test
    @DisplayName("CENTRAL: gradient approximation (higher accuracy)")
    void testCentralDifference() {
        double[] x = {1.0, -2.0, 0.5};
        double[] g = new double[3];
        
        Evaluation wrapped = NumericalGradient.CENTRAL.wrap(QUADRATIC);
        wrapped.evaluate(x, g);
        
        double[] expected = analyticalGradient(x);
        for (int i = 0; i < 3; i++) {
            assertThat(g[i]).isCloseTo(expected[i], within(1e-9));
        }
    }

    @Test
    @DisplayName("FIVE_POINT: gradient approximation (highest accuracy)")
    void testFivePointDifference() {
        double[] x = {1.0, -2.0, 0.5};
        double[] g = new double[3];
        
        Evaluation wrapped = NumericalGradient.FIVE_POINT.wrap(QUADRATIC);
        wrapped.evaluate(x, g);
        
        double[] expected = analyticalGradient(x);
        for (int i = 0; i < 3; i++) {
            assertThat(g[i]).isCloseTo(expected[i], within(1e-11));
        }
    }

    @Test
    @DisplayName("FIVE_POINT should be more accurate than CENTRAL")
    void testFivePointMoreAccurateThanCentral() {
        // Use a more complex function: f(x) = sin(x[0]) + cos(x[1])
        // Analytical gradient: g = [cos(x[0]), -sin(x[1])]
        ToDoubleFunction<double[]> sinCos = x -> Math.sin(x[0]) + Math.cos(x[1]);
        
        double[] x = {1.0, 2.0};
        double[] gCentral = new double[2];
        double[] gFivePoint = new double[2];
        
        NumericalGradient.CENTRAL.wrap(sinCos).evaluate(x, gCentral);
        NumericalGradient.FIVE_POINT.wrap(sinCos).evaluate(x, gFivePoint);
        
        double[] expected = {Math.cos(x[0]), -Math.sin(x[1])};
        
        double errorCentral = Math.abs(gCentral[0] - expected[0]) + Math.abs(gCentral[1] - expected[1]);
        double errorFivePoint = Math.abs(gFivePoint[0] - expected[0]) + Math.abs(gFivePoint[1] - expected[1]);
        
        assertThat(errorFivePoint).isLessThan(errorCentral);
    }

    @Test
    @DisplayName("Accuracy comparison: FORWARD < CENTRAL < FIVE_POINT")
    void testAccuracyOrdering() {
        // f(x) = exp(x[0]) + x[1]^3, gradient = [exp(x[0]), 3*x[1]^2]
        ToDoubleFunction<double[]> func = x -> Math.exp(x[0]) + Math.pow(x[1], 3);
        
        double[] x = {0.5, 1.5};
        double[] expected = {Math.exp(x[0]), 3 * x[1] * x[1]};
        
        double[] gForward = new double[2];
        double[] gBackward = new double[2];
        double[] gCentral = new double[2];
        double[] gFivePoint = new double[2];
        
        NumericalGradient.FORWARD.wrap(func).evaluate(x, gForward);
        NumericalGradient.BACKWARD.wrap(func).evaluate(x, gBackward);
        NumericalGradient.CENTRAL.wrap(func).evaluate(x, gCentral);
        NumericalGradient.FIVE_POINT.wrap(func).evaluate(x, gFivePoint);
        
        double errorForward = maxError(gForward, expected);
        double errorBackward = maxError(gBackward, expected);
        double errorCentral = maxError(gCentral, expected);
        double errorFivePoint = maxError(gFivePoint, expected);
        
        // Print errors for visibility
        System.out.printf("Gradient errors: FORWARD=%.2e, BACKWARD=%.2e, CENTRAL=%.2e, FIVE_POINT=%.2e%n",
            errorForward, errorBackward, errorCentral, errorFivePoint);
        
        // Verify ordering: FORWARD/BACKWARD ~ O(h), CENTRAL ~ O(h²), FIVE_POINT ~ O(h⁴)
        // FORWARD/BACKWARD should be around 1e-7 to 1e-8
        assertThat(errorForward).isBetween(1e-9, 1e-5);
        assertThat(errorBackward).isBetween(1e-9, 1e-5);
        
        // CENTRAL should be much better, around 1e-10 to 1e-11
        assertThat(errorCentral).isLessThan(errorForward * 1e-2);
        assertThat(errorCentral).isLessThan(errorBackward * 1e-2);
        
        // FIVE_POINT should be best, around 1e-12 to 1e-14
        assertThat(errorFivePoint).isLessThan(errorCentral * 1e-1);
    }
    
    private static double maxError(double[] computed, double[] expected) {
        double maxErr = 0;
        for (int i = 0; i < computed.length; i++) {
            maxErr = Math.max(maxErr, Math.abs(computed[i] - expected[i]));
        }
        return maxErr;
    }

    @Test
    @DisplayName("Gradient should be null-safe (no computation when g is null)")
    void testNullGradient() {
        double[] x = {1.0, 2.0, 3.0};
        
        for (NumericalGradient method : NumericalGradient.values()) {
            Evaluation wrapped = method.wrap(QUADRATIC);
            double f = wrapped.evaluate(x, null);
            
            // Function value should still be computed
            assertThat(f).isCloseTo(1 + 8 + 27, within(1e-10));
        }
    }

    @Test
    @DisplayName("All methods should work with L-BFGS-B optimizer")
    void testWithLbfgsbOptimizer() {
        for (NumericalGradient method : NumericalGradient.values()) {
            double[] x = {5.0, 5.0};
            OptimizationResult result = LbfgsbOptimizer.builder()
                .dimension(2)
                .objective(method.wrap(p -> p[0] * p[0] + p[1] * p[1]))
                .build()
                .optimize(x);
            
            assertThat(result.isConverged())
                .as("Method %s should converge", method)
                .isTrue();
            assertThat(x[0]).isCloseTo(0.0, within(1e-4));
            assertThat(x[1]).isCloseTo(0.0, within(1e-4));
        }
    }

    @Test
    @DisplayName("All methods should work with SLSQP optimizer")
    void testWithSlsqpOptimizer() {
        for (NumericalGradient method : NumericalGradient.values()) {
            double[] x = {5.0, 5.0};
            OptimizationResult result = SlsqpOptimizer.builder()
                .dimension(2)
                .objective(method.wrap(p -> p[0] * p[0] + p[1] * p[1]))
                .build()
                .optimize(x);
            
            assertThat(result.isConverged())
                .as("Method %s should converge", method)
                .isTrue();
            assertThat(x[0]).isCloseTo(0.0, within(1e-4));
            assertThat(x[1]).isCloseTo(0.0, within(1e-4));
        }
    }

    @Test
    @DisplayName("Rosenbrock function optimization with FIVE_POINT")
    void testRosenbrockWithFivePoint() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // Minimum at (1, 1)
        ToDoubleFunction<double[]> rosenbrock = x -> 
            Math.pow(1 - x[0], 2) + 100 * Math.pow(x[1] - x[0] * x[0], 2);
        
        double[] x = {0.5, 0.5};  // Start closer to optimum
        OptimizationResult result = LbfgsbOptimizer.builder()
            .dimension(2)
            .objective(NumericalGradient.FIVE_POINT.wrap(rosenbrock))
            .termination(Termination.builder()
                .maxIterations(2000)
                .accuracy(1e-12)
                .gradientTolerance(1e-8)
                .build())
            .build()
            .optimize(x);
        
        // Check solution is close to (1, 1)
        assertThat(x[0]).isCloseTo(1.0, within(1e-3));
        assertThat(x[1]).isCloseTo(1.0, within(1e-3));
    }
}
