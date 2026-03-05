/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

import net.jqwik.api.ForAll;
import net.jqwik.api.Label;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.IntRange;
import net.jqwik.api.constraints.LongRange;

import java.util.function.BiConsumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Property-based tests for Jacobian method convergence.
 * 
 * <p>Tests that forward and central difference methods produce similar Jacobian values
 * for smooth differentiable functions, and that central difference is more accurate
 * than forward difference when compared to analytical Jacobians.</p>
 * 
 */
public class JacobianMethodProperties {

    private static final double SIMILARITY_TOLERANCE = 1e-3;  // Tolerance for method similarity
    private static final double ACCURACY_TOLERANCE = 1e-6;  // Tolerance for accuracy comparison

    // ========================================================================
    // Property 4: Jacobian Method Convergence - Similarity Test
    // ========================================================================

    /**
     * Property 4: Jacobian Method Convergence - Forward and Central Difference Similarity
     * 
     * <p><i>For any</i> smooth differentiable function (polynomial), forward and central 
     * difference SHALL produce similar Jacobian values.</p>
     * 
     * <p>This test generates random polynomial models of the form:
     * ŷ = a₀ + a₁t + a₂t² + ... + aₙtⁿ
     * and verifies that both numerical differentiation methods produce similar results.</p>
     * 
     * 
     * @param numCoeffs number of polynomial coefficients (2 to 5)
     * @param numPoints number of data points (5 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Feature: levenberg-marquardt-optimizer, Property 4: Jacobian Method Convergence - Polynomial Similarity")
    void forwardAndCentralDifferenceProduceSimilarJacobiansForPolynomials(
            @ForAll @IntRange(min = 2, max = 5) int numCoeffs,
            @ForAll @IntRange(min = 5, max = 20) int numPoints,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random polynomial coefficients in reasonable range [-5, 5]
        double[] coeffs = new double[numCoeffs];
        for (int i = 0; i < numCoeffs; i++) {
            coeffs[i] = (random.nextDouble() - 0.5) * 10.0;
        }
        
        // Generate random data points t in range [0, 2]
        double[] tData = new double[numPoints];
        double[] yData = new double[numPoints];
        for (int i = 0; i < numPoints; i++) {
            tData[i] = random.nextDouble() * 2.0;
            // Generate y values with some noise
            yData[i] = evaluatePolynomial(tData[i], coeffs) + random.nextGaussian() * 0.1;
        }
        
        // Create polynomial residual function
        BiConsumer<double[], double[]> polyFunc = createPolynomialResidualFunction(tData, yData);
        
        // Compute Jacobians using both methods
        int m = numPoints;
        int n = numCoeffs;
        double[] residuals = new double[m];
        double[] jacobianForward = new double[m * n];
        double[] jacobianCentral = new double[m * n];
        
        Multivariate forwardEval = NumericalJacobian.FORWARD.wrap(polyFunc, m, n);
        Multivariate centralEval = NumericalJacobian.CENTRAL.wrap(polyFunc, m, n);
        forwardEval.evaluate(coeffs, residuals, jacobianForward);
        centralEval.evaluate(coeffs, residuals, jacobianCentral);
        
        // Verify both methods produce similar results
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                double forward = jacobianForward[idx];
                double central = jacobianCentral[idx];
                
                // Use relative tolerance for larger values, absolute for smaller
                double tolerance = Math.max(SIMILARITY_TOLERANCE, 
                                           SIMILARITY_TOLERANCE * Math.max(Math.abs(forward), Math.abs(central)));
                
                assertThat(forward)
                        .as("Forward and central difference should be similar at J[%d,%d]", i, j)
                        .isCloseTo(central, within(tolerance));
            }
        }
    }

    /**
     * Property 4: Jacobian Method Convergence - Exponential Model Similarity
     * 
     * <p><i>For any</i> smooth differentiable function (exponential), forward and central 
     * difference SHALL produce similar Jacobian values.</p>
     * 
     * <p>This test generates random exponential models of the form:
     * ŷ = a₀ * exp(-a₁ * t)
     * and verifies that both numerical differentiation methods produce similar results.</p>
     * 
     * 
     * @param numPoints number of data points (5 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Feature: levenberg-marquardt-optimizer, Property 4: Jacobian Method Convergence - Exponential Similarity")
    void forwardAndCentralDifferenceProduceSimilarJacobiansForExponentials(
            @ForAll @IntRange(min = 5, max = 20) int numPoints,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random exponential coefficients
        // a0 in [0.5, 5], a1 in [0.1, 2] to ensure reasonable decay
        double[] coeffs = new double[2];
        coeffs[0] = 0.5 + random.nextDouble() * 4.5;  // amplitude
        coeffs[1] = 0.1 + random.nextDouble() * 1.9;  // decay rate
        
        // Generate random data points t in range [0, 5]
        double[] tData = new double[numPoints];
        double[] yData = new double[numPoints];
        for (int i = 0; i < numPoints; i++) {
            tData[i] = random.nextDouble() * 5.0;
            // Generate y values with some noise
            yData[i] = coeffs[0] * Math.exp(-coeffs[1] * tData[i]) + random.nextGaussian() * 0.01;
        }
        
        // Create exponential residual function
        BiConsumer<double[], double[]> expFunc = createExponentialResidualFunction(tData, yData);
        
        // Compute Jacobians using both methods
        int m = numPoints;
        int n = 2;
        double[] residuals = new double[m];
        double[] jacobianForward = new double[m * n];
        double[] jacobianCentral = new double[m * n];
        
        Multivariate forwardEval = NumericalJacobian.FORWARD.wrap(expFunc, m, n);
        Multivariate centralEval = NumericalJacobian.CENTRAL.wrap(expFunc, m, n);
        forwardEval.evaluate(coeffs, residuals, jacobianForward);
        centralEval.evaluate(coeffs, residuals, jacobianCentral);
        
        // Verify both methods produce similar results
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                double forward = jacobianForward[idx];
                double central = jacobianCentral[idx];
                
                // Use relative tolerance for larger values, absolute for smaller
                double tolerance = Math.max(SIMILARITY_TOLERANCE, 
                                           SIMILARITY_TOLERANCE * Math.max(Math.abs(forward), Math.abs(central)));
                
                assertThat(forward)
                        .as("Forward and central difference should be similar at J[%d,%d]", i, j)
                        .isCloseTo(central, within(tolerance));
            }
        }
    }

    // ========================================================================
    // Property 4: Jacobian Method Convergence - Accuracy Test
    // ========================================================================

    /**
     * Property 4: Jacobian Method Convergence - Central Difference More Accurate (Polynomial)
     * 
     * <p><i>For any</i> smooth function (polynomial), central difference SHALL be more accurate 
     * than forward difference when compared to the analytical Jacobian.</p>
     * 
     * <p>For polynomial models, the analytical Jacobian is known exactly:
     * J_ij = ∂ŷ_i/∂a_j = t_i^j</p>
     * 
     * 
     * @param numCoeffs number of polynomial coefficients (2 to 4)
     * @param numPoints number of data points (5 to 15)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Feature: levenberg-marquardt-optimizer, Property 4: Jacobian Method Convergence - Polynomial Accuracy")
    void centralDifferenceIsMoreAccurateThanForwardForPolynomials(
            @ForAll @IntRange(min = 2, max = 4) int numCoeffs,
            @ForAll @IntRange(min = 5, max = 15) int numPoints,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random polynomial coefficients in reasonable range [-3, 3]
        double[] coeffs = new double[numCoeffs];
        for (int i = 0; i < numCoeffs; i++) {
            coeffs[i] = (random.nextDouble() - 0.5) * 6.0;
        }
        
        // Generate random data points t in range [0.1, 2] (avoid t=0 for numerical stability)
        double[] tData = new double[numPoints];
        double[] yData = new double[numPoints];
        for (int i = 0; i < numPoints; i++) {
            tData[i] = 0.1 + random.nextDouble() * 1.9;
            yData[i] = evaluatePolynomial(tData[i], coeffs);
        }
        
        // Create polynomial residual function
        BiConsumer<double[], double[]> polyFunc = createPolynomialResidualFunction(tData, yData);
        
        // Compute analytical Jacobian
        int m = numPoints;
        int n = numCoeffs;
        double[] analyticalJacobian = computePolynomialAnalyticalJacobian(tData, m, n);
        
        // Compute numerical Jacobians
        double[] residuals = new double[m];
        double[] jacobianForward = new double[m * n];
        double[] jacobianCentral = new double[m * n];
        
        Multivariate forwardEval = NumericalJacobian.FORWARD.wrap(polyFunc, m, n);
        Multivariate centralEval = NumericalJacobian.CENTRAL.wrap(polyFunc, m, n);
        forwardEval.evaluate(coeffs, residuals, jacobianForward);
        centralEval.evaluate(coeffs, residuals, jacobianCentral);
        
        // Compute errors
        double errorForward = computeMaxAbsoluteError(jacobianForward, analyticalJacobian);
        double errorCentral = computeMaxAbsoluteError(jacobianCentral, analyticalJacobian);
        
        // Central difference should be more accurate (or at least as accurate)
        // Allow small tolerance for numerical edge cases
        assertThat(errorCentral)
                .as("Central difference error (%e) should be <= forward difference error (%e)", 
                    errorCentral, errorForward)
                .isLessThanOrEqualTo(errorForward + ACCURACY_TOLERANCE);
    }

    /**
     * Property 4: Jacobian Method Convergence - Central Difference More Accurate (Exponential)
     * 
     * <p><i>For any</i> smooth function (exponential), central difference SHALL be more accurate 
     * than forward difference when compared to the analytical Jacobian.</p>
     * 
     * <p>For exponential model ŷ = a₀ * exp(-a₁ * t), the analytical Jacobian is:
     * J_i0 = exp(-a₁ * t_i)
     * J_i1 = -a₀ * t_i * exp(-a₁ * t_i)</p>
     * 
     * 
     * @param numPoints number of data points (5 to 15)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Feature: levenberg-marquardt-optimizer, Property 4: Jacobian Method Convergence - Exponential Accuracy")
    void centralDifferenceIsMoreAccurateThanForwardForExponentials(
            @ForAll @IntRange(min = 5, max = 15) int numPoints,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random exponential coefficients
        // a0 in [1, 3], a1 in [0.2, 1] to ensure reasonable values
        double[] coeffs = new double[2];
        coeffs[0] = 1.0 + random.nextDouble() * 2.0;  // amplitude
        coeffs[1] = 0.2 + random.nextDouble() * 0.8;  // decay rate
        
        // Generate random data points t in range [0.1, 3]
        double[] tData = new double[numPoints];
        double[] yData = new double[numPoints];
        for (int i = 0; i < numPoints; i++) {
            tData[i] = 0.1 + random.nextDouble() * 2.9;
            yData[i] = coeffs[0] * Math.exp(-coeffs[1] * tData[i]);
        }
        
        // Create exponential residual function
        BiConsumer<double[], double[]> expFunc = createExponentialResidualFunction(tData, yData);
        
        // Compute analytical Jacobian
        int m = numPoints;
        int n = 2;
        double[] analyticalJacobian = computeExponentialAnalyticalJacobian(tData, coeffs, m);
        
        // Compute numerical Jacobians
        double[] residuals = new double[m];
        double[] jacobianForward = new double[m * n];
        double[] jacobianCentral = new double[m * n];
        
        Multivariate forwardEval = NumericalJacobian.FORWARD.wrap(expFunc, m, n);
        Multivariate centralEval = NumericalJacobian.CENTRAL.wrap(expFunc, m, n);
        forwardEval.evaluate(coeffs, residuals, jacobianForward);
        centralEval.evaluate(coeffs, residuals, jacobianCentral);
        
        // Compute errors
        double errorForward = computeMaxAbsoluteError(jacobianForward, analyticalJacobian);
        double errorCentral = computeMaxAbsoluteError(jacobianCentral, analyticalJacobian);
        
        // Central difference should be more accurate (or at least as accurate)
        // Allow small tolerance for numerical edge cases
        assertThat(errorCentral)
                .as("Central difference error (%e) should be <= forward difference error (%e)", 
                    errorCentral, errorForward)
                .isLessThanOrEqualTo(errorForward + ACCURACY_TOLERANCE);
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Evaluates a polynomial at point t.
     * ŷ = a₀ + a₁t + a₂t² + ... + aₙtⁿ
     */
    private double evaluatePolynomial(double t, double[] coeffs) {
        double result = 0.0;
        double tPower = 1.0;
        for (double coeff : coeffs) {
            result += coeff * tPower;
            tPower *= t;
        }
        return result;
    }

    /**
     * Creates a residual function for polynomial model.
     * ŷ_i = a₀ + a₁t_i + a₂t_i² + ...
     */
    private BiConsumer<double[], double[]> createPolynomialResidualFunction(double[] tData, double[] yData) {
        return (coefficients, residuals) -> {
            for (int i = 0; i < tData.length; i++) {
                double yPred = evaluatePolynomial(tData[i], coefficients);
                residuals[i] = yData[i] - yPred;
            }
        };
    }

    /**
     * Creates a residual function for exponential model.
     * ŷ_i = a₀ * exp(-a₁ * t_i)
     */
    private BiConsumer<double[], double[]> createExponentialResidualFunction(double[] tData, double[] yData) {
        return (coefficients, residuals) -> {
            for (int i = 0; i < tData.length; i++) {
                double yPred = coefficients[0] * Math.exp(-coefficients[1] * tData[i]);
                residuals[i] = yData[i] - yPred;
            }
        };
    }

    /**
     * Computes the analytical Jacobian for polynomial model.
     * J_ij = ∂ŷ_i/∂a_j = t_i^j
     */
    private double[] computePolynomialAnalyticalJacobian(double[] tData, int m, int n) {
        double[] jacobian = new double[m * n];
        for (int i = 0; i < m; i++) {
            double tPower = 1.0;
            for (int j = 0; j < n; j++) {
                jacobian[i * n + j] = tPower;
                tPower *= tData[i];
            }
        }
        return jacobian;
    }

    /**
     * Computes the analytical Jacobian for exponential model.
     * J_i0 = exp(-a₁ * t_i)
     * J_i1 = -a₀ * t_i * exp(-a₁ * t_i)
     */
    private double[] computeExponentialAnalyticalJacobian(double[] tData, double[] coeffs, int m) {
        int n = 2;
        double[] jacobian = new double[m * n];
        for (int i = 0; i < m; i++) {
            double expTerm = Math.exp(-coeffs[1] * tData[i]);
            jacobian[i * n + 0] = expTerm;
            jacobian[i * n + 1] = -coeffs[0] * tData[i] * expTerm;
        }
        return jacobian;
    }

    /**
     * Computes the maximum absolute error between two arrays.
     */
    private double computeMaxAbsoluteError(double[] computed, double[] expected) {
        double maxError = 0.0;
        for (int i = 0; i < computed.length; i++) {
            double error = Math.abs(computed[i] - expected[i]);
            maxError = Math.max(maxError, error);
        }
        return maxError;
    }
}
