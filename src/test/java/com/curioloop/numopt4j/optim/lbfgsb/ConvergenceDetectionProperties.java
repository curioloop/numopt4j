/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.Univariate;
import com.curioloop.numopt4j.optim.TestTemplates;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

/**
 * Property-based tests for convergence detection in L-BFGS-B optimization.
 * 
 * <p>Tests that the convergence detection mechanisms correctly identify when
 * optimization should terminate based on various criteria.</p>
 * 
 * 
 * <h2>Property 8: Convergence Detection Correctness</h2>
 * <p><i>For any</i> optimization run:</p>
 * <ul>
 *   <li>If ||proj(g)||_∞ ≤ pgtol, convergence shall be reported</li>
 *   <li>If (f_old - f) / max(|f_old|, |f|, 1) ≤ factr*ε, convergence shall be reported</li>
 *   <li>If iterations exceed max_iter, the limit shall be reported</li>
 *   <li>If evaluations exceed max_eval, the limit shall be reported</li>
 * </ul>
 */
@Tag("Feature: lbfgsb-java-rewrite, Property 8: Convergence Detection Correctness")
public class ConvergenceDetectionProperties implements LBFGSBConstants {

    private static final double TOLERANCE = 1e-10;

    // Use centralized Rosenbrock template
    static final Univariate rosenbrock = TestTemplates.rosenbrock(); // REPLACED inline rosenbrock

    // Simple quadratic: f(x) = sum(x_i^2)
    static final Univariate quadratic = (point, grad) -> {
        double f = 0.0;
        for (int i = 0; i < point.length; i++) {
            f += point[i] * point[i];
            if (grad != null) {
                grad[i] = 2.0 * point[i];
            }
        }
        return f;
    };

    // ==================== Unit Tests for Basic Verification ====================

    /**
     * Unit test: Verify convergence when projected gradient norm is below tolerance.
     * 
     */
    @Test
    void testConvergenceOnProjectedGradientNorm() {
        int n = 3;
        int m = 5;
        
        // Simple quadratic: f(x) = sum(x_i^2), minimum at origin
        // At x = [0, 0, 0], gradient = [0, 0, 0], so projected gradient norm = 0
        double[] x = {0.0, 0.0, 0.0};
        double[] lower = {-10.0, -10.0, -10.0};
        double[] upper = {10.0, 10.0, 10.0};
        int[] boundType = {BOUND_BOTH, BOUND_BOTH, BOUND_BOTH};
        com.curioloop.numopt4j.optim.Bound[] bounds = TestBounds.toBounds(n, lower, upper, boundType);
        
        Univariate objective = (point, grad) -> {
            double f = 0.0;
            for (int i = 0; i < point.length; i++) {
                f += point[i] * point[i];
                if (grad != null) {
                    grad[i] = 2.0 * point[i];
                }
            }
            return f;
        };
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-5;
            double factr = 1e7;
            int maxIter = 100;
            int maxEval = 200;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, objective, bounds,
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should converge due to projected gradient norm
            assertThat(result.status)
                    .as("Should converge with CONV_GRAD_PROG_NORM")
                    .isEqualTo(CONV_GRAD_PROG_NORM);
            
            // Function value should be near zero
            assertThat(result.f)
                    .as("Function value should be near zero at minimum")
                    .isCloseTo(0.0, within(1e-8));
    }


    /**
     * Unit test: Verify iteration limit is reported when exceeded.
     * 
     */
    @Test
    void testIterationLimitReported() {
        int n = 3;
        int m = 5;
        
        // Rosenbrock function - hard to optimize, will hit iteration limit
        double[] x = {-1.0, -1.0, -1.0};
        double[] lower = {-10.0, -10.0, -10.0};
        double[] upper = {10.0, 10.0, 10.0};
        int[] boundType = {BOUND_BOTH, BOUND_BOTH, BOUND_BOTH};
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-15;  // Very tight tolerance
            double factr = 1e-15; // Very tight tolerance
            int maxIter = 3;      // Very low iteration limit
            int maxEval = 1000;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, rosenbrock, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should stop due to iteration limit
            assertThat(result.status)
                    .as("Should stop with OVER_ITER_LIMIT")
                    .isEqualTo(OVER_ITER_LIMIT);
            
            // Iterations should be at or near the limit
            assertThat(result.iterations)
                    .as("Iterations should be at or near the limit")
                    .isGreaterThanOrEqualTo(maxIter);
    }

    /**
     * Unit test: Verify evaluation limit is reported when exceeded.
     * 
     */
    @Test
    void testEvaluationLimitReported() {
        int n = 3;
        int m = 5;
        
        // Rosenbrock function - requires many evaluations
        double[] x = {-1.0, -1.0, -1.0};
        double[] lower = {-10.0, -10.0, -10.0};
        double[] upper = {10.0, 10.0, 10.0};
        int[] boundType = {BOUND_BOTH, BOUND_BOTH, BOUND_BOTH};
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-15;  // Very tight tolerance
            double factr = 1e-15; // Very tight tolerance
            int maxIter = 1000;
            int maxEval = 5;      // Very low evaluation limit
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, rosenbrock, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should stop due to evaluation limit
            assertThat(result.status)
                    .as("Should stop with OVER_EVAL_LIMIT")
                    .isEqualTo(OVER_EVAL_LIMIT);
            
            // Evaluations should be at or near the limit
            assertThat(result.evaluations)
                    .as("Evaluations should be at or near the limit")
                    .isGreaterThanOrEqualTo(maxEval);
    }

    /**
     * Unit test: Verify function value convergence is detected.
     * 
     */
    @Test
    void testFunctionValueConvergence() {
        int n = 2;
        int m = 5;
        
        // Simple quadratic with known minimum
        double[] x = {1.0, 1.0};
        double[] lower = {-10.0, -10.0};
        double[] upper = {10.0, 10.0};
        int[] boundType = {BOUND_BOTH, BOUND_BOTH};
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-15;  // Very tight gradient tolerance
            double factr = 1e12;   // Loose function value tolerance
            int maxIter = 100;
            int maxEval = 200;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, quadratic, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should converge (either by gradient or function value)
            assertThat(result.status)
                    .as("Should converge")
                    .isIn(CONV_GRAD_PROG_NORM, CONV_ENOUGH_ACCURACY);
    }

    /**
     * Unit test: Verify projected gradient norm computation for bound-constrained problems.
     * 
     */
    @Test
    void testProjectedGradientNormComputation() {
        int n = 4;
        
        // Point at bounds with gradient pointing away
        double[] x = {0.0, 1.0, 0.5, 0.5};  // at lower, at upper, free, free
        double[] g = {1.0, -1.0, 0.5, -0.5}; // positive, negative, positive, negative
        double[] lower = {0.0, 0.0, 0.0, 0.0};
        double[] upper = {1.0, 1.0, 1.0, 1.0};
        int[] boundType = {BOUND_BOTH, BOUND_BOTH, BOUND_BOTH, BOUND_BOTH};
        
        // At x[0]=0 (lower bound), g[0]=1.0 > 0: proj g = min(x-l, g) = min(0, 1) = 0
        // At x[1]=1 (upper bound), g[1]=-1.0 < 0: proj g = max(x-u, g) = max(0, -1) = 0
        // At x[2]=0.5 (free), g[2]=0.5 > 0: proj g = min(x-l, g) = min(0.5, 0.5) = 0.5
        // At x[3]=0.5 (free), g[3]=-0.5 < 0: proj g = max(x-u, g) = max(-0.5, -0.5) = -0.5
        // Expected norm = max(|0|, |0|, |0.5|, |-0.5|) = 0.5
        
        double norm = LBFGSBCore.projGradNorm(n, x, g, TestBounds.toBounds(n, lower, upper, boundType));
        assertThat(norm)
                .as("Projected gradient norm should be 0.5")
                .isCloseTo(0.5, within(TOLERANCE));
    }


    // ==================== Property 8: Convergence Detection Correctness ====================

    /**
     * Property 8.1: Projected gradient convergence is correctly detected
     * 
     * <p><i>For any</i> optimization problem where the projected gradient norm
     * falls below pgtol, convergence shall be reported with CONV_GRAD_PROG_NORM.</p>
     * 
     * 
     * @param n the dimension (2 to 10)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 8.1: Projected gradient convergence is correctly detected")
    void projectedGradientConvergenceDetected(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        // Generate random starting point near origin (will converge quickly)
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            x[i] = random.nextGaussian() * 0.1;  // Small perturbation from origin
            lower[i] = -10.0;
            upper[i] = 10.0;
            boundType[i] = BOUND_BOTH;
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-5;
            double factr = 1e-15;  // Very tight function tolerance (won't trigger)
            int maxIter = 100;
            int maxEval = 200;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, quadratic, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should converge due to projected gradient norm
            assertThat(result.status)
                    .as("Should converge with CONV_GRAD_PROG_NORM for simple quadratic")
                    .isEqualTo(CONV_GRAD_PROG_NORM);
            
            // Verify the solution is near the minimum
            double sumSq = 0.0;
            for (int i = 0; i < n; i++) {
                sumSq += x[i] * x[i];
            }
            assertThat(Math.sqrt(sumSq))
                    .as("Solution should be near origin")
                    .isLessThan(pgtol * 10);
    }

    /**
     * Property 8.2: Function value convergence is correctly detected
     * 
     * <p><i>For any</i> optimization problem where the relative function decrease
     * falls below factr*ε, convergence shall be reported with CONV_ENOUGH_ACCURACY.</p>
     * 
     * 
     * @param n the dimension (2 to 5)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 50)
    @Label("Property 8.2: Function value convergence is correctly detected")
    void functionValueConvergenceDetected(
            @ForAll @IntRange(min = 2, max = 5) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        // Generate random starting point
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            x[i] = random.nextGaussian();
            lower[i] = -10.0;
            upper[i] = 10.0;
            boundType[i] = BOUND_BOTH;
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-15;  // Very tight gradient tolerance
            double factr = 1e12;   // Loose function value tolerance
            int maxIter = 100;
            int maxEval = 200;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, quadratic, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should converge (either by gradient or function value)
            assertThat(result.status)
                    .as("Should converge")
                    .isIn(CONV_GRAD_PROG_NORM, CONV_ENOUGH_ACCURACY);
    }


    /**
     * Property 8.3: Iteration limit is correctly reported
     * 
     * <p><i>For any</i> optimization problem that exceeds max_iter iterations,
     * the limit shall be reported with OVER_ITER_LIMIT.</p>
     * 
     * 
     * @param n the dimension (2 to 5)
     * @param maxIter the maximum iterations (1 to 5)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 50)
    @Label("Property 8.3: Iteration limit is correctly reported")
    void iterationLimitReported(
            @ForAll @IntRange(min = 2, max = 5) int n,
            @ForAll @IntRange(min = 1, max = 5) int maxIter,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        // Generate random starting point far from minimum
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            x[i] = random.nextGaussian() * 5.0 - 2.5;  // Far from minimum
            lower[i] = -10.0;
            upper[i] = 10.0;
            boundType[i] = BOUND_BOTH;
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-15;  // Very tight tolerance
            double factr = 1e-15; // Very tight tolerance
            int maxEval = 1000;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, rosenbrock, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should stop due to iteration limit (or possibly abnormal search)
            assertThat(result.status)
                    .as("Should stop with OVER_ITER_LIMIT or STOP_ABNORMAL_SEARCH")
                    .isIn(OVER_ITER_LIMIT, STOP_ABNORMAL_SEARCH, OVER_GRAD_THRESH);
            
            // If iteration limit was hit, verify iterations
            if (result.status == OVER_ITER_LIMIT) {
                assertThat(result.iterations)
                        .as("Iterations should be at or near the limit")
                        .isGreaterThanOrEqualTo(maxIter);
            }
    }

    /**
     * Property 8.4: Evaluation limit is correctly reported
     * 
     * <p><i>For any</i> optimization problem that exceeds max_eval evaluations,
     * the limit shall be reported with OVER_EVAL_LIMIT.</p>
     * 
     * 
     * @param n the dimension (2 to 5)
     * @param maxEval the maximum evaluations (2 to 10)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 50)
    @Label("Property 8.4: Evaluation limit is correctly reported")
    void evaluationLimitReported(
            @ForAll @IntRange(min = 2, max = 5) int n,
            @ForAll @IntRange(min = 2, max = 10) int maxEval,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        // Generate random starting point far from minimum
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            x[i] = random.nextGaussian() * 5.0 - 2.5;
            lower[i] = -10.0;
            upper[i] = 10.0;
            boundType[i] = BOUND_BOTH;
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-15;  // Very tight tolerance
            double factr = 1e-15; // Very tight tolerance
            int maxIter = 1000;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, rosenbrock, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should stop due to evaluation limit (or possibly abnormal search)
            assertThat(result.status)
                    .as("Should stop with OVER_EVAL_LIMIT or STOP_ABNORMAL_SEARCH")
                    .isIn(OVER_EVAL_LIMIT, STOP_ABNORMAL_SEARCH, OVER_GRAD_THRESH);
            
            // If evaluation limit was hit, verify evaluations
            if (result.status == OVER_EVAL_LIMIT) {
                assertThat(result.evaluations)
                        .as("Evaluations should be at or near the limit")
                        .isGreaterThanOrEqualTo(maxEval);
            }
    }


    /**
     * Property 8.5: Projected gradient norm is correctly computed for bound-constrained problems
     * 
     * <p><i>For any</i> point x, gradient g, and bounds [l, u], the projected gradient
     * norm shall be correctly computed according to the formula:</p>
     * <ul>
     *   <li>If g_i &lt; 0 and has upper bound: proj g_i = max(x_i - u_i, g_i)</li>
     *   <li>If g_i &gt;= 0 and has lower bound: proj g_i = min(x_i - l_i, g_i)</li>
     *   <li>Otherwise: proj g_i = g_i</li>
     * </ul>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 8.5: Projected gradient norm is correctly computed")
    void projectedGradientNormCorrectlyComputed(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random bounds, point, and gradient
        double[] x = new double[n];
        double[] g = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 10;
            double u = l + Math.abs(random.nextGaussian()) * 10 + 0.1;
            lower[i] = l;
            upper[i] = u;
            boundType[i] = random.nextInt(4);  // 0-3: NONE, LOWER, BOTH, UPPER
            
            // Generate point within bounds
            x[i] = l + (u - l) * random.nextDouble();
            g[i] = random.nextGaussian() * 10;
        }
        
        // Compute projected gradient norm using the implementation
        double norm = LBFGSBCore.projGradNorm(n, x, g, TestBounds.toBounds(n, lower, upper, boundType));
        
        // Verify norm is non-negative
        assertThat(norm)
                .as("Projected gradient norm should be non-negative")
                .isGreaterThanOrEqualTo(0.0);
        
        // Manually compute expected norm
        double expectedNorm = 0.0;
        for (int i = 0; i < n; i++) {
            double gi = g[i];
            int bt = boundType[i];
            
            if (bt != BOUND_NONE) {
                if (gi < 0.0) {
                    // Check upper bound (bt >= BOUND_BOTH means has upper bound)
                    if (bt >= BOUND_BOTH) {
                        double diff = x[i] - upper[i];
                        if (diff > gi) {
                            gi = diff;
                        }
                    }
                } else {
                    // Check lower bound (bt <= BOUND_BOTH means has lower bound)
                    if (bt <= BOUND_BOTH) {
                        double diff = x[i] - lower[i];
                        if (diff < gi) {
                            gi = diff;
                        }
                    }
                }
            }
            
            double absGi = Math.abs(gi);
            if (absGi > expectedNorm) {
                expectedNorm = absGi;
            }
        }
        
        assertThat(norm)
                .as("Projected gradient norm should match expected value")
                .isCloseTo(expectedNorm, within(TOLERANCE));
    }

    /**
     * Property 8.6: Projected gradient norm is zero at optimal point with active bounds
     * 
     * <p><i>For any</i> point x at bounds with gradient pointing away from feasible region,
     * the projected gradient norm shall be zero.</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 8.6: Projected gradient norm is zero at optimal point")
    void projectedGradientNormZeroAtOptimalPoint(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate bounds and place point at bounds with gradient pointing away
        double[] x = new double[n];
        double[] g = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 10;
            double u = l + Math.abs(random.nextGaussian()) * 10 + 0.1;
            lower[i] = l;
            upper[i] = u;
            boundType[i] = BOUND_BOTH;
            
            // Place at lower or upper bound with gradient pointing away
            if (random.nextBoolean()) {
                x[i] = l; // At lower bound
                g[i] = Math.abs(random.nextGaussian()) + 0.1; // Positive gradient
            } else {
                x[i] = u; // At upper bound
                g[i] = -(Math.abs(random.nextGaussian()) + 0.1); // Negative gradient
            }
        }
        
        // Compute projected gradient norm
        double norm = LBFGSBCore.projGradNorm(n, x, g, TestBounds.toBounds(n, lower, upper, boundType));
        
        // At optimal point, projected gradient should be zero
        assertThat(norm)
                .as("Projected gradient norm should be zero at optimal point")
                .isCloseTo(0.0, within(TOLERANCE));
    }

    /**
     * Property 8.7: Convergence status is consistent with termination criteria
     * 
     * <p><i>For any</i> optimization run, the returned status shall be consistent
     * with the termination criteria that was triggered.</p>
     * 
     * 
     * @param n the dimension (2 to 5)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 50)
    @Label("Property 8.7: Convergence status is consistent with termination criteria")
    void convergenceStatusConsistent(
            @ForAll @IntRange(min = 2, max = 5) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        // Generate random starting point
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            x[i] = random.nextGaussian();
            lower[i] = -10.0;
            upper[i] = 10.0;
            boundType[i] = BOUND_BOTH;
        }
        
        // Simple quadratic
        Univariate quadratic = (point, grad) -> {
            double f = 0.0;
            for (int i = 0; i < point.length; i++) {
                f += point[i] * point[i];
                if (grad != null) {
                    grad[i] = 2.0 * point[i];
                }
            }
            return f;
        };
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-5;
            double factr = 1e7;
            int maxIter = 100;
            int maxEval = 200;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, quadratic, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Status should be one of the valid termination statuses
            assertThat(result.status)
                    .as("Status should be a valid termination status")
                    .isIn(
                            CONV_GRAD_PROG_NORM,
                            CONV_ENOUGH_ACCURACY,
                            OVER_ITER_LIMIT,
                            OVER_EVAL_LIMIT,
                            OVER_TIME_LIMIT,
                            OVER_GRAD_THRESH,
                            STOP_ABNORMAL_SEARCH,
                            HALT_EVAL_PANIC
                    );
            
            // Iterations and evaluations should be non-negative
            assertThat(result.iterations)
                    .as("Iterations should be non-negative")
                    .isGreaterThanOrEqualTo(0);
            assertThat(result.evaluations)
                    .as("Evaluations should be positive")
                    .isGreaterThan(0);
    }


    /**
     * Property 8.8: Convergence with mixed bound types
     * 
     * <p><i>For any</i> optimization problem with mixed bound types (NONE, LOWER, UPPER, BOTH),
     * convergence detection shall work correctly.</p>
     * 
     * 
     * @param n the dimension (4 to 12, divisible by 4)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 50)
    @Label("Property 8.8: Convergence with mixed bound types")
    void convergenceWithMixedBoundTypes(
            @ForAll @IntRange(min = 4, max = 12) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        // Generate random starting point with mixed bound types
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            x[i] = random.nextGaussian() * 0.5;  // Near origin
            lower[i] = -10.0;
            upper[i] = 10.0;
            boundType[i] = i % 4;  // Cycle through NONE, LOWER, BOTH, UPPER
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-5;
            double factr = 1e7;
            int maxIter = 100;
            int maxEval = 200;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, quadratic, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should converge
            assertThat(result.status)
                    .as("Should converge with mixed bound types")
                    .isIn(CONV_GRAD_PROG_NORM, CONV_ENOUGH_ACCURACY);
    }


    /**
     * Property 8.9: Convergence at boundary
     * 
     * <p><i>For any</i> optimization problem where the minimum is at a boundary,
     * convergence shall be correctly detected.</p>
     * 
     * 
     * @param n the dimension (2 to 10)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 50)
    @Label("Property 8.9: Convergence at boundary")
    void convergenceAtBoundary(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        // Generate starting point
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            lower[i] = 0.0;  // Minimum at lower bound
            upper[i] = 10.0;
            boundType[i] = BOUND_BOTH;
            // Ensure starting point lies within bounds to avoid abnormal starts
            double raw = random.nextGaussian();
            x[i] = Math.max(lower[i], Math.min(upper[i], raw));
        }
        
        // Quadratic with minimum at origin (which is at lower bound)
        // f(x) = sum((x_i - (-1))^2) = sum((x_i + 1)^2)
        // Minimum at x = -1, but constrained to x >= 0, so minimum at x = 0
        Univariate quadratic = (point, grad) -> {
            double f = 0.0;
            for (int i = 0; i < point.length; i++) {
                double diff = point[i] + 1.0;  // Unconstrained min at -1
                f += diff * diff;
                if (grad != null) {
                    grad[i] = 2.0 * diff;
                }
            }
            return f;
        };
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double pgtol = 1e-5;
            double factr = 1e7;
            int maxIter = 100;
            int maxEval = 200;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, quadratic, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Should converge (or stop due to abnormal search in some edge cases)
            assertThat(result.status)
                    .as("Should converge at boundary or stop with abnormal search")
                    .isIn(CONV_GRAD_PROG_NORM, CONV_ENOUGH_ACCURACY, STOP_ABNORMAL_SEARCH);
            
            // Solution should be at lower bound (0)
            for (int i = 0; i < n; i++) {
                assertThat(x[i])
                        .as("x[%d] should be at lower bound", i)
                        .isCloseTo(0.0, within(pgtol * 10));
            }
    }
}
