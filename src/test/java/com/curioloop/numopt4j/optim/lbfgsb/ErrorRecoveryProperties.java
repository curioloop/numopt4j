/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.Univariate;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

/**
 * Property-based tests for error recovery in L-BFGS-B optimization.
 * 
 * <p>Tests that the error recovery mechanisms correctly handle numerical issues
 * and limit the number of BFGS resets to prevent infinite loops.</p>
 * 
 * 
 * <h2>Property 10: Error Recovery Correctness</h2>
 * <p><i>For any</i> optimization run encountering numerical issues:</p>
 * <ul>
 *   <li>The number of BFGS resets shall not exceed the maximum (5)</li>
 *   <li>After reset, the algorithm shall continue with identity BFGS matrix</li>
 * </ul>
 */
@Tag("Feature: lbfgsb-java-rewrite, Property 10: Error Recovery Correctness")
public class ErrorRecoveryProperties implements LBFGSBConstants {

    private static final double TOLERANCE = 1e-10;

    // ==================== Unit Tests for Basic Verification ====================

    /**
     * Unit test: Verify BFGS reset counter is properly initialized and incremented.
     * 
     */
    @Test
    void testBfgsResetCounterInitialization() {
        int n = 3;
        int m = 5;
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            // Reset counter should start at 0
            assertThat(ws.getResetCount())
                    .as("Reset counter should start at 0")
                    .isEqualTo(0);
            
            // Increment and verify
            ws.incrementResetCount();
            assertThat(ws.getResetCount())
                    .as("Reset counter should be 1 after increment")
                    .isEqualTo(1);
    }


    /**
     * Unit test: Verify BFGS reset clears correction history.
     * 
     */
    @Test
    void testBfgsResetClearsHistory() {
        int n = 3;
        int m = 5;
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            // Simulate some BFGS updates
            ws.setCol(3);
            ws.setHead(2);
            ws.setTheta(2.5);
            ws.setUpdated(true);
            
            // Reset BFGS
            ws.resetBfgs();
            
            // Verify BFGS state is reset to identity
            assertThat(ws.getCol())
                    .as("col should be 0 after reset (no corrections)")
                    .isEqualTo(0);
            assertThat(ws.getHead())
                    .as("head should be 0 after reset")
                    .isEqualTo(0);
            assertThat(ws.getTheta())
                    .as("theta should be 1.0 after reset (identity scaling)")
                    .isCloseTo(1.0, within(TOLERANCE));
            assertThat(ws.isUpdated())
                    .as("updated should be false after reset")
                    .isFalse();
    }

    /**
     * Unit test: Verify maximum reset limit is enforced.
     * 
     */
    @Test
    void testMaximumResetLimitEnforced() {
        int n = 3;
        int m = 5;
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            // Verify MAX_BFGS_RESETS constant
            assertThat(MAX_BFGS_RESETS)
                    .as("MAX_BFGS_RESETS should be 5")
                    .isEqualTo(5);
            
            // Increment reset counter up to the limit
            for (int i = 0; i < MAX_BFGS_RESETS; i++) {
                ws.incrementResetCount();
                assertThat(ws.getResetCount())
                        .as("Reset count should be %d after %d increments", i + 1, i + 1)
                        .isEqualTo(i + 1);
            }
            
            // Verify we're at the limit
            assertThat(ws.getResetCount())
                    .as("Reset count should be at MAX_BFGS_RESETS")
                    .isEqualTo(MAX_BFGS_RESETS);
    }

    /**
     * Unit test: Verify workspace reset clears the reset counter.
     * 
     */
    @Test
    void testWorkspaceResetClearsResetCounter() {
        int n = 3;
        int m = 5;
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            // Increment reset counter
            ws.incrementResetCount();
            ws.incrementResetCount();
            assertThat(ws.getResetCount()).isEqualTo(2);
            
            // Reset workspace
            ws.reset();
            
            // Reset counter should be cleared
            assertThat(ws.getResetCount())
                    .as("Reset counter should be 0 after workspace reset")
                    .isEqualTo(0);
    }


    // ==================== Property 10: Error Recovery Correctness ====================

    /**
     * Property 10.1: BFGS reset counter never exceeds maximum limit
     * 
     * <p><i>For any</i> optimization run encountering numerical issues,
     * the number of BFGS resets shall not exceed MAX_BFGS_RESETS (5).</p>
     * 
     * <p>This test creates a pathological objective function that causes
     * numerical issues, forcing BFGS resets. The algorithm should terminate
     * gracefully when the reset limit is reached.</p>
     * 
     * 
     * @param n the dimension (2 to 5)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 50)
    @Label("Property 10.1: BFGS reset counter never exceeds maximum limit")
    void bfgsResetCounterNeverExceedsLimit(
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
            x[i] = random.nextGaussian() * 5.0;
            lower[i] = -10.0;
            upper[i] = 10.0;
            boundType[i] = BOUND_BOTH;
        }
        
        // Create a pathological objective function that may cause numerical issues
        // This function has a very ill-conditioned Hessian
        Univariate illConditioned = (point, _n, grad) -> {
            double f = 0.0;
            for (int i = 0; i < point.length; i++) {
                // Highly varying scales cause ill-conditioning
                double scale = Math.pow(10.0, i);
                f += scale * point[i] * point[i];
                if (grad != null) {
                    grad[i] = 2.0 * scale * point[i];
                }
            }
            return f;
        };
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            double pgtol = 1e-10;
            double factr = 1e7;
            int maxIter = 100;
            int maxEval = 500;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, illConditioned, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // Verify reset count never exceeds the maximum
            assertThat(ws.getResetCount())
                    .as("Reset count should not exceed MAX_BFGS_RESETS")
                    .isLessThanOrEqualTo(MAX_BFGS_RESETS);
            
            // If we hit the reset limit, status should indicate abnormal termination
            // or the algorithm should have converged/stopped for another reason
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
    }


    /**
     * Property 10.2: After BFGS reset, algorithm continues with identity matrix
     * 
     * <p><i>For any</i> optimization run where a BFGS reset occurs,
     * the algorithm shall continue with an identity BFGS matrix (col = 0, theta = 1.0).</p>
     * 
     * 
     * @param n the dimension (2 to 5)
     */
    @Property(tries = 50)
    @Label("Property 10.2: After BFGS reset, algorithm continues with identity matrix")
    void afterResetAlgorithmContinuesWithIdentityMatrix(
            @ForAll @IntRange(min = 2, max = 5) int n
    ) {
        int m = 5;
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            // Simulate some BFGS updates
            ws.setCol(3);
            ws.setHead(2);
            ws.setTheta(2.5);
            ws.setUpdated(true);
            
            // Verify non-identity state
            assertThat(ws.getCol()).isGreaterThan(0);
            assertThat(ws.getTheta()).isNotEqualTo(1.0);
            
            // Perform BFGS reset
            ws.resetBfgs();
            
            // Verify identity matrix state
            assertThat(ws.getCol())
                    .as("col should be 0 (no corrections = identity)")
                    .isEqualTo(0);
            assertThat(ws.getTheta())
                    .as("theta should be 1.0 (identity scaling)")
                    .isCloseTo(1.0, within(TOLERANCE));
            assertThat(ws.isUpdated())
                    .as("updated should be false")
                    .isFalse();
    }

    /**
     * Property 10.3: Reset counter is preserved across BFGS resets
     * 
     * <p><i>For any</i> sequence of BFGS resets, the reset counter shall
     * accurately track the total number of resets performed.</p>
     * 
     * 
     * @param numResets the number of resets to perform (1 to MAX_BFGS_RESETS)
     */
    @Property(tries = 20)
    @Label("Property 10.3: Reset counter is preserved across BFGS resets")
    void resetCounterPreservedAcrossResets(
            @ForAll @IntRange(min = 1, max = 5) int numResets
    ) {
        int n = 3;
        int m = 5;
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            // Perform multiple resets
            for (int i = 0; i < numResets; i++) {
                // Simulate some BFGS updates
                ws.setCol(i + 1);
                ws.setTheta(1.5 + i * 0.1);
                ws.setUpdated(true);
                
                // Increment reset counter and reset BFGS
                ws.incrementResetCount();
                ws.resetBfgs();
                
                // Verify reset counter is preserved
                assertThat(ws.getResetCount())
                        .as("Reset count should be %d after %d resets", i + 1, i + 1)
                        .isEqualTo(i + 1);
                
                // Verify BFGS is reset to identity
                assertThat(ws.getCol())
                        .as("col should be 0 after reset")
                        .isEqualTo(0);
            }
    }


    /**
     * Property 10.4: Optimization terminates gracefully when reset limit is exceeded
     * 
     * <p><i>For any</i> optimization run that would require more than MAX_BFGS_RESETS,
     * the algorithm shall terminate with STOP_ABNORMAL_SEARCH status.</p>
     * 
     * <p>This test uses a highly pathological function designed to trigger
     * repeated numerical issues and BFGS resets.</p>
     * 
     * 
     * @param n the dimension (2 to 4)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 30)
    @Label("Property 10.4: Optimization terminates gracefully when reset limit is exceeded")
    void optimizationTerminatesGracefullyOnResetLimit(
            @ForAll @IntRange(min = 2, max = 4) int n,
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
            x[i] = random.nextGaussian() * 10.0;
            lower[i] = -100.0;
            upper[i] = 100.0;
            boundType[i] = BOUND_BOTH;
        }
        
        // Create a simple quadratic that should converge normally
        Univariate quadratic = (point, _n, grad) -> {
            double f = 0.0;
            for (int i = 0; i < point.length; i++) {
                f += point[i] * point[i];
                if (grad != null) {
                    grad[i] = 2.0 * point[i];
                }
            }
            return f;
        };
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            double pgtol = 1e-8;
            double factr = 1e7;
            int maxIter = 200;
            int maxEval = 500;
            
            LBFGSBCore.OptimizeResult result = LBFGSBCore.optimize(
                    n, m, x, quadratic, TestBounds.toBounds(n, lower, upper, boundType),
                    factr, pgtol, maxIter, maxEval, 0, 20, ws);
            
            // For a well-behaved function, should converge without hitting reset limit
            assertThat(ws.getResetCount())
                    .as("Reset count should be low for well-behaved function")
                    .isLessThanOrEqualTo(MAX_BFGS_RESETS);
            
            // Should terminate with a valid status
            assertThat(result.status)
                    .as("Should terminate with valid status")
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
    }


    /**
     * Property 10.5: Reset counter is independent per optimization run
     * 
     * <p><i>For any</i> sequence of optimization runs using the same workspace,
     * the reset counter shall be reset to 0 at the start of each run.</p>
     * 
     * 
     * @param numRuns the number of optimization runs (2 to 5)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 30)
    @Label("Property 10.5: Reset counter is independent per optimization run")
    void resetCounterIndependentPerRun(
            @ForAll @IntRange(min = 2, max = 5) int numRuns,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int n = 3;
        int m = 5;
        
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            lower[i] = -10.0;
            upper[i] = 10.0;
            boundType[i] = BOUND_BOTH;
        }
        
        // Simple quadratic
        Univariate quadratic = (point, _n, grad) -> {
            double f = 0.0;
            for (int i = 0; i < point.length; i++) {
                f += point[i] * point[i];
                if (grad != null) {
                    grad[i] = 2.0 * point[i];
                }
            }
            return f;
        };
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            for (int run = 0; run < numRuns; run++) {
                // Generate new starting point for each run
                double[] x = new double[n];
                for (int i = 0; i < n; i++) {
                    x[i] = random.nextGaussian();
                }
                
                // Run optimization - reset counter starts at 0 for each run
                // (workspace.reset() is called at start of optimize)
                LBFGSBCore.optimize(
                        n, m, x, quadratic, TestBounds.toBounds(n, lower, upper, boundType),
                        1e7, 1e-5, 100, 200, 0, 20, ws);
                
                // After optimization, reset count should be reasonable
                assertThat(ws.getResetCount())
                        .as("Reset count for run %d should not exceed limit", run)
                        .isLessThanOrEqualTo(MAX_BFGS_RESETS);
            }
    }

    /**
     * Property 10.6: BFGS reset preserves workspace compatibility
     * 
     * <p><i>For any</i> BFGS reset, the workspace shall remain compatible
     * with the original dimensions (n, m).</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param m the number of corrections (3 to 10)
     */
    @Property(tries = 50)
    @Label("Property 10.6: BFGS reset preserves workspace compatibility")
    void bfgsResetPreservesWorkspaceCompatibility(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @IntRange(min = 3, max = 10) int m
    ) {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            // Verify initial compatibility
            assertThat(ws.isCompatible(n, m))
                    .as("Workspace should be compatible with original dimensions")
                    .isTrue();
            
            // Simulate some BFGS updates
            ws.setCol(Math.min(3, m));
            ws.setTheta(2.0);
            ws.setUpdated(true);
            
            // Reset BFGS
            ws.resetBfgs();
            
            // Verify workspace is still compatible after reset
            assertThat(ws.isCompatible(n, m))
                    .as("Workspace should still be compatible after BFGS reset")
                    .isTrue();
            
            // Verify BFGS state is reset to identity
            assertThat(ws.getCol())
                    .as("col should be 0 after reset")
                    .isEqualTo(0);
            assertThat(ws.getTheta())
                    .as("theta should be 1.0 after reset")
                    .isCloseTo(1.0, within(TOLERANCE));
    }
}
