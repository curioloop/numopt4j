/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBConstants;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBCore;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBWorkspace;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

/**
 * Property-based tests for projection operations.
 * 
 * <p>Tests that the projection operations (projectX, projGradNorm, projInitActive)
 * correctly handle bound-constrained optimization.</p>
 * 
 * 
 * <h2>Property 3: Projection Correctness</h2>
 * <p><i>For any</i> point x and bounds [l, u]:</p>
 * <ul>
 *   <li>The projected point x_proj shall satisfy l ≤ x_proj ≤ u</li>
 *   <li>Variables with no bounds shall remain unchanged</li>
 *   <li>Variables with equal bounds (l = u) shall be fixed at that value</li>
 *   <li>The projected gradient shall correctly reflect the active constraints</li>
 * </ul>
 */
@Tag("Feature: lbfgsb-java-rewrite, Property 3: Projection Correctness")
public class ProjectionProperties implements LBFGSBConstants {

    private static final double TOLERANCE = 1e-15;

    // ==================== Unit Tests for Basic Verification ====================

    /**
     * Simple unit test to verify projectX with known values.
     * 
     * x = [0.5, -1.0, 3.0, 2.0]
     * l = [0.0, 0.0, 1.0, 1.5]
     * u = [1.0, 1.0, 2.0, 1.5]
     * boundType = [BOTH, LOWER, UPPER, BOTH]
     * 
     * Expected: [0.5, 0.0, 2.0, 1.5]
     */
    @Test
    void testProjectXSimple() {
        int n = 4;
        double[] x = {0.5, -1.0, 3.0, 2.0};
        double[] lower = {0.0, 0.0, 1.0, 1.5};
        double[] upper = {1.0, 1.0, 2.0, 1.5};
        int[] boundType = {BOUND_BOTH, BOUND_LOWER, BOUND_UPPER, BOUND_BOTH};

        LBFGSBCore.projectX(n, x, TestBounds.toBounds(n, lower, upper, boundType));
        
        assertThat(x[0]).as("x[0] should remain unchanged (within bounds)").isCloseTo(0.5, within(TOLERANCE));
        assertThat(x[1]).as("x[1] should be projected to lower bound").isCloseTo(0.0, within(TOLERANCE));
        assertThat(x[2]).as("x[2] should be projected to upper bound").isCloseTo(2.0, within(TOLERANCE));
        assertThat(x[3]).as("x[3] should be fixed at l=u=1.5").isCloseTo(1.5, within(TOLERANCE));
    }

    /**
     * Unit test for projectX with unbounded variables.
     */
    @Test
    void testProjectXUnbounded() {
        int n = 3;
        double[] x = {-100.0, 0.0, 100.0};
        double[] lower = {0.0, 0.0, 0.0};
        double[] upper = {1.0, 1.0, 1.0};
        int[] boundType = {BOUND_NONE, BOUND_NONE, BOUND_NONE};
        
        double[] xOriginal = x.clone();
        LBFGSBCore.projectX(n, x, TestBounds.toBounds(n, lower, upper, boundType));
        
        for (int i = 0; i < n; i++) {
            assertThat(x[i]).as("Unbounded x[%d] should remain unchanged", i)
                    .isCloseTo(xOriginal[i], within(TOLERANCE));
        }
    }

    /**
     * Unit test for projGradNorm with known values.
     */
    @Test
    void testProjGradNormSimple() {
        int n = 4;
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
        assertThat(norm).as("Projected gradient norm").isCloseTo(0.5, within(TOLERANCE));
    }

    /**
     * Unit test for projInitActive with known values.
     */
    @Test
    void testProjInitActiveSimple() {
        int n = 5;
        double[] x = {-1.0, 2.0, 0.5, 0.5, 0.5};
        double[] lower = {0.0, 0.0, 0.0, 0.0, 0.5};
        double[] upper = {1.0, 1.0, 1.0, 1.0, 0.5};
        int[] boundType = {BOUND_BOTH, BOUND_BOTH, BOUND_BOTH, BOUND_NONE, BOUND_BOTH};

        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, 1);
        boolean projected = LBFGSBCore.projInitActive(n, x, TestBounds.toBounds(n, lower, upper, boundType), ws);
        
        // x[0] should be projected to 0.0 (lower bound)
        assertThat(x[0]).as("x[0] should be projected to lower bound").isCloseTo(0.0, within(TOLERANCE));
        // x[1] should be projected to 1.0 (upper bound)
        assertThat(x[1]).as("x[1] should be projected to upper bound").isCloseTo(1.0, within(TOLERANCE));
        // x[2] should remain unchanged (within bounds)
        assertThat(x[2]).as("x[2] should remain unchanged").isCloseTo(0.5, within(TOLERANCE));
        // x[3] should remain unchanged (unbounded)
        assertThat(x[3]).as("x[3] should remain unchanged (unbounded)").isCloseTo(0.5, within(TOLERANCE));
        // x[4] should be at fixed value 0.5
        assertThat(x[4]).as("x[4] should be at fixed value").isCloseTo(0.5, within(TOLERANCE));
        
        // Check where array
        int[] where = ws.getIntBuffer();
        int whereOffset = ws.getWhereOffset();
        assertThat(where[whereOffset+0]).as("where[0] should be VAR_FREE").isEqualTo(VAR_FREE);
        assertThat(where[whereOffset+1]).as("where[1] should be VAR_FREE").isEqualTo(VAR_FREE);
        assertThat(where[whereOffset+2]).as("where[2] should be VAR_FREE").isEqualTo(VAR_FREE);
        assertThat(where[whereOffset+3]).as("where[3] should be VAR_UNBOUND").isEqualTo(VAR_UNBOUND);
        assertThat(where[whereOffset+4]).as("where[4] should be VAR_FIXED").isEqualTo(VAR_FIXED);
        
        // Check result flags
        assertThat(projected).as("Should be projected").isTrue();
        assertThat(ws.isConstrained()).as("Should be constrained").isTrue();
        assertThat(ws.isBoxed()).as("Should not be boxed (has unbounded var)").isFalse();
    }

    // ==================== Property 3: Projection Correctness ====================

    /**
     * Property 3.1: projectX produces points within bounds (l ≤ x_proj ≤ u)
     * 
     * <p><i>For any</i> point x and bounds [l, u], the projected point x_proj
     * shall satisfy l ≤ x_proj ≤ u for all bounded variables.</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 3.1: projectX produces points within bounds")
    void projectXProducesPointsWithinBounds(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random bounds and point
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            // Generate random bounds
            double l = random.nextGaussian() * 10;
            double u = l + Math.abs(random.nextGaussian()) * 10 + 0.1; // Ensure u > l
            lower[i] = l;
            upper[i] = u;
            
            // Generate random bound type
            boundType[i] = random.nextInt(4); // 0-3: NONE, LOWER, BOTH, UPPER
            
            // Generate random point (possibly outside bounds)
            x[i] = l + (u - l) * (random.nextDouble() * 3 - 1); // Range: [l - (u-l), l + 2*(u-l)]
        }
        
        // Project the point
        LBFGSBCore.projectX(n, x, TestBounds.toBounds(n, lower, upper, boundType));
        
        // Verify bounds are satisfied
        for (int i = 0; i < n; i++) {
            int bt = boundType[i];
            
            if (bt == BOUND_LOWER || bt == BOUND_BOTH) {
                assertThat(x[i])
                        .as("x[%d] should be >= lower bound after projection", i)
                        .isGreaterThanOrEqualTo(lower[i] - TOLERANCE);
            }
            
            if (bt == BOUND_UPPER || bt == BOUND_BOTH) {
                assertThat(x[i])
                        .as("x[%d] should be <= upper bound after projection", i)
                        .isLessThanOrEqualTo(upper[i] + TOLERANCE);
            }
        }
    }

    /**
     * Property 3.2: Unbounded variables remain unchanged after projection
     * 
     * <p><i>For any</i> point x with unbounded variables (BOUND_NONE),
     * those variables shall remain unchanged after projection.</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 3.2: Unbounded variables remain unchanged")
    void unboundedVariablesRemainUnchanged(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random point with all unbounded variables
        double[] x = new double[n];
        double[] xOriginal = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            x[i] = random.nextGaussian() * 100; // Large range
            xOriginal[i] = x[i];
            lower[i] = 0.0;
            upper[i] = 1.0;
            boundType[i] = BOUND_NONE; // All unbounded
        }
        
        // Project the point
        LBFGSBCore.projectX(n, x, TestBounds.toBounds(n, lower, upper, boundType));
        
        // Verify all variables remain unchanged
        for (int i = 0; i < n; i++) {
            assertThat(x[i])
                    .as("Unbounded x[%d] should remain unchanged", i)
                    .isCloseTo(xOriginal[i], within(TOLERANCE));
        }
    }

    /**
     * Property 3.3: Fixed variables (l = u) are set to that value
     * 
     * <p><i>For any</i> point x with fixed variables (l = u),
     * those variables shall be set to the fixed value after projection.</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 3.3: Fixed variables are set to fixed value")
    void fixedVariablesAreSetToFixedValue(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random point with all fixed variables
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            double fixedValue = random.nextGaussian() * 10;
            lower[i] = fixedValue;
            upper[i] = fixedValue; // l = u (fixed)
            boundType[i] = BOUND_BOTH;
            x[i] = random.nextGaussian() * 100; // Random initial value
        }
        
        // Project the point
        LBFGSBCore.projectX(n, x, TestBounds.toBounds(n, lower, upper, boundType));
        
        // Verify all variables are set to fixed value
        for (int i = 0; i < n; i++) {
            assertThat(x[i])
                    .as("Fixed x[%d] should be set to fixed value %f", i, lower[i])
                    .isCloseTo(lower[i], within(TOLERANCE));
        }
    }

    /**
     * Property 3.4: projGradNorm correctly computes the projected gradient norm
     * 
     * <p><i>For any</i> point x, gradient g, and bounds [l, u], the projected
     * gradient norm shall be non-negative and correctly reflect active constraints.</p>
     * 
     * <p>The projected gradient is defined as:</p>
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
    @Label("Property 3.4: projGradNorm computes correct projected gradient norm")
    void projGradNormComputesCorrectNorm(
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
            boundType[i] = random.nextInt(4);
            
            // Generate point within bounds
            x[i] = l + (u - l) * random.nextDouble();
            g[i] = random.nextGaussian() * 10;
        }
        
        // Compute projected gradient norm
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
     * Property 3.5: projInitActive correctly initializes variable status
     * 
     * <p><i>For any</i> point x and bounds [l, u], projInitActive shall:</p>
     * <ul>
     *   <li>Project x to the feasible region</li>
     *   <li>Set where[i] = VAR_UNBOUND for unbounded variables</li>
     *   <li>Set where[i] = VAR_FIXED for fixed variables (l = u)</li>
     *   <li>Set where[i] = VAR_FREE for other bounded variables</li>
     *   <li>Set constrained = true if any variable has bounds</li>
     *   <li>Set boxed = true only if all variables have both bounds</li>
     * </ul>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 3.5: projInitActive correctly initializes variable status")
    void projInitActiveCorrectlyInitializesStatus(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random bounds and point
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];

        boolean expectConstrained = false;
        boolean expectBoxed = true;
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 10;
            double u = l + Math.abs(random.nextGaussian()) * 10 + 0.1;
            
            // Randomly decide bound type
            int bt = random.nextInt(5); // 0-4
            if (bt == 4) {
                // Fixed variable (l = u)
                bt = BOUND_BOTH;
                u = l;
            } else {
                bt = bt % 4; // 0-3
            }
            
            lower[i] = l;
            upper[i] = u;
            boundType[i] = bt;
            
            // Track expected flags
            if (bt != BOUND_NONE) {
                expectConstrained = true;
            }
            if (bt != BOUND_BOTH) {
                expectBoxed = false;
            }
            
            // Generate random point (possibly outside bounds)
            x[i] = l + (u - l) * (random.nextDouble() * 3 - 1);
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, 1);
        boolean projected = LBFGSBCore.projInitActive(n, x, TestBounds.toBounds(n, lower, upper, boundType), ws);
        
        // Verify projection
        for (int i = 0; i < n; i++) {
            int bt = boundType[i];
            
            if (bt == BOUND_LOWER || bt == BOUND_BOTH) {
                assertThat(x[i])
                        .as("x[%d] should be >= lower bound after projection", i)
                        .isGreaterThanOrEqualTo(lower[i] - TOLERANCE);
            }
            
            if (bt == BOUND_UPPER || bt == BOUND_BOTH) {
                assertThat(x[i])
                        .as("x[%d] should be <= upper bound after projection", i)
                        .isLessThanOrEqualTo(upper[i] + TOLERANCE);
            }
        }
        int[] where = ws.getIntBuffer();
        int whereOffset = ws.getWhereOffset();

        // Verify where array
        for (int i = 0; i < n; i++) {
            int bt = boundType[i];
            
            if (bt == BOUND_NONE) {
                assertThat(where[whereOffset+i])
                        .as("where[%d] should be VAR_UNBOUND for unbounded variable", i)
                        .isEqualTo(VAR_UNBOUND);
            } else if (bt == BOUND_BOTH && upper[i] - lower[i] <= 0.0) {
                assertThat(where[whereOffset+i])
                        .as("where[%d] should be VAR_FIXED for fixed variable", i)
                        .isEqualTo(VAR_FIXED);
            } else {
                assertThat(where[whereOffset+i])
                        .as("where[%d] should be VAR_FREE for bounded variable", i)
                        .isEqualTo(VAR_FREE);
            }
        }
        
        // Verify result flags
        assertThat(ws.isConstrained())
                .as("constrained flag should be correct")
                .isEqualTo(expectConstrained);
        assertThat(ws.isBoxed())
                .as("boxed flag should be correct")
                .isEqualTo(expectBoxed);
    }

    /**
     * Property 3.6: Projected gradient norm is zero at optimal point
     * 
     * <p><i>For any</i> point x at the optimal location (gradient points away from
     * feasible region at bounds), the projected gradient norm shall be zero.</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 3.6: Projected gradient norm is zero at optimal point")
    void projGradNormIsZeroAtOptimalPoint(
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
                g[i] = Math.abs(random.nextGaussian()) + 0.1; // Positive gradient (points to decrease x)
            } else {
                x[i] = u; // At upper bound
                g[i] = -(Math.abs(random.nextGaussian()) + 0.1); // Negative gradient (points to increase x)
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
     * Property 3.7: Idempotence of projection
     * 
     * <p><i>For any</i> point x and bounds [l, u], projecting twice shall
     * produce the same result as projecting once.</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 3.7: Projection is idempotent")
    void projectionIsIdempotent(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random bounds and point
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 10;
            double u = l + Math.abs(random.nextGaussian()) * 10 + 0.1;
            lower[i] = l;
            upper[i] = u;
            boundType[i] = random.nextInt(4);
            x[i] = random.nextGaussian() * 100;
        }
        
        // Project once
        LBFGSBCore.projectX(n, x, TestBounds.toBounds(n, lower, upper, boundType));
        double[] xAfterFirst = x.clone();
        
        // Project again
        LBFGSBCore.projectX(n, x, TestBounds.toBounds(n, lower, upper, boundType));
        
        // Should be the same
        for (int i = 0; i < n; i++) {
            assertThat(x[i])
                    .as("x[%d] should be unchanged after second projection", i)
                    .isCloseTo(xAfterFirst[i], within(TOLERANCE));
        }
    }

    /**
     * Property 3.8: Mixed bound types are handled correctly
     * 
     * <p><i>For any</i> point x with mixed bound types (NONE, LOWER, UPPER, BOTH),
     * each variable shall be projected according to its specific bound type.</p>
     * 
     * 
     * @param n the dimension (4 to 20, must be divisible by 4)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 3.8: Mixed bound types are handled correctly")
    void mixedBoundTypesHandledCorrectly(
            @ForAll @IntRange(min = 4, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random bounds and point with all bound types
        double[] x = new double[n];
        double[] xOriginal = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 10;
            double u = l + Math.abs(random.nextGaussian()) * 10 + 0.1;
            lower[i] = l;
            upper[i] = u;
            
            // Cycle through bound types
            boundType[i] = i % 4;
            
            // Generate point outside bounds
            x[i] = l - 10 + random.nextDouble() * (u - l + 20);
            xOriginal[i] = x[i];
        }
        
        // Project the point
        LBFGSBCore.projectX(n, x, TestBounds.toBounds(n, lower, upper, boundType));
        
        // Verify each bound type
        for (int i = 0; i < n; i++) {
            int bt = boundType[i];
            
            switch (bt) {
                case BOUND_NONE:
                    assertThat(x[i])
                            .as("BOUND_NONE: x[%d] should remain unchanged", i)
                            .isCloseTo(xOriginal[i], within(TOLERANCE));
                    break;
                case BOUND_LOWER:
                    assertThat(x[i])
                            .as("BOUND_LOWER: x[%d] should be >= lower", i)
                            .isGreaterThanOrEqualTo(lower[i] - TOLERANCE);
                    if (xOriginal[i] >= lower[i]) {
                        assertThat(x[i])
                                .as("BOUND_LOWER: x[%d] should remain unchanged if within bounds", i)
                                .isCloseTo(xOriginal[i], within(TOLERANCE));
                    }
                    break;
                case BOUND_BOTH:
                    assertThat(x[i])
                            .as("BOUND_BOTH: x[%d] should be >= lower", i)
                            .isGreaterThanOrEqualTo(lower[i] - TOLERANCE);
                    assertThat(x[i])
                            .as("BOUND_BOTH: x[%d] should be <= upper", i)
                            .isLessThanOrEqualTo(upper[i] + TOLERANCE);
                    break;
                case BOUND_UPPER:
                    assertThat(x[i])
                            .as("BOUND_UPPER: x[%d] should be <= upper", i)
                            .isLessThanOrEqualTo(upper[i] + TOLERANCE);
                    if (xOriginal[i] <= upper[i]) {
                        assertThat(x[i])
                                .as("BOUND_UPPER: x[%d] should remain unchanged if within bounds", i)
                                .isCloseTo(xOriginal[i], within(TOLERANCE));
                    }
                    break;
            }
        }
    }

    /**
     * Property 3.9: projInitActive with workspace correctly updates workspace state
     * 
     * <p><i>For any</i> point x and bounds [l, u], projInitActive with workspace
     * shall correctly update the workspace's constrained and boxed flags.</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 3.9: projInitActive with workspace updates state correctly")
    void projInitActiveWithWorkspaceUpdatesState(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5; // Number of corrections
        
        // Generate random bounds and point
        double[] x = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        boolean expectConstrained = false;
        boolean expectBoxed = true;
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 10;
            double u = l + Math.abs(random.nextGaussian()) * 10 + 0.1;
            lower[i] = l;
            upper[i] = u;
            boundType[i] = random.nextInt(4);
            
            if (boundType[i] != BOUND_NONE) {
                expectConstrained = true;
            }
            if (boundType[i] != BOUND_BOTH) {
                expectBoxed = false;
            }
            
            x[i] = random.nextGaussian() * 100;
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(); ws.ensure(n, m);
            LBFGSBCore.projInitActive(n, x, TestBounds.toBounds(n, lower, upper, boundType), ws);
            
            // Verify workspace state
            assertThat(ws.isConstrained())
                    .as("Workspace constrained flag should be correct")
                    .isEqualTo(expectConstrained);
            assertThat(ws.isBoxed())
                    .as("Workspace boxed flag should be correct")
                    .isEqualTo(expectBoxed);
    }
}
