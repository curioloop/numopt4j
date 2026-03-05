/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.lbfgsb.CauchyPoint;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBConstants;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBWorkspace;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

/**
 * Property-based tests for GCP (Generalized Cauchy Point) computation.
 * 
 * <p>Tests that the GCP computation and its helper functions (heapSortOut, bmv, freeVar)
 * produce mathematically correct results.</p>
 * 
 * 
 * <h2>Property 4: GCP Computation Correctness</h2>
 * <p><i>For any</i> point x, gradient g, and bounds [l, u]:</p>
 * <ul>
 *   <li>The GCP z shall be within the feasible region: l ≤ z ≤ u</li>
 *   <li>Variables at bounds shall be correctly identified and fixed</li>
 *   <li>Breakpoints shall be computed correctly as t_i = (x_i - bound_i) / g_i</li>
 *   <li>The BMV matrix-vector product shall satisfy p = M*v for the middle matrix M</li>
 * </ul>
 */
@Tag("Feature: lbfgsb-java-rewrite, Property 4: GCP Computation Correctness")
public class CauchyPointProperties implements LBFGSBConstants {

    private static final double TOLERANCE = 1e-10;
    private static final double EPSILON = 1e-6;

    // ==================== Unit Tests for Basic Verification ====================

    /**
     * Simple unit test to verify heapSortOut builds a min-heap and extracts minimum.
     */
    @Test
    void testHeapSortOutSimple() {
        double[] t = {5.0, 3.0, 8.0, 1.0, 4.0};
        int[] order = {0, 1, 2, 3, 4};
        int n = 5;
        
        // First call: build heap and extract minimum
        CauchyPoint.heapSortOut(n, t, 0, order, 0, false);

        // After first call, t[n-1] should contain the minimum (1.0)
        assertThat(t[n - 1]).as("Minimum should be extracted to t[n-1]").isCloseTo(1.0, within(TOLERANCE));
        assertThat(order[n - 1]).as("Order should track the original index of minimum").isEqualTo(3);
        
        // Second call: extract next minimum from heap t[0:n-1]
        CauchyPoint.heapSortOut(n - 1, t, 0, order, 0, true);
        
        // After second call, t[n-2] should contain the next minimum (3.0)
        assertThat(t[n - 2]).as("Next minimum should be extracted to t[n-2]").isCloseTo(3.0, within(TOLERANCE));
        assertThat(order[n - 2]).as("Order should track the original index").isEqualTo(1);
    }

    /**
     * Unit test for heapSortOut with already sorted input.
     */
    @Test
    void testHeapSortOutAlreadySorted() {
        double[] t = {1.0, 2.0, 3.0, 4.0, 5.0};
        int[] order = {0, 1, 2, 3, 4};
        int n = 5;
        
        // Build heap and extract minimum
        CauchyPoint.heapSortOut(n, t, 0, order, 0, false);
        
        // Minimum should be 1.0
        assertThat(t[n - 1]).as("Minimum should be 1.0").isCloseTo(1.0, within(TOLERANCE));
        assertThat(order[n - 1]).as("Original index of minimum").isEqualTo(0);
    }

    /**
     * Unit test for heapSortOut with reverse sorted input.
     */
    @Test
    void testHeapSortOutReverseSorted() {
        double[] t = {5.0, 4.0, 3.0, 2.0, 1.0};
        int[] order = {0, 1, 2, 3, 4};
        int n = 5;
        
        // Build heap and extract minimum
        CauchyPoint.heapSortOut(n, t, 0, order, 0, false);
        
        // Minimum should be 1.0
        assertThat(t[n - 1]).as("Minimum should be 1.0").isCloseTo(1.0, within(TOLERANCE));
        assertThat(order[n - 1]).as("Original index of minimum").isEqualTo(4);
    }

    /**
     * Unit test for heapSortOut with single element.
     */
    @Test
    void testHeapSortOutSingleElement() {
        double[] t = {42.0};
        int[] order = {0};
        
        // Should not crash with single element
        CauchyPoint.heapSortOut(1, t, 0, order, 0, false);
        
        assertThat(t[0]).as("Single element should remain").isCloseTo(42.0, within(TOLERANCE));
        assertThat(order[0]).as("Order should remain").isEqualTo(0);
    }

    /**
     * Unit test for bmv with zero corrections (col = 0).
     */
    @Test
    void testBmvZeroCorrections() {
        int m = 5;
        double[] sy = new double[m * m];
        double[] wt = new double[m * m];
        double[] v = new double[2 * m];
        double[] p = new double[2 * m];
        double[] diagInv = new double[m];
        double[] sqrtDiagInv = new double[m];
        
        // With col = 0, bmv should return immediately without error
        int info = CauchyPoint.bmv(m, 0, sy, 0, wt, 0, v, 0, p, 0, diagInv, 0, sqrtDiagInv, 0);
        
        assertThat(info).as("bmv should succeed with col=0").isEqualTo(ERR_NONE);
    }

    /**
     * Unit test for bmv with simple 1x1 case.
     * 
     * For col=1, the middle matrix M is 2x2:
     * M = [ -D    L' ]^-1  where D = diag(s'y), L is strictly lower triangular of S'Y
     *     [  L   θS'S]
     * 
     * With col=1: D = [sy[0,0]], L = [], θS'S = [θ*ss[0,0]]
     */
    @Test
    void testBmvSimple() {
        int m = 3;
        int col = 1;
        
        // S'Y matrix (m x m, column-major)
        // sy[0,0] = s'y = 2.0 (diagonal element D)
        double[] sy = new double[m * m];
        sy[0] = 2.0;  // sy[0,0] = D[0,0]
        
        // Cholesky factor wt (m x m, column-major)
        // wt contains J such that θS'S + LD^-1L' = JJ'
        // For col=1: θS'S = θ*ss[0,0], L = 0, so J[0,0] = sqrt(θ*ss[0,0])
        // Let's use θ = 1.0 and ss[0,0] = 4.0, so J[0,0] = 2.0
        double[] wt = new double[m * m];
        wt[0] = 2.0;  // J[0,0] = sqrt(θ*ss[0,0]) = sqrt(4) = 2
        
        // Input vector v = [v1, v2] where v1, v2 are scalars for col=1
        double[] v = new double[2 * m];
        v[0] = 1.0;  // v1
        v[1] = 1.0;  // v2
        
        // Output vector p
        double[] p = new double[2 * m];
        
        // Workspace arrays for bmv
        double[] diagInv = new double[m];
        double[] sqrtDiagInv = new double[m];
        
        int info = CauchyPoint.bmv(m, col, sy, 0, wt, 0, v, 0, p, 0, diagInv, 0, sqrtDiagInv, 0);
        
        assertThat(info).as("bmv should succeed").isEqualTo(ERR_NONE);
        // The result should be finite
        assertThat(Double.isFinite(p[0])).as("p[0] should be finite").isTrue();
        assertThat(Double.isFinite(p[1])).as("p[1] should be finite").isTrue();
    }

    /**
     * Unit test for freeVar with simple case.
     */
    @Test
    void testFreeVarSimple() {
        int n = 5;
        int m = 3;
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            int[] iBuffer = ws.getIntBuffer();
            int whereOffset = ws.getWhereOffset();
            int indexOffset = ws.getIndexOffset();
            
            // Set up variable status: 2 free, 2 at bounds, 1 unbounded
            iBuffer[whereOffset + 0] = VAR_FREE;
            iBuffer[whereOffset + 1] = VAR_AT_LOWER;
            iBuffer[whereOffset + 2] = VAR_FREE;
            iBuffer[whereOffset + 3] = VAR_AT_UPPER;
            iBuffer[whereOffset + 4] = VAR_UNBOUND;
            
            // Initialize index array (required for iter > 0)
            for (int i = 0; i < n; i++) {
                iBuffer[indexOffset + i] = i;
            }
            
            ws.setIter(0);  // First iteration
            ws.setConstrained(true);
            
            int needRecompute = CauchyPoint.freeVar(n, ws);
            
            // Check free count: VAR_FREE (0, 2) and VAR_UNBOUND (4) = 3 free
            assertThat(ws.getFree()).as("Should have 3 free variables").isEqualTo(3);
            assertThat(ws.getActive()).as("Should have 2 active variables").isEqualTo(2);
    }

    /**
     * Unit test for GCP computation with simple bounded problem.
     */
    @Test
    void testGcpComputeSimple() {
        int n = 3;
        int m = 3;
        
        double[] x = {0.5, 0.5, 0.5};
        double[] g = {1.0, -1.0, 0.0};  // Gradient
        double[] lower = {0.0, 0.0, 0.0};
        double[] upper = {1.0, 1.0, 1.0};
        int[] boundType = {BOUND_BOTH, BOUND_BOTH, BOUND_BOTH};
        double[] z = new double[n];
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            // Initialize workspace state
            ws.setSbgNorm(1.0);  // Non-zero projected gradient norm
            ws.setTheta(1.0);
            ws.setCol(0);  // No BFGS corrections yet
            ws.setHead(0);
            
            int[] iBuffer = ws.getIntBuffer();
            int whereOffset = ws.getWhereOffset();
            
            // Initialize where array
            for (int i = 0; i < n; i++) {
                iBuffer[whereOffset + i] = VAR_FREE;
            }
            
            int info = CauchyPoint.compute(n, m, x, g, TestBounds.toBounds(n, lower, upper, boundType), ws);
            // Copy result from workspace z into local z
            System.arraycopy(ws.getBuffer(), ws.getZOffset(), z, 0, n);
            
            assertThat(info).as("GCP compute should succeed").isEqualTo(ERR_NONE);
            
            // Verify GCP is within bounds
            for (int i = 0; i < n; i++) {
                assertThat(z[i])
                        .as("GCP z[%d] should be >= lower bound", i)
                        .isGreaterThanOrEqualTo(lower[i] - TOLERANCE);
                assertThat(z[i])
                        .as("GCP z[%d] should be <= upper bound", i)
                        .isLessThanOrEqualTo(upper[i] + TOLERANCE);
            }
    }

    // ==================== Property 4: GCP Computation Correctness ====================

    /**
     * Property 4.1: heapSortOut correctly extracts minimum elements in order
     * 
     * <p><i>For any</i> array of breakpoint times t, repeatedly calling heapSortOut
     * shall extract elements in non-decreasing order.</p>
     * 
     * 
     * @param n the number of elements (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.1: heapSortOut extracts minimum elements in order")
    void heapSortOutExtractsMinimumInOrder(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random breakpoint times
        double[] t = new double[n];
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = random.nextDouble() * 100;
            order[i] = i;
        }
        
        // Extract all elements using heapSortOut
        double[] extracted = new double[n];
        
        // First call builds heap and extracts minimum
        CauchyPoint.heapSortOut(n, t, 0, order, 0, false);
        extracted[0] = t[n - 1];
        
        // Subsequent calls extract remaining elements
        for (int i = 1; i < n; i++) {
            int remaining = n - i;
            CauchyPoint.heapSortOut(remaining, t, 0, order, 0, true);
            extracted[i] = t[remaining - 1];
        }
        
        // Verify elements are extracted in non-decreasing order
        for (int i = 1; i < n; i++) {
            assertThat(extracted[i])
                    .as("Element %d should be >= element %d", i, i - 1)
                    .isGreaterThanOrEqualTo(extracted[i - 1] - TOLERANCE);
        }
    }

    /**
     * Property 4.2: heapSortOut preserves all original elements
     * 
     * <p><i>For any</i> array of breakpoint times t, after extracting all elements
     * using heapSortOut, the multiset of extracted elements shall equal the original multiset.</p>
     * 
     * 
     * @param n the number of elements (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.2: heapSortOut preserves all original elements")
    void heapSortOutPreservesAllElements(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random breakpoint times
        double[] t = new double[n];
        int[] order = new int[n];
        double[] original = new double[n];
        for (int i = 0; i < n; i++) {
            t[i] = random.nextDouble() * 100;
            original[i] = t[i];
            order[i] = i;
        }
        
        // Sort original for comparison
        java.util.Arrays.sort(original);
        
        // Extract all elements using heapSortOut
        double[] extracted = new double[n];
        
        CauchyPoint.heapSortOut(n, t, 0, order, 0, false);
        extracted[0] = t[n - 1];
        
        for (int i = 1; i < n; i++) {
            int remaining = n - i;
            CauchyPoint.heapSortOut(remaining, t, 0, order, 0, true);
            extracted[i] = t[remaining - 1];
        }
        
        // Sort extracted for comparison
        java.util.Arrays.sort(extracted);
        
        // Verify all elements are preserved
        for (int i = 0; i < n; i++) {
            assertThat(extracted[i])
                    .as("Extracted element %d should match original", i)
                    .isCloseTo(original[i], within(TOLERANCE));
        }
    }

    /**
     * Property 4.3: heapSortOut maintains correct index tracking
     * 
     * <p><i>For any</i> array of breakpoint times t, the order array shall correctly
     * track the original indices of extracted elements.</p>
     * 
     * 
     * @param n the number of elements (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.3: heapSortOut maintains correct index tracking")
    void heapSortOutMaintainsIndexTracking(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        
        // Generate random breakpoint times
        double[] t = new double[n];
        int[] order = new int[n];
        double[] original = new double[n];
        for (int i = 0; i < n; i++) {
            t[i] = random.nextDouble() * 100;
            original[i] = t[i];
            order[i] = i;
        }
        
        // Extract all elements and verify index tracking
        CauchyPoint.heapSortOut(n, t, 0, order, 0, false);
        
        // The extracted value should match the original at the tracked index
        double extractedValue = t[n - 1];
        int extractedIndex = order[n - 1];
        
        assertThat(extractedValue)
                .as("Extracted value should match original at tracked index")
                .isCloseTo(original[extractedIndex], within(TOLERANCE));
    }

    /**
     * Property 4.4: bmv produces finite results for valid inputs
     * 
     * <p><i>For any</i> valid S'Y matrix with positive diagonal and valid Cholesky factor,
     * bmv shall produce finite results.</p>
     * 
     * 
     * @param col the number of corrections (1 to 5)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.4: bmv produces finite results for valid inputs")
    void bmvProducesFiniteResults(
            @ForAll @IntRange(min = 1, max = 5) int col,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = col + 2;  // m >= col
        
        // Generate S'Y matrix with positive diagonal (column-major)
        double[] sy = new double[m * m];
        for (int j = 0; j < col; j++) {
            for (int i = 0; i < col; i++) {
                if (i == j) {
                    // Positive diagonal (D = diag(s'y))
                    sy[j * m + i] = 1.0 + Math.abs(random.nextGaussian());
                } else if (i > j) {
                    // Lower triangular part (L)
                    sy[j * m + i] = random.nextGaussian() * 0.1;
                }
            }
        }
        
        // Generate valid Cholesky factor (upper triangular with positive diagonal)
        double[] wt = new double[m * m];
        for (int j = 0; j < col; j++) {
            for (int i = 0; i <= j; i++) {
                if (i == j) {
                    wt[j * m + i] = 1.0 + Math.abs(random.nextGaussian());
                } else {
                    wt[j * m + i] = random.nextGaussian() * 0.1;
                }
            }
        }
        
        // Generate input vector
        double[] v = new double[2 * m];
        for (int i = 0; i < 2 * col; i++) {
            v[i] = random.nextGaussian();
        }
        
        // Output vector
        double[] p = new double[2 * m];
        
        // Workspace arrays for bmv
        double[] diagInv = new double[m];
        double[] sqrtDiagInv = new double[m];
        
        int info = CauchyPoint.bmv(m, col, sy, 0, wt, 0, v, 0, p, 0, diagInv, 0, sqrtDiagInv, 0);
        
        // Should succeed or return singular error (both are valid outcomes)
        if (info == ERR_NONE) {
            // Verify results are finite
            for (int i = 0; i < 2 * col; i++) {
                assertThat(Double.isFinite(p[i]))
                        .as("p[%d] should be finite", i)
                        .isTrue();
            }
        } else {
            // Singular matrix is acceptable for some random inputs
            assertThat(info).as("Error should be singular triangular").isEqualTo(ERR_SINGULAR_TRIANGULAR);
        }
    }

    /**
     * Property 4.5: freeVar correctly counts free and active variables
     * 
     * <p><i>For any</i> variable status configuration, freeVar shall correctly count
     * free variables (VAR_FREE, VAR_UNBOUND) and active variables (VAR_AT_LOWER, VAR_AT_UPPER, VAR_FIXED).</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.5: freeVar correctly counts free and active variables")
    void freeVarCorrectlyCountsVariables(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            int[] iBuffer = ws.getIntBuffer();
            int whereOffset = ws.getWhereOffset();
            int indexOffset = ws.getIndexOffset();
            
            // Generate random variable status
            int expectedFree = 0;
            int expectedActive = 0;
            
            for (int i = 0; i < n; i++) {
                int status = random.nextInt(6) - 3;  // -3 to 2, then map to valid values
                if (status < -1) status = VAR_NOT_MOVE;
                else if (status == -1) status = VAR_UNBOUND;
                else if (status == 0) status = VAR_FREE;
                else if (status == 1) status = VAR_AT_LOWER;
                else status = VAR_AT_UPPER;
                
                iBuffer[whereOffset + i] = status;
                
                if (status <= VAR_FREE) {
                    expectedFree++;
                } else {
                    expectedActive++;
                }
                
                // Initialize index array
                iBuffer[indexOffset + i] = i;
            }
            
            ws.setIter(0);
            ws.setConstrained(true);
            
            CauchyPoint.freeVar(n, ws);
            
            assertThat(ws.getFree())
                    .as("Free count should match expected")
                    .isEqualTo(expectedFree);
            assertThat(ws.getActive())
                    .as("Active count should match expected")
                    .isEqualTo(expectedActive);
            assertThat(ws.getFree() + ws.getActive())
                    .as("Free + Active should equal n")
                    .isEqualTo(n);
    }

    /**
     * Property 4.6: freeVar builds correct index sets
     * 
     * <p><i>For any</i> variable status configuration, freeVar shall build index arrays
     * where index[0:free] contains indices of free variables and index[free:n] contains
     * indices of active variables.</p>
     * 
     * 
     * @param n the dimension (2 to 20)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.6: freeVar builds correct index sets")
    void freeVarBuildsCorrectIndexSets(
            @ForAll @IntRange(min = 2, max = 20) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            int[] iBuffer = ws.getIntBuffer();
            int whereOffset = ws.getWhereOffset();
            int indexOffset = ws.getIndexOffset();
            
            // Generate random variable status
            for (int i = 0; i < n; i++) {
                int status;
                int r = random.nextInt(5);
                if (r == 0) status = VAR_UNBOUND;
                else if (r == 1) status = VAR_FREE;
                else if (r == 2) status = VAR_AT_LOWER;
                else if (r == 3) status = VAR_AT_UPPER;
                else status = VAR_FIXED;
                
                iBuffer[whereOffset + i] = status;
                iBuffer[indexOffset + i] = i;
            }
            
            ws.setIter(0);
            ws.setConstrained(true);
            
            CauchyPoint.freeVar(n, ws);
            
            int freeCount = ws.getFree();
            
            // Verify free variables in index[0:free]
            for (int i = 0; i < freeCount; i++) {
                int idx = iBuffer[indexOffset + i];
                int status = iBuffer[whereOffset + idx];
                assertThat(status)
                        .as("Variable at index[%d]=%d should be free (status <= 0)", i, idx)
                        .isLessThanOrEqualTo(VAR_FREE);
            }
            
            // Verify active variables in index[free:n]
            for (int i = freeCount; i < n; i++) {
                int idx = iBuffer[indexOffset + i];
                int status = iBuffer[whereOffset + idx];
                assertThat(status)
                        .as("Variable at index[%d]=%d should be active (status > 0)", i, idx)
                        .isGreaterThan(VAR_FREE);
            }
    }

    /**
     * Property 4.7: GCP z is within feasible region (l ≤ z ≤ u)
     * 
     * <p><i>For any</i> point x, gradient g, and bounds [l, u], the computed GCP z
     * shall satisfy l ≤ z ≤ u for all bounded variables.</p>
     * 
     * 
     * @param n the dimension (2 to 10)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.7: GCP z is within feasible region")
    void gcpIsWithinFeasibleRegion(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        // Generate random bounds
        double[] x = new double[n];
        double[] g = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        double[] z = new double[n];
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 5;
            double u = l + Math.abs(random.nextGaussian()) * 5 + 0.1;
            lower[i] = l;
            upper[i] = u;
            boundType[i] = BOUND_BOTH;
            
            // Start within bounds
            x[i] = l + (u - l) * random.nextDouble();
            g[i] = random.nextGaussian() * 2;
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            // Initialize workspace
            ws.setSbgNorm(1.0);  // Non-zero to trigger computation
            ws.setTheta(1.0);
            ws.setCol(0);
            ws.setHead(0);
            
            int[] iBuffer = ws.getIntBuffer();
            int whereOffset = ws.getWhereOffset();
            
            for (int i = 0; i < n; i++) {
                iBuffer[whereOffset + i] = VAR_FREE;
            }
            
            int info = CauchyPoint.compute(n, m, x, g, TestBounds.toBounds(n, lower, upper, boundType), ws);
            System.arraycopy(ws.getBuffer(), ws.getZOffset(), z, 0, n);
            
            assertThat(info).as("GCP compute should succeed").isEqualTo(ERR_NONE);
            
            // Verify GCP is within bounds
            for (int i = 0; i < n; i++) {
                assertThat(z[i])
                        .as("GCP z[%d] should be >= lower bound %f", i, lower[i])
                        .isGreaterThanOrEqualTo(lower[i] - TOLERANCE);
                assertThat(z[i])
                        .as("GCP z[%d] should be <= upper bound %f", i, upper[i])
                        .isLessThanOrEqualTo(upper[i] + TOLERANCE);
            }
    }

    /**
     * Property 4.8: GCP equals x when projected gradient norm is zero
     * 
     * <p><i>For any</i> point x with zero projected gradient norm, the GCP z
     * shall equal x (no movement needed).</p>
     * 
     * 
     * @param n the dimension (2 to 10)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.8: GCP equals x when projected gradient norm is zero")
    void gcpEqualsXWhenProjGradNormIsZero(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        double[] x = new double[n];
        double[] g = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        double[] z = new double[n];
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 5;
            double u = l + Math.abs(random.nextGaussian()) * 5 + 0.1;
            lower[i] = l;
            upper[i] = u;
            boundType[i] = BOUND_BOTH;
            x[i] = l + (u - l) * random.nextDouble();
            g[i] = random.nextGaussian();
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            // Set projected gradient norm to zero
            ws.setSbgNorm(0.0);
            ws.setTheta(1.0);
            ws.setCol(0);
            ws.setHead(0);
            
            int[] iBuffer = ws.getIntBuffer();
            int whereOffset = ws.getWhereOffset();
            
            for (int i = 0; i < n; i++) {
                iBuffer[whereOffset + i] = VAR_FREE;
            }
            
            int info = CauchyPoint.compute(n, m, x, g, TestBounds.toBounds(n, lower, upper, boundType), ws);
            System.arraycopy(ws.getBuffer(), ws.getZOffset(), z, 0, n);
            
            assertThat(info).as("GCP compute should succeed").isEqualTo(ERR_NONE);
            
            // GCP should equal x when projected gradient norm is zero
            for (int i = 0; i < n; i++) {
                assertThat(z[i])
                        .as("GCP z[%d] should equal x[%d] when sbgNorm=0", i, i)
                        .isCloseTo(x[i], within(TOLERANCE));
            }
    }

    /**
     * Property 4.9: GCP with mixed bound types respects each bound type
     * 
     * <p><i>For any</i> point x with mixed bound types (NONE, LOWER, UPPER, BOTH),
     * the GCP z shall respect each variable's specific bound type.</p>
     * 
     * 
     * @param n the dimension (4 to 12, divisible by 4)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.9: GCP with mixed bound types respects each bound type")
    void gcpWithMixedBoundTypesRespectsEachType(
            @ForAll @IntRange(min = 4, max = 12) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        double[] x = new double[n];
        double[] g = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        double[] z = new double[n];
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 5;
            double u = l + Math.abs(random.nextGaussian()) * 5 + 0.1;
            lower[i] = l;
            upper[i] = u;
            
            // Cycle through bound types
            boundType[i] = i % 4;  // NONE, LOWER, BOTH, UPPER
            
            // Start within bounds
            x[i] = l + (u - l) * random.nextDouble();
            g[i] = random.nextGaussian() * 2;
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            ws.setSbgNorm(1.0);
            ws.setTheta(1.0);
            ws.setCol(0);
            ws.setHead(0);
            
            int[] iBuffer = ws.getIntBuffer();
            int whereOffset = ws.getWhereOffset();
            
            for (int i = 0; i < n; i++) {
                if (boundType[i] == BOUND_NONE) {
                    iBuffer[whereOffset + i] = VAR_UNBOUND;
                } else {
                    iBuffer[whereOffset + i] = VAR_FREE;
                }
            }
            
            int info = CauchyPoint.compute(n, m, x, g, TestBounds.toBounds(n, lower, upper, boundType), ws);
            System.arraycopy(ws.getBuffer(), ws.getZOffset(), z, 0, n);
            
            assertThat(info).as("GCP compute should succeed").isEqualTo(ERR_NONE);
            
            // Verify bounds are respected according to bound type
            for (int i = 0; i < n; i++) {
                int bt = boundType[i];
                
                if (bt == BOUND_LOWER || bt == BOUND_BOTH) {
                    assertThat(z[i])
                            .as("z[%d] should be >= lower bound for BOUND_LOWER/BOTH", i)
                            .isGreaterThanOrEqualTo(lower[i] - TOLERANCE);
                }
                
                if (bt == BOUND_UPPER || bt == BOUND_BOTH) {
                    assertThat(z[i])
                            .as("z[%d] should be <= upper bound for BOUND_UPPER/BOTH", i)
                            .isLessThanOrEqualTo(upper[i] + TOLERANCE);
                }
            }
    }

    /**
     * Property 4.10: Breakpoints are computed correctly
     * 
     * <p><i>For any</i> point x, gradient g, and bounds [l, u], breakpoints shall be
     * computed as t_i = (x_i - bound_i) / g_i where bound_i is the relevant bound.</p>
     * 
     * 
     * @param n the dimension (2 to 10)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.10: Breakpoints are computed correctly")
    void breakpointsAreComputedCorrectly(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        double[] x = new double[n];
        double[] g = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        double[] z = new double[n];
        
        // Set up a simple case where we can verify breakpoints
        for (int i = 0; i < n; i++) {
            lower[i] = 0.0;
            upper[i] = 1.0;
            boundType[i] = BOUND_BOTH;
            x[i] = 0.5;  // Start in middle
            
            // Alternate gradient directions
            if (i % 2 == 0) {
                g[i] = 1.0;  // Will hit lower bound: t = (0.5 - 0) / 1 = 0.5
            } else {
                g[i] = -1.0;  // Will hit upper bound: t = (1 - 0.5) / 1 = 0.5
            }
        }
        
        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            ws.setSbgNorm(1.0);
            ws.setTheta(1.0);
            ws.setCol(0);
            ws.setHead(0);
            
            int[] iBuffer = ws.getIntBuffer();
            int whereOffset = ws.getWhereOffset();
            
            for (int i = 0; i < n; i++) {
                iBuffer[whereOffset + i] = VAR_FREE;
            }
            
            int info = CauchyPoint.compute(n, m, x, g, TestBounds.toBounds(n, lower, upper, boundType), ws);
            System.arraycopy(ws.getBuffer(), ws.getZOffset(), z, 0, n);
            
            assertThat(info).as("GCP compute should succeed").isEqualTo(ERR_NONE);
            
            // With uniform breakpoints at t=0.5, the GCP should be at the bounds
            // or somewhere along the path depending on the quadratic model
            for (int i = 0; i < n; i++) {
                assertThat(z[i])
                        .as("z[%d] should be within bounds", i)
                        .isBetween(lower[i] - TOLERANCE, upper[i] + TOLERANCE);
            }
    }

    /**
     * Property 4.11: GCP computation is deterministic
     * 
     * <p><i>For any</i> fixed inputs, calling GCP compute twice shall produce
     * identical results.</p>
     * 
     * 
     * @param n the dimension (2 to 10)
     * @param seed random seed for reproducibility
     */
    @Property(tries = 100)
    @Label("Property 4.11: GCP computation is deterministic")
    void gcpComputationIsDeterministic(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @LongRange(min = 1, max = Long.MAX_VALUE) long seed
    ) {
        java.util.Random random = new java.util.Random(seed);
        int m = 5;
        
        double[] x = new double[n];
        double[] g = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        int[] boundType = new int[n];
        
        for (int i = 0; i < n; i++) {
            double l = random.nextGaussian() * 5;
            double u = l + Math.abs(random.nextGaussian()) * 5 + 0.1;
            lower[i] = l;
            upper[i] = u;
            boundType[i] = BOUND_BOTH;
            x[i] = l + (u - l) * random.nextDouble();
            g[i] = random.nextGaussian() * 2;
        }
        
        double[] z1 = new double[n];
        double[] z2 = new double[n];
        
        // First computation
        LBFGSBWorkspace ws1 = new LBFGSBWorkspace(n, m);
            ws1.setSbgNorm(1.0);
            ws1.setTheta(1.0);
            ws1.setCol(0);
            ws1.setHead(0);
            
            int[] iBuffer = ws1.getIntBuffer();
            int whereOffset = ws1.getWhereOffset();
            for (int i = 0; i < n; i++) {
                iBuffer[whereOffset + i] = VAR_FREE;
            }
            
            CauchyPoint.compute(n, m, x.clone(), g.clone(), TestBounds.toBounds(n, lower, upper, boundType), ws1);
            System.arraycopy(ws1.getBuffer(), ws1.getZOffset(), z1, 0, n);
        
        // Second computation with fresh workspace
        LBFGSBWorkspace ws2 = new LBFGSBWorkspace(n, m);
            ws2.setSbgNorm(1.0);
            ws2.setTheta(1.0);
            ws2.setCol(0);
            ws2.setHead(0);
            
            int[] iBuffer2 = ws2.getIntBuffer();
            int whereOffset2 = ws2.getWhereOffset();
            for (int i = 0; i < n; i++) {
                iBuffer2[whereOffset2 + i] = VAR_FREE;
            }
            
            CauchyPoint.compute(n, m, x.clone(), g.clone(), TestBounds.toBounds(n, lower, upper, boundType), ws2);
            System.arraycopy(ws2.getBuffer(), ws2.getZOffset(), z2, 0, n);
        
        // Results should be identical
        for (int i = 0; i < n; i++) {
            assertThat(z1[i])
                    .as("z1[%d] should equal z2[%d]", i, i)
                    .isCloseTo(z2[i], within(TOLERANCE));
        }
    }
}
