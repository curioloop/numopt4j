/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DpttrfTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic3x3() {
        // A = [4 1 0; 1 4 1; 0 1 4] — SPD tridiagonal
        double[] d = {4, 4, 4};
        double[] e = {1, 1};

        boolean ok = Dptsv.dpttrf(3, d, 0, e, 0);
        assertTrue(ok);

        // Verify L*D*Lt = A by reconstructing
        // L is unit lower bidiagonal with subdiag e (after factorization)
        // D is diagonal d (after factorization)
        // A[i][i]   = d[i] + e[i-1]^2 * d[i-1]  (for i>0, before factorization)
        // A[i][i+1] = e[i] * d[i]
        // Check A[0][1] = e[0]*d[0] = 1
        assertEquals(1.0, e[0] * d[0], TOL);
        // Check A[1][2] = e[1]*d[1] = 1
        assertEquals(1.0, e[1] * d[1], TOL);
    }

    @Test
    void testNotPositiveDefinite() {
        double[] d = {4, -1, 4};
        double[] e = {1, 1};
        boolean ok = Dptsv.dpttrf(3, d, 0, e, 0);
        assertFalse(ok);
    }

    @Test
    void testSingleElement() {
        double[] d = {5};
        boolean ok = Dptsv.dpttrf(1, d, 0, new double[0], 0);
        assertTrue(ok);
        assertEquals(5.0, d[0], TOL);
    }

    @Test
    void testEmpty() {
        boolean ok = Dptsv.dpttrf(0, new double[0], 0, new double[0], 0);
        assertTrue(ok);
    }

    @Test
    void testZeroDiagonal() {
        double[] d = {0, 4};
        double[] e = {1};
        boolean ok = Dptsv.dpttrf(2, d, 0, e, 0);
        assertFalse(ok);
    }

    @Test
    void testLarger() {
        int n = 8;
        double[] d = new double[n];
        double[] e = new double[n - 1];
        for (int i = 0; i < n; i++) d[i] = 4;
        for (int i = 0; i < n - 1; i++) e[i] = 1;

        boolean ok = Dptsv.dpttrf(n, d, 0, e, 0);
        assertTrue(ok);
        // All diagonal elements of D must be positive
        for (int i = 0; i < n; i++) assertTrue(d[i] > 0, "d[" + i + "] = " + d[i]);
    }
}
