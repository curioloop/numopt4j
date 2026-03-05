/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DsterfTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        int n = 3;
        double[] d = {4, 3, 2};
        double[] e = {1, 1};

        boolean ok = Dsterf.dsterf(n, d, 0, e, 0);

        assertTrue(ok);
        assertTrue(d[0] <= d[1]);
        assertTrue(d[1] <= d[2]);
    }

    @Test
    void testEmpty() {
        boolean ok = Dsterf.dsterf(0, new double[0], 0, new double[0], 0);
        assertTrue(ok);
    }

    @Test
    void testSingleElement() {
        double[] d = {5};
        double[] e = new double[0];

        boolean ok = Dsterf.dsterf(1, d, 0, e, 0);

        assertTrue(ok);
        assertEquals(5, d[0], TOL);
    }

    @Test
    void testIdentityMatrix() {
        int n = 4;
        double[] d = {1, 1, 1, 1};
        double[] e = {0, 0, 0};

        boolean ok = Dsterf.dsterf(n, d, 0, e, 0);

        assertTrue(ok);
        for (int i = 0; i < n; i++) {
            assertEquals(1.0, d[i], TOL);
        }
    }

    @Test
    void testNegativeEigenvalues() {
        double[] d = {-1, -2, -3};
        double[] e = {0, 0};

        boolean ok = Dsterf.dsterf(3, d, 0, e, 0);

        assertTrue(ok);
        assertTrue(d[0] <= d[1]);
        assertTrue(d[1] <= d[2]);
    }

    @Test
    void testRepeatedEigenvalues() {
        double[] d = {2, 2, 2};
        double[] e = {0, 0};

        boolean ok = Dsterf.dsterf(3, d, 0, e, 0);

        assertTrue(ok);
        for (int i = 0; i < 3; i++) {
            assertEquals(2.0, d[i], TOL);
        }
    }
}
