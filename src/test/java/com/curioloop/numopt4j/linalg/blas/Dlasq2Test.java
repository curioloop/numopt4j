/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dlasq2Test {

    private static final double TOL = 1e-10;

    @Test
    void testSingleElement() {
        double[] z = {0, 5, 0};
        int[] istate = new int[7];
        double[] dstate = new double[17];
        int info = Dlasq2.dlasq2(1, z, 0, istate, 0, dstate, 0);
        assertEquals(0, info);
    }

    @Test
    void testEmpty() {
        double[] z = new double[0];
        int[] istate = new int[7];
        double[] dstate = new double[17];
        int info = Dlasq2.dlasq2(0, z, 0, istate, 0, dstate, 0);
        assertEquals(0, info);
    }

    @Test
    void testTwoElementsIdentity() {
        double[] z = new double[4 * 2];
        z[0] = 1;
        z[1] = 0;
        z[2] = 1;
        int[] istate = new int[7];
        double[] dstate = new double[17];

        int info = Dlasq2.dlasq2(2, z, 0, istate, 0, dstate, 0);

        assertEquals(0, info);
    }

    @Test
    void testZeroOffDiagonal() {
        int n = 5;
        double[] z = new double[4 * n];
        double[] diag = {5, 4, 3, 2, 1};
        for (int i = 0; i < n; i++) {
            z[2 * i] = diag[i];
        }
        int[] istate = new int[7];
        double[] dstate = new double[17];

        int info = Dlasq2.dlasq2(n, z, 0, istate, 0, dstate, 0);

        assertEquals(0, info);
        for (int i = 0; i < n - 1; i++) {
            assertTrue(z[i] >= z[i + 1], "Eigenvalues should be sorted");
        }
    }

    @Test
    void testIdentityMatrix() {
        int n = 5;
        double[] z = new double[4 * n];
        for (int i = 0; i < n; i++) {
            z[2 * i] = 1.0;
        }
        int[] istate = new int[7];
        double[] dstate = new double[17];

        int info = Dlasq2.dlasq2(n, z, 0, istate, 0, dstate, 0);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(1.0, z[i], TOL);
        }
    }

    @Test
    void testDiagonalMatrix() {
        int n = 6;
        double[] z = new double[4 * n];
        double[] diag = {6, 5, 4, 3, 2, 1};
        for (int i = 0; i < n; i++) {
            z[2 * i] = diag[i];
        }
        int[] istate = new int[7];
        double[] dstate = new double[17];

        int info = Dlasq2.dlasq2(n, z, 0, istate, 0, dstate, 0);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(diag[i], z[i], TOL);
        }
    }
}
