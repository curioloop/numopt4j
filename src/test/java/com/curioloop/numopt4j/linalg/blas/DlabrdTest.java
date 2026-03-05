/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlabrdTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        int m = 4, n = 5, nb = 2;
        double[] A = {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        };
        double[] d = new double[nb];
        double[] e = new double[nb];
        double[] tauQ = new double[nb];
        double[] tauP = new double[nb];
        double[] x = new double[m * nb];
        double[] y = new double[n * nb];

        Dlabrd.dlabrd(m, n, nb, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, x, 0, nb, y, 0, nb);

        assertTrue(Double.isFinite(d[0]));
        assertTrue(Double.isFinite(tauQ[0]));
    }

    @Test
    void testSquare() {
        int m = 5, n = 5, nb = 2;
        double[] A = {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25
        };
        double[] d = new double[nb];
        double[] e = new double[nb];
        double[] tauQ = new double[nb];
        double[] tauP = new double[nb];
        double[] x = new double[m * nb];
        double[] y = new double[n * nb];

        Dlabrd.dlabrd(m, n, nb, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, x, 0, nb, y, 0, nb);

        assertTrue(Double.isFinite(d[0]));
        assertTrue(Double.isFinite(tauQ[0]));
    }

    @Test
    void testEmpty() {
        Dlabrd.dlabrd(0, 0, 0, new double[0], 0, 0,
                new double[0], 0, new double[0], 0,
                new double[0], 0, new double[0], 0,
                new double[0], 0, 0, new double[0], 0, 0);
    }

    @Test
    void testNbZero() {
        int m = 3, n = 3, nb = 0;
        double[] A = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        Dlabrd.dlabrd(m, n, nb, A, 0, n,
                new double[0], 0, new double[0], 0,
                new double[0], 0, new double[0], 0,
                new double[0], 0, 0, new double[0], 0, 0);
    }

    @Test
    void testMgeN_NbEqualsN() {
        int m = 5, n = 4, nb = 4;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random();
        }
        double[] d = new double[nb];
        double[] e = new double[nb];
        double[] tauQ = new double[nb];
        double[] tauP = new double[nb];
        double[] x = new double[m * nb];
        double[] y = new double[n * nb];

        Dlabrd.dlabrd(m, n, nb, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, x, 0, nb, y, 0, nb);

        for (int i = 0; i < nb; i++) {
            assertTrue(Double.isFinite(d[i]));
        }
    }

    @Test
    void testMltN_NbEqualsM() {
        int m = 4, n = 5, nb = 4;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random();
        }
        double[] d = new double[nb];
        double[] e = new double[nb];
        double[] tauQ = new double[nb];
        double[] tauP = new double[nb];
        double[] x = new double[m * nb];
        double[] y = new double[n * nb];

        Dlabrd.dlabrd(m, n, nb, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, x, 0, nb, y, 0, nb);

        for (int i = 0; i < nb; i++) {
            assertTrue(Double.isFinite(d[i]));
        }
    }

    @Test
    void testSingleBlock() {
        int m = 4, n = 5, nb = 4;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random();
        }
        double[] d = new double[nb];
        double[] e = new double[nb];
        double[] tauQ = new double[nb];
        double[] tauP = new double[nb];
        double[] x = new double[m * nb];
        double[] y = new double[n * nb];

        Dlabrd.dlabrd(m, n, nb, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, x, 0, nb, y, 0, nb);

        assertTrue(Double.isFinite(d[0]));
        assertTrue(Double.isFinite(d[1]));
    }

    @Test
    void testLargeMatrix() {
        int m = 10, n = 12, nb = 3;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random();
        }
        double[] d = new double[nb];
        double[] e = new double[nb];
        double[] tauQ = new double[nb];
        double[] tauP = new double[nb];
        double[] x = new double[m * nb];
        double[] y = new double[n * nb];

        Dlabrd.dlabrd(m, n, nb, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, x, 0, nb, y, 0, nb);

        for (int i = 0; i < nb; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }
}
