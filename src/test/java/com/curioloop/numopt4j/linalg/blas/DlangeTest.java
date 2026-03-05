/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlangeTest {

    private static final double TOL = 1e-14;

    @Test
    void testMaxNorm() {
        double[] A = {
            1, -2, 3,
            -4, 5, -6,
            7, -8, 9
        };
        int n = 3;

        double norm = Dlange.dlange('M', n, n, A, 0, n);
        assertEquals(9.0, norm, TOL);
    }

    @Test
    void testOneNorm() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int n = 3;

        double norm = Dlange.dlange('1', n, n, A, 0, n);
        assertEquals(18.0, norm, TOL);
    }

    @Test
    void testInfNorm() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int n = 3;

        double norm = Dlange.dlange('I', n, n, A, 0, n);
        assertEquals(24.0, norm, TOL);
    }

    @Test
    void testFrobeniusNorm() {
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        int n = 3;

        double norm = Dlange.dlange('F', n, n, A, 0, n);
        assertEquals(Math.sqrt(14), norm, TOL);
    }

    @Test
    void testZeroMatrix() {
        double[] A = new double[9];
        int n = 3;

        assertEquals(0.0, Dlange.dlange('M', n, n, A, 0, n), TOL);
        assertEquals(0.0, Dlange.dlange('1', n, n, A, 0, n), TOL);
        assertEquals(0.0, Dlange.dlange('I', n, n, A, 0, n), TOL);
        assertEquals(0.0, Dlange.dlange('F', n, n, A, 0, n), TOL);
    }

    @Test
    void testNonSquare() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        int m = 2, n = 3;

        double maxNorm = Dlange.dlange('M', m, n, A, 0, n);
        assertEquals(6.0, maxNorm, TOL);

        double oneNorm = Dlange.dlange('1', m, n, A, 0, n);
        assertEquals(9.0, oneNorm, TOL);

        double infNorm = Dlange.dlange('I', m, n, A, 0, n);
        assertEquals(15.0, infNorm, TOL);
    }

    @Test
    void testEmptyMatrix() {
        assertEquals(0.0, Dlange.dlange('M', 0, 0, new double[0], 0, 0), TOL);
        assertEquals(0.0, Dlange.dlange('1', 0, 3, new double[0], 0, 3), TOL);
        assertEquals(0.0, Dlange.dlange('I', 3, 0, new double[0], 0, 0), TOL);
    }
}
