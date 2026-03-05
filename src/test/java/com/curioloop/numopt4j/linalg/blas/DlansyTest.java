/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlansyTest {

    private static final double TOL = 1e-10;

    @Test
    void testFrobeniusNormUpper() {
        double[] A = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };

        double norm = Dlansy.dlansy('F', BLAS.Uplo.Upper, 3, A, 3, new double[3]);

        assertTrue(norm > 0);
    }

    @Test
    void testFrobeniusNormLower() {
        double[] A = {
            1, 0, 0,
            2, 4, 0,
            3, 5, 6
        };

        double norm = Dlansy.dlansy('F', BLAS.Uplo.Lower, 3, A, 3, new double[3]);

        assertTrue(norm > 0);
    }

    @Test
    void testEmpty() {
        double norm = Dlansy.dlansy('F', BLAS.Uplo.Upper, 0, new double[0], 0, new double[0]);
        assertEquals(0.0, norm, TOL);
    }
}
