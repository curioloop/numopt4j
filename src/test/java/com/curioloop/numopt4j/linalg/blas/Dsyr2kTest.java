/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dsyr2kTest {

    private static final double TOL = 1e-10;

    @Test
    void testUpperNoTrans() {
        // C := alpha*A*B^T + alpha*B*A^T + beta*C
        // A is n x k = 3 x 2, B is n x k = 3 x 2
        double[] A = {
            1, 2,
            3, 4,
            5, 6
        };
        double[] B = {
            7, 8,
            9, 10,
            11, 12
        };
        double[] C = new double[9];

        Dsyr2k.dsyr2k(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, 3, 2, 1.0, A, 2, B, 2, 0.0, C, 3);

        // Expected: C[i,j] = A[i,0]*B[j,0] + A[i,1]*B[j,1] + B[i,0]*A[j,0] + B[i,1]*A[j,1]
        // C[0,0] = 1*7 + 2*8 + 7*1 + 8*2 = 7 + 16 + 7 + 16 = 46
        assertEquals(46.0, C[0], TOL);
        // C[0,1] = 1*9 + 2*10 + 7*3 + 8*4 = 9 + 20 + 21 + 32 = 82
        assertEquals(82.0, C[1], TOL);
        // C[0,2] = 1*11 + 2*12 + 7*5 + 8*6 = 11 + 24 + 35 + 48 = 118
        assertEquals(118.0, C[2], TOL);
        // C[1,1] = 3*9 + 4*10 + 9*3 + 10*4 = 27 + 40 + 27 + 40 = 134
        assertEquals(134.0, C[4], TOL);
    }

    @Test
    void testLowerNoTrans() {
        double[] A = {
            1, 2,
            3, 4,
            5, 6
        };
        double[] B = {
            7, 8,
            9, 10,
            11, 12
        };
        double[] C = new double[9];

        Dsyr2k.dsyr2k(BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, 3, 2, 1.0, A, 2, B, 2, 0.0, C, 3);

        assertEquals(46.0, C[0], TOL);
        assertEquals(82.0, C[3], TOL);
        assertEquals(118.0, C[6], TOL);
        assertEquals(134.0, C[4], TOL);
    }

    @Test
    void testUpperTrans() {
        // C := alpha*A^T*B + alpha*B^T*A + beta*C
        // A is k x n = 2 x 3, B is k x n = 2 x 3
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] B = {
            7, 8, 9,
            10, 11, 12
        };
        double[] C = new double[9];

        Dsyr2k.dsyr2k(BLAS.Uplo.Upper, BLAS.Transpose.Trans, 3, 2, 1.0, A, 3, B, 3, 0.0, C, 3);

        // Expected: C[i,j] = A[0,i]*B[0,j] + A[1,i]*B[1,j] + B[0,i]*A[0,j] + B[1,i]*A[1,j]
        // C[0,0] = 1*7 + 4*10 + 7*1 + 10*4 = 7 + 40 + 7 + 40 = 94
        assertEquals(94.0, C[0], TOL);
        // C[0,1] = 1*8 + 4*11 + 7*2 + 10*5 = 8 + 44 + 14 + 50 = 116
        assertEquals(116.0, C[1], TOL);
    }

    @Test
    void testEmpty() {
        Dsyr2k.dsyr2k(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, 0, 0, 1.0, new double[0], 0, 0, new double[0], 0, 0, 0.0, new double[0], 0, 0);
    }

    @Test
    void testBeta() {
        double[] A = {1, 2, 3, 4, 5, 6};
        double[] B = {7, 8, 9, 10, 11, 12};
        double[] C = new double[9];
        for (int i = 0; i < 9; i++) C[i] = 1.0;

        Dsyr2k.dsyr2k(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, 3, 2, 1.0, A, 2, B, 2, 0.5, C, 3);

        // C[0,0] = 0.5*1 + 46 = 46.5
        assertEquals(46.5, C[0], TOL);
    }
}
