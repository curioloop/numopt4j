/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DtrsmTest {

    private static final double TOL = 1e-13;

    @Test
    void testRightLowerNoTransNonUnit() {
        int m = 4, n = 3;
        double alpha = 3.0;
        
        double[] A = {
            2, 0, 0,
            3, 5, 0,
            4, 6, 8
        };
        
        double[] B = {
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21
        };
        
        double[] expected = {
            4.2, 1.2, 4.5,
            5.775, 1.65, 5.625,
            7.35, 2.1, 6.75,
            8.925, 2.55, 7.875
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, n, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testRightLowerNoTransUnit() {
        int m = 4, n = 3;
        double alpha = 3.0;
        
        double[] A = {
            2, 0, 0,
            3, 5, 0,
            4, 6, 8
        };
        
        double[] B = {
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21
        };
        
        double[] expected = {
            435, -183, 36,
            543, -228, 45,
            651, -273, 54,
            759, -318, 63
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, n, alpha, A, 0, n, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testRightUpperNoTransNonUnit() {
        int m = 4, n = 3;
        double alpha = 3.0;
        
        double[] A = {
            2, 3, 4,
            0, 5, 6,
            0, 0, 7
        };
        
        double[] B = {
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21
        };
        
        double[] expected = {
            15, -2.4, -48.0 / 35,
            19.5, -3.3, -66.0 / 35,
            24, -4.2, -2.4,
            28.5, -5.1, -102.0 / 35
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, n, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testRightUpperNoTransUnit() {
        int m = 4, n = 3;
        double alpha = 3.0;
        
        double[] A = {
            2, 3, 4,
            0, 5, 6,
            0, 0, 8
        };
        
        double[] B = {
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21
        };
        
        double[] expected = {
            30, -57, 258,
            39, -75, 339,
            48, -93, 420,
            57, -111, 501
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, n, alpha, A, 0, n, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testLeftUpperNoTransNonUnit() {
        int m = 3, n = 2;
        double alpha = 2.0;
        
        double[] A = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 5
        };
        
        double[] B = {
            3, 6,
            4, 7,
            5, 8
        };
        
        double[] expected = {
            1, 3.4,
            -0.5, -0.5,
            2, 3.2
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, m, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testLeftLowerNoTransNonUnit() {
        int m = 3, n = 2;
        double alpha = 3.0;
        
        double[] A = {
            2, 0, 0,
            3, 4, 0,
            5, 6, 7
        };
        
        double[] B = {
            3, 6,
            4, 7,
            5, 8
        };
        
        double[] expected = {
            4.5, 9,
            -0.375, -1.5,
            -0.75, -12.0 / 7
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, m, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testRightLowerTransNonUnit() {
        int m = 4, n = 3;
        double alpha = 3.0;
        
        double[] A = {
            2, 0, 0,
            3, 5, 0,
            4, 6, 8
        };
        
        double[] B = {
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21
        };
        
        double[] expected = {
            15, -2.4, -1.2,
            19.5, -3.3, -1.65,
            24, -4.2, -2.1,
            28.5, -5.1, -2.55
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, n, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testRightUpperTransNonUnit() {
        int m = 4, n = 3;
        double alpha = 3.0;
        
        double[] A = {
            2, 3, 4,
            0, 5, 6,
            0, 0, 8
        };
        
        double[] B = {
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21
        };
        
        double[] expected = {
            4.2, 1.2, 4.5,
            5.775, 1.65, 5.625,
            7.35, 2.1, 6.75,
            8.925, 2.55, 7.875
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, n, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testRightLowerNoTransWithLda() {
        int m = 2, n = 3;
        int lda = 5;
        int ldb = 4;
        double alpha = 3.0;
        
        double[] A = new double[3 * lda];
        A[0 * lda + 0] = 2;
        A[1 * lda + 0] = 3; A[1 * lda + 1] = 5;
        A[2 * lda + 0] = 4; A[2 * lda + 1] = 6; A[2 * lda + 2] = 8;
        
        double[] B = new double[m * ldb];
        B[0 * ldb + 0] = 10; B[0 * ldb + 1] = 11; B[0 * ldb + 2] = 12;
        B[1 * ldb + 0] = 13; B[1 * ldb + 1] = 14; B[1 * ldb + 2] = 15;
        
        double[] expected = {
            4.2, 1.2, 4.5,
            5.775, 1.65, 5.625
        };
        
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, lda, B, 0, ldb);
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals(expected[i * n + j], B[i * ldb + j], TOL, 
                    "Mismatch at (" + i + "," + j + ")");
            }
        }
    }

    @Test
    void testAlphaZero() {
        int m = 2, n = 3;
        double alpha = 0.0;
        
        double[] A = {
            2, 0, 0,
            3, 5, 0,
            4, 6, 8
        };
        
        double[] B = {
            10, 11, 12,
            13, 14, 15
        };
        
        double[] expected = new double[m * n];
        
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, n, B, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], B[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testLeftUpperNoTransUnit() {
        int m = 3, n = 2;
        double alpha = 2.0;
        
        double[] A = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 5
        };
        
        double[] B = {
            3, 6,
            4, 7,
            5, 8
        };
        
        double[] expected = {
            60, 96,
            -42, -66,
            10, 16
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, n, alpha, A, 0, m, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testLeftUpperTransNonUnit() {
        int m = 3, n = 2;
        double alpha = 3.0;
        
        double[] A = {
            2, 3, 4,
            0, 5, 6,
            0, 0, 7
        };
        
        double[] B = {
            3, 6,
            4, 7,
            5, 8
        };
        
        double[] expected = {
            4.5, 9,
            -0.30, -1.2,
            -6.0 / 35, -24.0 / 35
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, m, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testLeftUpperTransUnit() {
        int m = 3, n = 2;
        double alpha = 3.0;
        
        double[] A = {
            2, 3, 4,
            0, 5, 6,
            0, 0, 7
        };
        
        double[] B = {
            3, 6,
            4, 7,
            5, 8
        };
        
        double[] expected = {
            9, 18,
            -15, -33,
            69, 150
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, m, n, alpha, A, 0, m, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testLeftLowerNoTransUnit() {
        int m = 3, n = 2;
        double alpha = 3.0;
        
        double[] A = {
            2, 0, 0,
            3, 4, 0,
            5, 6, 7
        };
        
        double[] B = {
            3, 6,
            4, 7,
            5, 8
        };
        
        double[] expected = {
            9, 18,
            -15, -33,
            60, 132
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, n, alpha, A, 0, m, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testLeftLowerTransNonUnit() {
        int m = 3, n = 2;
        double alpha = 3.0;
        
        double[] A = {
            2, 0, 0,
            3, 4, 0,
            5, 6, 8
        };
        
        double[] B = {
            3, 6,
            4, 7,
            5, 8
        };
        
        double[] expected = {
            -0.46875, 0.375,
            0.1875, 0.75,
            1.875, 3
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, m, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testLeftLowerTransUnit() {
        int m = 3, n = 2;
        double alpha = 3.0;
        
        double[] A = {
            2, 0, 0,
            3, 4, 0,
            5, 6, 8
        };
        
        double[] B = {
            3, 6,
            4, 7,
            5, 8
        };
        
        double[] expected = {
            168, 267,
            -78, -123,
            15, 24
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, m, n, alpha, A, 0, m, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testRightUpperTransUnit() {
        int m = 4, n = 3;
        double alpha = 3.0;
        
        double[] A = {
            2, 3, 4,
            0, 5, 6,
            0, 0, 8
        };
        
        double[] B = {
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21
        };
        
        double[] expected = {
            435, -183, 36,
            543, -228, 45,
            651, -273, 54,
            759, -318, 63
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, m, n, alpha, A, 0, n, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testRightLowerTransUnit() {
        int m = 4, n = 3;
        double alpha = 3.0;
        
        double[] A = {
            2, 0, 0,
            3, 5, 0,
            4, 6, 8
        };
        
        double[] B = {
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21
        };
        
        double[] expected = {
            30, -57, 258,
            39, -75, 339,
            48, -93, 420,
            57, -111, 501
        };
        
        double[] Bcopy = B.clone();
        Dtrsm.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, m, n, alpha, A, 0, n, Bcopy, 0, n);
        
        for (int i = 0; i < m * n; i++) {
            assertEquals(expected[i], Bcopy[i], TOL, "Mismatch at index " + i);
        }
    }

    @Test
    void testLeftWideRandomParity() {
        verifyRandomLeftParity(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, 24, 160, 29, 167);
        verifyRandomLeftParity(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, 24, 160, 29, 167);
        verifyRandomLeftParity(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 40, 96, 44, 101);
        verifyRandomLeftParity(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, 40, 96, 44, 101);
    }

    @Test
    void testRightWideRandomParity() {
        verifyRandomRightParity(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 128, 64, 69, 71);
        verifyRandomRightParity(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 128, 64, 69, 71);
        verifyRandomRightParity(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, 160, 24, 29, 31);
        verifyRandomRightParity(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, 160, 24, 29, 31);
        verifyRandomRightParity(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, 96, 40, 47, 53);
        verifyRandomRightParity(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, 96, 40, 47, 53);
    }

    private static void verifyRandomLeftParity(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                                               int m, int n, int lda, int ldb) {
        Random random = new Random(20260423L + 31L * uplo.ordinal() + 17L * trans.ordinal() + 13L * diag.ordinal() + m + n);
        double[] a = new double[m * lda];
        double[] baseline = new double[m * ldb];
        double[] current = new double[m * ldb];
        fillTriangular(random, a, uplo, diag, m, lda);
        fillRandom(random, baseline);
        System.arraycopy(baseline, 0, current, 0, baseline.length);

        BlasTestSupport.dtrsmScalar(BLAS.Side.Left, uplo, trans, diag, m, n, a, 0, lda, baseline, 0, ldb);
        Dtrsm.dtrsm(BLAS.Side.Left, uplo, trans, diag, m, n, 1.0, a, 0, lda, current, 0, ldb);
        assertArrayClose(baseline, current);
    }

    private static void verifyRandomRightParity(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                                                int m, int n, int lda, int ldb) {
        Random random = new Random(20260423L + 43L * uplo.ordinal() + 19L * trans.ordinal() + 11L * diag.ordinal() + m + n);
        double[] a = new double[n * lda];
        double[] baseline = new double[m * ldb];
        double[] current = new double[m * ldb];
        fillTriangular(random, a, uplo, diag, n, lda);
        fillRandom(random, baseline);
        System.arraycopy(baseline, 0, current, 0, baseline.length);

        BlasTestSupport.dtrsmScalar(BLAS.Side.Right, uplo, trans, diag, m, n, a, 0, lda, baseline, 0, ldb);
        Dtrsm.dtrsm(BLAS.Side.Right, uplo, trans, diag, m, n, 1.0, a, 0, lda, current, 0, ldb);
        assertArrayClose(baseline, current);
    }

    private static void fillTriangular(Random random, double[] a, BLAS.Uplo uplo, BLAS.Diag diag, int m, int lda) {
        for (int i = 0; i < m; i++) {
            int rowOff = i * lda;
            for (int j = 0; j < m; j++) {
                if (uplo == BLAS.Uplo.Upper ? j >= i : j <= i) {
                    a[rowOff + j] = random.nextDouble() - 0.5;
                } else {
                    a[rowOff + j] = 0.0;
                }
            }
            a[rowOff + i] = diag == BLAS.Diag.Unit ? 1.0 : a[rowOff + i] + m;
        }
    }

    private static void fillRandom(Random random, double[] values) {
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble() - 0.5;
        }
    }

    private static void assertArrayClose(double[] expected, double[] actual) {
        for (int i = 0; i < expected.length; i++) {
            double tolerance = Math.max(1e-12, Math.abs(expected[i]) * 1e-10);
            assertEquals(expected[i], actual[i], tolerance, "Mismatch at index " + i);
        }
    }
}
