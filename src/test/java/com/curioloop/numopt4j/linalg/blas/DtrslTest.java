/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DtrslTest {

    @Test
    void testUpper() {
        double[] A = {
            2, 1, 0,
            0, 3, 1,
            0, 0, 4
        };
        double[] b = {3, 7, 4};
        int n = 3;

        int info = Dtrsl.dtrsl(A, 0, n, n, b, 0, BLAS.Uplo.Upper, BLAS.Trans.NoTrans);

        assertEquals(0, info);
        assertArrayEquals(new double[] {0.5, 2.0, 1.0}, b, 1e-12);
    }

    @Test
    void testLower() {
        double[] A = {
            2, 0, 0,
            1, 3, 0,
            0, 1, 4
        };
        double[] b = {2, 4, 8};
        int n = 3;

        int info = Dtrsl.dtrsl(A, 0, n, n, b, 0, BLAS.Uplo.Lower, BLAS.Trans.NoTrans);

        assertEquals(0, info);
        assertArrayEquals(new double[] {1.0, 1.0, 1.75}, b, 1e-12);
    }

    @Test
    void testEmpty() {
        int info = Dtrsl.dtrsl(new double[0], 0, 0, 0, new double[0], 0, BLAS.Uplo.Upper, BLAS.Trans.NoTrans);
        assertEquals(0, info);
    }

    @Test
    void testUnitDiagonalUpperNoTransIgnoresStoredDiagonal() {
        double[] A = {
            7.0, 0.25, -0.5,
            0.0, 8.0, 0.75,
            0.0, 0.0, 9.0
        };
        double[] b = {-1.0, 0.25, 3.0};

        int info = Dtrsl.dtrsl(A, 0, 3, 3, b, 0, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit);

        assertEquals(0, info);
        assertArrayEquals(new double[] {1.0, -2.0, 3.0}, b, 1e-12);
    }

    @Test
    void testUnitDiagonalUpperTransIgnoresStoredDiagonal() {
        double[] A = {
            7.0, 0.25, -0.5,
            0.0, 8.0, 0.75,
            0.0, 0.0, 9.0
        };
        double[] b = {1.0, -1.75, 1.0};

        int info = Dtrsl.dtrsl(A, 0, 3, 3, b, 0, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit);

        assertEquals(0, info);
        assertArrayEquals(new double[] {1.0, -2.0, 3.0}, b, 1e-12);
    }

    @Test
    void testUnitDiagonalLowerNoTransIgnoresStoredDiagonal() {
        double[] A = {
            7.0, 0.0, 0.0,
            0.25, 8.0, 0.0,
            -0.5, 0.75, 9.0
        };
        double[] b = {1.0, -1.75, 1.0};

        int info = Dtrsl.dtrsl(A, 0, 3, 3, b, 0, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit);

        assertEquals(0, info);
        assertArrayEquals(new double[] {1.0, -2.0, 3.0}, b, 1e-12);
    }

    @Test
    void testUnitDiagonalLowerTransIgnoresStoredDiagonal() {
        double[] A = {
            7.0, 0.0, 0.0,
            0.25, 8.0, 0.0,
            -0.5, 0.75, 9.0
        };
        double[] b = {-1.0, 0.25, 3.0};

        int info = Dtrsl.dtrsl(A, 0, 3, 3, b, 0, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit);

        assertEquals(0, info);
        assertArrayEquals(new double[] {1.0, -2.0, 3.0}, b, 1e-12);
    }

    @Test
    void testRandomParityAllVariants() {
        BLAS.Uplo[] uplos = {BLAS.Uplo.Upper, BLAS.Uplo.Lower};
        BLAS.Trans[] transposes = {BLAS.Trans.NoTrans, BLAS.Trans.Trans};
        BLAS.Diag[] diags = {BLAS.Diag.NonUnit, BLAS.Diag.Unit};
        int[] sizes = {1, 2, 7, 33};

        for (BLAS.Uplo uplo : uplos) {
            for (BLAS.Trans trans : transposes) {
                for (BLAS.Diag diag : diags) {
                    for (int n : sizes) {
                        for (int caseIndex = 0; caseIndex < 12; caseIndex++) {
                            verifyParity(uplo, trans, diag, n, 20260423L + 97L * caseIndex + 13L * n);
                        }
                    }
                }
            }
        }
    }

    private static void verifyParity(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, int n, long seed) {
        int ldt = n + 2;
        double[] matrix = new double[n * ldt];
        double[] baseline = new double[n];
        double[] current = new double[n];
        Random random = new Random(seed + uplo.ordinal() * 31L + trans.ordinal() * 131L + diag.ordinal() * 521L);

        fillTriangular(random, matrix, uplo, n, ldt);
        fillVector(random, baseline);
        System.arraycopy(baseline, 0, current, 0, n);

        int expectedInfo = BlasTestSupport.scalarDtrsl(matrix, 0, ldt, n, baseline, 0, uplo, trans, diag);
        int actualInfo = Dtrsl.dtrsl(matrix, 0, ldt, n, current, 0, uplo, trans, diag);

        assertEquals(expectedInfo, actualInfo, () -> "Info mismatch for " + uplo + "/" + trans + "/" + diag + " n=" + n);
        assertVectorClose(baseline, current, 1e-11);
    }

    private static void fillTriangular(Random random, double[] matrix, BLAS.Uplo uplo, int n, int ldt) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        for (int row = 0; row < n; row++) {
            double rowSum = 0.0;
            for (int col = 0; col < n; col++) {
                boolean used = upper ? col >= row : col <= row;
                if (!used) {
                    matrix[row * ldt + col] = 0.0;
                    continue;
                }
                if (row == col) {
                    continue;
                }
                double value = (random.nextDouble() - 0.5) * 0.3;
                matrix[row * ldt + col] = value;
                rowSum += Math.abs(value);
            }
            matrix[row * ldt + row] = rowSum + 1.25 + random.nextDouble();
        }
    }

    private static void fillVector(Random random, double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (random.nextDouble() - 0.5) * 2.0;
        }
    }

    private static void assertVectorClose(double[] expected, double[] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            final int index = i;
            double diff = Math.abs(expected[i] - actual[i]);
            double scale = Math.max(1.0, Math.max(Math.abs(expected[i]), Math.abs(actual[i])));
            assertTrue(diff <= tolerance * scale,
                () -> "Mismatch at index " + index + ": expected=" + expected[index] + ", actual=" + actual[index]);
        }
    }
}
