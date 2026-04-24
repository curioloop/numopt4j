/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

public class DtrmvTest {

    private static final double EPS = 1e-14;

    @Test
    void testDtrmvLowerNoTransStrideLda() {
        double[] A = {
            2, 0, 0, 0,
            1, 3, 0, 0,
            2, 1, 4, 0,
            0, 2, 1, 5
        };
        
        double[] x = {1, 2, 3, 4};
        
        Dtrmv.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 4, A, 0, 4, x, 0, 1);
        
        double[] expected = {
            2 * 1,
            1 * 1 + 3 * 2,
            2 * 1 + 1 * 2 + 4 * 3,
            0 * 1 + 2 * 2 + 1 * 3 + 5 * 4
        };
        
        for (int i = 0; i < 4; i++) {
            assertThat(x[i]).isCloseTo(expected[i], within(EPS));
        }
    }

    @Test
    void testDtrmvLowerNoTransWithStride() {
        double[] A = {
            2, 0, 0, 0,
            1, 3, 0, 0,
            2, 1, 4, 0,
            0, 2, 1, 5
        };
        
        double[] x = {0, 1, 0, 2, 0, 3, 0, 4};
        
        Dtrmv.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 4, A, 0, 4, x, 1, 2);
        
        double[] expected = {
            0, 2 * 1,
            0, 1 * 1 + 3 * 2,
            0, 2 * 1 + 1 * 2 + 4 * 3,
            0, 0 * 1 + 2 * 2 + 1 * 3 + 5 * 4
        };
        
        for (int i = 0; i < 8; i++) {
            assertThat(x[i]).isCloseTo(expected[i], within(EPS));
        }
    }

    @Test
    void testDtrmvSubmatrixLowerNoTrans() {
        double[] A = {
            0, 0, 0, 0, 0,
            0, 2, 0, 0, 0,
            0, 1, 3, 0, 0,
            0, 2, 1, 4, 0,
            0, 0, 0, 0, 0
        };
        
        double[] x = {0, 0, 1, 2, 3};
        
        Dtrmv.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 3, A, 6, 5, x, 2, 1);
        
        double[] expected = {
            0, 0,
            2 * 1,
            1 * 1 + 3 * 2,
            2 * 1 + 1 * 2 + 4 * 3
        };
        
        for (int i = 0; i < 5; i++) {
            assertThat(x[i]).isCloseTo(expected[i], within(EPS));
        }
    }

    @Test
    void testDtrmvSubmatrixLowerNoTransWithStride() {
        double[] A = {
            0, 0, 0, 0, 0,
            0, 2, 0, 0, 0,
            0, 1, 3, 0, 0,
            0, 2, 1, 4, 0,
            0, 0, 0, 0, 0
        };
        
        double[] x = {0, 0, 0, 1, 0, 2, 0, 3, 0};
        
        Dtrmv.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 3, A, 6, 5, x, 3, 2);
        
        double[] expected = {
            0, 0, 0,
            2 * 1,
            0, 1 * 1 + 3 * 2,
            0, 2 * 1 + 1 * 2 + 4 * 3,
            0
        };
        
        for (int i = 0; i < 9; i++) {
            assertThat(x[i]).isCloseTo(expected[i], within(EPS));
        }
    }

    @Test
    void testDtrmvUpperNoTrans() {
        double[] A = {
            2, 1, 2, 0,
            0, 3, 1, 2,
            0, 0, 4, 1,
            0, 0, 0, 5
        };
        
        double[] x = {1, 2, 3, 4};
        
        Dtrmv.dtrmv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 4, A, 0, 4, x, 0, 1);
        
        double[] expected = {
            2 * 1 + 1 * 2 + 2 * 3 + 0 * 4,
            3 * 2 + 1 * 3 + 2 * 4,
            4 * 3 + 1 * 4,
            5 * 4
        };
        
        for (int i = 0; i < 4; i++) {
            assertThat(x[i]).isCloseTo(expected[i], within(EPS));
        }
    }

    @Test
    void testDtrmvLowerTrans() {
        double[] A = {
            2, 0, 0, 0,
            1, 3, 0, 0,
            2, 1, 4, 0,
            0, 2, 1, 5
        };
        
        double[] x = {1, 2, 3, 4};
        
        Dtrmv.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, 4, A, 0, 4, x, 0, 1);
        
        double[] expected = {
            2 * 1 + 1 * 2 + 2 * 3 + 0 * 4,
            3 * 2 + 1 * 3 + 2 * 4,
            4 * 3 + 1 * 4,
            5 * 4
        };
        
        for (int i = 0; i < 4; i++) {
            assertThat(x[i]).isCloseTo(expected[i], within(EPS));
        }
    }

    @Test
    void testDtrmvUpperTrans() {
        double[] A = {
            2, 1, 2, 0,
            0, 3, 1, 2,
            0, 0, 4, 1,
            0, 0, 0, 5
        };
        
        double[] x = {1, 2, 3, 4};
        
        Dtrmv.dtrmv(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, 4, A, 0, 4, x, 0, 1);
        
        double[] expected = {
            2 * 1,
            1 * 1 + 3 * 2,
            2 * 1 + 1 * 2 + 4 * 3,
            0 * 1 + 2 * 2 + 1 * 3 + 5 * 4
        };
        
        for (int i = 0; i < 4; i++) {
            assertThat(x[i]).isCloseTo(expected[i], within(EPS));
        }
    }

    @Test
    void testUpperNoTransAliasingColumnViewParity() {
        verifyAliasingParity(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, 32, 33, 1, 33, 20260426L);
    }

    @Test
    void testLowerTransAliasingColumnViewParity() {
        verifyAliasingParity(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, 32, 33, 0, 33, 20260427L);
    }

    @Test
    void testUpperNoTransAliasingColumnViewNonUnitParity() {
        verifyAliasingParity(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 64, 65, 1, 65, 20260430L);
    }

    @Test
    void testLowerNoTransAliasingColumnViewNonUnitParity() {
        verifyAliasingParity(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 64, 65, 0, 65, 20260431L);
    }

    @Test
    void testUpperNoTransAdjacentRightColumnViewParity() {
        verifyAdjacentColumnViewParity(BLAS.Uplo.Upper, BLAS.Diag.NonUnit, 64, 65, 0, 20260501L);
    }

    @Test
    void testUpperNoTransAdjacentRightColumnViewUnitParity() {
        verifyAdjacentColumnViewParity(BLAS.Uplo.Upper, BLAS.Diag.Unit, 64, 65, 0, 20260503L);
    }

    @Test
    void testLowerNoTransAdjacentLeftColumnViewParity() {
        verifyAdjacentColumnViewParity(BLAS.Uplo.Lower, BLAS.Diag.NonUnit, 64, 65, 1, 20260502L);
    }

    @Test
    void testLowerNoTransAdjacentLeftColumnViewUnitParity() {
        verifyAdjacentColumnViewParity(BLAS.Uplo.Lower, BLAS.Diag.Unit, 64, 65, 1, 20260504L);
    }

    @Test
    void testUpperNoTransLargeParity() {
        verifyRandomParity(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, 128, 128, 1, 20260428L);
    }

    @Test
    void testLowerTransLargeParity() {
        verifyRandomParity(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, 128, 128, 1, 20260429L);
    }

    private static void verifyRandomParity(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                                           int n, int lda, int incX, long seed) {
        Random random = new Random(seed);
        double[] a = new double[n * lda];
        double[] expected = new double[1 + (n - 1) * incX];
        double[] current = new double[expected.length];

        fillTriangular(random, a, uplo, n, lda);
        fillRandom(random, expected);
        System.arraycopy(expected, 0, current, 0, expected.length);

        dtrmvScalar(uplo, trans, diag, n, a, 0, lda, expected, 0, incX);
        Dtrmv.dtrmv(uplo, trans, diag, n, a, 0, lda, current, 0, incX);

        assertArrayClose(expected, current);
    }

    private static void verifyAliasingParity(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                                             int n, int lda, int xOff, int incX, long seed) {
        Random random = new Random(seed);
        double[] template = new double[n * lda];
        double[] expected = new double[template.length];
        double[] current = new double[template.length];

        fillTriangular(random, template, uplo, n, lda);
        System.arraycopy(template, 0, expected, 0, template.length);
        System.arraycopy(template, 0, current, 0, template.length);

        dtrmvScalar(uplo, trans, diag, n, expected, 0, lda, expected, xOff, incX);
        Dtrmv.dtrmv(uplo, trans, diag, n, current, 0, lda, current, xOff, incX);

        assertArrayClose(expected, current);
    }

    private static void verifyAdjacentColumnViewParity(BLAS.Uplo uplo, BLAS.Diag diag,
                                                       int n, int lda, int aOff, long seed) {
        Random random = new Random(seed);
        int xOff = uplo == BLAS.Uplo.Upper ? aOff + n : aOff - 1;
        double[] template = new double[aOff + n * lda];
        double[] expected = new double[template.length];
        double[] current = new double[template.length];

        fillTriangular(random, template, uplo, aOff, n, lda);
        fillAliasedVector(random, template, xOff, lda, n);
        System.arraycopy(template, 0, expected, 0, template.length);
        System.arraycopy(template, 0, current, 0, template.length);

        dtrmvScalar(uplo, BLAS.Trans.NoTrans, diag, n, expected, aOff, lda, expected, xOff, lda);
        Dtrmv.dtrmv(uplo, BLAS.Trans.NoTrans, diag, n, current, aOff, lda, current, xOff, lda);

        assertArrayClose(expected, current);
    }

    private static void fillTriangular(Random random, double[] matrix, BLAS.Uplo uplo, int n, int lda) {
        fillTriangular(random, matrix, uplo, 0, n, lda);
    }

    private static void fillTriangular(Random random, double[] matrix, BLAS.Uplo uplo, int aOff, int n, int lda) {
        for (int i = 0; i < n; i++) {
            int rowOff = aOff + i * lda;
            for (int j = 0; j < n; j++) {
                if (uplo == BLAS.Uplo.Upper ? j >= i : j <= i) {
                    matrix[rowOff + j] = random.nextDouble() - 0.5;
                } else {
                    matrix[rowOff + j] = 0.0;
                }
            }
            matrix[rowOff + i] += n;
        }
    }

    private static void fillAliasedVector(Random random, double[] matrix, int xOff, int incX, int n) {
        int xi = xOff;
        for (int i = 0; i < n; i++) {
            matrix[xi] = random.nextDouble() - 0.5;
            xi += incX;
        }
    }

    private static void fillRandom(Random random, double[] values) {
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble() - 0.5;
        }
    }

    private static void dtrmvScalar(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, int n,
                                    double[] A, int aOff, int lda,
                                    double[] x, int xOff, int incX) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean transA = trans == BLAS.Trans.Trans || trans == BLAS.Trans.Conj;
        boolean unit = diag == BLAS.Diag.Unit;

        if (!transA) {
            if (upper) {
                for (int i = 0; i < n; i++) {
                    int rowOff = aOff + i * lda;
                    double tmp = unit ? x[xOff + i * incX] : A[rowOff + i] * x[xOff + i * incX];
                    for (int j = i + 1; j < n; j++) {
                        tmp = Math.fma(A[rowOff + j], x[xOff + j * incX], tmp);
                    }
                    x[xOff + i * incX] = tmp;
                }
            } else {
                for (int i = n - 1; i >= 0; i--) {
                    int rowOff = aOff + i * lda;
                    double tmp = unit ? x[xOff + i * incX] : A[rowOff + i] * x[xOff + i * incX];
                    for (int j = 0; j < i; j++) {
                        tmp = Math.fma(A[rowOff + j], x[xOff + j * incX], tmp);
                    }
                    x[xOff + i * incX] = tmp;
                }
            }
            return;
        }

        if (upper) {
            for (int i = n - 1; i >= 0; i--) {
                int rowOff = aOff + i * lda;
                double xi = x[xOff + i * incX];
                for (int j = i + 1; j < n; j++) {
                    x[xOff + j * incX] = Math.fma(xi, A[rowOff + j], x[xOff + j * incX]);
                }
                if (!unit) {
                    x[xOff + i * incX] *= A[rowOff + i];
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int rowOff = aOff + i * lda;
                double xi = x[xOff + i * incX];
                for (int j = 0; j < i; j++) {
                    x[xOff + j * incX] = Math.fma(xi, A[rowOff + j], x[xOff + j * incX]);
                }
                if (!unit) {
                    x[xOff + i * incX] *= A[rowOff + i];
                }
            }
        }
    }

    private static void assertArrayClose(double[] expected, double[] actual) {
        for (int i = 0; i < expected.length; i++) {
            double tolerance = Math.max(EPS, Math.abs(expected[i]) * 1e-13);
            assertThat(actual[i]).isCloseTo(expected[i], within(tolerance));
        }
    }

    private static org.assertj.core.data.Offset<Double> within(double value) {
        return org.assertj.core.data.Offset.offset(value);
    }
}
