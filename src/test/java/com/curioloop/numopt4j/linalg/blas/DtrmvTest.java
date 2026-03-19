/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
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

    private static org.assertj.core.data.Offset<Double> within(double value) {
        return org.assertj.core.data.Offset.offset(value);
    }
}
