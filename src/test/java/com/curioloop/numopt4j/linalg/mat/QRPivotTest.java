/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class QRPivotTest {

    private static final double EPSILON = 1e-10;

    @Test
    void testBasicDecomposition() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        
        QR qr = QR.decompose(A, 2, 2, true, null);
        
        assertThat(qr.ok()).isTrue();
        assertThat(qr.rank()).isEqualTo(2);
        assertThat(qr.isPivoting()).isTrue();
    }

    @Test
    void testRankDeficientMatrix() {
        double[] A = {
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            1.0, 2.0, 4.0
        };
        
        QR qr = QR.decompose(A, 3, 3, true, null);
        
        assertThat(qr.ok()).isTrue();
        assertThat(qr.rank()).isLessThan(3);
    }

    @Test
    void testSolve() {
        double[] A = {
            4.0, 1.0,
            3.0, 2.0
        };
        double[] b = {1.0, 2.0};
        
        QR qr = QR.decompose(A, 2, 2, true, null);
        double[] x = qr.solve(b, null);
        
        assertThat(qr.ok()).isTrue();
        
        double[] Aorig = {
            4.0, 1.0,
            3.0, 2.0
        };
        double[] result = new double[2];
        for (int i = 0; i < 2; i++) {
            result[i] = Aorig[i * 2] * x[0] + Aorig[i * 2 + 1] * x[1];
        }
        
        assertThat(result[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(result[1]).isCloseTo(2.0, offset(EPSILON));
    }

    @Test
    void testLeastSquares() {
        double[] A = {
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0
        };
        double[] b = {1.0, 2.0, 3.0};
        double[] x = new double[2];
        
        QR qr = QR.decompose(A, 3, 2, true, null);
        qr.leastSquares(b, x);
        
        assertThat(qr.ok()).isTrue();
        
        assertThat(x[0]).isCloseTo(0.0, offset(1e-8));
        assertThat(x[1]).isCloseTo(1.0, offset(1e-8));
    }

    @Test
    void testPivot() {
        double[] A = {
            1.0, 100.0,
            1.0, 1.0
        };
        
        QR qr = QR.decompose(A, 2, 2, true, null);
        
        assertThat(qr.ok()).isTrue();
        assertThat(qr.piv()).isNotNull();
        assertThat(qr.piv().length).isEqualTo(2);
    }

    @Test
    void testExtractR() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        
        QR qr = QR.decompose(A, 2, 2, true, null);
        double[] R = qr.toR().data;
        
        assertThat(R[0]).isNotCloseTo(0.0, offset(EPSILON));
        assertThat(R[3]).isNotCloseTo(0.0, offset(EPSILON));
    }

    @Test
    void testRankWithTolerance() {
        double[] A = {
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            1.0, 2.0, 4.0
        };
        
        QR qr = QR.decompose(A, 3, 3, true, null);
        
        assertThat(qr.rank(1e-10)).isGreaterThanOrEqualTo(1);
    }

    @Test
    void testNonPivotingDecomposition() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        
        QR qr = QR.decompose(A, 2, 2);
        
        assertThat(qr.ok()).isTrue();
        assertThat(qr.isPivoting()).isFalse();
        assertThat(qr.piv()).isNull();
    }

    @Test
    void testExtractP() {
        double[] A = {
            1.0, 100.0,
            1.0, 1.0
        };
        
        QR qr = QR.decompose(A, 2, 2, true, null);
        double[] P = qr.toP().data;
        
        int[] jpvt = qr.piv();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertThat(P[i * 2 + j]).isEqualTo((jpvt[j] == i) ? 1.0 : 0.0);
            }
        }
    }

    @Test
    void testWithWorkspace() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        
        QR.Pool ws = (QR.Pool) QR.workspace(2, 2, true);
        QR qr = QR.decompose(A, 2, 2, true, ws);
        
        assertThat(qr.ok()).isTrue();
        assertThat(qr.pool()).isSameAs(ws);
    }

    @Test
    void testReuseWorkspace() {
        double[] A1 = {
            1.0, 2.0,
            3.0, 4.0
        };
        double[] A2 = {
            5.0, 6.0,
            7.0, 8.0
        };
        
        QR.Pool ws = (QR.Pool) QR.workspace(2, 2, false);
        
        QR qr1 = QR.decompose(A1, 2, 2, ws);
        assertThat(qr1.ok()).isTrue();
        
        QR qr2 = QR.decompose(A2, 2, 2, ws);
        assertThat(qr2.ok()).isTrue();
        
        assertThat(qr1.pool()).isSameAs(qr2.pool());
    }
}
