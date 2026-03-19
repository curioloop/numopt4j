/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DgehrdTest {

    @Test
    @DisplayName("dgehrd: reduce 3x3 matrix to Hessenberg form")
    void testReduce3x3() {
        int n = 3;
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] AOrig = A.clone();
        
        double[] tau = new double[n - 1];
        double[] work = new double[n];
        
        Dgehrd.dgehrd(n, 0, n - 1, A, 0, n, tau, 0, work, 0, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j >= i - 1) {
                    assertTrue(Double.isFinite(A[i * n + j]), 
                        "A[" + i + "," + j + "] should be finite (part of H)");
                }
            }
        }
        
        double traceOrig = 0;
        double traceHess = 0;
        for (int i = 0; i < n; i++) {
            traceOrig += AOrig[i * n + i];
            traceHess += A[i * n + i];
        }
        assertEquals(traceOrig, traceHess, Math.abs(traceOrig) * 1e-10, "Trace should be preserved");
    }

    @Test
    @DisplayName("dgehrd: reduce 4x4 matrix to Hessenberg form")
    void testReduce4x4() {
        int n = 4;
        double[] A = {
            4, 1, 2, 3,
            1, 5, 1, 2,
            2, 1, 6, 1,
            3, 2, 1, 7
        };
        double[] AOrig = A.clone();
        
        double[] tau = new double[n - 1];
        double[] work = new double[n];
        
        Dgehrd.dgehrd(n, 0, n - 1, A, 0, n, tau, 0, work, 0, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j >= i - 1) {
                    assertTrue(Double.isFinite(A[i * n + j]), 
                        "A[" + i + "," + j + "] should be finite (part of H)");
                }
            }
        }
        
        double traceOrig = 0;
        double traceHess = 0;
        for (int i = 0; i < n; i++) {
            traceOrig += AOrig[i * n + i];
            traceHess += A[i * n + i];
        }
        assertEquals(traceOrig, traceHess, Math.abs(traceOrig) * 1e-10, "Trace should be preserved");
    }

    @Test
    @DisplayName("dgehrd: reduce 5x5 matrix to Hessenberg form")
    void testReduce5x5() {
        int n = 5;
        double[] A = {
            1, 2, 3, 4, 5,
            2, 3, 4, 5, 6,
            3, 4, 5, 6, 7,
            4, 5, 6, 7, 8,
            5, 6, 7, 8, 9
        };
        double[] AOrig = A.clone();
        
        double[] tau = new double[n - 1];
        double[] work = new double[n];
        
        Dgehrd.dgehrd(n, 0, n - 1, A, 0, n, tau, 0, work, 0, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j >= i - 1) {
                    assertTrue(Double.isFinite(A[i * n + j]), 
                        "A[" + i + "," + j + "] should be finite (part of H)");
                }
            }
        }
        
        double traceOrig = 0;
        double traceHess = 0;
        for (int i = 0; i < n; i++) {
            traceOrig += AOrig[i * n + i];
            traceHess += A[i * n + i];
        }
        assertEquals(traceOrig, traceHess, Math.abs(traceOrig) * 1e-10, "Trace should be preserved");
    }

    @Test
    @DisplayName("dgehrd: test with ilo=1, ihi=3")
    void testPartialReduction() {
        int n = 5;
        double[] A = {
            1, 2, 3, 4, 5,
            2, 3, 4, 5, 6,
            3, 4, 5, 6, 7,
            4, 5, 6, 7, 8,
            5, 6, 7, 8, 9
        };
        
        int ilo = 1;
        int ihi = 3;
        
        double[] tau = new double[n - 1];
        double[] work = new double[n];
        
        Dgehrd.dgehrd(n, ilo, ihi, A, 0, n, tau, 0, work, 0, n);
        
        for (int i = ilo; i <= ihi; i++) {
            for (int j = ilo; j <= ihi; j++) {
                if (j >= i - 1) {
                    assertTrue(Double.isFinite(A[i * n + j]), 
                        "A[" + i + "," + j + "] should be finite in reduced block");
                }
            }
        }
    }

    @Test
    @DisplayName("dgehrd: 2x2 matrix (no reduction needed)")
    void test2x2() {
        int n = 2;
        double[] A = {
            1, 2,
            3, 4
        };
        double[] AOrig = A.clone();
        
        double[] tau = new double[n - 1];
        double[] work = new double[n];
        
        Dgehrd.dgehrd(n, 0, n - 1, A, 0, n, tau, 0, work, 0, n);
        
        assertEquals(AOrig[0], A[0], 1e-14, "A[0,0] should be unchanged");
        assertEquals(AOrig[1], A[1], 1e-14, "A[0,1] should be unchanged");
        assertEquals(AOrig[2], A[2], 1e-14, "A[1,0] should be unchanged");
        assertEquals(AOrig[3], A[3], 1e-14, "A[1,1] should be unchanged");
    }

    @Test
    @DisplayName("dgehrd: 1x1 matrix (trivial)")
    void test1x1() {
        int n = 1;
        double[] A = {5.0};
        
        double[] tau = new double[n];
        double[] work = new double[n];
        
        Dgehrd.dgehrd(n, 0, 0, A, 0, n, tau, 0, work, 0, n);
        
        assertEquals(5.0, A[0], 1e-14, "A[0,0] should be unchanged");
    }

    @Test
    @DisplayName("dgehrd: verify Q*H*Q^T = A for 3x3")
    void testOrthogonalSimilarity3x3() {
        int n = 3;
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] AOrig = A.clone();
        
        double[] tau = new double[n - 1];
        double[] work = new double[n];
        
        Dgehrd.dgehrd(n, 0, n - 1, A, 0, n, tau, 0, work, 0, n);
        
        double[] Q = new double[n * n];
        for (int i = 0; i < n; i++) {
            Q[i * n + i] = 1.0;
        }
        
        for (int i = n - 2; i >= 0; i--) {
            double[] v = new double[n];
            v[i + 1] = 1.0;
            for (int k = i + 2; k < n; k++) {
                v[k] = A[k * n + i];
            }
            
            double[] work2 = new double[n];
            Dlarf.dlarf(BLAS.Side.Left, n - i - 1, n, v, i + 1, 1, tau[i], Q, (i + 1) * n, n, work2, 0);
        }
        
        double[] H = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j >= i - 1) {
                    H[i * n + j] = A[i * n + j];
                }
            }
        }
        
        double[] QH = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    QH[i * n + j] += Q[i * n + k] * H[k * n + j];
                }
            }
        }
        
        double[] QHQ = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    QHQ[i * n + j] += QH[i * n + k] * Q[j * n + k];
                }
            }
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals(AOrig[i * n + j], QHQ[i * n + j], Math.abs(AOrig[i * n + j]) * 1e-10 + 1e-10,
                    "Q*H*Q^T[" + i + "," + j + "] should equal A[" + i + "," + j + "]");
            }
        }
    }
}
