/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DlarftTest {

    private static final double TOL = 1e-10;

    @Nested
    @DisplayName("Forward Direction")
    class ForwardTests {

        @Test
        @DisplayName("ColumnWise storage")
        void testForwardColWise() {
            Random rnd = new Random(12345);
            
            int[][] sizes = {{6, 6}, {8, 6}, {6, 8}};
            
            for (int[] size : sizes) {
                int m = size[0], n = size[1];
                int k = Math.min(m, n);
                int lda = n, ldv = k, ldt = k;
                
                double[] a = randomMatrix(m, lda, rnd);
                double[] tau = new double[k];
                double[] work = new double[n];
                
                Dgeqr.dgeqr2(m, k, a, 0, lda, tau, 0, work, 0);
                
                double[] v = extractVForwardColWise(m, k, a, lda);
                double[] t = new double[k * ldt];
                Dlarft.dlarftForward(v, 0, ldv, tau, 0, t, 0, ldt, m, k);
                
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < i; j++) {
                        double val = t[i * ldt + j];
                        assertTrue(Math.abs(val) < TOL, 
                            String.format("Lower triangle should be zero at (%d,%d), got %g", i, j, val));
                    }
                }
                
                double[] h = constructHForwardColWise(tau, v, m, k);
                double[] hFromT = constructHFromTForwardColWise(v, t, m, k);
                
                assertMatrixClose(h, hFromT, m, m, TOL, 
                    String.format("Forward/ColWise m=%d, n=%d", m, n));
            }
        }

        @Test
        @DisplayName("RowWise storage")
        void testForwardRowWise() {
            Random rnd = new Random(12345);
            
            int[][] sizes = {{6, 6}, {8, 6}, {6, 8}};
            
            for (int[] size : sizes) {
                int m = size[0], n = size[1];
                int k = Math.min(m, n);
                int lda = n, ldv = m, ldt = k;
                
                double[] a = randomMatrix(m, lda, rnd);
                double[] tau = new double[k];
                double[] work = new double[n];
                
                Dgeqr.dgeqr2(m, k, a, 0, lda, tau, 0, work, 0);
                
                double[] v = extractVForwardRowWise(m, k, a, lda);
                double[] t = new double[k * ldt];
                Dlarft.dlarft('F', 'R', m, k, v, 0, ldv, tau, 0, t, 0, ldt);
                
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < i; j++) {
                        double val = t[i * ldt + j];
                        assertTrue(Math.abs(val) < TOL, 
                            String.format("Lower triangle should be zero at (%d,%d), got %g", i, j, val));
                    }
                }
                
                double[] h = constructHForwardRowWise(tau, v, m, k);
                double[] hFromT = constructHFromTForwardRowWise(v, t, m, k);
                
                assertMatrixClose(h, hFromT, m, m, TOL, 
                    String.format("Forward/RowWise m=%d, n=%d", m, n));
            }
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("Forward with leading dimension")
        void testForwardWithLd() {
            Random rnd = new Random(12345);
            
            int m = 6, n = 8, k = Math.min(m, n);
            int lda = n, ldv = 10, ldt = 15;
            
            double[] a = randomMatrix(m, lda, rnd);
            double[] tau = new double[k];
            double[] work = new double[n];
            
            Dgeqr.dgeqr2(m, k, a, 0, lda, tau, 0, work, 0);
            
            double[] v = new double[m * ldv];
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < i; j++) {
                    v[j * ldv + i] = 0;
                }
                v[i * ldv + i] = 1;
                for (int j = i + 1; j < m; j++) {
                    v[j * ldv + i] = a[j * lda + i];
                }
            }
            
            double[] t = new double[k * ldt];
            Dlarft.dlarftForward(v, 0, ldv, tau, 0, t, 0, ldt, m, k);
            
            double[] h = constructHFromVColWise(tau, v, m, k, ldv);
            double[] vCompact = new double[m * k];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < k; j++) {
                    vCompact[i * k + j] = v[i * ldv + j];
                }
            }
            
            double[] hFromT = constructHFromTForwardColWise(vCompact, t, m, k, ldt);
            assertMatrixClose(h, hFromT, m, m, TOL, "Forward with ld");
        }

        @Test
        @DisplayName("Zero tau case")
        void testZeroTau() {
            int m = 6, k = 3;
            int ldv = k, ldt = k;
            
            double[] v = new double[m * k];
            for (int i = 0; i < k; i++) {
                v[i * ldv + i] = 1;
            }
            
            double[] tau = new double[k];
            double[] t = new double[k * ldt];
            Dlarft.dlarftForward(v, 0, ldv, tau, 0, t, 0, ldt, m, k);
            
            for (int i = 0; i < k; i++) {
                for (int j = 0; j <= i; j++) {
                    assertEquals(0, t[j * ldt + i], TOL, 
                        String.format("T should be zero when tau=0 at (%d,%d)", j, i));
                }
            }
        }

        @Test
        @DisplayName("Single reflector")
        void testSingleReflector() {
            Random rnd = new Random(12345);
            
            int m = 6, k = 1;
            int ldv = k, ldt = k;
            
            double[] v = new double[m * k];
            v[0] = 1;
            for (int i = 1; i < m; i++) {
                v[i * ldv] = rnd.nextDouble();
            }
            
            double[] tau = new double[]{0.5 + rnd.nextDouble() * 0.5};
            double[] t = new double[k * ldt];
            Dlarft.dlarftForward(v, 0, ldv, tau, 0, t, 0, ldt, m, k);
            
            assertEquals(tau[0], t[0], TOL, "T[0,0] should equal tau[0]");
            
            double[] h = new double[m * m];
            for (int i = 0; i < m; i++) h[i * m + i] = 1;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    h[i * m + j] -= tau[0] * v[i * ldv] * v[j * ldv];
                }
            }
            
            double[] hFromT = constructHFromTForwardColWise(v, t, m, k);
            assertMatrixClose(h, hFromT, m, m, TOL, "Single reflector");
        }
    }

    @Nested
    @DisplayName("Backward Direction")
    class BackwardTests {

        @Test
        @DisplayName("ColumnWise storage")
        void testBackwardColWise() {
            Random rnd = new Random(12345);
            
            int[][] sizes = {{6, 6}, {8, 6}, {6, 8}};
            
            for (int[] size : sizes) {
                int m = size[0], n = size[1];
                int k = Math.min(m, n);
                int ldv = k, ldt = k;
                
                double[] tau = new double[k];
                for (int i = 0; i < k; i++) {
                    tau[i] = 0.5 + rnd.nextDouble() * 0.5;
                }
                
                double[] v = new double[m * k];
                for (int col = 0; col < k; col++) {
                    for (int row = 0; row < m - k + col; row++) {
                        v[row * ldv + col] = rnd.nextDouble();
                    }
                    v[(m - k + col) * ldv + col] = 1;
                    for (int row = m - k + col + 1; row < m; row++) {
                        v[row * ldv + col] = 0;
                    }
                }
                
                double[] t = new double[k * ldt];
                Dlarft.dlarftBackward(v, 0, ldv, tau, 0, t, 0, ldt, m, k);
                
                for (int i = 0; i < k; i++) {
                    for (int j = i + 1; j < k; j++) {
                        double val = t[i * ldt + j];
                        assertTrue(Math.abs(val) < TOL, 
                            String.format("Upper triangle should be zero at (%d,%d), got %g", i, j, val));
                    }
                }
                
                double[] h = constructHBackwardColWise(tau, v, m, k);
                double[] hFromT = constructHFromTBackwardColWise(v, t, m, k);
                
                assertMatrixClose(h, hFromT, m, m, TOL, 
                    String.format("Backward/ColWise m=%d, n=%d", m, n));
            }
        }

        @Test
        @DisplayName("RowWise storage")
        void testBackwardRowWise() {
            Random rnd = new Random(12345);
            
            int[][] sizes = {{6, 6}, {8, 6}, {6, 8}};
            
            for (int[] size : sizes) {
                int m = size[0], n = size[1];
                int k = Math.min(m, n);
                int ldv = m, ldt = k;
                
                double[] tau = new double[k];
                for (int i = 0; i < k; i++) {
                    tau[i] = 0.5 + rnd.nextDouble() * 0.5;
                }
                
                double[] v = new double[k * m];
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < m - k + i; j++) {
                        v[i * ldv + j] = rnd.nextDouble();
                    }
                    v[i * ldv + (m - k + i)] = 1;
                    for (int j = m - k + i + 1; j < m; j++) {
                        v[i * ldv + j] = 0;
                    }
                }
                
                double[] t = new double[k * ldt];
                Dlarft.dlarft('B', 'R', m, k, v, 0, ldv, tau, 0, t, 0, ldt);
                
                for (int i = 0; i < k; i++) {
                    for (int j = i + 1; j < k; j++) {
                        double val = t[i * ldt + j];
                        assertTrue(Math.abs(val) < TOL, 
                            String.format("Upper triangle should be zero at (%d,%d), got %g", i, j, val));
                    }
                }
                
                double[] h = constructHBackwardRowWise(tau, v, m, k);
                double[] hFromT = constructHFromTBackwardRowWise(v, t, m, k);
                
                assertMatrixClose(h, hFromT, m, m, TOL, 
                    String.format("Backward/RowWise m=%d, n=%d", m, n));
            }
        }
    }

    private static double[] randomMatrix(int rows, int cols, Random rnd) {
        double[] a = new double[rows * cols];
        for (int i = 0; i < a.length; i++) {
            a[i] = rnd.nextDouble();
        }
        return a;
    }

    private static double[] extractVForwardColWise(int m, int k, double[] a, int lda) {
        double[] v = new double[m * k];
        int stride = k;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++) {
                v[j * stride + i] = 0;
            }
            v[i * stride + i] = 1;
            for (int j = i + 1; j < m; j++) {
                v[j * stride + i] = a[j * lda + i];
            }
        }
        return v;
    }

    private static double[] extractVForwardRowWise(int m, int k, double[] a, int lda) {
        double[] v = new double[k * m];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++) {
                v[i * m + j] = 0;
            }
            v[i * m + i] = 1;
            for (int j = i + 1; j < m; j++) {
                v[i * m + j] = a[j * lda + i];
            }
        }
        return v;
    }

    private static double[] constructHForwardColWise(double[] tau, double[] v, int m, int k) {
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            h[i * m + i] = 1;
        }
        
        for (int i = 0; i < k; i++) {
            double[] vec = new double[m];
            for (int j = 0; j < m; j++) {
                vec[j] = v[j * k + i];
            }
            
            applyHouseholderLeft(h, m, tau[i], vec);
        }
        return h;
    }

    private static double[] constructHForwardRowWise(double[] tau, double[] v, int m, int k) {
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            h[i * m + i] = 1;
        }
        
        for (int i = 0; i < k; i++) {
            double[] vec = new double[m];
            for (int j = 0; j < m; j++) {
                vec[j] = v[i * m + j];
            }
            
            applyHouseholderLeft(h, m, tau[i], vec);
        }
        return h;
    }

    private static void applyHouseholderLeft(double[] h, int m, double tau, double[] vec) {
        double[] hi = new double[m * m];
        for (int j = 0; j < m; j++) {
            hi[j * m + j] = 1;
        }
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < m; col++) {
                hi[row * m + col] -= tau * vec[row] * vec[col];
            }
        }
        
        double[] hCopy = h.clone();
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < m; col++) {
                h[row * m + col] = 0;
                for (int l = 0; l < m; l++) {
                    h[row * m + col] += hCopy[row * m + l] * hi[l * m + col];
                }
            }
        }
    }

    private static double[] constructHFromVColWise(double[] tau, double[] v, int m, int k, int ldv) {
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            h[i * m + i] = 1;
        }
        
        for (int i = 0; i < k; i++) {
            double[] vec = new double[m];
            for (int j = 0; j < m; j++) {
                vec[j] = v[j * ldv + i];
            }
            
            double[] hi = new double[m * m];
            for (int j = 0; j < m; j++) {
                hi[j * m + j] = 1;
            }
            for (int row = 0; row < m; row++) {
                for (int col = 0; col < m; col++) {
                    hi[row * m + col] -= tau[i] * vec[row] * vec[col];
                }
            }
            
            double[] hCopy = h.clone();
            for (int row = 0; row < m; row++) {
                for (int col = 0; col < m; col++) {
                    h[row * m + col] = 0;
                    for (int l = 0; l < m; l++) {
                        h[row * m + col] += hCopy[row * m + l] * hi[l * m + col];
                    }
                }
            }
        }
        return h;
    }

    private static double[] constructHFromTForwardColWise(double[] v, double[] t, int m, int k) {
        return constructHFromTForwardColWise(v, t, m, k, k);
    }

    private static double[] constructHFromTForwardColWise(double[] v, double[] t, int m, int k, int ldt) {
        double[] vt = new double[m * k];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0;
                for (int l = 0; l <= j; l++) {
                    sum += v[i * k + l] * t[l * ldt + j];
                }
                vt[i * k + j] = sum;
            }
        }
        
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int l = 0; l < k; l++) {
                    sum += vt[i * k + l] * v[j * k + l];
                }
                h[i * m + j] = (i == j ? 1 : 0) - sum;
            }
        }
        return h;
    }

    private static double[] constructHFromTForwardRowWise(double[] v, double[] t, int m, int k) {
        double[] vt = new double[k * m];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int l = 0; l <= i; l++) {
                    sum += v[l * m + j] * t[l * k + i];
                }
                vt[i * m + j] = sum;
            }
        }
        
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int l = 0; l < k; l++) {
                    sum += vt[l * m + i] * v[l * m + j];
                }
                h[i * m + j] = (i == j ? 1 : 0) - sum;
            }
        }
        return h;
    }

    private static void assertMatrixClose(double[] expected, double[] actual, 
                                          int rows, int cols, double tol, String msg) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = Math.abs(expected[i * cols + j] - actual[i * cols + j]);
                if (diff > tol) {
                    fail(String.format("%s: mismatch at (%d,%d): expected %g, got %g, diff %g",
                        msg, i, j, expected[i * cols + j], actual[i * cols + j], diff));
                }
            }
        }
    }

    private static double[] constructHBackwardColWise(double[] tau, double[] v, int m, int k) {
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            h[i * m + i] = 1;
        }
        
        for (int i = k - 1; i >= 0; i--) {
            double[] vec = new double[m];
            for (int j = 0; j < m; j++) {
                vec[j] = v[j * k + i];
            }
            applyHouseholderLeft(h, m, tau[i], vec);
        }
        return h;
    }

    private static double[] constructHBackwardRowWise(double[] tau, double[] v, int m, int k) {
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            h[i * m + i] = 1;
        }
        
        for (int i = k - 1; i >= 0; i--) {
            double[] vec = new double[m];
            for (int j = 0; j < m; j++) {
                vec[j] = v[i * m + j];
            }
            applyHouseholderLeft(h, m, tau[i], vec);
        }
        return h;
    }

    private static double[] constructHFromTBackwardColWise(double[] v, double[] t, int m, int k) {
        double[] vt = new double[m * k];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0;
                for (int l = j; l < k; l++) {
                    sum += v[i * k + l] * t[l * k + j];
                }
                vt[i * k + j] = sum;
            }
        }
        
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int l = 0; l < k; l++) {
                    sum += vt[i * k + l] * v[j * k + l];
                }
                h[i * m + j] = (i == j ? 1 : 0) - sum;
            }
        }
        return h;
    }

    private static double[] constructHFromTBackwardRowWise(double[] v, double[] t, int m, int k) {
        double[] vt = new double[k * m];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int l = i; l < k; l++) {
                    sum += v[l * m + j] * t[l * k + i];
                }
                vt[i * m + j] = sum;
            }
        }
        
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int l = 0; l < k; l++) {
                    sum += vt[l * m + i] * v[l * m + j];
                }
                h[i * m + j] = (i == j ? 1 : 0) - sum;
            }
        }
        return h;
    }
}
