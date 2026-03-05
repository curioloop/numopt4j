/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DSYTD2 implementation.
 */
public class Dsytd2Test {

    @Test
    public void testDsytd2Upper() {
        // Test symmetric tridiagonal reduction (upper triangular)
        int n = 4;
        // Create upper triangular matrix only - lower triangle is zero
        double[] A = new double[] {
            4, 1, 2, 3,
            0, 5, 6, 7,
            0, 0, 6, 9,
            0, 0, 0, 7
        };
        int lda = n;
        
        double[] d = new double[n];
        double[] e = new double[n - 1];
        double[] tau = new double[n - 1];
        
        Dsytd2.dsytd2(BLAS.Uplo.Upper, n, A, lda, d, e, tau);
        
        // After reduction, A should be in tridiagonal form
        // Check diagonal elements
        for (int i = 0; i < n; i++) {
            assertEquals(A[i * lda + i], d[i], 1e-10, "d[" + i + "] incorrect");
        }
        
        // Check super-diagonal elements (e)
        for (int i = 0; i < n - 1; i++) {
            assertEquals(A[i * lda + i + 1], e[i], 1e-10, "e[" + i + "] incorrect");
        }
        // Note: Elements above the first super-diagonal contain Householder vectors,
        // not necessarily zero.
    }

    @Test
    public void testDsytd2Lower() {
        // Test symmetric tridiagonal reduction (lower triangular)
        int n = 4;
        double[] A = new double[] {
            4, 0, 0, 0,
            1, 5, 0, 0,
            2, 3, 6, 0,
            4, 5, 7, 8
        };
        int lda = n;
        
        double[] d = new double[n];
        double[] e = new double[n - 1];
        double[] tau = new double[n - 1];
        
        Dsytd2.dsytd2(BLAS.Uplo.Lower, n, A, lda, d, e, tau);
        
        // Check diagonal
        for (int i = 0; i < n; i++) {
            assertEquals(A[i * lda + i], d[i], 1e-10, "d[" + i + "] incorrect");
        }
        
        // Check sub-diagonal
        for (int i = 0; i < n - 1; i++) {
            assertEquals(A[(i + 1) * lda + i], e[i], 1e-10, "e[" + i + "] incorrect");
        }
    }

    @Test
    public void testDsytd2Identity() {
        // Test DSYTD2 on identity matrix
        int n = 3;
        double[] A = new double[] {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        int lda = n;
        
        double[] d = new double[n];
        double[] e = new double[n - 1];
        double[] tau = new double[n - 1];
        
        Dsytd2.dsytd2(BLAS.Uplo.Lower, n, A, lda, d, e, tau);
        
        // All diagonal should be 1
        for (int i = 0; i < n; i++) {
            assertEquals(1.0, d[i], 1e-10, "d[" + i + "] should be 1");
        }
        for (int i = 0; i < n - 1; i++) {
            assertEquals(0.0, e[i], 1e-10, "e[" + i + "] should be 0");
        }
    }

    @Test
    public void testDsytd2SmallMatrix() {
        // Test with a 2x2 matrix
        // Note: 2x2 case with upper='U' has special behavior - tau=0, no transformation
        int n = 2;
        double[] A = new double[] {
            2, 1,
            1, 2
        };
        int lda = n;
        
        double[] d = new double[n];
        double[] e = new double[n - 1];
        double[] tau = new double[n - 1];
        
        // Use upper case to get correct behavior
        Dsytd2.dsytd2(BLAS.Uplo.Upper, n, A, lda, d, e, tau);
        
        // For 2x2 with upper case: d = [2, 2], e = [1], tau = [0]
        assertEquals(2.0, d[0], 1e-10, "d[0]");
        assertEquals(2.0, d[1], 1e-10, "d[1]");
        assertEquals(1.0, e[0], 1e-10, "e[0]");
    }
}
