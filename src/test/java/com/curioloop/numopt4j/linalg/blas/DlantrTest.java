/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlantrTest {

    private static final double TOL = 1e-10;

    // -----------------------------------------------------------------------
    // Existing NonUnit tests
    // -----------------------------------------------------------------------

    @Test
    void testFrobeniusNormUpper() {
        double[] A = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        double norm = Dlantr.dlantr('F', BLAS.Uplo.Upper, BLAS.Diag.NonUnit, 3, 3, A, 3, new double[3]);
        assertTrue(norm > 0);
    }

    @Test
    void testFrobeniusNormLower() {
        double[] A = {
            1, 0, 0,
            2, 4, 0,
            3, 5, 6
        };
        double norm = Dlantr.dlantr('F', BLAS.Uplo.Lower, BLAS.Diag.NonUnit, 3, 3, A, 3, new double[3]);
        assertTrue(norm > 0);
    }

    @Test
    void testEmpty() {
        double norm = Dlantr.dlantr('F', BLAS.Uplo.Upper, BLAS.Diag.NonUnit, 0, 0, new double[0], 0, new double[0]);
        assertEquals(0.0, norm, TOL);
    }

    // -----------------------------------------------------------------------
    // Unit diagonal: diagonal elements must be treated as 1.0 regardless of
    // the actual stored value.
    //
    // Strategy: store garbage (99.0) on the diagonal so any test that reads
    // the diagonal directly will produce a wrong answer.
    // -----------------------------------------------------------------------

    /**
     * 3×3 upper triangular, unit diagonal.
     * Stored diagonal = 99 (should be ignored), off-diagonal = 2.
     *
     *   [99  2  2]
     *   [ 0 99  2]
     *   [ 0  0 99]
     *
     * Treated as:
     *   [ 1  2  2]
     *   [ 0  1  2]
     *   [ 0  0  1]
     */
    @Test
    void testUnitDiagUpperNorm1() {
        double[] A = {
            99,  2,  2,
             0, 99,  2,
             0,  0, 99
        };
        // Column sums: col0=1, col1=1+2=3, col2=1+2+2=5  → max = 5
        double norm = Dlantr.dlantr('1', BLAS.Uplo.Upper, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
        assertEquals(5.0, norm, TOL);
    }

    @Test
    void testUnitDiagUpperNormInf() {
        double[] A = {
            99,  2,  2,
             0, 99,  2,
             0,  0, 99
        };
        // Row sums: row0=1+2+2=5, row1=1+2=3, row2=1  → max = 5
        double norm = Dlantr.dlantr('I', BLAS.Uplo.Upper, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
        assertEquals(5.0, norm, TOL);
    }

    @Test
    void testUnitDiagUpperNormFro() {
        double[] A = {
            99,  2,  2,
             0, 99,  2,
             0,  0, 99
        };
        // Frobenius: sqrt(1+4+4 + 1+4 + 1) = sqrt(15)
        double expected = Math.sqrt(15.0);
        double norm = Dlantr.dlantr('F', BLAS.Uplo.Upper, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
        assertEquals(expected, norm, TOL);
    }

    @Test
    void testUnitDiagUpperNormMax() {
        double[] A = {
            99,  2,  2,
             0, 99,  2,
             0,  0, 99
        };
        // Max abs: off-diagonal max = 2, diagonal treated as 1 → max = 2
        double norm = Dlantr.dlantr('M', BLAS.Uplo.Upper, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
        assertEquals(2.0, norm, TOL);
    }

    /**
     * 3×3 lower triangular, unit diagonal.
     *
     *   [99  0  0]
     *   [ 2 99  0]
     *   [ 2  2 99]
     *
     * Treated as:
     *   [ 1  0  0]
     *   [ 2  1  0]
     *   [ 2  2  1]
     */
    @Test
    void testUnitDiagLowerNorm1() {
        double[] A = {
            99,  0,  0,
             2, 99,  0,
             2,  2, 99
        };
        // Column sums: col0=1+2+2=5, col1=1+2=3, col2=1  → max = 5
        double norm = Dlantr.dlantr('1', BLAS.Uplo.Lower, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
        assertEquals(5.0, norm, TOL);
    }

    @Test
    void testUnitDiagLowerNormInf() {
        double[] A = {
            99,  0,  0,
             2, 99,  0,
             2,  2, 99
        };
        // Row sums: row0=1, row1=2+1=3, row2=2+2+1=5  → max = 5
        double norm = Dlantr.dlantr('I', BLAS.Uplo.Lower, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
        assertEquals(5.0, norm, TOL);
    }

    @Test
    void testUnitDiagLowerNormFro() {
        double[] A = {
            99,  0,  0,
             2, 99,  0,
             2,  2, 99
        };
        // Frobenius: sqrt(1 + 4+1 + 4+4+1) = sqrt(15)
        double expected = Math.sqrt(15.0);
        double norm = Dlantr.dlantr('F', BLAS.Uplo.Lower, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
        assertEquals(expected, norm, TOL);
    }

    @Test
    void testUnitDiagLowerNormMax() {
        double[] A = {
            99,  0,  0,
             2, 99,  0,
             2,  2, 99
        };
        // Max abs: off-diagonal max = 2, diagonal treated as 1 → max = 2
        double norm = Dlantr.dlantr('M', BLAS.Uplo.Lower, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
        assertEquals(2.0, norm, TOL);
    }

    /**
     * Unit diagonal where off-diagonal values are all zero.
     * Every norm should equal 1.0 (identity matrix).
     */
    @Test
    void testUnitDiagIdentityAllNorms() {
        double[] A = {
            99,  0,  0,
             0, 99,  0,
             0,  0, 99
        };
        for (char c : new char[]{'1', 'I', 'M'}) {
            double norm = Dlantr.dlantr(c, BLAS.Uplo.Upper, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
            assertEquals(1.0, norm, TOL, "norm '" + c + "' of identity should be 1");
        }
        // Frobenius of identity = sqrt(3)
        double fro = Dlantr.dlantr('F', BLAS.Uplo.Upper, BLAS.Diag.Unit, 3, 3, A, 3, new double[3]);
        assertEquals(Math.sqrt(3.0), fro, TOL);
    }

    /**
     * NonUnit vs Unit: same matrix, diagonal = 1.0.
     * Both should produce identical results.
     */
    @Test
    void testNonUnitEqualsUnitWhenDiagIsOne() {
        double[] A = {
            1,  3,  5,
            0,  1,  7,
            0,  0,  1
        };
        for (char c : new char[]{'1', 'I', 'F', 'M'}) {
            double nonUnit = Dlantr.dlantr(c, BLAS.Uplo.Upper, BLAS.Diag.NonUnit, 3, 3, A, 3, new double[3]);
            double unit    = Dlantr.dlantr(c, BLAS.Uplo.Upper, BLAS.Diag.Unit,    3, 3, A, 3, new double[3]);
            assertEquals(nonUnit, unit, TOL, "norm '" + c + "' should match when diag=1");
        }
    }

    /**
     * Rectangular (trapezoidal) matrix: 4×3 lower, unit diagonal.
     *
     *   [99  0  0]
     *   [ 3 99  0]
     *   [ 3  3 99]
     *   [ 3  3  3]
     *
     * Treated as:
     *   [ 1  0  0]
     *   [ 3  1  0]
     *   [ 3  3  1]
     *   [ 3  3  3]
     */
    @Test
    void testUnitDiagRectangularLower() {
        double[] A = {
            99,  0,  0,
             3, 99,  0,
             3,  3, 99,
             3,  3,  3
        };
        // Norm-1 (max col sum): col0=1+3+3+3=10, col1=1+3+3=7, col2=1+3=4 → 10
        double n1 = Dlantr.dlantr('1', BLAS.Uplo.Lower, BLAS.Diag.Unit, 4, 3, A, 3, new double[3]);
        assertEquals(10.0, n1, TOL);

        // Norm-Inf (max row sum): row0=1, row1=3+1=4, row2=3+3+1=7, row3=3+3+3=9 → 9
        double ni = Dlantr.dlantr('I', BLAS.Uplo.Lower, BLAS.Diag.Unit, 4, 3, A, 3, new double[4]);
        assertEquals(9.0, ni, TOL);
    }
}
