/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for illegal dimension parameter error handling consistency.
 *
 */
package com.curioloop.numopt4j.linalg.blas;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests verifying consistent error handling for illegal dimension parameters.
 *
 * <p>Property 7: For all BLAS/LAPACK routines and all illegal dimension inputs (n &lt; 0, m &lt; 0),
 * the error handling behavior must be consistent:
 * <ul>
 *   <li>BLAS Level 1/2/3 routines: silently return (no exception, no error code)</li>
 *   <li>LAPACK routines: return a negative info value</li>
 * </ul>
 *
 */
class IllegalDimErrorHandlingProperties {

    // ========================================================================
    // Helper: minimal arrays for routine calls
    // ========================================================================

    private static double[] vec(int size) {
        return new double[Math.max(size, 1)];
    }

    private static double[] mat(int rows, int cols) {
        return new double[Math.max(rows * cols, 1)];
    }

    // ========================================================================
    // Property 7.1: BLAS Level 1 routines silently return on n < 0
    // ========================================================================

    /**
     * For all negative n values, BLAS Level 1 routines must silently return
     * without throwing any exception.
     *
     */
    @Property(tries = 100)
    @Label("Feature: blas-api-refactor, Property 7: BLAS Level 1 silently returns on n < 0")
    void blasLevel1SilentlyReturnsOnNegativeN(
            @ForAll @IntRange(min = -1000, max = -1) int n
    ) {
        double[] x = vec(1);
        double[] y = vec(1);

        // All Level 1 void routines should silently return without exception
        assertDoesNotThrow(() -> BLAS.daxpy(n, 1.0, x, 0, 1, y, 0, 1),
                "daxpy should silently return on n=" + n);
        assertDoesNotThrow(() -> BLAS.dcopy(n, x, 0, 1, y, 0, 1),
                "dcopy should silently return on n=" + n);
        assertDoesNotThrow(() -> BLAS.dswap(n, x, 0, 1, y, 0, 1),
                "dswap should silently return on n=" + n);
        assertDoesNotThrow(() -> BLAS.dscal(n, 2.0, x, 0, 1),
                "dscal should silently return on n=" + n);
        assertDoesNotThrow(() -> BLAS.drot(n, x, 0, 1, y, 0, 1, 1.0, 0.0),
                "drot should silently return on n=" + n);

        // Scalar-returning routines should also return without exception
        assertDoesNotThrow(() -> BLAS.ddot(n, x, 0, 1, y, 0, 1),
                "ddot should silently return on n=" + n);
        assertDoesNotThrow(() -> BLAS.dnrm2(n, x, 0, 1),
                "dnrm2 should silently return on n=" + n);
        assertDoesNotThrow(() -> BLAS.dasum(n, x, 0, 1),
                "dasum should silently return on n=" + n);
        assertDoesNotThrow(() -> BLAS.idamax(n, x, 0, 1),
                "idamax should silently return on n=" + n);
    }

    // ========================================================================
    // Property 7.2: BLAS Level 2 routines silently return on negative m or n
    // ========================================================================

    /**
     * For all negative m or n values, BLAS Level 2 routines must silently return
     * without throwing any exception.
     *
     */
    @Property(tries = 100)
    @Label("Feature: blas-api-refactor, Property 7: BLAS Level 2 silently returns on negative m/n")
    void blasLevel2SilentlyReturnsOnNegativeDimensions(
            @ForAll @IntRange(min = -100, max = -1) int negDim,
            @ForAll @IntRange(min = 1, max = 10) int posDim
    ) {
        double[] A = mat(posDim, posDim);
        double[] x = vec(posDim);
        double[] y = vec(posDim);

        // dgemv with negative m
        assertDoesNotThrow(() ->
                BLAS.dgemv(BLAS.Transpose.NoTrans, negDim, posDim, 1.0, A, 0, posDim, x, 0, 1, 0.0, y, 0, 1),
                "dgemv should silently return on m=" + negDim);

        // dgemv with negative n
        assertDoesNotThrow(() ->
                BLAS.dgemv(BLAS.Transpose.NoTrans, posDim, negDim, 1.0, A, 0, posDim, x, 0, 1, 0.0, y, 0, 1),
                "dgemv should silently return on n=" + negDim);

        // dger with negative m
        assertDoesNotThrow(() ->
                BLAS.dger(negDim, posDim, 1.0, x, 0, 1, y, 0, 1, A, 0, posDim),
                "dger should silently return on m=" + negDim);

        // dger with negative n
        assertDoesNotThrow(() ->
                BLAS.dger(posDim, negDim, 1.0, x, 0, 1, y, 0, 1, A, 0, posDim),
                "dger should silently return on n=" + negDim);

        // dsymv with negative n
        assertDoesNotThrow(() ->
                BLAS.dsymv(BLAS.Uplo.Upper, negDim, 1.0, A, 0, posDim, x, 0, 1, 0.0, y, 0, 1),
                "dsymv should silently return on n=" + negDim);

        // dsyr2 with negative n
        assertDoesNotThrow(() ->
                BLAS.dsyr2(BLAS.Uplo.Upper, negDim, 1.0, x, 0, 1, y, 0, 1, A, 0, posDim),
                "dsyr2 should silently return on n=" + negDim);
    }

    // ========================================================================
    // Property 7.3: BLAS Level 3 routines silently return on negative m, n, or k
    // ========================================================================

    /**
     * For all negative m, n, or k values, BLAS Level 3 routines must silently return
     * without throwing any exception.
     *
     */
    @Property(tries = 100)
    @Label("Feature: blas-api-refactor, Property 7: BLAS Level 3 silently returns on negative m/n/k")
    void blasLevel3SilentlyReturnsOnNegativeDimensions(
            @ForAll @IntRange(min = -100, max = -1) int negDim,
            @ForAll @IntRange(min = 1, max = 10) int posDim
    ) {
        double[] A = mat(posDim, posDim);
        double[] B = mat(posDim, posDim);
        double[] C = mat(posDim, posDim);

        // dgemm with negative m
        assertDoesNotThrow(() ->
                BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans,
                        negDim, posDim, posDim, 1.0, A, 0, posDim, B, 0, posDim, 0.0, C, 0, posDim),
                "dgemm should silently return on m=" + negDim);

        // dgemm with negative n
        assertDoesNotThrow(() ->
                BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans,
                        posDim, negDim, posDim, 1.0, A, 0, posDim, B, 0, posDim, 0.0, C, 0, posDim),
                "dgemm should silently return on n=" + negDim);

        // dsyrk with negative n
        assertDoesNotThrow(() ->
                BLAS.dsyrk(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans,
                        negDim, posDim, 1.0, A, 0, posDim, 0.0, C, 0, posDim),
                "dsyrk should silently return on n=" + negDim);

        // dsyrk with negative k
        assertDoesNotThrow(() ->
                BLAS.dsyrk(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans,
                        posDim, negDim, 1.0, A, 0, posDim, 0.0, C, 0, posDim),
                "dsyrk should silently return on k=" + negDim);

        // dtrsm with negative m
        assertDoesNotThrow(() ->
                BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit,
                        negDim, posDim, 1.0, A, 0, posDim, B, 0, posDim),
                "dtrsm should silently return on m=" + negDim);

        // dtrsm with negative n
        assertDoesNotThrow(() ->
                BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit,
                        posDim, negDim, 1.0, A, 0, posDim, B, 0, posDim),
                "dtrsm should silently return on n=" + negDim);
    }

    // ========================================================================
    // Property 7.4: LAPACK dgetrf returns negative info on negative m or n
    // ========================================================================

    /**
     * For all negative m or n values, LAPACK dgetrf must return a negative info value.
     *
     */
    @Property(tries = 100)
    @Label("Feature: blas-api-refactor, Property 7: dgetrf returns negative info on m < 0 or n < 0")
    void dgetrfReturnsNegativeInfoOnNegativeDimensions(
            @ForAll @IntRange(min = -1000, max = -1) int negDim
    ) {
        double[] A = vec(1);
        int[] ipiv = new int[1];

        // dgetrf: m < 0 should return negative info
        int infoNegM = BLAS.dgetrf(negDim, 1, A, 0, 1, ipiv, 0);
        assertTrue(infoNegM < 0,
                "dgetrf should return negative info on m=" + negDim + ", got " + infoNegM);

        // dgetrf: n < 0 should return negative info
        int infoNegN = BLAS.dgetrf(1, negDim, A, 0, 1, ipiv, 0);
        assertTrue(infoNegN < 0,
                "dgetrf should return negative info on n=" + negDim + ", got " + infoNegN);
    }

    // ========================================================================
    // Property 7.5: LAPACK dpotrf returns negative info on negative n
    // ========================================================================

    /**
     * For all negative n values, LAPACK dpotrf must return a negative info value.
     *
     */
    @Property(tries = 100)
    @Label("Feature: blas-api-refactor, Property 7: dpotrf returns negative info on n < 0")
    void dpotrfReturnsNegativeInfoOnNegativeN(
            @ForAll @IntRange(min = -1000, max = -1) int n
    ) {
        double[] A = vec(1);

        int info = BLAS.dpotrf(BLAS.Uplo.Upper, n, A, 0, 1);
        assertTrue(info < 0,
                "dpotrf should return negative info on n=" + n + ", got " + info);

        int infoLower = BLAS.dpotrf(BLAS.Uplo.Lower, n, A, 0, 1);
        assertTrue(infoLower < 0,
                "dpotrf (Lower) should return negative info on n=" + n + ", got " + infoLower);
    }

    // ========================================================================
    // Property 7.6: LAPACK dgeqrf returns negative info on negative m or n
    // ========================================================================

    /**
     * For all negative m or n values, LAPACK dgeqrf must return a negative info value.
     *
     */
    @Property(tries = 100)
    @Label("Feature: blas-api-refactor, Property 7: dgeqrf returns negative info on m < 0 or n < 0")
    void dgeqrfReturnsNegativeInfoOnNegativeDimensions(
            @ForAll @IntRange(min = -1000, max = -1) int negDim
    ) {
        double[] A = vec(1);
        double[] tau = vec(1);
        double[] work = vec(1);

        // dgeqrf: m < 0 should return negative info
        int infoNegM = BLAS.dgeqrf(negDim, 1, A, 0, 1, tau, 0, work, 0, 1);
        assertTrue(infoNegM < 0,
                "dgeqrf should return negative info on m=" + negDim + ", got " + infoNegM);

        // dgeqrf: n < 0 should return negative info
        int infoNegN = BLAS.dgeqrf(1, negDim, A, 0, 1, tau, 0, work, 0, 1);
        assertTrue(infoNegN < 0,
                "dgeqrf should return negative info on n=" + negDim + ", got " + infoNegN);
    }

    // ========================================================================
    // Property 7.7: BLAS Level 1 routines do not modify output on n < 0
    // ========================================================================

    /**
     * For all negative n values, BLAS Level 1 routines must not modify the output
     * arrays (silent return means no side effects).
     *
     */
    @Property(tries = 100)
    @Label("Feature: blas-api-refactor, Property 7: BLAS Level 1 no side effects on n < 0")
    void blasLevel1NoSideEffectsOnNegativeN(
            @ForAll @IntRange(min = -1000, max = -1) int n,
            @ForAll @DoubleRange(min = -100.0, max = 100.0) double initialValue
    ) {
        double[] x = new double[]{initialValue};
        double[] y = new double[]{initialValue};

        // daxpy should not modify y
        BLAS.daxpy(n, 2.0, x, 0, 1, y, 0, 1);
        assertEquals(initialValue, y[0], 0.0,
                "daxpy should not modify y on n=" + n);

        // dcopy should not modify y
        y[0] = initialValue;
        BLAS.dcopy(n, x, 0, 1, y, 0, 1);
        assertEquals(initialValue, y[0], 0.0,
                "dcopy should not modify y on n=" + n);

        // dscal should not modify x
        x[0] = initialValue;
        BLAS.dscal(n, 2.0, x, 0, 1);
        assertEquals(initialValue, x[0], 0.0,
                "dscal should not modify x on n=" + n);
    }

    // ========================================================================
    // Property 7.8: BLAS Level 3 dgemm does not modify C on negative m or n
    // ========================================================================

    /**
     * For all negative m or n values, dgemm must not modify the output matrix C.
     *
     */
    @Property(tries = 100)
    @Label("Feature: blas-api-refactor, Property 7: dgemm no side effects on negative m/n")
    void dgemmNoSideEffectsOnNegativeDimensions(
            @ForAll @IntRange(min = -100, max = -1) int negDim,
            @ForAll @IntRange(min = 1, max = 5) int posDim,
            @ForAll @DoubleRange(min = -10.0, max = 10.0) double cValue
    ) {
        double[] A = mat(posDim, posDim);
        double[] B = mat(posDim, posDim);
        double[] C = new double[]{cValue};

        // dgemm with negative m should not modify C
        BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans,
                negDim, posDim, posDim, 1.0, A, 0, posDim, B, 0, posDim, 0.0, C, 0, posDim);
        assertEquals(cValue, C[0], 0.0,
                "dgemm should not modify C on m=" + negDim);

        // dgemm with negative n should not modify C
        C[0] = cValue;
        BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans,
                posDim, negDim, posDim, 1.0, A, 0, posDim, B, 0, posDim, 0.0, C, 0, posDim);
        assertEquals(cValue, C[0], 0.0,
                "dgemm should not modify C on n=" + negDim);
    }
}
