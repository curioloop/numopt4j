/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Applies a block reflector H = I - V*T*V^T or its transpose H^T to a matrix C.
 * LAPACK DLARFB algorithm (adapted for row-major storage).
 *
 * <p>This operation is fundamental to blocked QR factorization algorithms.</p>
 * 
 * <p>Important: V is stored in row-major format with column-wise storage of Householder vectors.
 * V[row * ldv + col] = V(row, col), where col is the reflector index.</p>
 *
 * <p>Storage conventions (row-major):</p>
 * <ul>
 *   <li>ColumnWise: V columns contain Householder vectors</li>
 *   <li>RowWise: V rows contain Householder vectors</li>
 * </ul>
 *
 * <p>Direction conventions:</p>
 * <ul>
 *   <li>Forward: H = H_0 * H_1 * ... * H_{k-1}</li>
 *   <li>Backward: H = H_{k-1} * ... * H_1 * H_0</li>
 * </ul>
 */
public interface Dlarfb {

    /**
     * Applies a block reflector H or H^T to a matrix C.
     *
     * @param side    'L' for left multiplication (H*C or H^T*C), 'R' for right (C*H or C*H^T)
     * @param trans   'N' for no transpose (H), 'T' for transpose (H^T)
     * @param direct  'F' for forward direction, 'B' for backward direction
     * @param storev  'C' for column-wise storage, 'R' for row-wise storage
     * @param m       number of rows of C
     * @param n       number of columns of C
     * @param k       number of elementary reflectors
     * @param V       matrix containing Householder vectors
     * @param vOff    offset into V
     * @param ldv     leading dimension of V
     * @param T       triangular factor matrix (k x k)
     * @param tOff    offset into T
     * @param ldt     leading dimension of T
     * @param C       matrix to be multiplied (m x n), overwritten with result
     * @param cOff    offset into C
     * @param ldc     leading dimension of C
     * @param work    workspace array (at least n*k for left side, m*k for right side)
     * @param wOff    offset into work
     * @param ldwork  leading dimension of work
     */
    static void dlarfb(char side, char trans, char direct, char storev,
                       int m, int n, int k,
                       double[] V, int vOff, int ldv,
                       double[] T, int tOff, int ldt,
                       double[] C, int cOff, int ldc,
                       double[] work, int wOff, int ldwork) {
        boolean left = side == 'L' || side == 'l';
        boolean notrans = trans == 'N' || trans == 'n';
        boolean forward = direct == 'F' || direct == 'f';
        boolean colwise = storev == 'C' || storev == 'c';

        if (colwise) {
            if (forward) {
                if (left) {
                    if (notrans) {
                        dlarfbLeftForwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        dlarfbLeftTransForwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                } else {
                    if (notrans) {
                        dlarfbRightForwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        dlarfbRightTransForwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                }
            } else {
                if (left) {
                    if (notrans) {
                        dlarfbLeftBackwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        dlarfbLeftTransBackwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                } else {
                    if (notrans) {
                        dlarfbRightBackwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        dlarfbRightTransBackwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                }
            }
        } else {
            if (forward) {
                if (left) {
                    if (notrans) {
                        dlarfbLeftForwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        dlarfbLeftTransForwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                } else {
                    if (notrans) {
                        dlarfbRightForwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        dlarfbRightTransForwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                }
            } else {
                if (left) {
                    if (notrans) {
                        dlarfbLeftBackwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        dlarfbLeftTransBackwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                } else {
                    if (notrans) {
                        dlarfbRightBackwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        dlarfbRightTransBackwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                }
            }
        }
    }

    // ==================== ColumnWise + Forward ====================

    /**
     * Applies H*C where H is a block reflector with forward direction and column-wise storage.
     * V has the form: V = [1; v1; v1; v1; ...] where V1 (first k rows) is unit lower triangular.
     */
    static void dlarfbLeftForwardColWise(double[] V, int vOff, int ldv,
                                         double[] T, int tOff, int ldt,
                                         double[] C, int cOff, int ldc,
                                         double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // W = C1^T (copy first k rows of C to work)
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + i * ldwork + j] = C[cOff + j * ldc + i];
            }
        }

        // W = W * V1 (V1 is unit lower triangular)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // W = W + C2^T * V2 (if m > k)
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n, k, m - k, 1.0, C, cOff + k * ldc, ldc, V, vOff + k * ldv, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T^T (T is upper triangular for forward direction)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, n, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C2 = C2 - V2 * W^T (if m > k)
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m - k, n, k, -1.0, V, vOff + k * ldv, ldv, work, wOff, ldwork, 1.0, C, cOff + k * ldc, ldc);
        }

        // W = W * V1^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // C1 = C1 - W^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + j * ldc + i] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies H^T*C where H is a block reflector with forward direction and column-wise storage.
     */
    static void dlarfbLeftTransForwardColWise(double[] V, int vOff, int ldv,
                                              double[] T, int tOff, int ldt,
                                              double[] C, int cOff, int ldc,
                                              double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // W = C1^T
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + i * ldwork + j] = C[cOff + j * ldc + i];
            }
        }

        // W = W * V1
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // W = W + C2^T * V2
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n, k, m - k, 1.0, C, cOff + k * ldc, ldc, V, vOff + k * ldv, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T (not T^T for transpose case)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C2 = C2 - V2 * W^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m - k, n, k, -1.0, V, vOff + k * ldv, ldv, work, wOff, ldwork, 1.0, C, cOff + k * ldc, ldc);
        }

        // W = W * V1^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // C1 = C1 - W^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + j * ldc + i] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies C*H where H is a block reflector with forward direction and column-wise storage.
     */
    static void dlarfbRightForwardColWise(double[] V, int vOff, int ldv,
                                          double[] T, int tOff, int ldt,
                                          double[] C, int cOff, int ldc,
                                          double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // W = C1 (copy first k columns of C to work, row-major: work[i*ldwork+j] = C[i*ldc+j]
        copyLeadingColumnsToWork(C, cOff, ldc, work, wOff, ldwork, m, k);

        // W = W * V1 (V1 is k x k unit lower triangular)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // W = W + C2 * V2^T
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, k, n - k, 1.0, C, cOff + k, ldc, V, vOff + k * ldv, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C2 = C2 - W * V2^T
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m, n - k, k, -1.0, work, wOff, ldwork, V, vOff + k * ldv, ldv, 1.0, C, cOff + k, ldc);
        }

        // W = W * V1^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // C1 = C1 - W
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + j] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies C*H^T where H is a block reflector with forward direction and column-wise storage.
     */
    static void dlarfbRightTransForwardColWise(double[] V, int vOff, int ldv,
                                               double[] T, int tOff, int ldt,
                                               double[] C, int cOff, int ldc,
                                               double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // W = C1 (row-major)
        copyLeadingColumnsToWork(C, cOff, ldc, work, wOff, ldwork, m, k);

        // W = W * V1
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // W = W + C2 * V2^T
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, k, n - k, 1.0, C, cOff + k, ldc, V, vOff + k * ldv, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C2 = C2 - W * V2^T
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m, n - k, k, -1.0, work, wOff, ldwork, V, vOff + k * ldv, ldv, 1.0, C, cOff + k, ldc);
        }

        // W = W * V1^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // C1 = C1 - W
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + j] -= work[wOff + i * ldwork + j];
            }
        }
    }

    // ==================== ColumnWise + Backward ====================

    /**
     * Applies H*C where H is a block reflector with backward direction and column-wise storage.
     * V has the form: V = [v1 v2 v3; v1 v2 v3; 1 v2 v3; 1 1 v3; 1 1 1]
     * where V2 (last k rows) is unit upper triangular.
     */
    static void dlarfbLeftBackwardColWise(double[] V, int vOff, int ldv,
                                          double[] T, int tOff, int ldt,
                                          double[] C, int cOff, int ldc,
                                          double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (m - k) * ldv;

        // W = C2^T (copy last k rows of C to work)
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + i * ldwork + j] = C[cOff + (m - k + j) * ldc + i];
            }
        }

        // W = W * V2 (V2 is unit upper triangular)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // W = W + C1^T * V1 (if m > k)
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n, k, m - k, 1.0, C, cOff, ldc, V, vOff, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T^T (T is lower triangular for backward direction, trans=NoTrans means use T^T)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, n, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C1 = C1 - V1 * W^T (if m > k)
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m - k, n, k, -1.0, V, vOff, ldv, work, wOff, ldwork, 1.0, C, cOff, ldc);
        }

        // W = W * V2^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, n, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // C2 = C2 - W^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + (m - k + j) * ldc + i] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies H^T*C where H is a block reflector with backward direction and column-wise storage.
     */
    static void dlarfbLeftTransBackwardColWise(double[] V, int vOff, int ldv,
                                               double[] T, int tOff, int ldt,
                                               double[] C, int cOff, int ldc,
                                               double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (m - k) * ldv;

        // W = C2^T
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + i * ldwork + j] = C[cOff + (m - k + j) * ldc + i];
            }
        }

        // W = W * V2
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // W = W + C1^T * V1
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n, k, m - k, 1.0, C, cOff, ldc, V, vOff, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T (trans=Trans means use T, not T^T)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C1 = C1 - V1 * W^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m - k, n, k, -1.0, V, vOff, ldv, work, wOff, ldwork, 1.0, C, cOff, ldc);
        }

        // W = W * V2^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, n, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // C2 = C2 - W^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + (m - k + j) * ldc + i] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies C*H where H is a block reflector with backward direction and column-wise storage.
     */
    static void dlarfbRightBackwardColWise(double[] V, int vOff, int ldv,
                                           double[] T, int tOff, int ldt,
                                           double[] C, int cOff, int ldc,
                                           double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (n - k) * ldv;

        // W = C2 (copy last k columns of C to work, row-major)
        copyTrailingColumnsToWork(C, cOff, ldc, work, wOff, ldwork, m, n, k);

        // W = W * V2 (V2 is unit upper triangular)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // W = W + C1 * V1^T
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, k, n - k, 1.0, C, cOff, ldc, V, vOff, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C1 = C1 - W * V1^T
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m, n - k, k, -1.0, work, wOff, ldwork, V, vOff, ldv, 1.0, C, cOff, ldc);
        }

        // W = W * V2^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, m, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // C2 = C2 - W
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + (n - k + j)] -= work[wOff + i * ldwork + j];
            }
        }
    }

    static void dlarfbRightTransBackwardColWise(double[] V, int vOff, int ldv,
                                                double[] T, int tOff, int ldt,
                                                double[] C, int cOff, int ldc,
                                                double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (n - k) * ldv;

        // W = C2 (row-major)
        copyTrailingColumnsToWork(C, cOff, ldc, work, wOff, ldwork, m, n, k);

        // W = W * V2
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // W = W + C1 * V1^T
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, k, n - k, 1.0, C, cOff, ldc, V, vOff, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C1 = C1 - W * V1^T
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m, n - k, k, -1.0, work, wOff, ldwork, V, vOff, ldv, 1.0, C, cOff, ldc);
        }

        // W = W * V2^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, m, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // C2 = C2 - W
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + (n - k + j)] -= work[wOff + i * ldwork + j];
            }
        }
    }

    // ==================== RowWise + Forward ====================

    /**
     * Applies H*C where H is a block reflector with forward direction and row-wise storage.
     * V has the form: V = [1 v1 v1 v1 v1; 1 v2 v2 v2; 1 v3 v3]
     * where V1 (first k columns) is unit upper triangular.
     */
    static void dlarfbLeftForwardRowWise(double[] V, int vOff, int ldv,
                                         double[] T, int tOff, int ldt,
                                         double[] C, int cOff, int ldc,
                                         double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // W = C1^T
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + i * ldwork + j] = C[cOff + j * ldc + i];
            }
        }

        // W = W * V1^T (V1 is unit upper triangular)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // W = W + C2^T * V2^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, n, k, m - k, 1.0, C, cOff + k * ldc, ldc, V, vOff + k, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, n, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C2 = C2 - V2^T * W^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, m - k, n, k, -1.0, V, vOff + k, ldv, work, wOff, ldwork, 1.0, C, cOff + k * ldc, ldc);
        }

        // W = W * V1
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // C1 = C1 - W^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + j * ldc + i] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies H^T*C where H is a block reflector with forward direction and row-wise storage.
     */
    static void dlarfbLeftTransForwardRowWise(double[] V, int vOff, int ldv,
                                              double[] T, int tOff, int ldt,
                                              double[] C, int cOff, int ldc,
                                              double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // W = C1^T
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + i * ldwork + j] = C[cOff + j * ldc + i];
            }
        }

        // W = W * V1^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // W = W + C2^T * V2^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, n, k, m - k, 1.0, C, cOff + k * ldc, ldc, V, vOff + k, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C2 = C2 - V2^T * W^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, m - k, n, k, -1.0, V, vOff + k, ldv, work, wOff, ldwork, 1.0, C, cOff + k * ldc, ldc);
        }

        // W = W * V1
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // C1 = C1 - W^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + j * ldc + i] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies C*H where H is a block reflector with forward direction and row-wise storage.
     */
    static void dlarfbRightForwardRowWise(double[] V, int vOff, int ldv,
                                          double[] T, int tOff, int ldt,
                                          double[] C, int cOff, int ldc,
                                          double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // W = C1 (row-major: work[i*ldwork+j] = C[i*ldc+j])
        copyLeadingColumnsToWork(C, cOff, ldc, work, wOff, ldwork, m, k);

        // W = W * V1^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // W = W + C2 * V2
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m, k, n - k, 1.0, C, cOff + k, ldc, V, vOff + k, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C2 = C2 - W * V2
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n - k, k, -1.0, work, wOff, ldwork, V, vOff + k, ldv, 1.0, C, cOff + k, ldc);
        }

        // W = W * V1
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // C1 = C1 - W
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + j] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies C*H^T where H is a block reflector with forward direction and row-wise storage.
     */
    static void dlarfbRightTransForwardRowWise(double[] V, int vOff, int ldv,
                                               double[] T, int tOff, int ldt,
                                               double[] C, int cOff, int ldc,
                                               double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // W = C1 (row-major: work[i*ldwork+j] = C[i*ldc+j])
        copyLeadingColumnsToWork(C, cOff, ldc, work, wOff, ldwork, m, k);

        // W = W * V1^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // W = W + C2 * V2
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m, k, n - k, 1.0, C, cOff + k, ldc, V, vOff + k, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T^T (Right side Forward Trans uses T^T)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C2 = C2 - W * V2
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n - k, k, -1.0, work, wOff, ldwork, V, vOff + k, ldv, 1.0, C, cOff + k, ldc);
        }

        // W = W * V1
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        // C1 = C1 - W
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + j] -= work[wOff + i * ldwork + j];
            }
        }
    }

    // ==================== RowWise + Backward ====================

    /**
     * Applies H*C where H is a block reflector with backward direction and row-wise storage.
     * V has the form: V = [v1 v1 1; v2 v2 v2 1; v3 v3 v3 v3 1]
     * where V2 (last k columns) is unit lower triangular.
     */
    static void dlarfbLeftBackwardRowWise(double[] V, int vOff, int ldv,
                                          double[] T, int tOff, int ldt,
                                          double[] C, int cOff, int ldc,
                                          double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (m - k);

        // W = C2^T
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + i * ldwork + j] = C[cOff + (m - k + j) * ldc + i];
            }
        }

        // W = W * V2^T (V2 is unit lower triangular)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, n, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // W = W + C1^T * V1^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, n, k, m - k, 1.0, C, cOff, ldc, V, vOff, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, n, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C1 = C1 - V1^T * W^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, m - k, n, k, -1.0, V, vOff, ldv, work, wOff, ldwork, 1.0, C, cOff, ldc);
        }

        // W = W * V2
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // C2 = C2 - W^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + (m - k + j) * ldc + i] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies H^T*C where H is a block reflector with backward direction and row-wise storage.
     */
    static void dlarfbLeftTransBackwardRowWise(double[] V, int vOff, int ldv,
                                               double[] T, int tOff, int ldt,
                                               double[] C, int cOff, int ldc,
                                               double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (m - k);

        // W = C2^T
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + i * ldwork + j] = C[cOff + (m - k + j) * ldc + i];
            }
        }

        // W = W * V2^T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, n, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // W = W + C1^T * V1^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, n, k, m - k, 1.0, C, cOff, ldc, V, vOff, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C1 = C1 - V1^T * W^T
        if (m > k) {
            BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, m - k, n, k, -1.0, V, vOff, ldv, work, wOff, ldwork, 1.0, C, cOff, ldc);
        }

        // W = W * V2
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // C2 = C2 - W^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + (m - k + j) * ldc + i] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies C*H where H is a block reflector with backward direction and row-wise storage.
     */
    static void dlarfbRightBackwardRowWise(double[] V, int vOff, int ldv,
                                           double[] T, int tOff, int ldt,
                                           double[] C, int cOff, int ldc,
                                           double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // W = C2 (row-major: work[i*ldwork+j] = C[i*ldc+(n-k+j)])
        // Copy last k columns of C to work
        copyTrailingColumnsToWork(C, cOff, ldc, work, wOff, ldwork, m, n, k);

        // W *= V2ᵀ.
        // V2 is the last k columns of V, stored as k×k lower unit triangular.
        // v[n-k:] points to the start of V2, with stride ldv
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, m, k, 1.0, V, vOff + n - k, ldv, work, wOff, ldwork);

        // W = W + C1 * V1^T
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m, k, n - k, 1.0, C, cOff, ldc, V, vOff, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T (T is lower triangular for backward direction)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C1 = C1 - W * V1
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n - k, k, -1.0, work, wOff, ldwork, V, vOff, ldv, 1.0, C, cOff, ldc);
        }

        // W *= V2.
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, vOff + n - k, ldv, work, wOff, ldwork);

        // C2 = C2 - W
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + (n - k + j)] -= work[wOff + i * ldwork + j];
            }
        }
    }

    /**
     * Applies C*H^T where H is a block reflector with backward direction and row-wise storage.
     */
    static void dlarfbRightTransBackwardRowWise(double[] V, int vOff, int ldv,
                                                double[] T, int tOff, int ldt,
                                                double[] C, int cOff, int ldc,
                                                double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (n - k);

        // W = C2 (row-major: work[i*ldwork+j] = C[i*ldc+(n-k+j)])
        copyTrailingColumnsToWork(C, cOff, ldc, work, wOff, ldwork, m, n, k);

        // W = W * V2^T (V2 is unit lower triangular)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // W = W + C1 * V1
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m, k, n - k, 1.0, C, cOff, ldc, V, vOff, ldv, 1.0, work, wOff, ldwork);
        }

        // W = W * T^T (Right side Backward Trans uses T^T)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        // C1 = C1 - W * V1
        if (n > k) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n - k, k, -1.0, work, wOff, ldwork, V, vOff, ldv, 1.0, C, cOff, ldc);
        }

        // W = W * V2 (V2 is unit lower triangular)
        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, V, v2Off, ldv, work, wOff, ldwork);

        // C2 = C2 - W
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + (n - k + j)] -= work[wOff + i * ldwork + j];
            }
        }
    }

    private static void copyLeadingColumnsToWork(double[] C, int cOff, int ldc,
                                                 double[] work, int wOff, int ldwork,
                                                 int rows, int width) {
        for (int i = 0; i < rows; i++) {
            System.arraycopy(C, cOff + i * ldc, work, wOff + i * ldwork, width);
        }
    }

    private static void copyTrailingColumnsToWork(double[] C, int cOff, int ldc,
                                                  double[] work, int wOff, int ldwork,
                                                  int rows, int n, int width) {
        int colOff = n - width;
        for (int i = 0; i < rows; i++) {
            System.arraycopy(C, cOff + i * ldc + colOff, work, wOff + i * ldwork, width);
        }
    }

}
