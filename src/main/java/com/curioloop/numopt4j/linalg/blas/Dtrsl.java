/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * LINPACK triangular system solver - optimized implementation.
 *
 * This file contains triangular system solvers used by optimization algorithms.
 * The implementations are optimized for small matrices (n ≤ 40) typical in
 * L-BFGS-B and similar algorithms.
 *
 * Reference: LINPACK Users' Guide, Dongarra et al., SIAM, 1979.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Triangular system solver (LINPACK dtrsl / BLAS dtrsv style).
 *
 * <p>Solves systems of the form T*x = b or Tᵀ*x = b where T is a triangular
 * matrix of order n.</p>
 *
 * <h2>Solve Options</h2>
 * <p>Uses standard BLAS-style character parameters:</p>
 * <ul>
 *   <li>uplo: 'L' for lower triangular, 'U' for upper triangular</li>
 *   <li>trans: 'N' for no transpose (T*x = b), 'T' for transpose (Tᵀ*x = b)</li>
 * </ul>
 *
 * <h2>Optimization Notes</h2>
 * <p>This implementation is optimized for small matrices (n ≤ 40):</p>
 * <ul>
 *   <li>Inlined BLAS operations to reduce function call overhead</li>
 *   <li>4-way unrolled scalar row/column walks in the hot solve kernels</li>
 *   <li>Zero-skip on axpy-style no-trans updates when the solved entry is zero</li>
 *   <li>Delayed singularity check for better branch prediction</li>
 *   <li>Specialized methods for each solve type</li>
 * </ul>
 *
 * @see <a href="https://www.netlib.org/linpack/">LINPACK</a>
 */
interface Dtrsl {

    /**
     * Solve triangular system T*x = b or Tᵀ*x = b.
     *
     * <p>Solves systems of the form T * x = b or Tᵀ * x = b where T is a triangular
     * matrix of order n.</p>
     *
     * <h3>On entry:</h3>
     * <ul>
     *   <li>t - Contains the matrix of the system. The zero elements of the
     *       matrix are not referenced, and the corresponding elements of
     *       the array can be used to store other information.</li>
     *   <li>tOff - Offset into array t</li>
     *   <li>ldt - The leading dimension of the array t</li>
     *   <li>n - The order of the system</li>
     *   <li>b - Contains the right hand side of the system</li>
     *   <li>bOff - Offset into array b</li>
     *   <li>uplo - 'L' for lower triangular, 'U' for upper triangular</li>
     *   <li>trans - 'N' for T*x = b, 'T' for Tᵀ*x = b</li>
     * </ul>
     *
     * <h3>On return:</h3>
     * <ul>
     *   <li>b - Contains the solution, if info = 0. Otherwise b is unaltered.</li>
     * </ul>
     *
     * @param t     the triangular matrix (column-major storage)
     * @param tOff  offset into array t
     * @param ldt   the leading dimension of the array t
     * @param n     the order of the system
     * @param b     the right hand side vector (modified in place with solution)
     * @param bOff  offset into array b
     * @param uplo  'L' for lower triangular, 'U' for upper triangular
     * @param trans 'N' for no transpose, 'T' for transpose
     * @return 0 if the system is nonsingular, otherwise the index of the first
     *         zero diagonal element of t
     */
    static int dtrsl(double[] t, int tOff, int ldt, int n,
                     double[] b, int bOff, BLAS.Uplo uplo, BLAS.Trans trans) {
        return dtrsl(t, tOff, ldt, n, b, bOff, uplo, trans, BLAS.Diag.NonUnit);
    }

    static int dtrsl(double[] t, int tOff, int ldt, int n,
                     double[] b, int bOff, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag) {
        boolean upper = (uplo == BLAS.Uplo.Upper);
        boolean transpose = (trans == BLAS.Trans.Trans || trans == BLAS.Trans.Conj);
        boolean nounit = (diag == BLAS.Diag.NonUnit);

        if (!upper) {
            if (!transpose) {
                return solveLowerN(t, tOff, ldt, n, b, bOff, nounit);
            } else {
                return solveLowerT(t, tOff, ldt, n, b, bOff, nounit);
            }
        } else {
            if (!transpose) {
                return solveUpperN(t, tOff, ldt, n, b, bOff, nounit);
            } else {
                return solveUpperT(t, tOff, ldt, n, b, bOff, nounit);
            }
        }
    }

    /**
     * Solve T*x = b where T is lower triangular.
     * Optimized with delayed singularity check and inlined operations.
     */
    static int solveLowerN(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff, boolean nounit) {
        if (n <= 0) return 0;  // Handle empty case

        double diag = 1.0;
        if (nounit) {
            diag = t[tOff];
            if (diag == 0.0) return 1;
            b[bOff] /= diag;
        }

        for (int j = 1; j < n; j++) {
            int jCol = tOff + j * ldt;
            if (nounit) {
                diag = t[jCol + j];
                if (diag == 0.0) return j + 1;
            }

            double scale = -b[bOff + j - 1];
            int bIndex = bOff + j;
            if (scale != 0.0) {
                int tIndex = jCol + j - 1;
                int limit = n - j;
                int i = 0;
                int unrolled = limit & ~3;
                for (; i < unrolled; i += 4) {
                    b[bIndex + i] = Math.fma(scale, t[tIndex], b[bIndex + i]);
                    b[bIndex + i + 1] = Math.fma(scale, t[tIndex + ldt], b[bIndex + i + 1]);
                    b[bIndex + i + 2] = Math.fma(scale, t[tIndex + 2 * ldt], b[bIndex + i + 2]);
                    b[bIndex + i + 3] = Math.fma(scale, t[tIndex + 3 * ldt], b[bIndex + i + 3]);
                    tIndex += 4 * ldt;
                }
                for (; i < limit; i++) {
                    b[bIndex + i] = Math.fma(scale, t[tIndex], b[bIndex + i]);
                    tIndex += ldt;
                }
            }
            if (nounit) {
                b[bIndex] /= diag;
            }
        }
        return 0;
    }

    /**
     * Solve Tᵀ*x = b where T is lower triangular.
     * Optimized with delayed singularity check and inlined ddot.
     */
    static int solveLowerT(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff, boolean nounit) {
        if (n <= 0) return 0;  // Handle empty case

        int lastCol = tOff + (n - 1) * ldt;
        double diag = 1.0;
        if (nounit) {
            diag = t[lastCol + (n - 1)];
            if (diag == 0.0) return n;
            b[bOff + n - 1] /= diag;
        }

        for (int jj = 1; jj < n; jj++) {
            int j = n - 1 - jj;
            int jCol = tOff + j * ldt;
            if (nounit) {
                diag = t[jCol + j];
                if (diag == 0.0) return j + 1;
            }

            int tIndex = tOff + (j + 1) * ldt + j;
            int bIndex = bOff + j + 1;
            double dot0 = 0.0;
            double dot1 = 0.0;
            double dot2 = 0.0;
            double dot3 = 0.0;
            int i = 0;
            int unrolled = jj & ~3;
            for (; i < unrolled; i += 4) {
                dot0 = Math.fma(t[tIndex], b[bIndex], dot0);
                dot1 = Math.fma(t[tIndex + ldt], b[bIndex + 1], dot1);
                dot2 = Math.fma(t[tIndex + 2 * ldt], b[bIndex + 2], dot2);
                dot3 = Math.fma(t[tIndex + 3 * ldt], b[bIndex + 3], dot3);
                tIndex += 4 * ldt;
                bIndex += 4;
            }
            double dot = (dot0 + dot1) + (dot2 + dot3);
            for (; i < jj; i++) {
                dot = Math.fma(t[tIndex], b[bIndex], dot);
                tIndex += ldt;
                bIndex++;
            }
            if (nounit) {
                b[bOff + j] = (b[bOff + j] - dot) / diag;
            } else {
                b[bOff + j] -= dot;
            }
        }
        return 0;
    }

    /**
     * Solve T*x = b where T is upper triangular.
     * Optimized with delayed singularity check and inlined daxpy.
     */
    static int solveUpperN(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff, boolean nounit) {
        if (n <= 0) return 0;  // Handle empty case

        int lastCol = tOff + (n - 1) * ldt;
        double diag = 1.0;
        if (nounit) {
            diag = t[lastCol + (n - 1)];
            if (diag == 0.0) return n;
            b[bOff + n - 1] /= diag;
        }

        for (int jj = 1; jj < n; jj++) {
            int j = n - 1 - jj;
            int jCol = tOff + j * ldt;
            if (nounit) {
                diag = t[jCol + j];
                if (diag == 0.0) return j + 1;
            }

            double scale = -b[bOff + j + 1];
            if (scale != 0.0) {
                int tIndex = tOff + j + 1;
                int i = 0;
                int limit = j + 1;
                int unrolled = limit & ~3;
                for (; i < unrolled; i += 4) {
                    b[bOff + i] = Math.fma(scale, t[tIndex], b[bOff + i]);
                    b[bOff + i + 1] = Math.fma(scale, t[tIndex + ldt], b[bOff + i + 1]);
                    b[bOff + i + 2] = Math.fma(scale, t[tIndex + 2 * ldt], b[bOff + i + 2]);
                    b[bOff + i + 3] = Math.fma(scale, t[tIndex + 3 * ldt], b[bOff + i + 3]);
                    tIndex += 4 * ldt;
                }
                for (; i < limit; i++) {
                    b[bOff + i] = Math.fma(scale, t[tIndex], b[bOff + i]);
                    tIndex += ldt;
                }
            }
            if (nounit) {
                b[bOff + j] /= diag;
            }
        }
        return 0;
    }

    /**
     * Solve Tᵀ*x = b where T is upper triangular.
     * Optimized with delayed singularity check and inlined ddot.
     */
    static int solveUpperT(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff, boolean nounit) {
        if (n <= 0) return 0;  // Handle empty case

        double diag = 1.0;
        if (nounit) {
            diag = t[tOff];
            if (diag == 0.0) return 1;
            b[bOff] /= diag;
        }

        for (int j = 1; j < n; j++) {
            int jCol = tOff + j * ldt;
            if (nounit) {
                diag = t[jCol + j];
                if (diag == 0.0) return j + 1;
            }

            int tIndex = tOff + j;
            int bIndex = bOff;
            double dot0 = 0.0;
            double dot1 = 0.0;
            double dot2 = 0.0;
            double dot3 = 0.0;
            int i = 0;
            int unrolled = j & ~3;
            for (; i < unrolled; i += 4) {
                dot0 = Math.fma(t[tIndex], b[bIndex], dot0);
                dot1 = Math.fma(t[tIndex + ldt], b[bIndex + 1], dot1);
                dot2 = Math.fma(t[tIndex + 2 * ldt], b[bIndex + 2], dot2);
                dot3 = Math.fma(t[tIndex + 3 * ldt], b[bIndex + 3], dot3);
                tIndex += 4 * ldt;
                bIndex += 4;
            }
            double dot = (dot0 + dot1) + (dot2 + dot3);
            for (; i < j; i++) {
                dot = Math.fma(t[tIndex], b[bIndex], dot);
                tIndex += ldt;
                bIndex++;
            }
            if (nounit) {
                b[bOff + j] = (b[bOff + j] - dot) / diag;
            } else {
                b[bOff + j] -= dot;
            }
        }
        return 0;
    }
}
