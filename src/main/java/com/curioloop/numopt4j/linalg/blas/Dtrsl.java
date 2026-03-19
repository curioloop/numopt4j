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
        boolean upper = (uplo == BLAS.Uplo.Upper);
        boolean transpose = (trans == BLAS.Trans.Trans || trans == BLAS.Trans.Conj);

        if (!upper) {
            if (!transpose) {
                return solveLowerN(t, tOff, ldt, n, b, bOff);
            } else {
                return solveLowerT(t, tOff, ldt, n, b, bOff);
            }
        } else {
            if (!transpose) {
                return solveUpperN(t, tOff, ldt, n, b, bOff);
            } else {
                return solveUpperT(t, tOff, ldt, n, b, bOff);
            }
        }
    }

    /**
     * Solve T*x = b where T is lower triangular.
     * Optimized with delayed singularity check and inlined operations.
     */
    static int solveLowerN(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff) {
        if (n <= 0) return 0;  // Handle empty case

        double diag = t[tOff];
        if (diag == 0.0) return 1;
        b[bOff] /= diag;

        for (int j = 1; j < n; j++) {
            int jCol = tOff + j * ldt;
            diag = t[jCol + j];
            if (diag == 0.0) return j + 1;

            // Inline daxpy: b[j:n] += (-b[j-1]) * t[j*ldt+(j-1) + i*ldt] for i=0..n-j-1
            double scale = -b[bOff + j - 1];
            for (int i = 0; i < n - j; i++) {
                b[bOff + j + i] = FMA.op(scale, t[jCol + (j - 1) + i * ldt], b[bOff + j + i]);
            }
            b[bOff + j] /= diag;
        }
        return 0;
    }

    /**
     * Solve Tᵀ*x = b where T is lower triangular.
     * Optimized with delayed singularity check and inlined ddot.
     */
    static int solveLowerT(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff) {
        if (n <= 0) return 0;  // Handle empty case

        int lastCol = tOff + (n - 1) * ldt;
        double diag = t[lastCol + (n - 1)];
        if (diag == 0.0) return n;
        b[bOff + n - 1] /= diag;

        for (int jj = 1; jj < n; jj++) {
            int j = n - 1 - jj;
            int jCol = tOff + j * ldt;
            diag = t[jCol + j];
            if (diag == 0.0) return j + 1;

            // Inline ddot: sum of t[(j+1)*ldt + j + i*ldt] * b[j+1+i] for i=0..jj-1
            double dot = 0.0;
            for (int i = 0; i < jj; i++) {
                dot = FMA.op(t[tOff + (j + 1) * ldt + j + i * ldt], b[bOff + j + 1 + i], dot);
            }
            b[bOff + j] = (b[bOff + j] - dot) / diag;
        }
        return 0;
    }

    /**
     * Solve T*x = b where T is upper triangular.
     * Optimized with delayed singularity check and inlined daxpy.
     */
    static int solveUpperN(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff) {
        if (n <= 0) return 0;  // Handle empty case

        int lastCol = tOff + (n - 1) * ldt;
        double diag = t[lastCol + (n - 1)];
        if (diag == 0.0) return n;
        b[bOff + n - 1] /= diag;

        for (int jj = 1; jj < n; jj++) {
            int j = n - 1 - jj;
            int jCol = tOff + j * ldt;
            diag = t[jCol + j];
            if (diag == 0.0) return j + 1;

            // Inline daxpy: b[0:j+1] += (-b[j+1]) * t[j+1 + i*ldt] for i=0..j
            double scale = -b[bOff + j + 1];
            for (int i = 0; i <= j; i++) {
                b[bOff + i] = FMA.op(scale, t[tOff + j + 1 + i * ldt], b[bOff + i]);
            }
            b[bOff + j] /= diag;
        }
        return 0;
    }

    /**
     * Solve Tᵀ*x = b where T is upper triangular.
     * Optimized with delayed singularity check and inlined ddot.
     */
    static int solveUpperT(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff) {
        if (n <= 0) return 0;  // Handle empty case

        double diag = t[tOff];
        if (diag == 0.0) return 1;
        b[bOff] /= diag;

        for (int j = 1; j < n; j++) {
            int jCol = tOff + j * ldt;
            diag = t[jCol + j];
            if (diag == 0.0) return j + 1;

            // Inline ddot: sum of t[j + i*ldt] * b[i] for i=0..j-1
            double dot = 0.0;
            for (int i = 0; i < j; i++) {
                dot = FMA.op(t[tOff + j + i * ldt], b[bOff + i], dot);
            }
            b[bOff + j] = (b[bOff + j] - dot) / diag;
        }
        return 0;
    }
}
