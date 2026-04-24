/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Tridiagonal linear system solvers.
 *
 * <ul>
 *   <li>DGTSV  — general tridiagonal A*X=B via Gaussian elimination with partial pivoting</li>
 *   <li>DPTTRF — L*D*Lᵀ factorization of a symmetric positive definite tridiagonal matrix</li>
 *   <li>DPTTRS — solve A*X=B using L*D*Lᵀ factorization (DPTTS2 core)</li>
 *   <li>DPTSV  — combined factorize + solve for SPD tridiagonal systems</li>
 * </ul>
 *
 */
interface Dptsv {

    // ==================== DGTSV ====================

    /**
     * Solves A * X = B where A is an n×n tridiagonal matrix.
     * Uses Gaussian elimination with partial pivoting.
     * The equation Aᵀ * X = B may be solved by swapping dl and du.
     *
     * <p>On entry, dl, d, du contain the sub-diagonal, diagonal, and super-diagonal of A.
     * On return, the first n-2 elements of dl, first n-1 of du, and all of d may be overwritten.</p>
     *
     * @param n     order of matrix A
     * @param nrhs  number of right-hand sides
     * @param dl    sub-diagonal (length n-1), modified on exit
     * @param dlOff offset into dl
     * @param d     diagonal (length n), modified on exit
     * @param dOff  offset into d
     * @param du    super-diagonal (length n-1), modified on exit
     * @param duOff offset into du
     * @param B     right-hand side matrix (n × nrhs, row-major), overwritten with solution
     * @param bOff  offset into B
     * @param ldb   leading dimension of B (>= nrhs)
     * @return true on success, false if A is singular
     */
    static boolean dgtsv(int n, int nrhs,
                         double[] dl, int dlOff,
                         double[] d,  int dOff,
                         double[] du, int duOff,
                         double[] B,  int bOff, int ldb) {
        if (n == 0 || nrhs == 0) return true;

        for (int i = 0; i < n - 1; i++) {
            double di  = d[dOff + i];
            double dli = dl[dlOff + i];
            if (Math.abs(di) >= Math.abs(dli)) {
                // No row interchange required.
                if (di == 0) return false;
                double fact = dli / di;
                d[dOff + i + 1] = Math.fma(-fact, du[duOff + i], d[dOff + i + 1]);
                for (int j = 0; j < nrhs; j++)
                    B[bOff + (i + 1) * ldb + j] = Math.fma(-fact, B[bOff + i * ldb + j], B[bOff + (i + 1) * ldb + j]);
                dl[dlOff + i] = 0;
            } else {
                // Interchange rows i and i+1.
                double fact = di / dli;
                d[dOff + i] = dli;
                double tmp = d[dOff + i + 1];
                d[dOff + i + 1] = Math.fma(-fact, tmp, du[duOff + i]);
                du[duOff + i] = tmp;
                if (i + 1 < n - 1) {
                    dl[dlOff + i] = du[duOff + i + 1];
                    du[duOff + i + 1] = -fact * dl[dlOff + i];
                }
                for (int j = 0; j < nrhs; j++) {
                    tmp = B[bOff + i * ldb + j];
                    B[bOff + i * ldb + j] = B[bOff + (i + 1) * ldb + j];
                    B[bOff + (i + 1) * ldb + j] = Math.fma(-fact, B[bOff + (i + 1) * ldb + j], tmp);
                }
            }
        }
        if (d[dOff + n - 1] == 0) return false;

        // Back solve with the upper triangular matrix U.
        for (int j = 0; j < nrhs; j++) {
            B[bOff + (n - 1) * ldb + j] /= d[dOff + n - 1];
            if (n > 1)
                B[bOff + (n - 2) * ldb + j] = Math.fma(-du[duOff + n - 2], B[bOff + (n - 1) * ldb + j],
                        B[bOff + (n - 2) * ldb + j]) / d[dOff + n - 2];
            for (int i = n - 3; i >= 0; i--)
                B[bOff + i * ldb + j] = Math.fma(-dl[dlOff + i], B[bOff + (i + 2) * ldb + j],
                        Math.fma(-du[duOff + i], B[bOff + (i + 1) * ldb + j], B[bOff + i * ldb + j]))
                        / d[dOff + i];
        }
        return true;
    }

    // ==================== DPTTRF ====================

    /**
     * Computes the L*D*Lᵀ factorization of an n×n symmetric positive definite tridiagonal matrix A.
     *
     * <p>On entry, d contains the n diagonal elements and e contains the (n-1) subdiagonal elements of A.
     * On return, d contains the diagonal of D and e contains the subdiagonal of the unit bidiagonal L.</p>
     *
     * @param n    order of matrix A
     * @param d    diagonal elements (length n), overwritten with D on exit
     * @param dOff offset into d
     * @param e    subdiagonal elements (length n-1), overwritten with L on exit
     * @param eOff offset into e
     * @return true on success, false if A is not positive definite
     */
    static boolean dpttrf(int n, double[] d, int dOff, double[] e, int eOff) {
        if (n == 0) return true;

        int i4 = (n - 1) % 4;
        for (int i = 0; i < i4; i++) {
            if (d[dOff + i] <= 0) return false;
            double ei = e[eOff + i];
            e[eOff + i] /= d[dOff + i];
            d[dOff + i + 1] = Math.fma(-e[eOff + i], ei, d[dOff + i + 1]);
        }
        for (int i = i4; i < n - 4; i += 4) {
            if (d[dOff + i] <= 0) return false;
            double ei = e[eOff + i];
            e[eOff + i] /= d[dOff + i];
            d[dOff + i + 1] = Math.fma(-e[eOff + i], ei, d[dOff + i + 1]);

            if (d[dOff + i + 1] <= 0) return false;
            ei = e[eOff + i + 1];
            e[eOff + i + 1] /= d[dOff + i + 1];
            d[dOff + i + 2] = Math.fma(-e[eOff + i + 1], ei, d[dOff + i + 2]);

            if (d[dOff + i + 2] <= 0) return false;
            ei = e[eOff + i + 2];
            e[eOff + i + 2] /= d[dOff + i + 2];
            d[dOff + i + 3] = Math.fma(-e[eOff + i + 2], ei, d[dOff + i + 3]);

            if (d[dOff + i + 3] <= 0) return false;
            ei = e[eOff + i + 3];
            e[eOff + i + 3] /= d[dOff + i + 3];
            d[dOff + i + 4] = Math.fma(-e[eOff + i + 3], ei, d[dOff + i + 4]);
        }
        return d[dOff + n - 1] > 0;
    }

    // ==================== DPTTRS / DPTTS2 ====================

    /**
     * Solves A * X = B using the L*D*Lᵀ factorization from {@link #dpttrf}.
     *
     * @param n    order of matrix A
     * @param nrhs number of right-hand sides
     * @param d    diagonal of D from dpttrf (length n)
     * @param dOff offset into d
     * @param e    subdiagonal of L from dpttrf (length n-1)
     * @param eOff offset into e
     * @param B    right-hand side matrix (n × nrhs, row-major), overwritten with solution
     * @param bOff offset into B
     * @param ldb  leading dimension of B (>= nrhs)
     */
    static void dpttrs(int n, int nrhs,
                       double[] d, int dOff,
                       double[] e, int eOff,
                       double[] B, int bOff, int ldb) {
        if (n == 0 || nrhs == 0) return;
        if (n == 1) {
            double inv = 1.0 / d[dOff];
            for (int j = 0; j < nrhs; j++) B[bOff + j] *= inv;
            return;
        }
        for (int j = 0; j < nrhs; j++) {
            // Solve L * x = b.
            for (int i = 1; i < n; i++)
                B[bOff + i * ldb + j] = Math.fma(-e[eOff + i - 1], B[bOff + (i - 1) * ldb + j], B[bOff + i * ldb + j]);
            // Solve D * Lᵀ * x = b.
            B[bOff + (n - 1) * ldb + j] /= d[dOff + n - 1];
            for (int i = n - 2; i >= 0; i--)
                B[bOff + i * ldb + j] = B[bOff + i * ldb + j] / d[dOff + i]
                        - B[bOff + (i + 1) * ldb + j] * e[eOff + i];
        }
    }

    // ==================== DPTSV ====================

    /**
     * Computes the solution to A * X = B where A is an n×n symmetric positive definite tridiagonal matrix.
     * Factors A = L*D*Lᵀ via {@link #dpttrf} then solves via {@link #dpttrs}.
     *
     * <p>On return, d and e are overwritten with the L*D*Lᵀ factorization.</p>
     *
     * @param n    order of matrix A
     * @param nrhs number of right-hand sides
     * @param d    diagonal elements (length n), overwritten with D on exit
     * @param dOff offset into d
     * @param e    subdiagonal elements (length n-1), overwritten with L on exit
     * @param eOff offset into e
     * @param B    right-hand side matrix (n × nrhs, row-major), overwritten with solution
     * @param bOff offset into B
     * @param ldb  leading dimension of B (>= nrhs)
     * @return true on success, false if A is not positive definite
     */
    static boolean dptsv(int n, int nrhs,
                         double[] d, int dOff,
                         double[] e, int eOff,
                         double[] B, int bOff, int ldb) {
        if (n == 0 || nrhs == 0) return true;
        if (!dpttrf(n, d, dOff, e, eOff)) return false;
        dpttrs(n, nrhs, d, dOff, e, eOff, B, bOff, ldb);
        return true;
    }
}
