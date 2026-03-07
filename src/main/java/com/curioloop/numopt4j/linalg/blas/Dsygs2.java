/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DSYGS2 — reduces a real symmetric-definite generalized eigenproblem to standard form (unblocked).
 *
 * <p>Corresponds to LAPACK {@code DSYGS2}.
 *
 * <p>Given the Cholesky factor of B (from {@code DPOTRF}), overwrites A with the
 * equivalent standard symmetric eigenproblem:
 * <ul>
 *   <li>Type 1 ({@code A·x = λ·B·x}):  A ← inv(L)·A·inv(Lᵀ)  (lower)</li>
 *   <li>Type 2/3 ({@code A·B·x = λ·x} or {@code B·A·x = λ·x}):  A ← Lᵀ·A·L  (lower)</li>
 * </ul>
 *
 * <p>Only the lower (or upper) triangle of A is updated on exit; the caller must
 * symmetrize before passing to a standard eigensolver.
 *
 * <p>All matrices are stored in <b>row-major</b> order with leading dimension {@code n}.
 *
 * <p>Reference: <a href="https://netlib.org/lapack/explore-html/">LAPACK DSYGS2</a>
 */
interface Dsygs2 {

    /**
     * Reduces generalized symmetric eigenproblem to standard form (unblocked, row-major).
     *
     * @param itype problem type: 1 for A·x=λ·B·x; 2 or 3 for A·B·x=λ·x / B·A·x=λ·x
     * @param uplo  {@link BLAS.Uplo#Lower} or {@link BLAS.Uplo#Upper}
     * @param n     order of A and B
     * @param A     symmetric matrix (n×n, row-major), overwritten with reduced form
     * @param aOff  offset into A
     * @param lda   leading dimension of A (typically n)
     * @param B     Cholesky factor of B from dpotrf (n×n, row-major)
     * @param bOff  offset into B
     * @param ldb   leading dimension of B (typically n)
     */
    static void dsygs2(int itype, BLAS.Uplo uplo, int n,
                       double[] A, int aOff, int lda,
                       double[] B, int bOff, int ldb) {
        if (n == 0) return;

        boolean lower = (uplo == BLAS.Uplo.Lower);

        if (itype == 1) {
            if (lower) {
                dsygs2Type1Lower(n, A, aOff, lda, B, bOff, ldb);
            } else {
                dsygs2Type1Upper(n, A, aOff, lda, B, bOff, ldb);
            }
        } else {
            // type 2 and 3 use the same forward transform
            if (lower) {
                dsygs2Type23Lower(n, A, aOff, lda, B, bOff, ldb);
            } else {
                dsygs2Type23Upper(n, A, aOff, lda, B, bOff, ldb);
            }
        }
    }

    /**
     * Type 1, lower: A ← inv(L)·A·inv(Lᵀ).
     *
     * <p>Fortran reference (column-major, 1-indexed):
     * <pre>
     *   DO K = 1, N
     *     AKK = A(K,K) / BKK^2
     *     A(K+1:N, K) /= BKK
     *     A(K+1:N, K) += -0.5*AKK * B(K+1:N, K)
     *     A(K+1:N, K+1:N) += dsyr2(-1, A(K+1:N,K), B(K+1:N,K))
     *     A(K+1:N, K) += -0.5*AKK * B(K+1:N, K)
     *     A(K+1:N, K) = solve L(K+1:N,K+1:N) * x = A(K+1:N,K)
     *   END DO
     * </pre>
     *
     * <p>Row-major mapping: Fortran column vector A(K+1:N, K) with stride 1 (col-major)
     * maps to Java column vector A[(k+1)*lda+k .. (n-1)*lda+k] with stride lda.
     * We use dtrsm (n×1 system) to handle the non-unit stride triangular solve.
     */
    static void dsygs2Type1Lower(int n, double[] A, int aOff, int lda,
                                     double[] B, int bOff, int ldb) {
        for (int k = 0; k < n; k++) {
            double akk = A[aOff + k * lda + k];
            double bkk = B[bOff + k * ldb + k];
            akk = akk / (bkk * bkk);
            A[aOff + k * lda + k] = akk;

            int rem = n - k - 1;
            if (rem <= 0) continue;

            // Column vector: A(k+1:n-1, k), offset=(k+1)*lda+k, stride=lda
            int aColOff = aOff + (k + 1) * lda + k;
            int bColOff = bOff + (k + 1) * ldb + k;

            // A(k+1:n, k) /= bkk
            BLAS.dscal(rem, 1.0 / bkk, A, aColOff, lda);

            double ct = -0.5 * akk;

            // A(k+1:n, k) += ct * B(k+1:n, k)
            BLAS.daxpy(rem, ct, B, bColOff, ldb, A, aColOff, lda);

            // A(k+1:n, k+1:n) lower += dsyr2(-1, A(k+1:n,k), B(k+1:n,k))
            // Both vectors have stride lda (column vectors in row-major)
            BLAS.dsyr2(BLAS.Uplo.Lower, rem, -1.0,
                    A, aColOff, lda,
                    B, bColOff, ldb,
                    A, aOff + (k + 1) * lda + (k + 1), lda);

            // A(k+1:n, k) += ct * B(k+1:n, k)
            BLAS.daxpy(rem, ct, B, bColOff, ldb, A, aColOff, lda);

            // Solve L(k+1:n, k+1:n) * x = A(k+1:n, k)
            // dtrsm(Left, Lower, NoTrans, NonUnit, rem, 1, 1.0, B(k+1,k+1), ldb, A(k+1,k), lda)
            // Treats A(k+1:n, k) as a rem×1 matrix with ldb=lda
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit,
                    rem, 1, 1.0,
                    B, bOff + (k + 1) * ldb + (k + 1), ldb,
                    A, aColOff, lda);
        }
    }

    /**
     * Type 1, upper: A ← inv(Uᵀ)·A·inv(U).
     *
     * <p>Fortran reference (column-major, 1-indexed):
     * <pre>
     *   DO K = 1, N
     *     AKK = A(K,K) / BKK^2
     *     A(K, 1:K-1) /= BKK          ! col-major: row K, cols 1..K-1, stride LDA
     *     A(K, 1:K-1) += -0.5*AKK * B(K, 1:K-1)
     *     A(1:K-1, 1:K-1) += dsyr2(-1, A(K,1:K-1), B(K,1:K-1))
     *     A(K, 1:K-1) += -0.5*AKK * B(K, 1:K-1)
     *     A(K, 1:K-1) = solve U(1:K-1,1:K-1)^T * x = A(K,1:K-1)
     *   END DO
     * </pre>
     *
     * <p>Row-major mapping: Fortran "row K, cols 1:K-1" with stride LDA (col-major)
     * is the upper-triangle portion of row K. In row-major, the upper triangle of
     * column k (rows 0..k-1) is A[0*lda+k, 1*lda+k, ...], stride=lda.
     * So Fortran A(K, 1:K-1) maps to Java column vector A[col=k, rows=0..k-1], stride=lda.
     */
    static void dsygs2Type1Upper(int n, double[] A, int aOff, int lda,
                                     double[] B, int bOff, int ldb) {
        for (int k = 0; k < n; k++) {
            double akk = A[aOff + k * lda + k];
            double bkk = B[bOff + k * ldb + k];
            akk = akk / (bkk * bkk);
            A[aOff + k * lda + k] = akk;

            if (k == 0) continue;

            // Column vector: A(0:k-1, k), offset=aOff+k (col k, rows 0..k-1), stride=lda
            int aColOff = aOff + k;
            int bColOff = bOff + k;

            // A(0:k-1, k) /= bkk
            BLAS.dscal(k, 1.0 / bkk, A, aColOff, lda);

            double ct = -0.5 * akk;

            // A(0:k-1, k) += ct * B(0:k-1, k)
            BLAS.daxpy(k, ct, B, bColOff, ldb, A, aColOff, lda);

            // A(0:k-1, 0:k-1) upper += dsyr2(-1, A(0:k-1,k), B(0:k-1,k))
            // Both vectors are column vectors with stride lda
            BLAS.dsyr2(BLAS.Uplo.Upper, k, -1.0,
                    A, aColOff, lda,
                    B, bColOff, ldb,
                    A, aOff, lda);

            // A(0:k-1, k) += ct * B(0:k-1, k)
            BLAS.daxpy(k, ct, B, bColOff, ldb, A, aColOff, lda);

            // Solve U(0:k-1, 0:k-1)^T * x = A(0:k-1, k)
            // dtrsm(Left, Upper, Trans, NonUnit, k, 1, 1.0, B(0,0), ldb, A(0:k-1,k), lda)
            // Treats A(0:k-1, k) as a k×1 matrix with leading dim lda
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.Trans, BLAS.Diag.NonUnit,
                    k, 1, 1.0,
                    B, bOff, ldb,
                    A, aColOff, lda);
        }
    }

    /**
     * Type 2/3, lower: A ← Lᵀ·A·L.
     *
     * <p>Fortran reference (column-major, 1-indexed):
     * <pre>
     *   DO K = 1, N
     *     AKK = A(K,K);  BKK = B(K,K)
     *     A(K, 1:K-1) = L(1:K-1,1:K-1)^T * A(K, 1:K-1)   [dtrmv Lower Trans]
     *     A(K, 1:K-1) += 0.5*AKK * B(K, 1:K-1)
     *     A(1:K-1,1:K-1) += dsyr2(+1, A(K,1:K-1), B(K,1:K-1))
     *     A(K, 1:K-1) += 0.5*AKK * B(K, 1:K-1)
     *     A(K, 1:K-1) *= BKK
     *     A(K,K) = AKK * BKK^2
     *   END DO
     * </pre>
     *
     * <p>Row-major mapping: Fortran row vector A(K, 1:K-1) with stride LDA (col-major)
     * maps to Java row vector A[k*lda .. k*lda+k-1] with stride 1.
     * L(1:K-1,1:K-1) is the top-left (k)×(k) block of B, starting at B[bOff].
     */
    static void dsygs2Type23Lower(int n, double[] A, int aOff, int lda,
                                      double[] B, int bOff, int ldb) {
        for (int k = 0; k < n; k++) {
            double akk = A[aOff + k * lda + k];
            double bkk = B[bOff + k * ldb + k];

            if (k > 0) {
                // Row vector: A(k, 0:k-1), offset=k*lda, stride=1
                int aRowOff = aOff + k * lda;
                int bRowOff = bOff + k * ldb;

                // A(k, 0:k-1) = L(0:k-1, 0:k-1)^T * A(k, 0:k-1)
                BLAS.dtrmv(BLAS.Uplo.Lower, BLAS.Transpose.Trans, BLAS.Diag.NonUnit,
                        k, B, bOff, ldb, A, aRowOff, 1);

                double ct = 0.5 * akk;

                BLAS.daxpy(k, ct, B, bRowOff, 1, A, aRowOff, 1);

                BLAS.dsyr2(BLAS.Uplo.Lower, k, 1.0,
                        A, aRowOff, 1,
                        B, bRowOff, 1,
                        A, aOff, lda);

                BLAS.daxpy(k, ct, B, bRowOff, 1, A, aRowOff, 1);

                BLAS.dscal(k, bkk, A, aRowOff, 1);
            }

            A[aOff + k * lda + k] = akk * bkk * bkk;
        }
    }

    /**
     * Type 2/3, upper: A ← U·A·Uᵀ.
     *
     * <p>Fortran reference (column-major, 1-indexed):
     * <pre>
     *   DO K = 1, N
     *     AKK = A(K,K);  BKK = B(K,K)
     *     A(1:K-1, K) = U(1:K-1,1:K-1) * A(1:K-1, K)   [dtrmv Upper NoTrans]
     *     A(1:K-1, K) += 0.5*AKK * B(1:K-1, K)
     *     A(1:K-1,1:K-1) += dsyr2(+1, A(1:K-1,K), B(1:K-1,K))
     *     A(1:K-1, K) += 0.5*AKK * B(1:K-1, K)
     *     A(1:K-1, K) *= BKK
     *     A(K,K) = AKK * BKK^2
     *   END DO
     * </pre>
     *
     * <p>Row-major mapping: Fortran column vector A(1:K-1, K) with stride 1 (col-major)
     * maps to Java column vector A[0*lda+k, 1*lda+k, ...] with stride lda.
     */
    static void dsygs2Type23Upper(int n, double[] A, int aOff, int lda,
                                      double[] B, int bOff, int ldb) {
        for (int k = 0; k < n; k++) {
            double akk = A[aOff + k * lda + k];
            double bkk = B[bOff + k * ldb + k];

            if (k > 0) {
                // Column vector: A(0:k-1, k), offset=aOff+k (col k, rows 0..k-1), stride=lda
                int aColOff = aOff + k;
                int bColOff = bOff + k;

                // A(0:k-1, k) = U(0:k-1, 0:k-1) * A(0:k-1, k)
                BLAS.dtrmv(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit,
                        k, B, bOff, ldb, A, aColOff, lda);

                double ct = 0.5 * akk;

                BLAS.daxpy(k, ct, B, bColOff, ldb, A, aColOff, lda);

                BLAS.dsyr2(BLAS.Uplo.Upper, k, 1.0,
                        A, aColOff, lda,
                        B, bColOff, ldb,
                        A, aOff, lda);

                BLAS.daxpy(k, ct, B, bColOff, ldb, A, aColOff, lda);

                BLAS.dscal(k, bkk, A, aColOff, lda);
            }

            A[aOff + k * lda + k] = akk * bkk * bkk;
        }
    }
}
