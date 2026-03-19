/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DSYGST — reduces a real symmetric-definite generalized eigenproblem to standard form.
 *
 * <p>Corresponds to LAPACK {@code DSYGST} (blocked algorithm).
 *
 * <p>Given the Cholesky factor of B (from {@code DPOTRF}), overwrites A with the
 * equivalent standard symmetric eigenproblem:
 * <ul>
 *   <li>Type 1 ({@code A·x = λ·B·x}):  A ← inv(L)·A·inv(Lᵀ)  (lower)</li>
 *   <li>Type 2 ({@code A·B·x = λ·x}):  A ← Lᵀ·A·L             (lower)</li>
 *   <li>Type 3 ({@code B·A·x = λ·x}):  A ← Lᵀ·A·L             (lower)</li>
 * </ul>
 *
 * <p>Uses a blocked algorithm for large matrices (NB from ilaenv), falling back to
 * the unblocked {@link Dsygs2} for small matrices or when NB >= N.
 *
 * <p>All matrices are stored in <b>row-major</b> order.
 *
 * <p>Reference: <a href="https://netlib.org/lapack/explore-html/">LAPACK DSYGST</a>
 */
public interface Dsygst {

    /**
     * Reduces generalized symmetric eigenproblem to standard form (blocked, row-major).
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
     * @return 0 on success, negative value if argument error
     */
    static int dsygst(int itype, BLAS.Uplo uplo, int n,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {
        if (itype < 1 || itype > 3) return -1;
        if (uplo == null) return -2;
        if (n < 0) return -3;
        if (n == 0) return 0;

        int nb = BLAS.ilaenv(1, "DSYGST", uplo == BLAS.Uplo.Lower ? "L" : "U", n, -1, -1, -1);

        // Fall back to unblocked if NB <= 1 or NB >= N
        if (nb <= 1 || nb >= n) {
            Dsygs2.dsygs2(itype, uplo, n, A, aOff, lda, B, bOff, ldb);
            return 0;
        }

        boolean lower = (uplo == BLAS.Uplo.Lower);

        if (itype == 1) {
            if (lower) {
                dsygstType1Lower(n, nb, A, aOff, lda, B, bOff, ldb);
            } else {
                dsygstType1Upper(n, nb, A, aOff, lda, B, bOff, ldb);
            }
        } else {
            if (lower) {
                dsygstType23Lower(n, nb, A, aOff, lda, B, bOff, ldb);
            } else {
                dsygstType23Upper(n, nb, A, aOff, lda, B, bOff, ldb);
            }
        }
        return 0;
    }

    // =========================================================================
    // Type 1, Lower: A ← inv(L)·A·inv(Lᵀ)
    //
    // Fortran DSYGST Type 1 Lower (0-indexed, k=block start, kb=block size):
    //   dsygs2(1, L, kb, A[k,k], B[k,k])
    //   if rem > 0:
    //     dtrsm(Right, L, Trans, NonUnit, rem, kb, 1, B[k,k], A[k+kb,k])
    //     dsymm(Right, L, rem, kb, -0.5, A[k,k], B[k+kb,k], 1, A[k+kb,k])
    //     dsyr2k(L, NoTrans, rem, kb, -1, A[k+kb,k], B[k+kb,k], 1, A[k+kb,k+kb])
    //     dsymm(Right, L, rem, kb, -0.5, A[k,k], B[k+kb,k], 1, A[k+kb,k])
    //     dtrsm(Left, L, NoTrans, NonUnit, rem, kb, 1, B[k+kb,k+kb], A[k+kb,k])
    //
    // Row-major mapping (same rules as Dsygs2):
    //   A[k,k] submatrix: aOff + k*lda + k
    //   A[k+kb, k] block (rem×kb): aOff + (k+kb)*lda + k
    //   B[k+kb, k] block (rem×kb): bOff + (k+kb)*ldb + k
    //   B[k+kb, k+kb] submatrix: bOff + (k+kb)*ldb + (k+kb)
    // =========================================================================
    static void dsygstType1Lower(int n, int nb,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb) {
        for (int k = 0; k < n; k += nb) {
            int kb = Math.min(n - k, nb);
            int rem = n - k - kb;

            // Unblocked reduction of the kb×kb diagonal block
            Dsygs2.dsygs2(1, BLAS.Uplo.Lower, kb,
                    A, aOff + k * lda + k, lda,
                    B, bOff + k * ldb + k, ldb);

            if (rem <= 0) continue;

            // Offsets for the off-diagonal block A[k+kb:n, k:k+kb] (rem×kb)
            int aBlkOff = aOff + (k + kb) * lda + k;
            int bBlkOff = bOff + (k + kb) * ldb + k;

            // dtrsm(Right, Lower, Trans, NonUnit, rem, kb, 1, B[k,k], ldb, A[k+kb,k], lda)
            // Solves A[k+kb,k] * L[k,k]^T = A[k+kb,k]
            BLAS.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                    rem, kb, 1.0,
                    B, bOff + k * ldb + k, ldb,
                    A, aBlkOff, lda);

            // dsymm(Right, Lower, rem, kb, -0.5, A[k,k], lda, B[k+kb,k], ldb, 1, A[k+kb,k], lda)
            // A[k+kb,k] += -0.5 * B[k+kb,k] * A[k,k]
            BLAS.dsymm(BLAS.Side.Right, BLAS.Uplo.Lower, rem, kb,
                    -0.5,
                    A, aOff + k * lda + k, lda,
                    B, bBlkOff, ldb,
                    1.0,
                    A, aBlkOff, lda);

            // dsyr2k(Lower, NoTrans, rem, kb, -1, A[k+kb,k], lda, B[k+kb,k], ldb, 1, A[k+kb,k+kb], lda)
            // A[k+kb,k+kb] += -1 * (A[k+kb,k]*B[k+kb,k]^T + B[k+kb,k]*A[k+kb,k]^T)
            BLAS.dsyr2k(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, rem, kb,
                    -1.0,
                    A, aBlkOff, lda,
                    B, bBlkOff, ldb,
                    1.0,
                    A, aOff + (k + kb) * lda + (k + kb), lda);

            // dsymm(Right, Lower, rem, kb, -0.5, A[k,k], lda, B[k+kb,k], ldb, 1, A[k+kb,k], lda)
            BLAS.dsymm(BLAS.Side.Right, BLAS.Uplo.Lower, rem, kb,
                    -0.5,
                    A, aOff + k * lda + k, lda,
                    B, bBlkOff, ldb,
                    1.0,
                    A, aBlkOff, lda);

            // dtrsm(Left, Lower, NoTrans, NonUnit, rem, kb, 1, B[k+kb,k+kb], ldb, A[k+kb,k], lda)
            // Solves L[k+kb,k+kb] * A[k+kb,k] = A[k+kb,k]
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                    rem, kb, 1.0,
                    B, bOff + (k + kb) * ldb + (k + kb), ldb,
                    A, aBlkOff, lda);
        }
    }

    // =========================================================================
    // Type 1, Upper: A ← inv(Uᵀ)·A·inv(U)
    //
    // Fortran DSYGST Type 1 Upper (0-indexed, k=block start, kb=block size):
    //   dsygs2(1, U, kb, A[k,k], B[k,k])
    //   if k > 0:
    //     dtrsm(Left, U, Trans, NonUnit, kb, k, 1, B[0,0], A[k,0])
    //     dsymm(Left, U, kb, k, -0.5, A[k,k], B[k,0], 1, A[k,0])
    //     dsyr2k(U, Trans, k, kb, -1, A[k,0], B[k,0], 1, A[0,0])
    //     dsymm(Left, U, kb, k, -0.5, A[k,k], B[k,0], 1, A[k,0])
    //     dtrsm(Right, U, NoTrans, NonUnit, kb, k, 1, B[0,0], A[k,0])
    //
    // Row-major mapping for Upper:
    //   A[k,0] block (kb×k): aOff + k*lda + 0
    //   B[k,0] block (kb×k): bOff + k*ldb + 0
    //   A[k,k] submatrix: aOff + k*lda + k
    //   B[0,0] submatrix: bOff
    //
    // Note: In Fortran col-major, "A(1:K-1, K:K+KB-1)" is a (K-1)×KB block.
    // In row-major, the equivalent is A[k, 0:k-1] which is a kb×k block at A[k*lda].
    // =========================================================================
    static void dsygstType1Upper(int n, int nb,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb) {
        for (int k = 0; k < n; k += nb) {
            int kb = Math.min(n - k, nb);

            // Unblocked reduction of the kb×kb diagonal block
            Dsygs2.dsygs2(1, BLAS.Uplo.Upper, kb,
                    A, aOff + k * lda + k, lda,
                    B, bOff + k * ldb + k, ldb);

            if (k == 0) continue;

            // Offsets for the off-diagonal block A[k:k+kb, 0:k] (kb×k)
            int aBlkOff = aOff + k * lda;      // A[k, 0]
            int bBlkOff = bOff + k * ldb;      // B[k, 0]

            // dtrsm(Left, Upper, Trans, NonUnit, kb, k, 1, B[0,0], ldb, A[k,0], lda)
            // Solves U[0,0]^T * A[k,0] = A[k,0]
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                    kb, k, 1.0,
                    B, bOff, ldb,
                    A, aBlkOff, lda);

            // dsymm(Left, Upper, kb, k, -0.5, A[k,k], lda, B[k,0], ldb, 1, A[k,0], lda)
            // A[k,0] += -0.5 * A[k,k] * B[k,0]
            BLAS.dsymm(BLAS.Side.Left, BLAS.Uplo.Upper, kb, k,
                    -0.5,
                    A, aOff + k * lda + k, lda,
                    B, bBlkOff, ldb,
                    1.0,
                    A, aBlkOff, lda);

            // dsyr2k(Upper, Trans, k, kb, -1, A[k,0], lda, B[k,0], ldb, 1, A[0,0], lda)
            // A[0,0] += -1 * (A[k,0]^T*B[k,0] + B[k,0]^T*A[k,0])
            BLAS.dsyr2k(BLAS.Uplo.Upper, BLAS.Trans.Trans, k, kb,
                    -1.0,
                    A, aBlkOff, lda,
                    B, bBlkOff, ldb,
                    1.0,
                    A, aOff, lda);

            // dsymm(Left, Upper, kb, k, -0.5, A[k,k], lda, B[k,0], ldb, 1, A[k,0], lda)
            BLAS.dsymm(BLAS.Side.Left, BLAS.Uplo.Upper, kb, k,
                    -0.5,
                    A, aOff + k * lda + k, lda,
                    B, bBlkOff, ldb,
                    1.0,
                    A, aBlkOff, lda);

            // dtrsm(Right, Upper, NoTrans, NonUnit, kb, k, 1, B[0,0], ldb, A[k,0], lda)
            // Solves A[k,0] * U[0,0] = A[k,0]
            BLAS.dtrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                    kb, k, 1.0,
                    B, bOff, ldb,
                    A, aBlkOff, lda);
        }
    }

    // =========================================================================
    // Type 2/3, Lower: A ← Lᵀ·A·L
    //
    // Fortran DSYGST Type 2/3 Lower (0-indexed, k=block start, kb=block size):
    //   dtrmm(Right, L, NoTrans, NonUnit, kb, k, 1, B[0,0], A[k,0])
    //   dsymm(Left, L, kb, k, 0.5, A[k,k], B[k,0], 1, A[k,0])
    //   dsyr2k(L, Trans, k, kb, 1, A[k,0], B[k,0], 1, A[0,0])
    //   dsymm(Left, L, kb, k, 0.5, A[k,k], B[k,0], 1, A[k,0])
    //   dtrmm(Left, L, Trans, NonUnit, kb, k, 1, B[k,k], A[k,0])
    //   dsygs2(2, L, kb, A[k,k], B[k,k])
    //
    // Row-major mapping:
    //   A[k, 0:k] block (kb×k): aOff + k*lda
    //   B[k, 0:k] block (kb×k): bOff + k*ldb
    //   A[k,k] submatrix: aOff + k*lda + k
    //   B[0,0] submatrix: bOff
    //   B[k,k] submatrix: bOff + k*ldb + k
    // =========================================================================
    static void dsygstType23Lower(int n, int nb,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb) {
        for (int k = 0; k < n; k += nb) {
            int kb = Math.min(n - k, nb);

            if (k > 0) {
                // Offsets for the off-diagonal block A[k:k+kb, 0:k] (kb×k)
                int aBlkOff = aOff + k * lda;   // A[k, 0]
                int bBlkOff = bOff + k * ldb;   // B[k, 0]

                // dtrmm(Right, Lower, NoTrans, NonUnit, kb, k, 1, B[0,0], ldb, A[k,0], lda)
                // A[k,0] = A[k,0] * L[0,0]
                BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                        kb, k, 1.0,
                        B, bOff, ldb,
                        A, aBlkOff, lda);

                // dsymm(Left, Lower, kb, k, 0.5, A[k,k], lda, B[k,0], ldb, 1, A[k,0], lda)
                // A[k,0] += 0.5 * A[k,k] * B[k,0]
                BLAS.dsymm(BLAS.Side.Left, BLAS.Uplo.Lower, kb, k,
                        0.5,
                        A, aOff + k * lda + k, lda,
                        B, bBlkOff, ldb,
                        1.0,
                        A, aBlkOff, lda);

                // dsyr2k(Lower, Trans, k, kb, 1, A[k,0], lda, B[k,0], ldb, 1, A[0,0], lda)
                // A[0,0] += A[k,0]^T*B[k,0] + B[k,0]^T*A[k,0]
                BLAS.dsyr2k(BLAS.Uplo.Lower, BLAS.Trans.Trans, k, kb,
                        1.0,
                        A, aBlkOff, lda,
                        B, bBlkOff, ldb,
                        1.0,
                        A, aOff, lda);

                // dsymm(Left, Lower, kb, k, 0.5, A[k,k], lda, B[k,0], ldb, 1, A[k,0], lda)
                BLAS.dsymm(BLAS.Side.Left, BLAS.Uplo.Lower, kb, k,
                        0.5,
                        A, aOff + k * lda + k, lda,
                        B, bBlkOff, ldb,
                        1.0,
                        A, aBlkOff, lda);

                // dtrmm(Left, Lower, Trans, NonUnit, kb, k, 1, B[k,k], ldb, A[k,0], lda)
                // A[k,0] = L[k,k]^T * A[k,0]
                BLAS.dtrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                        kb, k, 1.0,
                        B, bOff + k * ldb + k, ldb,
                        A, aBlkOff, lda);
            }

            // Unblocked reduction of the kb×kb diagonal block
            Dsygs2.dsygs2(2, BLAS.Uplo.Lower, kb,
                    A, aOff + k * lda + k, lda,
                    B, bOff + k * ldb + k, ldb);
        }
    }

    // =========================================================================
    // Type 2/3, Upper: A ← U·A·Uᵀ
    //
    // Fortran DSYGST Type 2/3 Upper (0-indexed, k=block start, kb=block size):
    //   dtrmm(Left, U, NoTrans, NonUnit, k, kb, 1, B[0,0], A[0,k])
    //   dsymm(Right, U, k, kb, 0.5, A[k,k], B[0,k], 1, A[0,k])
    //   dsyr2k(U, NoTrans, k, kb, 1, A[0,k], B[0,k], 1, A[0,0])
    //   dsymm(Right, U, k, kb, 0.5, A[k,k], B[0,k], 1, A[0,k])
    //   dtrmm(Right, U, Trans, NonUnit, k, kb, 1, B[k,k], A[0,k])
    //   dsygs2(2, U, kb, A[k,k], B[k,k])
    //
    // Row-major mapping:
    //   A[0:k, k:k+kb] block (k×kb): aOff + k  (col k, rows 0..k-1)
    //   B[0:k, k:k+kb] block (k×kb): bOff + k  (col k, rows 0..k-1)
    //   A[k,k] submatrix: aOff + k*lda + k
    //   B[0,0] submatrix: bOff
    //   B[k,k] submatrix: bOff + k*ldb + k
    // =========================================================================
    static void dsygstType23Upper(int n, int nb,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb) {
        for (int k = 0; k < n; k += nb) {
            int kb = Math.min(n - k, nb);

            if (k > 0) {
                // Offsets for the off-diagonal block A[0:k, k:k+kb] (k×kb)
                int aBlkOff = aOff + k;   // A[0, k]
                int bBlkOff = bOff + k;   // B[0, k]

                // dtrmm(Left, Upper, NoTrans, NonUnit, k, kb, 1, B[0,0], ldb, A[0,k], lda)
                // A[0,k] = U[0,0] * A[0,k]
                BLAS.dtrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                        k, kb, 1.0,
                        B, bOff, ldb,
                        A, aBlkOff, lda);

                // dsymm(Right, Upper, k, kb, 0.5, A[k,k], lda, B[0,k], ldb, 1, A[0,k], lda)
                // A[0,k] += 0.5 * B[0,k] * A[k,k]
                BLAS.dsymm(BLAS.Side.Right, BLAS.Uplo.Upper, k, kb,
                        0.5,
                        A, aOff + k * lda + k, lda,
                        B, bBlkOff, ldb,
                        1.0,
                        A, aBlkOff, lda);

                // dsyr2k(Upper, NoTrans, k, kb, 1, A[0,k], lda, B[0,k], ldb, 1, A[0,0], lda)
                // A[0,0] += A[0,k]*B[0,k]^T + B[0,k]*A[0,k]^T
                BLAS.dsyr2k(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, k, kb,
                        1.0,
                        A, aBlkOff, lda,
                        B, bBlkOff, ldb,
                        1.0,
                        A, aOff, lda);

                // dsymm(Right, Upper, k, kb, 0.5, A[k,k], lda, B[0,k], ldb, 1, A[0,k], lda)
                BLAS.dsymm(BLAS.Side.Right, BLAS.Uplo.Upper, k, kb,
                        0.5,
                        A, aOff + k * lda + k, lda,
                        B, bBlkOff, ldb,
                        1.0,
                        A, aBlkOff, lda);

                // dtrmm(Right, Upper, Trans, NonUnit, k, kb, 1, B[k,k], ldb, A[0,k], lda)
                // A[0,k] = A[0,k] * U[k,k]^T
                BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                        k, kb, 1.0,
                        B, bOff + k * ldb + k, ldb,
                        A, aBlkOff, lda);
            }

            // Unblocked reduction of the kb×kb diagonal block
            Dsygs2.dsygs2(2, BLAS.Uplo.Upper, kb,
                    A, aOff + k * lda + k, lda,
                    B, bOff + k * ldb + k, ldb);
        }
    }
}
