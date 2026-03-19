/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Unified BLAS Level 1/2 interface.
 *
 * <p>This interface aggregates all BLAS routines for external access.</p>
 *
 * <h2>Level 1 Operations</h2>
 * <ul>
 *   <li>dcopy, dset, dswap - vector copy/fill/swap</li>
 *   <li>dnrm2 - Euclidean norm</li>
 *   <li>damax - infinity norm</li>
 *   <li>dasum - L1 norm (sum of absolute values)</li>
 *   <li>idamax - index of max absolute value</li>
 *   <li>drot - Givens rotation</li>
 *   <li>daxpy - y += α×x</li>
 *   <li>dscal - x *= α</li>
 *   <li>ddot - dot product</li>
 * </ul>
 *
 * <h2>Level 2 Operations</h2>
 * <ul>
 *   <li>dgemv - matrix-vector multiply</li>
 *   <li>dtrsv - triangular solve</li>
 * </ul>
 */
public interface BLAS {

    // ========== Enum Types ==========

    enum Trans {
        NoTrans('N'), Trans('T'), Conj('C');
        final char code;
        Trans(char code) { this.code = code; }
    }

    enum Uplo {
        Upper('U'), Lower('L'), All('A');
        final char code;
        Uplo(char code) { this.code = code; }
    }

    enum Side {
        Left('L'), Right('R');
        final char code;
        Side(char code) { this.code = code; }
    }

    enum Diag {
        NonUnit('N'), Unit('U');
        final char code;
        Diag(char code) { this.code = code; }
    }

    enum Direct {
        Forward('F'), Backward('B');
        final char code;
        Direct(char code) { this.code = code; }
    }

    enum Wise {
        Row('R'), Col('C');
        final char code;
        Wise(char code) { this.code = code; }
    }

    enum GsvdJob {
        None(0), Compute(1), Initialize(2);
        final int code;
        GsvdJob(int code) { this.code = code; }
    }

    enum Norm {
        MaxAbs('M'), One('1'), Inf('I'), Frobenius('F');
        final char code;
        Norm(char code) { this.code = code; }
    }

    // ========== BLAS Level 1: Vector-Vector Operations ==========

    // ==================== Dlamv ====================

    /**
     * Copies a vector x to y: y := x (BLAS DCOPY).
     *
     * @param n    number of elements
     * @param x    source vector
     * @param xOff offset into x
     * @param incX stride in x
     * @param y    destination vector
     * @param yOff offset into y
     * @param incY stride in y
     */
    static void dcopy(int n, double[] x, int xOff, int incX, double[] y, int yOff, int incY) {
        Dlamv.dcopy(n, x, xOff, incX, y, yOff, incY);
    }

    /**
     * Sets all elements of a vector to a constant: x[i] := a (LAPACK DLASET vector form).
     *
     * @param n    number of elements
     * @param a    fill value
     * @param x    target vector
     * @param xOff offset into x
     * @param incX stride in x
     */
    static void dset(int n, double a, double[] x, int xOff, int incX) {
        Dlamv.dset(n, a, x, xOff, incX);
    }

    /**
     * Swaps the contents of two vectors: x ↔ y (BLAS DSWAP).
     *
     * @param n    number of elements
     * @param x    first vector
     * @param xOff offset into x
     * @param incX stride in x
     * @param y    second vector
     * @param yOff offset into y
     * @param incY stride in y
     */
    static void dswap(int n, double[] x, int xOff, int incX, double[] y, int yOff, int incY) {
        Dlamv.dswap(n, x, xOff, incX, y, yOff, incY);
    }

    /**
     * Applies row interchanges to a matrix (DLASWP).
     *
     * @param n number of columns to swap
     * @param A matrix (row-major)
     * @param aOff offset into A
     * @param lda leading dimension
     * @param k1 first row to swap
     * @param k2 last row to swap
     * @param ipiv pivot indices
     * @param ipivOff offset into ipiv
     * @param incX 1 for forward, -1 for reverse
     */
    static void dlaswp(int n, double[] A, int aOff, int lda, int k1, int k2, int[] ipiv, int ipivOff, int incX) {
        Dlamv.dlaswp(n, A, aOff, lda, k1, k2, ipiv, ipivOff, incX);
    }

    // ==================== Dnrm2 ====================

    /**
     * Computes the Euclidean norm of a vector: ||x||₂ (BLAS DNRM2).
     * Uses pairwise summation for improved numerical stability.
     *
     * @param n    number of elements
     * @param x    input vector
     * @param xOff offset into x
     * @param incX stride in x
     * @return Euclidean norm
     */
    static double dnrm2(int n, double[] x, int xOff, int incX) {
        return Dnrm2.dnrm2(n, x, xOff, incX);
    }

    // ==================== Damax ====================

    /**
     * Returns the infinity norm of a vector: max|x[i]| (non-standard extension).
     *
     * @param n    number of elements
     * @param x    input vector
     * @param xOff offset into x
     * @param incX stride in x
     * @return maximum absolute value
     */
    static double damax(int n, double[] x, int xOff, int incX) {
        return Damax.damax(n, x, xOff, incX);
    }

    // ==================== Dasum ====================

    /**
     * Computes the L1 norm (sum of absolute values): Σ|x[i]| (BLAS DASUM).
     *
     * @param n    number of elements
     * @param x    input vector
     * @param xOff offset into x
     * @param incX stride in x
     * @return sum of absolute values
     */
    static double dasum(int n, double[] x, int xOff, int incX) {
        return Dasum.dasum(n, x, xOff, incX);
    }

    // ==================== Idamax ====================

    /**
     * Returns the index of the element with maximum absolute value (BLAS IDAMAX).
     *
     * @param n    number of elements
     * @param x    input vector
     * @param xOff offset into x
     * @param incX stride in x
     * @return 0-based index of the element with maximum absolute value
     */
    static int idamax(int n, double[] x, int xOff, int incX) {
        return Damax.idamax(n, x, xOff, incX);
    }

    // ==================== Drot ====================

    /**
     * Applies a Givens plane rotation to vectors x and y (BLAS DROT).
     * Computes: x[i] := c*x[i] + s*y[i], y[i] := c*y[i] - s*x[i]
     *
     * @param n    number of elements
     * @param x    first vector (modified in place)
     * @param xOff offset into x
     * @param incX stride in x
     * @param y    second vector (modified in place)
     * @param yOff offset into y
     * @param incY stride in y
     * @param c    cosine of rotation angle
     * @param s    sine of rotation angle
     */
    static void drot(int n, double[] x, int xOff, int incX, double[] y, int yOff, int incY, double c, double s) {
        Drot.drot(n, x, xOff, incX, y, yOff, incY, c, s);
    }

    // ==================== Daxpy ====================

    /**
     * Computes y := alpha*x + y (BLAS DAXPY).
     *
     * @param n     number of elements
     * @param alpha scalar multiplier
     * @param x     input vector
     * @param xOff  offset into x
     * @param incX  stride in x
     * @param y     input/output vector (modified in place)
     * @param yOff  offset into y
     * @param incY  stride in y
     */
    static void daxpy(int n, double alpha, double[] x, int xOff, int incX, double[] y, int yOff, int incY) {
        Daxpy.daxpy(n, alpha, x, xOff, incX, y, yOff, incY);
    }

    // ==================== Dscal ====================

    /**
     * Scales a vector by a constant: x := alpha*x (BLAS DSCAL).
     *
     * @param n     number of elements
     * @param alpha scalar multiplier
     * @param x     vector (modified in place)
     * @param xOff  offset into x
     * @param incX  stride in x
     */
    static void dscal(int n, double alpha, double[] x, int xOff, int incX) {
        Dscal.dscal(n, alpha, x, xOff, incX);
    }

    // ==================== Ddot ====================

    /**
     * Computes the dot product of two vectors: Σ x[i]*y[i] (BLAS DDOT).
     * Uses FMA (fused multiply-add) for improved accuracy when available.
     *
     * @param n    number of elements
     * @param x    first vector
     * @param xOff offset into x
     * @param incX stride in x
     * @param y    second vector
     * @param yOff offset into y
     * @param incY stride in y
     * @return dot product value
     */
    static double ddot(int n, double[] x, int xOff, int incX, double[] y, int yOff, int incY) {
        return Ddot.ddot(n, x, xOff, incX, y, yOff, incY);
    }

    // ========== BLAS Level 2: Matrix-Vector Operations ==========

    // ==================== Dgemv ====================

    /**
     * Matrix-vector product with stride support for x and y.
     *
     * @param trans 'N' for no transpose, 'T' for transpose
     * @param m number of rows in A
     * @param n number of columns in A
     * @param alpha scalar
     * @param A matrix (m × n, row-major)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param x input vector
     * @param xOff offset into x
     * @param incX stride in x
     * @param beta scalar multiplier for y
     * @param y output vector
     * @param yOff offset into y
     * @param incY stride in y
     */
    static void dgemv(Trans trans, int m, int n, double alpha,
                      double[] A, int aOff, int lda,
                      double[] x, int xOff, int incX, double beta,
                      double[] y, int yOff, int incY) {
        Dgemv.dgemv(trans, m, n, alpha, A, aOff, lda, x, xOff, incX, beta, y, yOff, incY);
    }

    // ==================== Dger ====================

    /**
     * Performs rank-1 update: A := alpha * x * y^T + A
     *
     * @param m number of rows in A
     * @param n number of columns in A
     * @param alpha scalar
     * @param x vector (length m)
     * @param xOff offset into x
     * @param incX stride in x (use m for column access in row-major)
     * @param y vector (length n)
     * @param yOff offset into y
     * @param incY stride in y (use 1 for contiguous row access)
     * @param A matrix (m × n, row-major)
     * @param aOff offset into A
     * @param lda leading dimension (stride between rows = n for row-major)
     */
    static void dger(int m, int n, double alpha, double[] x, int xOff, int incX,
                     double[] y, int yOff, int incY, double[] A, int aOff, int lda) {
        Dger.dger(m, n, alpha, x, xOff, incX, y, yOff, incY, A, aOff, lda);
    }

    // ==================== Dtrsl ====================

    /**
     * Solve triangular system T*x = b or Tᵀ*x = b (LINPACK dtrsl / BLAS dtrsv style).
     *
     * @param t     the triangular matrix (column-major storage)
     * @param tOff  offset into array t
     * @param ldt   the leading dimension of the array t
     * @param n     the order of the system
     * @param b     the right hand side vector (modified in place with solution)
     * @param bOff  offset into array b
     * @param uplo  {@link Uplo#Upper} for upper triangular, {@link Uplo#Lower} for lower triangular
     * @param trans {@link Trans#NoTrans} for T*x = b, {@link Trans#Trans} for Tᵀ*x = b
     * @return 0 if the system is nonsingular, otherwise the index of the first
     *         zero diagonal element of t
     */
    static int dtrsl(double[] t, int tOff, int ldt, int n, double[] b, int bOff, Uplo uplo, Trans trans) {
        return Dtrsl.dtrsl(t, tOff, ldt, n, b, bOff, uplo, trans);
    }

    // ==================== Dlange ====================

    /**
     * Returns the value of the specified norm of a general rectangular matrix (LAPACK DLANGE).
     *
     * @param norm  norm type: {@link Norm#MaxAbs} max abs, {@link Norm#One} 1-norm,
     *              {@link Norm#Inf} inf-norm, {@link Norm#Frobenius} Frobenius norm
     * @param m     number of rows
     * @param n     number of columns
     * @param A     matrix (m × n, row-major)
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @return computed norm value
     */
    static double dlange(Norm norm, int m, int n, double[] A, int aOff, int lda) {
        return Dlange.dlange(norm.code, m, n, A, aOff, lda);
    }

    // ========== BLAS Level 3: Matrix-Matrix Operations ==========

    // ==================== Dsyrk ====================

    /**
     * Symmetric rank-k update: C := alpha*A*A^T + beta*C or C := alpha*A^T*A + beta*C (BLAS DSYRK).
     * Only the upper or lower triangular part of C is updated.
     *
     * @param uplo  {@link Uplo#Upper} or {@link Uplo#Lower} — which triangle of C to update
     * @param trans {@link Trans#NoTrans} for A*A^T, {@link Trans#Trans} for A^T*A
     * @param n     order of matrix C
     * @param k     number of columns of A (NoTrans) or rows of A (Trans)
     * @param alpha scalar
     * @param A     matrix A
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param beta  scalar for C
     * @param C     symmetric matrix C (n × n, row-major), modified in place
     * @param cOff  offset into C
     * @param ldc   leading dimension of C
     */
    static void dsyrk(Uplo uplo, Trans trans, int n, int k, double alpha,
                      double[] A, int aOff, int lda, double beta,
                      double[] C, int cOff, int ldc) {
        Dsyrk.dsyrk(uplo, trans, n, k, alpha, A, aOff, lda, beta, C, cOff, ldc);
    }

    /**
     * Matrix-matrix multiplication: C := alpha*op(A)*op(B) + beta*C
     *
     * @param transA 'N' for A, 'T' for A^T
     * @param transB 'N' for B, 'T' for B^T
     * @param m rows in C and op(A)
     * @param n columns in C and op(B)
     * @param k columns in op(A) / rows in op(B)
     * @param alpha scalar
     * @param A matrix A
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param B matrix B
     * @param bOff offset into B
     * @param ldb leading dimension of B
     * @param beta scalar for C
     * @param C result matrix
     * @param cOff offset into C
     * @param ldc leading dimension of C
     */
    static void dgemm(Trans transA, Trans transB, int m, int n, int k,
                      double alpha, double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb,
                      double beta, double[] C, int cOff, int ldc) {
        Dgemm.dgemm(transA, transB, m, n, k, alpha, A, aOff, lda, B, bOff, ldb, beta, C, cOff, ldc);
    }

    /**
     * Triangular matrix-matrix multiplication: B := alpha*op(A)*B or B := alpha*B*op(A)
     *
     * @param side 'L' for A*B, 'R' for B*A
     * @param uplo 'U' for upper, 'L' for lower triangular
     * @param trans 'N' or 'T' for transpose
     * @param diag 'U' for unit diagonal, 'N' for non-unit
     * @param m rows in B
     * @param n columns in B
     * @param alpha scalar
     * @param A triangular matrix
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param B matrix B (overwritten)
     * @param bOff offset into B
     * @param ldb leading dimension of B
     */
    static void dtrmm(Side side, Uplo uplo, Trans trans, Diag diag,
                      int m, int n, double alpha,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {
        Dtrmm.dtrmm(side, uplo, trans, diag, m, n, alpha, A, aOff, lda, B, bOff, ldb);
    }

    /**
     * Solves triangular matrix equations (in-place).
     * BLAS Level-3 DTRSM.
     *
     * <p>Solves one of:</p>
     * <ul>
     *   <li>op(A) * X = alpha * B</li>
     *   <li>X * op(A) = alpha * B</li>
     * </ul>
     *
     * @param side 'L' for op(A) * X = B, 'R' for X * op(A) = B
     * @param uplo 'U' for upper, 'L' for lower triangular
     * @param trans 'N' or 'T' for transpose
     * @param diag 'U' for unit diagonal, 'N' for non-unit
     * @param m rows in B
     * @param n columns in B
     * @param alpha scalar
     * @param A triangular matrix
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param B matrix B (overwritten with solution X)
     * @param bOff offset into B
     * @param ldb leading dimension of B
     */
    static void dtrsm(Side side, Uplo uplo, Trans trans, Diag diag,
                      int m, int n, double alpha,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {
        Dtrsm.dtrsm(side, uplo, trans, diag, m, n, alpha, A, aOff, lda, B, bOff, ldb);
    }

    /**
     * Triangular matrix-vector multiplication.
     * BLAS Level-2 DTRMV.
     *
     * <p>Computes x := A * x or x := A^T * x where A is triangular.</p>
     *
     * @param uplo 'U' for upper, 'L' for lower triangular
     * @param trans 'N' or 'T' for transpose
     * @param diag 'U' for unit diagonal, 'N' for non-unit
     * @param n dimension of A
     * @param A triangular matrix
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param x vector (overwritten)
     * @param xOff offset into x
     * @param incX stride of x
     */
    static void dtrmv(Uplo uplo, Trans trans, Diag diag, int n,
                      double[] A, int aOff, int lda,
                      double[] x, int xOff, int incX) {
        Dtrmv.dtrmv(uplo, trans, diag, n, A, aOff, lda, x, xOff, incX);
    }

    // ==================== QR Block Operations (LAPACK-style) ====================

    /**
     * Forms the triangular factor T in a block reflector H = I - V*T*V^T.
     * LAPACK DLARFT.
     *
     * @param direct  'F' for forward (H = H_0*H_1*...*H_{k-1}), 'B' for backward
     * @param storev  'C' for column-wise storage, 'R' for row-wise storage
     * @param m       number of rows in V
     * @param k       number of reflectors
     * @param V       matrix of Householder vectors
     * @param vOff    offset into V
     * @param ldv     leading dimension of V
     * @param tau     Householder scalars (length k)
     * @param tauOff  offset into tau
     * @param T       output triangular factor (k × k)
     * @param tOff    offset into T
     * @param ldt     leading dimension of T
     */
    static void dlarft(Direct direct, Wise storev, int m, int k,
                       double[] V, int vOff, int ldv,
                       double[] tau, int tauOff,
                       double[] T, int tOff, int ldt) {
        Dlarft.dlarft(direct.code, storev.code, m, k, V, vOff, ldv, tau, tauOff, T, tOff, ldt);
    }

    /**
     * Applies a block reflector H = I - V*T*V^T (or its transpose) to a matrix C.
     * LAPACK DLARFB.
     *
     * @param side    'L' to apply from the left (H*C or H^T*C), 'R' from the right (C*H or C*H^T)
     * @param trans   'N' to apply H, 'T' to apply H^T
     * @param direct  'F' for forward direction, 'B' for backward direction
     * @param storev  'C' for column-wise storage of V, 'R' for row-wise storage
     * @param m       number of rows in C
     * @param n       number of columns in C
     * @param k       number of reflectors
     * @param V       matrix of Householder vectors
     * @param vOff    offset into V
     * @param ldv     leading dimension of V
     * @param T       triangular factor from dlarft (k × k)
     * @param tOff    offset into T
     * @param ldt     leading dimension of T
     * @param C       matrix to update (m × n)
     * @param cOff    offset into C
     * @param ldc     leading dimension of C
     * @param work    workspace
     * @param workOff offset into work
     * @param ldwork  leading dimension of work
     */
    static void dlarfb(Side side, Trans trans, Direct direct, Wise storev,
                       int m, int n, int k,
                       double[] V, int vOff, int ldv,
                       double[] T, int tOff, int ldt,
                       double[] C, int cOff, int ldc,
                       double[] work, int workOff, int ldwork) {
        Dlarfb.dlarfb(side.code, trans.code, direct.code, storev.code, m, n, k, V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, workOff, ldwork);
    }

    // ==================== Dlarfg ====================

    /**
     * Generates a Householder reflection (LAPACK DLARFG).
     *
     * @param n       the order of the elementary reflection
     * @param alpha   the scalar alpha
     * @param x       the vector x
     * @param xOff   offset into x
     * @param incX    the increment between elements of x
     * @param tau     on exit: the scalar tau
     * @param tauOff  offset into tau
     * @return beta   the scalar beta
     */
    static double dlarfg(int n, double alpha,
                         double[] x, int xOff, int incX,
                         double[] tau, int tauOff) {
        return Dlarfg.dlarfg(n, alpha, x, xOff, incX, tau, tauOff);
    }

    // ==================== Dlarf ====================

    /**
     * Applies a Householder reflection (LAPACK DLARF).
     *
     * @param side   'L' to apply from left, 'R' to apply from right
     * @param m      number of rows of C
     * @param n      number of columns of C
     * @param v      the Householder vector v
     * @param vOff   offset into v
     * @param incv   increment between elements of v
     * @param tau    the scalar tau
     * @param C      the matrix C
     * @param cOff   offset into C
     * @param ldc    leading dimension of C
     * @param work   workspace array (size max(m, n))
     * @param workOff offset into work array
     */
    static void dlarf(Side side, int m, int n,
                      double[] v, int vOff, int incv, double tau,
                      double[] C, int cOff, int ldc, double[] work, int workOff) {
        Dlarf.dlarf(side, m, n, v, vOff, incv, tau, C, cOff, ldc, work, workOff);
    }

    // ==================== Dsteqr ====================

    /**
     * Computes all eigenvalues of a symmetric tridiagonal matrix (LAPACK DSTEQR).
     *
     * @param jobz 'N' for eigenvalues only, 'V' for eigenvalues and eigenvectors
     * @param n    the order of the matrix
     * @param d    the diagonal elements (length n), modified on output
     * @param e    the off-diagonal elements (length n-1), modified on output
     * @param z    the eigenvectors (n x n, row-major), if jobz='V'
     * @param ldz  leading dimension of z
     * @param work workspace
     * @return 0 if successful, >0 if convergence failure
     */
    static int dsteqr(char jobz, int n, double[] d, double[] e,
                      double[] z, int ldz, double[] work) {
        return Dsteqr.dsteqr(jobz, n, d, 0, e, 0, z, 0, ldz, work, 0);
    }

    // ==================== Dsymv ====================

    /**
     * DSYMV with matrix offset.
     */
    static void dsymv(Uplo uplo, int n, double alpha,
                      double[] A, int aOff, int lda,
                      double[] x, int xOff, int incX, double beta,
                      double[] y, int yOff, int incY) {
        Dsymv.dsymv(uplo, n, alpha, A, aOff, lda, x, xOff, incX, beta, y, yOff, incY);
    }

    // ==================== Dsyr2 ====================

    /**
     * DSYR2 with matrix offset.
     */
    static void dsyr2(Uplo uplo, int n, double alpha,
                      double[] x, int xOff, int incX, double[] y, int yOff, int incY,
                      double[] A, int aOff, int lda) {
        Dsyr2.dsyr2(uplo, n, alpha, x, xOff, incX, y, yOff, incY, A, aOff, lda);
    }

    // ==================== Dsyr2k ====================

    /**
     * Symmetric rank-2k update (BLAS DSYR2K) with offsets.
     * C := alpha*A*B^T + alpha*B*A^T + beta*C
     *
     * @param uplo  'U' for upper, 'L' for lower
     * @param trans 'N' for NoTrans, 'T' for Trans
     * @param n     order of matrix C
     * @param k     number of columns of A and B
     * @param alpha scalar
     * @param A     matrix (n x k for NoTrans, k x n for Trans)
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param B     matrix (n x k for NoTrans, k x n for Trans)
     * @param bOff  offset into B
     * @param ldb   leading dimension of B
     * @param beta  scalar
     * @param C     symmetric matrix (n x n, row-major)
     * @param cOff  offset into C
     * @param ldc   leading dimension of C
     */
    static void dsyr2k(Uplo uplo, Trans trans, int n, int k, double alpha,
                       double[] A, int aOff, int lda, double[] B, int bOff, int ldb,
                       double beta, double[] C, int cOff, int ldc) {
        Dsyr2k.dsyr2k(uplo, trans, n, k, alpha, A, aOff, lda, B, bOff, ldb, beta, C, cOff, ldc);
    }

    /**
     * Reduces a symmetric matrix to banded form (LAPACK DLATRD) with matrix offset.
     *
     * @param uplo  'U' for upper, 'L' for lower
     * @param n     order of matrix
     * @param nb    number of columns to reduce
     * @param A     symmetric matrix (n x n, row-major), overwritten
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param e     off-diagonal elements (output)
     * @param eOff  offset into e
     * @param tau   Householder scalars (output)
     * @param tauOff offset into tau
     * @param w     workspace matrix (n x nb)
     * @param workOff  offset into w
     * @param ldw   leading dimension of w
     */
    static void dlatrd(Uplo uplo, int n, int nb, double[] A, int aOff, int lda,
                       double[] e, int eOff, double[] tau, int tauOff,
                       double[] w, int workOff, int ldw) {
        Dlatrd.dlatrd(uplo, n, nb, A, aOff, lda, e, eOff, tau, tauOff, w, workOff, ldw);
    }

    // ==================== Dsytrd ====================

    /**
     * Reduces a symmetric matrix to tridiagonal form (LAPACK DSYTRD).
     *
     * @param uplo  'L' for lower triangular, 'U' for upper triangular
     * @param n     the order of the matrix
     * @param A     the symmetric matrix (n x n, row-major), overwritten
     * @param lda   leading dimension of A
     * @param d     the diagonal elements (output, length n)
     * @param e     the off-diagonal elements (output, length n-1)
     * @param tau   the Householder scalars (output, length n-1)
     * @param work  workspace
     * @param lwork size of work
     */
    static void dsytrd(Uplo uplo, int n, double[] A, int lda,
                       double[] d, double[] e, double[] tau,
                       double[] work, int lwork) {
        Dsytrd.dsytrd(uplo, n, A, lda, d, 0, e, 0, tau, 0, work, lwork);
    }

    // ==================== Dorgtr ====================

    /**
     * Generates the orthogonal matrix Q from symmetric tridiagonalization (LAPACK DORGTR).
     * Version with workspace offset.
     *
     * @param uplo  'L' for lower triangular, 'U' for upper triangular
     * @param n     the order of the matrix
     * @param A     the matrix from Dsytrd (n x n, row-major), overwritten with Q
     * @param lda   leading dimension of A
     * @param tau   the Householder scalars from Dsytrd
     * @param work  workspace
     * @param workOff offset into work array
     * @param lwork size of work
     */
    static void dorgtr(Uplo uplo, int n, double[] A, int lda,
                       double[] tau, double[] work, int workOff, int lwork) {
        Dorgtr.dorgtr(uplo, n, A, lda, tau, 0, work, workOff, lwork);
    }

    // ==================== Dlascl ====================

    /**
     * Multiplies a general matrix by a real scalar: A := (cto/cfrom) * A (LAPACK DLASCL).
     * Handles overflow/underflow by scaling in multiple steps if necessary.
     *
     * @param kind  matrix type: 'G' general, 'U' upper triangular, 'L' lower triangular, etc.
     * @param kl    number of subdiagonals (for banded matrices)
     * @param ku    number of superdiagonals (for banded matrices)
     * @param cfrom scale from value (denominator)
     * @param cto   scale to value (numerator)
     * @param m     number of rows
     * @param n     number of columns
     * @param a     matrix to scale (modified in place)
     * @param aOff  offset into a
     * @param lda   leading dimension of a
     */
    static void dlascl(char kind, int kl, int ku, double cfrom, double cto,
                       int m, int n, double[] a, int aOff, int lda) {
        Dlamv.dlascl(kind, kl, ku, cfrom, cto, m, n, a, aOff, lda);
    }

    /**
     * Scales a vector by a real scalar: x := (cto/cfrom) * x (LAPACK DLASCL vector form).
     *
     * @param kind matrix type (typically 'G')
     * @param cfrom scale from value (denominator)
     * @param cto   scale to value (numerator)
     * @param n     number of elements
     * @param a     vector to scale (modified in place)
     * @param off   offset into a
     * @param inc   stride in a
     */
    static void dlascl(char kind, double cfrom, double cto,
                       int n, double[] a, int off, int inc) {
        Dlamv.dlascl(kind, cfrom, cto, n, a, off, inc);
    }

    // ==================== Dgehrd ====================

    /**
     * Reduces a general matrix to upper Hessenberg form: A = Q*H*Q^T (LAPACK DGEHRD).
     * The orthogonal matrix Q is represented as a product of elementary reflectors.
     *
     * @param n       order of matrix A
     * @param ilo     lower index of the balanced submatrix (1-based, use 1 if not balanced)
     * @param ihi     upper index of the balanced submatrix (1-based, use n if not balanced)
     * @param A       matrix (n × n, row-major), overwritten with H and reflector vectors
     * @param lda     leading dimension of A
     * @param tau     Householder scalars (output, length n-1)
     * @param tauOff  offset into tau
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dgehrd(int n, int ilo, int ihi, double[] A, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        return Dgehrd.dgehrd(n, ilo, ihi, A, 0, lda, tau, tauOff, work, workOff, lwork);
    }

    // ==================== Dhseqr ====================

    /**
     * Computes eigenvalues and Schur factorization of an upper Hessenberg matrix (LAPACK DHSEQR).
     * Uses the multishift QR algorithm with aggressive early deflation.
     *
     * @param job    'E' for eigenvalues only, 'S' for Schur form
     * @param compz  'N' no Z, 'I' initialize Z to identity, 'V' accumulate into Z
     * @param n      order of H
     * @param ilo    lower index of the active submatrix (1-based)
     * @param ihi    upper index of the active submatrix (1-based)
     * @param H      upper Hessenberg matrix (n × n, row-major), overwritten
     * @param ldh    leading dimension of H
     * @param wr     real parts of eigenvalues (output, length n)
     * @param wi     imaginary parts of eigenvalues (output, length n)
     * @param z      Schur vectors (n × n, row-major); used if compz != 'N'
     * @param ldz    leading dimension of z
     * @param work   workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork  size of work; use -1 for workspace query
     * @return 0 on success; positive value i if eigenvalues i+1..n were not computed
     */
    static int dhseqr(char job, char compz, int n, int ilo, int ihi,
                      double[] H, int ldh, double[] wr, double[] wi,
                      double[] z, int ldz, double[] work, int workOff, int lwork) {
        return Dhseqr.dhseqr(job, compz, n, ilo, ihi, H, ldh, wr, wi, z, ldz, work, workOff, lwork);
    }

    // ==================== Dorghr ====================

    /**
     * Generates the orthogonal matrix Q from Hessenberg reduction (LAPACK DORGHR).
     * Q is the product of elementary reflectors produced by DGEHRD.
     *
     * @param n       order of matrix Q
     * @param ilo     lower index of the balanced submatrix (1-based)
     * @param ihi     upper index of the balanced submatrix (1-based)
     * @param A       matrix from DGEHRD (n × n, row-major), overwritten with Q
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     Householder scalars from DGEHRD
     * @param tauOff  offset into tau
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     */
    static void dorghr(int n, int ilo, int ihi, double[] A, int aOff, int lda,
                        double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        Dgees.dorghr(n, ilo, ihi, A, aOff, lda, tau, tauOff, work, workOff, lwork);
    }

    // ==================== Dlaset ====================

    /**
     * Initializes a matrix to diagonal and off-diagonal values (LAPACK DLASET).
     * Sets off-diagonal elements to alpha and diagonal elements to beta.
     *
     * @param uplo 'U' upper triangle, 'L' lower triangle, 'A' or other for full matrix
     * @param m    number of rows
     * @param n    number of columns
     * @param alpha value for off-diagonal elements
     * @param beta  value for diagonal elements
     * @param A    matrix (modified in place)
     * @param aOff offset into A
     * @param lda  leading dimension of A
     */
    static void dlaset(Uplo uplo, int m, int n, double alpha, double beta,
                       double[] A, int aOff, int lda) {
        Dlamv.dlaset(uplo.code, m, n, alpha, beta, A, aOff, lda);
    }

    // ==================== Dgebal ====================

    /**
     * Balances a general matrix to improve eigenvalue computation.
     *
     * @param job   'N' no operation, 'P' permute, 'S' scale, 'B' both
     * @param n     order of matrix
     * @param A     matrix (n x n, row-major), overwritten on output
     * @param lda   leading dimension of A
     * @param scale output: scaling factors
     * @param scaleOff offset in scale array
     * @param out   output: double[2] {ilo, ihi} indices of balanced submatrix
     * @param outOff offset in out array
     */
    static void dgebal(char job, int n, double[] A, int lda, double[] scale, int scaleOff, double[] out, int outOff) {
        Dgebal.dgebal(job, n, A, lda, scale, scaleOff, out, outOff);
    }

    // ==================== Dgebak ====================

    /**
     * Transforms eigenvectors back after matrix balancing (LAPACK DGEBAK).
     * Reverses the transformations applied by DGEBAL.
     *
     * @param job      balancing type used in DGEBAL: 'N', 'P', 'S', or 'B'
     * @param side     {@link Side#Right} for right eigenvectors, {@link Side#Left} for left
     * @param n        order of matrix
     * @param ilo      lower index from DGEBAL (1-based)
     * @param ihi      upper index from DGEBAL (1-based)
     * @param scale    scaling factors from DGEBAL
     * @param scaleOff offset into scale
     * @param m        number of eigenvectors
     * @param V        eigenvector matrix (n × m, row-major), modified in place
     * @param ldv      leading dimension of V
     */
    static void dgebak(char job, Side side, int n, int ilo, int ihi,
                       double[] scale, int scaleOff, int m, double[] V, int ldv) {
        Dgebak.dgebak(job, side, n, ilo, ihi, scale, scaleOff, m, V, ldv);
    }

    // ==================== Dlacpy ====================

    /**
     * Copies all or part of a matrix to another matrix (LAPACK DLACPY).
     *
     * @param uplo 'U' upper triangle, 'L' lower triangle, other for full matrix
     * @param m    number of rows
     * @param n    number of columns
     * @param A    source matrix
     * @param aOff offset into A
     * @param lda  leading dimension of A
     * @param B    destination matrix (modified in place)
     * @param bOff offset into B
     * @param ldb  leading dimension of B
     */
    static void dlacpy(Uplo uplo, int m, int n,
                       double[] A, int aOff, int lda,
                       double[] B, int bOff, int ldb) {
        Dlamv.dlacpy(uplo.code, m, n, A, aOff, lda, B, bOff, ldb);
    }

    // ==================== Dgeqr (QR decomposition) ====================

    /**
     * Unblocked QR factorization: A = Q*R (LAPACK DGEQR2).
     *
     * @param m      number of rows
     * @param n      number of columns
     * @param A      matrix (m × n, row-major), overwritten with R and reflector vectors
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param tau    Householder scalars (output, length min(m,n))
     * @param tauOff offset into tau
     * @param work   workspace (length n)
     * @param workOff offset into work
     */
    static void dgeqr2(int m, int n, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        Dgeqr.dgeqr2(m, n, A, aOff, lda, tau, tauOff, work, workOff);
    }

    /**
     * Blocked QR factorization: A = Q*R (LAPACK DGEQRF).
     *
     * @param m       number of rows
     * @param n       number of columns
     * @param A       matrix (m × n, row-major), overwritten with R and reflector vectors
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     Householder scalars (output, length min(m,n))
     * @param tauOff  offset into tau
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dgeqrf(int m, int n, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        return Dgeqr.dgeqrf(m, n, A, aOff, lda, tau, tauOff, work, workOff, lwork);
    }

    /**
     * Generates Q from unblocked QR factorization (LAPACK DORG2R).
     *
     * @param m      number of rows
     * @param n      number of columns
     * @param k      number of reflectors
     * @param A      matrix from DGEQR2 (m × n, row-major), overwritten with Q
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param tau    Householder scalars from DGEQR2
     * @param tauOff offset into tau
     * @param work   workspace (length n)
     * @param workOff offset into work
     */
    static void dorg2r(int m, int n, int k, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        Dgeqr.dorg2r(m, n, k, A, aOff, lda, tau, tauOff, work, workOff);
    }

    /**
     * Generates Q from blocked QR factorization (LAPACK DORGQR).
     *
     * @param m       number of rows
     * @param n       number of columns
     * @param k       number of reflectors
     * @param A       matrix from DGEQRF (m × n, row-major), overwritten with Q
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     Householder scalars from DGEQRF
     * @param tauOff  offset into tau
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dorgqr(int m, int n, int k, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        return Dgeqr.dorgqr(m, n, k, A, aOff, lda, tau, tauOff, work, workOff, lwork);
    }

    /**
     * Applies Q from QR factorization to a matrix (LAPACK DORMQR, blocked).
     *
     * @param side    {@link Side#Left} for Q*C or Q^T*C, {@link Side#Right} for C*Q or C*Q^T
     * @param trans   {@link Trans#NoTrans} for Q, {@link Trans#Trans} for Q^T
     * @param m       number of rows in C
     * @param n       number of columns in C
     * @param k       number of elementary reflectors
     * @param V       Householder vectors from DGEQRF
     * @param vOff    offset into V
     * @param ldv     leading dimension of V
     * @param tau     scalar factors of reflectors
     * @param tauOff  offset into tau
     * @param C       matrix C, overwritten with result
     * @param cOff    offset into C
     * @param ldc     leading dimension of C
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dormqr(Side side, Trans trans, int m, int n, int k,
                      double[] V, int vOff, int ldv, double[] tau, int tauOff,
                      double[] C, int cOff, int ldc, double[] work, int workOff, int lwork) {
        return Dormqr.dormqr(side, trans, m, n, k, V, vOff, ldv, tau, tauOff,
                            C, cOff, ldc, work, workOff, lwork);
    }

    /**
     * DTRTRS - solves a triangular system A * X = B or A^T * X = B.
     *
     * @param uplo  'U' for upper, 'L' for lower
     * @param trans 'N' for A*X=B, 'T' for A^T*X=B
     * @param diag  'U' for unit diagonal, 'N' otherwise
     * @param n     order of A
     * @param nrhs  number of right-hand sides
     * @param A     triangular matrix
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param B     right-hand side, overwritten with solution
     * @param bOff  offset into B
     * @param ldb   leading dimension of B
     * @return true if successful, false if singular
     */
    static boolean dtrtrs(Uplo uplo, Trans trans, Diag diag, int n, int nrhs,
                          double[] A, int aOff, int lda,
                          double[] B, int bOff, int ldb) {
        return Dtrtrs.dtrtrs(uplo, trans, diag, n, nrhs, A, aOff, lda, B, bOff, ldb);
    }

    // ==================== Dlamch ====================

    /**
     * Returns machine-dependent constants (LAPACK DLAMCH).
     *
     * @param cmach Specifies which constant: 'E' epsilon, 'S' safe min, etc.
     * @return the requested constant
     */
    static double dlamch(char cmach) {
        return Dlamch.dlamch(cmach);
    }

    /**
     * Returns the safe minimum (smallest normalized number).
     */
    static double safmin() {
        return Dlamch.safmin();
    }

    /**
     * Returns epsilon (relative machine precision).
     */
    static double eps() {
        return Dlamch.eps();
    }

    // ==================== Drscl ====================

    /**
     * Multiplies a vector by the reciprocal of a scalar: x := (1/a) * x (LAPACK DRSCL).
     * Avoids division by computing 1/a first, then scaling.
     *
     * @param n    number of elements
     * @param a    scalar divisor
     * @param x    vector (modified in place)
     * @param xOff offset into x
     * @param incX stride in x
     */
    static void drscl(int n, double a, double[] x, int xOff, int incX) {
        Drscl.drscl(n, a, x, xOff, incX);
    }

    // ==================== Dlartg ====================

    /**
     * Generates a Givens rotation (LAPACK DLARTG).
     *
     * @param f first component
     * @param g second component
     * @return [cs, sn, r] where cs = cos, sn = sin, r = sqrt(f^2 + g^2)
     */
    static double[] dlartg(double f, double g) {
        double[] out = new double[3];
        Dlartg.dlartg(f, g, out, 0);
        return out;
    }

    // ==================== Ilaenv ====================

    /**
     * Returns environment-dependent information about LAPACK (LAPACK ILAENV).
     *
     * @param ispec Specifies which parameter
     * @param name  Routine name
     * @param opts  Options string
     * @param n1    First dimension parameter
     * @param n2    Second dimension parameter
     * @param n3    Third dimension parameter
     * @param n4    Fourth dimension parameter
     * @return the requested parameter
     */
    static int ilaenv(int ispec, String name, String opts, int n1, int n2, int n3, int n4) {
        return Ilaenv.ilaenv(ispec, name, opts, n1, n2, n3, n4);
    }

    // ==================== Dlaln2 ====================

    /**
     * Solves a 1×1 or 2×2 linear system with scaling to avoid overflow (LAPACK DLALN2).
     * Solves (ca*A - wr*I - wi*J) * X = scale*B, where J = [0 1; -1 0].
     *
     * @param trans  {@link Trans#Trans} to solve with A^T, {@link Trans#NoTrans} for A
     * @param na     size of A: 1 or 2
     * @param nw     number of right-hand sides: 1 (real) or 2 (complex)
     * @param smin   lower bound on |A| to avoid singularity
     * @param ca     scalar multiplier for A
     * @param a      matrix A (na × na, row-major)
     * @param aOff   offset into a
     * @param lda    leading dimension of a
     * @param d1     (1,1) element of diagonal scaling matrix D
     * @param d2     (2,2) element of diagonal scaling matrix D (used if na=2)
     * @param b      right-hand side matrix (na × nw, row-major)
     * @param bOff   offset into b
     * @param ldb    leading dimension of b
     * @param wr     real part of shift
     * @param wi     imaginary part of shift (used if nw=2)
     * @param x      solution matrix (na × nw, row-major)
     * @param xOff   offset into x
     * @param ldx    leading dimension of x
     * @return true if the system is nearly singular (scale < 1)
     */
    static boolean dlaln2(Trans trans, int na, int nw, double smin, double ca,
                          double[] a, int aOff, int lda, double d1, double d2,
                          double[] b, int bOff, int ldb, double wr, double wi,
                          double[] x, int xOff, int ldx) {
        return Dlaln2.dlaln2(trans.code == Trans.Trans.code, na, nw, smin, ca, a, aOff, lda, d1, d2, b, bOff, ldb, wr, wi, x, xOff, ldx);
    }

    // ==================== Dtrevc3 ====================

    /**
     * Computes right and/or left eigenvectors of a quasi-triangular matrix (LAPACK DTREVC3).
     *
     * @param side    'R' for right eigenvectors, 'L' for left, 'B' for both
     * @param howmny  'A' for all, 'B' for backtransform, 'S' for selected
     * @param selected boolean array selecting which eigenvectors to compute (for howmny='S')
     * @param n       order of T
     * @param T       quasi-triangular matrix (n × n, row-major)
     * @param tOff    offset into T
     * @param ldt     leading dimension of T
     * @param VL      left eigenvectors (n × mm, row-major); used if side='L' or 'B'
     * @param vlOff   offset into VL
     * @param ldvl    leading dimension of VL
     * @param VR      right eigenvectors (n × mm, row-major); used if side='R' or 'B'
     * @param vrOff   offset into VR
     * @param ldvr    leading dimension of VR
     * @param mm      number of columns in VL/VR
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return number of eigenvectors computed
     */
    static int dtrevc3(char side, char howmny, boolean[] selected, int n,
                        double[] T, int tOff, int ldt,
                        double[] VL, int vlOff, int ldvl,
                        double[] VR, int vrOff, int ldvr,
                        int mm, double[] work, int workOff, int lwork) {
        return Dtrevc3.dtrevc3(side, howmny, selected, n, T, tOff, ldt, VL, vlOff, ldvl, VR, vrOff, ldvr, mm, work, workOff, lwork);
    }

    // ==================== Dgebrd ====================

    /**
     * Reduces a general matrix to bidiagonal form: A = Q*B*P^T (LAPACK DGEBRD).
     *
     * @param m       number of rows
     * @param n       number of columns
     * @param A       matrix (m × n, row-major), overwritten with bidiagonal form and reflectors
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param d       diagonal elements of B (output, length min(m,n))
     * @param dOff    offset into d
     * @param e       off-diagonal elements of B (output, length min(m,n)-1)
     * @param eOff    offset into e
     * @param tauQ    Householder scalars for Q (output, length min(m,n))
     * @param tauQOff offset into tauQ
     * @param tauP    Householder scalars for P (output, length min(m,n))
     * @param tauPOff offset into tauP
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dgebrd(int m, int n, double[] A, int aOff, int lda,
                      double[] d, int dOff, double[] e, int eOff,
                      double[] tauQ, int tauQOff, double[] tauP, int tauPOff,
                      double[] work, int workOff, int lwork) {
        return Dgebrd.dgebrd(m, n, A, aOff, lda, d, dOff, e, eOff, tauQ, tauQOff, tauP, tauPOff, work, workOff, lwork);
    }

    // ==================== Dgebd2 ====================

    /**
     * Unblocked bidiagonal reduction: A = Q*B*P^T (LAPACK DGEBD2).
     *
     * @param m       number of rows
     * @param n       number of columns
     * @param A       matrix (m × n, row-major), overwritten with bidiagonal form and reflectors
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param d       diagonal elements of B (output, length min(m,n))
     * @param dOff    offset into d
     * @param e       off-diagonal elements of B (output, length min(m,n)-1)
     * @param eOff    offset into e
     * @param tauQ    Householder scalars for Q (output, length min(m,n))
     * @param tauQOff offset into tauQ
     * @param tauP    Householder scalars for P (output, length min(m,n))
     * @param tauPOff offset into tauP
     * @param work    workspace (length max(m,n))
     * @param workOff offset into work
     */
    static void dgebd2(int m, int n, double[] A, int aOff, int lda,
                       double[] d, int dOff, double[] e, int eOff,
                       double[] tauQ, int tauQOff, double[] tauP, int tauPOff,
                       double[] work, int workOff) {
        Dgebd2.dgebd2(m, n, A, aOff, lda, d, dOff, e, eOff, tauQ, tauQOff, tauP, tauPOff, work, workOff);
    }

    // ==================== Dgelq (LQ decomposition) ====================

    /**
     * Unblocked LQ factorization: A = L*Q (LAPACK DGELQ2).
     *
     * @param m      number of rows
     * @param n      number of columns
     * @param A      matrix (m × n, row-major), overwritten with L and reflector vectors
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param tau    Householder scalars (output, length min(m,n))
     * @param tauOff offset into tau
     * @param work   workspace (length m)
     * @param workOff offset into work
     */
    static void dgelq2(int m, int n, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        Dgelq.dgelq2(m, n, A, aOff, lda, tau, tauOff, work, workOff);
    }

    /**
     * Blocked LQ factorization: A = L*Q (LAPACK DGELQF).
     *
     * @param m       number of rows
     * @param n       number of columns
     * @param A       matrix (m × n, row-major), overwritten with L and reflector vectors
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     Householder scalars (output, length min(m,n))
     * @param tauOff  offset into tau
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dgelqf(int m, int n, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        return Dgelq.dgelqf(m, n, A, aOff, lda, tau, tauOff, work, workOff, lwork);
    }

    /**
     * Generates Q from unblocked LQ factorization (LAPACK DORGL2).
     *
     * @param m      number of rows
     * @param n      number of columns
     * @param k      number of reflectors
     * @param A      matrix from DGELQ2 (m × n, row-major), overwritten with Q
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param tau    Householder scalars from DGELQ2
     * @param tauOff offset into tau
     * @param work   workspace (length m)
     * @param workOff offset into work
     */
    static void dorgl2(int m, int n, int k, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        Dgelq.dorgl2(m, n, k, A, aOff, lda, tau, tauOff, work, workOff);
    }

    /**
     * Generates Q from blocked LQ factorization (LAPACK DORGLQ).
     *
     * @param m      number of rows
     * @param n      number of columns
     * @param k      number of reflectors
     * @param A      matrix from DGELQF (m × n, row-major), overwritten with Q
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param tau    Householder scalars from DGELQF
     * @param tauOff offset into tau
     * @param work   workspace
     * @param workOff offset into work
     */
    static void dorglq(int m, int n, int k, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        Dgelq.dorglq(m, n, k, A, aOff, lda, tau, tauOff, work, workOff, m);
    }

    /**
     * Multiplies a matrix by Q from LQ factorization (LAPACK DORMLQ).
     *
     * @param side    {@link Side#Left} for Q*C, {@link Side#Right} for C*Q
     * @param trans   {@link Trans#NoTrans} for Q, {@link Trans#Trans} for Q^T
     * @param m       number of rows in C
     * @param n       number of columns in C
     * @param k       number of elementary reflectors
     * @param A       Householder vectors from DGELQF
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     scalar factors of reflectors
     * @param tauOff  offset into tau
     * @param C       matrix C, overwritten with result
     * @param cOff    offset into C
     * @param ldc     leading dimension of C
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dormlq(Side side, Trans trans, int m, int n, int k,
                      double[] A, int aOff, int lda, double[] tau, int tauOff,
                      double[] C, int cOff, int ldc, double[] work, int workOff, int lwork) {
        return Dgelq.dormlq(side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff, lwork);
    }

    // ==================== Dorgbr ====================

    /**
     * Generates Q or P^T from bidiagonal reduction (LAPACK DORGBR).
     *
     * @param vect    'Q' to generate Q, 'P' to generate P^T
     * @param m       number of rows
     * @param n       number of columns
     * @param k       number of reflectors (min(m,n) for Q, min(m,n) for P^T)
     * @param A       matrix from DGEBRD (m × n, row-major), overwritten with Q or P^T
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     Householder scalars from DGEBRD (tauQ for 'Q', tauP for 'P')
     * @param tauOff  offset into tau
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dorgbr(char vect, int m, int n, int k, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        return Dorgbr.dorgbr(vect, m, n, k, A, aOff, lda, tau, tauOff, work, workOff, lwork);
    }

    // ==================== Dlasv2 ====================

    /**
     * Computes the SVD of a 2×2 triangular matrix (LAPACK DLASV2).
     * Decomposes [f g; 0 h] = U * diag(ssmin, ssmax) * V^T.
     *
     * @param f   (1,1) element
     * @param g   (1,2) element
     * @param h   (2,2) element
     * @param out output array: [ssmin, ssmax, snr, csr, snl, csl]
     * @param off offset into out
     */
    static void dlasv2(double f, double g, double h, double[] out, int off) {
        Dlas2.dlasv2(f, g, h, out, off);
    }

    // ==================== Dlasr ====================

    /**
     * Applies a sequence of plane rotations to a matrix (LAPACK DLASR).
     *
     * @param side    {@link Side#Left} to apply from left, {@link Side#Right} from right
     * @param pivot   'V' variable pivot, 'T' top pivot, 'B' bottom pivot
     * @param direct  'F' forward sequence, 'B' backward sequence
     * @param m       number of rows
     * @param n       number of columns
     * @param c       cosines of rotations (length m-1 or n-1)
     * @param cOff    offset into c
     * @param s       sines of rotations (length m-1 or n-1)
     * @param sOff    offset into s
     * @param a       matrix (m × n, row-major), modified in place
     * @param aOff    offset into a
     * @param lda     leading dimension of a
     */
    static void dlasr(Side side, char pivot, char direct, int m, int n,
                      double[] c, int cOff, double[] s, int sOff,
                      double[] a, int aOff, int lda) {
        Dlasr.dlasr(side, pivot, direct, m, n, c, cOff, s, sOff, a, aOff, lda);
    }

    // ==================== Dbdsqr ====================

    /**
     * Computes the SVD of a real bidiagonal matrix using the QR algorithm (LAPACK DBDSQR).
     * Computes singular values and optionally updates matrices U, V^T, and C.
     *
     * @param uplo   {@link Uplo#Upper} for upper bidiagonal, {@link Uplo#Lower} for lower
     * @param n      order of the bidiagonal matrix
     * @param ncvt   number of columns in V^T (0 if not wanted)
     * @param nru    number of rows in U (0 if not wanted)
     * @param ncc    number of columns in C (0 if not wanted)
     * @param d      diagonal elements (length n), overwritten with singular values
     * @param dOff   offset into d
     * @param e      off-diagonal elements (length n-1), overwritten on exit
     * @param eOff   offset into e
     * @param vt     matrix V^T (n × ncvt, row-major); updated if ncvt > 0
     * @param vtOff  offset into vt
     * @param ldvt   leading dimension of vt
     * @param u      matrix U (nru × n, row-major); updated if nru > 0
     * @param uOff   offset into u
     * @param ldu    leading dimension of u
     * @param c      matrix C (n × ncc, row-major); updated if ncc > 0
     * @param cOff   offset into c
     * @param ldc    leading dimension of c
     * @param work   workspace (length 4*n)
     * @param workOff offset into work
     * @return true on success, false if convergence failed
     */
    static boolean dbdsqr(Uplo uplo, int n, int ncvt, int nru, int ncc,
                          double[] d, int dOff, double[] e, int eOff,
                          double[] vt, int vtOff, int ldvt,
                          double[] u, int uOff, int ldu,
                          double[] c, int cOff, int ldc,
                          double[] work, int workOff) {
        return Dbdsqr.dbdsqr(uplo, n, ncvt, nru, ncc, d, dOff, e, eOff, vt, vtOff, ldvt, u, uOff, ldu, c, cOff, ldc, work, workOff);
    }

    // ==================== Dgetr (LU decomposition) ====================

    /**
     * LU factorization with partial pivoting: A = P*L*U (LAPACK DGETRF).
     *
     * @param m       number of rows
     * @param n       number of columns
     * @param A       matrix (m × n, row-major), overwritten with L and U factors
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param ipiv    pivot indices (output, length min(m,n))
     * @param ipivOff offset into ipiv
     * @return 0 on success; positive value i if U[i,i] is exactly zero (singular)
     */
    static int dgetrf(int m, int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff) {
        return Dgetr.dgetrf(m, n, A, aOff, lda, ipiv, ipivOff);
    }

    /**
     * Computes the inverse of a matrix using LU factorization (LAPACK DGETRI).
     *
     * @param n       order of matrix A
     * @param A       LU factorization from DGETRF (n × n, row-major), overwritten with inverse
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param ipiv    pivot indices from DGETRF
     * @param ipivOff offset into ipiv
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return true on success, false if matrix is singular
     */
    static boolean dgetri(int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff,
                          double[] work, int workOff, int lwork) {
        return Dgetr.dgetri(n, A, aOff, lda, ipiv, ipivOff, work, workOff, lwork);
    }

    /**
     * Solves a system of linear equations using LU factorization: A*X = B (LAPACK DGETRS).
     *
     * @param trans   {@link Trans#NoTrans} for A*X=B, {@link Trans#Trans} for A^T*X=B
     * @param n       order of A
     * @param nrhs    number of right-hand sides
     * @param A       LU factorization from DGETRF (n × n, row-major)
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param ipiv    pivot indices from DGETRF
     * @param ipivOff offset into ipiv
     * @param B       right-hand side matrix (n × nrhs, row-major), overwritten with solution
     * @param bOff    offset into B
     * @param ldb     leading dimension of B
     */
    static void dgetrs(Trans trans, int n, int nrhs, double[] A, int aOff, int lda,
                       int[] ipiv, int ipivOff, double[] B, int bOff, int ldb) {
        Dgetr.dgetrs(trans, n, nrhs, A, aOff, lda, ipiv, ipivOff, B, bOff, ldb);
    }

    /**
     * Estimates the reciprocal condition number of a general matrix (LAPACK DGECON).
     *
     * @param norm  '1' for 1-norm, 'I' for infinity norm
     * @param n     order of matrix A
     * @param A     LU factorization from DGETRF (n × n, row-major)
     * @param lda   leading dimension of A
     * @param anorm 1-norm or infinity norm of the original matrix A
     * @param work  workspace (length 4*n)
     * @param iwork integer workspace (length n)
     * @return reciprocal condition number estimate (0 if singular)
     */
    static double dgecon(Norm norm, int n, double[] A, int lda, double anorm,
                         double[] work, int[] iwork) {
        return Dgetr.dgecon(norm.code, n, A, lda, anorm, work, iwork);
    }

    // ==================== Dpotr (Cholesky decomposition) ====================

    /**
     * Cholesky factorization: A = U^T*U or A = L*L^T (LAPACK DPOTRF).
     *
     * @param uplo {@link Uplo#Upper} for U^T*U, {@link Uplo#Lower} for L*L^T
     * @param n    order of matrix A
     * @param A    symmetric positive definite matrix (n × n, row-major), overwritten with factor
     * @param aOff offset into A
     * @param lda  leading dimension of A
     * @return 0 on success; positive value i if the leading minor of order i is not positive definite
     */
    static int dpotrf(Uplo uplo, int n, double[] A, int aOff, int lda) {
        return Dpotr.dpotrf(uplo, n, A, aOff, lda);
    }

    /**
     * Computes the inverse of a symmetric positive definite matrix (LAPACK DPOTRI).
     *
     * @param uplo {@link Uplo#Upper} or {@link Uplo#Lower}
     * @param n    order of matrix A
     * @param A    Cholesky factor from DPOTRF (n × n, row-major), overwritten with inverse
     * @param aOff offset into A
     * @param lda  leading dimension of A
     * @return true on success, false if matrix is singular
     */
    static boolean dpotri(Uplo uplo, int n, double[] A, int aOff, int lda) {
        return Dpotr.dpotri(uplo, n, A, aOff, lda);
    }

    /**
     * Solves a system using Cholesky factorization: A*X = B (LAPACK DPOTRS).
     *
     * @param uplo {@link Uplo#Upper} or {@link Uplo#Lower}
     * @param n    order of A
     * @param nrhs number of right-hand sides
     * @param A    Cholesky factor from DPOTRF (n × n, row-major)
     * @param aOff offset into A
     * @param lda  leading dimension of A
     * @param B    right-hand side matrix (n × nrhs, row-major), overwritten with solution
     * @param bOff offset into B
     * @param ldb  leading dimension of B
     */
    static void dpotrs(Uplo uplo, int n, int nrhs, double[] A, int aOff, int lda,
                       double[] B, int bOff, int ldb) {
        Dpotr.dpotrs(uplo, n, nrhs, A, aOff, lda, B, bOff, ldb);
    }

    /**
     * Estimates the reciprocal condition number of a symmetric positive definite matrix (LAPACK DPOCON).
     *
     * @param uplo  {@link Uplo#Upper} or {@link Uplo#Lower}
     * @param n     order of matrix A
     * @param A     Cholesky factor from DPOTRF (n × n, row-major)
     * @param lda   leading dimension of A
     * @param anorm 1-norm of the original matrix A
     * @param work  workspace (length 3*n)
     * @param iwork integer workspace (length n)
     * @return reciprocal condition number estimate (0 if singular)
     */
    static double dpocon(Uplo uplo, int n, double[] A, int lda, double anorm,
                         double[] work, int[] iwork) {
        return Dpotr.dpocon(uplo, n, A, lda, anorm, work, iwork);
    }

    // ==================== Dsymm ====================

    /**
     * Symmetric matrix-matrix multiplication: C := alpha*A*B + beta*C or C := alpha*B*A + beta*C (BLAS DSYMM).
     *
     * @param side  {@link Side#Left} for A*B, {@link Side#Right} for B*A
     * @param uplo  {@link Uplo#Upper} or {@link Uplo#Lower} — which triangle of A to use
     * @param m     rows of C and B
     * @param n     columns of C and B
     * @param alpha scalar
     * @param A     symmetric matrix (row-major); m×m for Left, n×n for Right
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param B     general matrix (m×n, row-major)
     * @param bOff  offset into B
     * @param ldb   leading dimension of B
     * @param beta  scalar for C
     * @param C     result matrix (m×n, row-major), modified in place
     * @param cOff  offset into C
     * @param ldc   leading dimension of C
     */
    static void dsymm(Side side, Uplo uplo, int m, int n,
                      double alpha,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb,
                      double beta,
                      double[] C, int cOff, int ldc) {
        Dsymm.dsymm(side, uplo, m, n, alpha, A, aOff, lda, B, bOff, ldb, beta, C, cOff, ldc);
    }

    // ==================== Dsygst (Generalized eigenproblem reduction) ====================

    /**
     * Reduces a real symmetric-definite generalized eigenproblem to standard form (LAPACK DSYGST).
     *
     * <p>Given the Cholesky factor of B (from dpotrf), overwrites A:
     * <ul>
     *   <li>Type 1: A ← inv(L)·A·inv(Lᵀ)  (lower) — for A·x = λ·B·x</li>
     *   <li>Type 2/3: A ← Lᵀ·A·L           (lower) — for A·B·x = λ·x or B·A·x = λ·x</li>
     * </ul>
     *
     * @param itype problem type: 1, 2, or 3
     * @param uplo  {@link Uplo#Lower} or {@link Uplo#Upper}
     * @param n     order of A and B
     * @param A     symmetric matrix (n×n, row-major), overwritten with reduced form
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param B     Cholesky factor from dpotrf (n×n, row-major)
     * @param bOff  offset into B
     * @param ldb   leading dimension of B
     * @return 0 on success
     */
    static int dsygst(int itype, Uplo uplo, int n,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {
        return Dsygst.dsygst(itype, uplo, n, A, aOff, lda, B, bOff, ldb);
    }

    // ==================== Dtrtri (Triangular matrix inverse) ====================

    /**
     * Computes the inverse of a triangular matrix (LAPACK DTRTRI).
     *
     * @param uplo {@link Uplo#Upper} for upper, {@link Uplo#Lower} for lower triangular
     * @param diag {@link Diag#Unit} for unit diagonal, {@link Diag#NonUnit} otherwise
     * @param n    order of matrix A
     * @param A    triangular matrix (n × n, row-major), overwritten with inverse
     * @param aOff offset into A
     * @param lda  leading dimension of A
     * @return 0 on success; positive value i if A[i,i] is exactly zero (singular)
     */
    static int dtrtri(Uplo uplo, Diag diag, int n, double[] A, int aOff, int lda) {
        return Dtrtri.dtrtri(uplo, diag, n, A, aOff, lda);
    }

    /**
     * Estimates the reciprocal condition number of a triangular matrix (LAPACK DTRCON).
     *
     * @param norm  '1' for 1-norm, 'I' for infinity norm
     * @param uplo  {@link Uplo#Upper} or {@link Uplo#Lower}
     * @param diag  {@link Diag#Unit} or {@link Diag#NonUnit}
     * @param n     order of matrix A
     * @param A     triangular matrix (n × n, row-major)
     * @param lda   leading dimension of A
     * @param work  workspace (length 3*n)
     * @param iwork integer workspace (length n)
     * @return reciprocal condition number estimate (0 if singular)
     */
    static double dtrcon(Norm norm, Uplo uplo, Diag diag, int n,
                         double[] A, int lda, double[] work, int[] iwork) {
        return Dtrtri.dtrcon(norm.code, uplo, diag, n, A, lda, work, iwork);
    }

    // ==================== Dlauum (Cholesky inverse multiplication) ====================

    /**
     * Computes the product U^T*U or L*L^T for a triangular matrix (LAPACK DLAUUM).
     * Used as a step in computing the inverse of a symmetric positive definite matrix.
     *
     * @param uplo {@link Uplo#Upper} to compute U^T*U, {@link Uplo#Lower} to compute L*L^T
     * @param n    order of matrix A
     * @param A    triangular matrix (n × n, row-major), overwritten with the product
     * @param aOff offset into A
     * @param lda  leading dimension of A
     */
    static void dlauum(Uplo uplo, int n, double[] A, int aOff, int lda) {
        Dlauum.dlauum(uplo, n, A, aOff, lda);
    }

    // ==================== Dgeev (Eigenvalue decomposition) ====================

    /**
     * Computes eigenvalues and optionally eigenvectors of a general matrix (LAPACK DGEEV).
     *
     * @param wantvl  true to compute left eigenvectors
     * @param wantvr  true to compute right eigenvectors
     * @param n       order of matrix A
     * @param A       general matrix (n × n, row-major), overwritten
     * @param lda     leading dimension of A
     * @param wr      real parts of eigenvalues (output, length n)
     * @param wi      imaginary parts of eigenvalues (output, length n)
     * @param vl      left eigenvectors (n × n, row-major); used if wantvl=true
     * @param ldvl    leading dimension of vl
     * @param vr      right eigenvectors (n × n, row-major); used if wantvr=true
     * @param ldvr    leading dimension of vr
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success; positive value i if QR algorithm failed to compute all eigenvalues
     */
    static int dgeev(boolean wantvl, boolean wantvr, int n, double[] A, int lda,
                     double[] wr, double[] wi, double[] vl, int ldvl,
                     double[] vr, int ldvr, double[] work, int workOff, int lwork) {
        return Dgeev.dgeev(wantvl, wantvr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, work, workOff, lwork);
    }

    // ==================== Dgees (Schur decomposition) ====================

    /**
     * Computes the Schur factorization of a general matrix: A = Z*T*Z^T (LAPACK DGEES).
     * Optionally reorders eigenvalues so that selected ones appear first.
     *
     * @param jobvs  'V' to compute Schur vectors Z, 'N' otherwise
     * @param sort   'S' to reorder eigenvalues, 'N' otherwise
     * @param select eigenvalue selection function (used if sort='S')
     * @param n      order of matrix A
     * @param A      general matrix (n × n, row-major), overwritten with quasi-triangular T
     * @param lda    leading dimension of A
     * @param wr     real parts of eigenvalues (output, length n)
     * @param wi     imaginary parts of eigenvalues (output, length n)
     * @param vs     Schur vectors (n × n, row-major); used if jobvs='V'
     * @param ldvs   leading dimension of vs
     * @param work   workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork  size of work; use -1 for workspace query
     * @param bwork  boolean workspace (length n); used if sort='S'
     * @return 0 on success; positive value if QR algorithm failed or reordering failed
     */
    static int dgees(char jobvs, char sort, Select select, int n,
                       double[] A, int lda,
                       double[] wr, double[] wi,
                       double[] vs, int ldvs,
                       double[] work, int workOff, int lwork,
                       boolean[] bwork) {
        return Dgees.dgees(jobvs, sort, select, n, A, lda, wr, wi, vs, ldvs, work, workOff, lwork, bwork);
    }

    // ==================== Dgesvd (Singular Value Decomposition) ====================

    /**
     * Computes the singular value decomposition: A = U*S*V^T (LAPACK DGESVD).
     *
     * @param jobU    'A' all columns of U, 'S' first min(m,n) columns, 'O' overwrite A, 'N' none
     * @param jobVT   'A' all rows of V^T, 'S' first min(m,n) rows, 'O' overwrite A, 'N' none
     * @param m       number of rows
     * @param n       number of columns
     * @param A       matrix (m × n, row-major), overwritten depending on jobU/jobVT
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param s       singular values (output, length min(m,n), in descending order)
     * @param sOff    offset into s
     * @param u       left singular vectors (m × ldu, row-major); used if jobU != 'N'
     * @param uOff    offset into u
     * @param ldu     leading dimension of u
     * @param vt      right singular vectors (ldvt × n, row-major); used if jobVT != 'N'
     * @param vtOff   offset into vt
     * @param ldvt    leading dimension of vt
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success; positive value i if DBDSQR did not converge
     */
    static int dgesvd(char jobU, char jobVT, int m, int n,
                      double[] A, int aOff, int lda,
                      double[] s, int sOff,
                      double[] u, int uOff, int ldu,
                      double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork) {
        return Dgesvd.dgesvd(jobU, jobVT, m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt, work, workOff, lwork);
    }

    // ==================== Dgeqp (QR with Column Pivoting) ====================

    /**
     * QR factorization with column pivoting: A*P = Q*R (LAPACK DGEQP3).
     *
     * @param m       number of rows
     * @param n       number of columns
     * @param A       matrix (m × n, row-major), overwritten with R and reflector vectors
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param jpvt    column pivot indices (input/output, length n); pre-set nonzero to fix columns
     * @param tau     Householder scalars (output, length min(m,n))
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dgeqp3(int m, int n, double[] A, int aOff, int lda,
                      int[] jpvt, double[] tau, double[] work, int workOff, int lwork) {
        return Dgeqp.dgeqp3(m, n, A, aOff, lda, jpvt, tau, work, workOff, lwork);
    }

    // ==================== Dlapmt (Permute columns) ====================

    /**
     * Performs a forward or backward permutation of matrix columns (LAPACK DLAPMT).
     *
     * @param forward true for forward permutation (A := A*P), false for backward (A := A*P^T)
     * @param m       number of rows
     * @param n       number of columns
     * @param A       matrix (m × n, row-major), modified in place
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param k       permutation indices (length n)
     * @param kOff    offset into k
     */
    static void dlapmt(boolean forward, int m, int n, double[] A, int aOff, int lda, int[] k, int kOff) {
        Dgeqp.dlapmt(forward, m, n, A, aOff, lda, k, kOff);
    }

    // ==================== Dgerq (RQ decomposition) ====================

    /**
     * Unblocked RQ factorization: A = R*Q (LAPACK DGERQ2).
     *
     * @param m      number of rows
     * @param n      number of columns
     * @param A      matrix (m × n, row-major), overwritten with R and reflector vectors
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param tau    Householder scalars (output, length min(m,n))
     * @param tauOff offset into tau
     * @param work   workspace (length m)
     * @param workOff offset into work
     */
    static void dgerq2(int m, int n, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        Dgerq.dgerq2(m, n, A, aOff, lda, tau, tauOff, work, workOff);
    }

    /**
     * Applies Q from RQ factorization to a matrix (LAPACK DORMR2, unblocked).
     *
     * @param side    {@link Side#Left} for Q*C, {@link Side#Right} for C*Q
     * @param trans   {@link Trans#NoTrans} for Q, {@link Trans#Trans} for Q^T
     * @param m       number of rows in C
     * @param n       number of columns in C
     * @param k       number of elementary reflectors
     * @param A       Householder vectors from DGERQ2
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     scalar factors of reflectors
     * @param tauOff  offset into tau
     * @param C       matrix C, overwritten with result
     * @param cOff    offset into C
     * @param ldc     leading dimension of C
     * @param work    workspace (length n or m)
     * @param workOff offset into work
     */
    static void dormr2(Side side, Trans trans, int m, int n, int k,
                       double[] A, int aOff, int lda, double[] tau, int tauOff,
                       double[] C, int cOff, int ldc, double[] work, int workOff) {
        Dgerq.dormr2(side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff);
    }

    /**
     * Applies Q from RQ factorization to a matrix (LAPACK DORMRQ, blocked).
     *
     * @param side    {@link Side#Left} for Q*C, {@link Side#Right} for C*Q
     * @param trans   {@link Trans#NoTrans} for Q, {@link Trans#Trans} for Q^T
     * @param m       number of rows in C
     * @param n       number of columns in C
     * @param k       number of elementary reflectors
     * @param A       Householder vectors from DGERQF
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     scalar factors of reflectors
     * @param tauOff  offset into tau
     * @param C       matrix C, overwritten with result
     * @param cOff    offset into C
     * @param ldc     leading dimension of C
     * @param work    workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork   size of work; use -1 for workspace query
     * @return 0 on success
     */
    static int dormrq(Side side, Trans trans, int m, int n, int k,
                      double[] A, int aOff, int lda, double[] tau, int tauOff,
                      double[] C, int cOff, int ldc, double[] work, int workOff, int lwork) {
        return Dgerq.dormrq(side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff, lwork);
    }

    /**
     * Generates Q from unblocked RQ factorization (LAPACK DORGR2).
     *
     * @param m      number of rows
     * @param n      number of columns
     * @param k      number of reflectors
     * @param A      matrix from DGERQ2 (m × n, row-major), overwritten with Q
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param tau    Householder scalars from DGERQ2
     * @param tauOff offset into tau
     * @param work   workspace (length m)
     * @param workOff offset into work
     */
    static void dorgr2(int m, int n, int k, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        Dgerq.dorgr2(m, n, k, A, aOff, lda, tau, tauOff, work, workOff);
    }

    // ==================== Dorm2r (Multiply by Q from QR) ====================

    /**
     * Applies Q from QR factorization to a matrix (LAPACK DORM2R, unblocked).
     *
     * @param side    {@link Side#Left} for Q*C, {@link Side#Right} for C*Q
     * @param trans   {@link Trans#NoTrans} for Q, {@link Trans#Trans} for Q^T
     * @param m       number of rows in C
     * @param n       number of columns in C
     * @param k       number of elementary reflectors
     * @param A       Householder vectors from DGEQR2
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     scalar factors of reflectors
     * @param tauOff  offset into tau
     * @param C       matrix C, overwritten with result
     * @param cOff    offset into C
     * @param ldc     leading dimension of C
     * @param work    workspace (length n or m)
     * @param workOff offset into work
     */
    static void dorm2r(Side side, Trans trans, int m, int n, int k,
                       double[] A, int aOff, int lda, double[] tau, int tauOff,
                       double[] C, int cOff, int ldc, double[] work, int workOff) {
        Dormqr.dorm2r(side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff);
    }

    // ==================== Dggsvp3 (GSVD preprocessing) ====================

    /**
     * Preprocessing step for GSVD: reduces A and B to upper triangular form (LAPACK DGGSVP3).
     * Computes orthogonal matrices U, V, Q such that U^T*A*Q and V^T*B*Q are upper triangular.
     *
     * @param jobU  {@link GsvdJob} for U: None, Compute, or Initialize
     * @param jobV  {@link GsvdJob} for V: None, Compute, or Initialize
     * @param jobQ  {@link GsvdJob} for Q: None, Compute, or Initialize
     * @param m     number of rows in A
     * @param p     number of rows in B
     * @param n     number of columns in A and B
     * @param A     matrix A (m × n, row-major), overwritten
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param B     matrix B (p × n, row-major), overwritten
     * @param bOff  offset into B
     * @param ldb   leading dimension of B
     * @param tola  tolerance for rank of A
     * @param tolb  tolerance for rank of B
     * @param U     orthogonal matrix U (m × m, row-major); used if jobU != None
     * @param uOff  offset into U
     * @param ldu   leading dimension of U
     * @param V     orthogonal matrix V (p × p, row-major); used if jobV != None
     * @param vOff  offset into V
     * @param ldv   leading dimension of V
     * @param Q     orthogonal matrix Q (n × n, row-major); used if jobQ != None
     * @param qOff  offset into Q
     * @param ldq   leading dimension of Q
     * @param iwork integer workspace (length n)
     * @param tau   Householder scalars workspace
     * @param work  workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork size of work; use -1 for workspace query
     * @return int[2] {k, l} where k+l is the effective rank of [A^T; B^T]^T
     */
    static int[] dggsvp3(GsvdJob jobU, GsvdJob jobV, GsvdJob jobQ, int m, int p, int n,
                         double[] A, int aOff, int lda, double[] B, int bOff, int ldb,
                         double tola, double tolb,
                         double[] U, int uOff, int ldu, double[] V, int vOff, int ldv,
                         double[] Q, int qOff, int ldq,
                         int[] iwork, double[] tau, double[] work, int workOff, int lwork) {
        return Dtgsja.dggsvp3(jobU, jobV, jobQ, m, p, n, A, aOff, lda, B, bOff, ldb,
                              tola, tolb, U, uOff, ldu, V, vOff, ldv, Q, qOff, ldq,
                              iwork, tau, work, workOff, lwork);
    }

    // ==================== Dtgsja (GSVD core algorithm) ====================

    /**
     * Computes the GSVD of upper triangular matrices A and B (LAPACK DTGSJA).
     * Computes alpha and beta such that A = U*diag(alpha)*R and B = V*diag(beta)*R.
     * The number of iterations (cycles) is written to work[workOff].
     *
     * @param jobU    {@link GsvdJob} for U: None, Compute, or Initialize
     * @param jobV    {@link GsvdJob} for V: None, Compute, or Initialize
     * @param jobQ    {@link GsvdJob} for Q: None, Compute, or Initialize
     * @param m       number of rows in A
     * @param p       number of rows in B
     * @param n       number of columns in A and B
     * @param k       number of rows of A in the upper trapezoidal part
     * @param l       number of rows of B in the upper trapezoidal part
     * @param A       matrix A (m × n, row-major), overwritten
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param B       matrix B (p × n, row-major), overwritten
     * @param bOff    offset into B
     * @param ldb     leading dimension of B
     * @param tola    tolerance for convergence of A
     * @param tolb    tolerance for convergence of B
     * @param alpha   generalized singular values alpha (output, length n)
     * @param alphaOff offset into alpha
     * @param beta    generalized singular values beta (output, length n)
     * @param betaOff offset into beta
     * @param U       orthogonal matrix U; used if jobU != None
     * @param uOff    offset into U
     * @param ldu     leading dimension of U
     * @param V       orthogonal matrix V; used if jobV != None
     * @param vOff    offset into V
     * @param ldv     leading dimension of V
     * @param Q       orthogonal matrix Q; used if jobQ != None
     * @param qOff    offset into Q
     * @param ldq     leading dimension of Q
     * @param work    workspace; work[workOff] receives the number of Jacobi iterations (cycles)
     * @param workOff offset into work
     * @return true on convergence, false if maximum iterations exceeded
     */
    static boolean dtgsja(GsvdJob jobU, GsvdJob jobV, GsvdJob jobQ, int m, int p, int n, int k, int l,
                           double[] A, int aOff, int lda, double[] B, int bOff, int ldb,
                           double tola, double tolb,
                           double[] alpha, int alphaOff, double[] beta, int betaOff,
                           double[] U, int uOff, int ldu, double[] V, int vOff, int ldv,
                           double[] Q, int qOff, int ldq, double[] work, int workOff) {
        return Dtgsja.dtgsja(jobU, jobV, jobQ, m, p, n, k, l, A, aOff, lda, B, bOff, ldb,
                             tola, tolb, alpha, alphaOff, beta, betaOff,
                             U, uOff, ldu, V, vOff, ldv, Q, qOff, ldq, work, workOff);
    }

    // ==================== Dsytrf (Bunch-Kaufman LDL factorization, blocked) ====================

    /**
     * Bunch-Kaufman LDL^T factorization of a symmetric matrix (LAPACK DSYTRF).
     *
     * @param uplo    {@link Uplo#Upper} or {@link Uplo#Lower}
     * @param n       order of matrix A
     * @param A       symmetric matrix (n × n, row-major), overwritten with LDL^T factors
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param ipiv    pivot indices (output, length n)
     * @param ipivOff offset into ipiv
     * @param work    workspace
     * @param lwork   size of work; use -1 for workspace query (writes optimal size to work[0])
     * @return 0 on success; positive value i if D[i,i] is exactly zero (singular)
     */
    static int dsytrf(Uplo uplo, int n, double[] A, int aOff, int lda,
                      int[] ipiv, int ipivOff, double[] work, int lwork) {
        return Dsytrf.dsytrf(uplo, n, A, aOff, lda, ipiv, ipivOff, work, lwork);
    }

    // ==================== Dsytrs (Solve using LDL factorization) ====================

    /**
     * Solves a symmetric system using LDL^T factorization: A*X = B (LAPACK DSYTRS).
     *
     * @param uplo    {@link Uplo#Upper} or {@link Uplo#Lower}
     * @param n       order of A
     * @param nrhs    number of right-hand sides
     * @param A       LDL^T factorization from DSYTRF (n × n, row-major)
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param ipiv    pivot indices from DSYTRF
     * @param ipivOff offset into ipiv
     * @param B       right-hand side matrix (n × nrhs, row-major), overwritten with solution
     * @param bOff    offset into B
     * @param ldb     leading dimension of B
     */
    static void dsytrs(Uplo uplo, int n, int nrhs, double[] A, int aOff, int lda,
                       int[] ipiv, int ipivOff, double[] B, int bOff, int ldb) {
        Dsytrs.dsytrs(uplo, n, nrhs, A, aOff, lda, ipiv, ipivOff, B, bOff, ldb);
    }

    // ==================== Dsycon (Condition number for LDL factorization) ====================

    /**
     * Estimates the reciprocal condition number of a symmetric matrix (LAPACK DSYCON).
     *
     * @param uplo    {@link Uplo#Upper} or {@link Uplo#Lower}
     * @param n       order of matrix A
     * @param A       LDL^T factorization from DSYTRF (n × n, row-major)
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param ipiv    pivot indices from DSYTRF
     * @param ipivOff offset into ipiv
     * @param anorm   1-norm of the original matrix A
     * @param work    workspace (length 2*n)
     * @param iwork   integer workspace (length n); must not overlap ipiv
     * @return reciprocal condition number estimate (0 if singular)
     */
    static double dsycon(Uplo uplo, int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff,
                         double anorm, double[] work, int[] iwork) {
        return Dsycon.dsycon(uplo, n, A, aOff, lda, ipiv, ipivOff, anorm, work, iwork);
    }

    // ==================== Dtrsyl (Sylvester equation) ====================

    /**
     * Solves the real Sylvester matrix equation: op(A)*X + isgn*X*op(B) = scale*C (LAPACK DTRSYL).
     * A and B must be in Schur canonical form (upper quasi-triangular).
     *
     * @param trana  true for A^T, false for A
     * @param tranb  true for B^T, false for B
     * @param isgn   +1 or -1 (sign in the equation)
     * @param m      number of rows in A and C
     * @param n      number of columns in B and C
     * @param A      upper quasi-triangular matrix A (m × m, row-major)
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param B      upper quasi-triangular matrix B (n × n, row-major)
     * @param bOff   offset into B
     * @param ldb    leading dimension of B
     * @param C      right-hand side matrix (m × n, row-major), overwritten with solution X
     * @param cOff   offset into C
     * @param ldc    leading dimension of C
     * @param okOut  if non-null, okOut[0] is set to false if A and B have a common eigenvalue
     * @return scale factor; solution satisfies op(A)*X + isgn*X*op(B) = scale*C
     */
    static double dtrsyl(boolean trana, boolean tranb, int isgn, int m, int n,
                         double[] A, int aOff, int lda,
                         double[] B, int bOff, int ldb,
                         double[] C, int cOff, int ldc,
                         boolean[] okOut) {
        return Dtrsyl.dtrsyl(trana, tranb, isgn, m, n, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, okOut);
    }

    // ==================== Dtrsen (Schur reordering) ====================

    /**
     * Reorders the real Schur factorization A = Q*T*Q^T (LAPACK DTRSEN).
     * Moves selected eigenvalues to the leading diagonal blocks of T.
     * Output values are written to arrays before returning:
     * <ul>
     *   <li>{@code iwork[1]} = m (number of selected eigenvalues)</li>
     *   <li>{@code work[1]} = s (condition number estimate, when job requires it)</li>
     *   <li>{@code work[2]} = sep (subspace separation, when job requires it)</li>
     * </ul>
     *
     * @param job      condition number job: {@link Dtrsen#NO_COND}, {@link Dtrsen#EIGENVAL_COND},
     *                 {@link Dtrsen#SUBSPACE_COND}, or {@link Dtrsen#BOTH_COND}
     * @param wantq    true to update Q
     * @param selects  boolean array selecting which eigenvalues to move (length n)
     * @param n        order of T
     * @param t        quasi-triangular matrix T (n × n, row-major), modified in place
     * @param tOff     offset into t
     * @param ldt      leading dimension of t
     * @param q        Schur vectors Q (n × n, row-major); used if wantq=true
     * @param qOff     offset into q
     * @param ldq      leading dimension of q
     * @param wr       real parts of eigenvalues (output, length n)
     * @param wi       imaginary parts of eigenvalues (output, length n)
     * @param work     workspace; work[1]=s, work[2]=sep on output
     * @param lwork    size of work; use -1 for workspace query
     * @param iwork    integer workspace; iwork[1]=m on output
     * @param liwork   size of iwork; use -1 for workspace query
     * @return true on success, false if eigenvalue swap failed
     */
    static boolean dtrsen(char job, boolean wantq, boolean[] selects, int n,
                          double[] t, int tOff, int ldt,
                          double[] q, int qOff, int ldq,
                          double[] wr, double[] wi,
                          double[] work, int lwork,
                          int[] iwork, int liwork) {
        return Dtrsen.dtrsen(job, wantq, selects, n, t, tOff, ldt, q, qOff, ldq, wr, wi, work, lwork, iwork, liwork);
    }

    // ==================== Dhgeqz (QZ iteration) ====================

    /**
     * QZ iteration for generalized upper Hessenberg matrix pair (LAPACK DHGEQZ).
     * Computes eigenvalues (alphar, alphai, beta) and optionally Schur vectors Q, Z.
     *
     * @param job    'E' eigenvalues only, 'S' Schur form
     * @param compq  'N' no Q, 'I' initialize Q=I, 'V' accumulate into Q
     * @param compz  'N' no Z, 'I' initialize Z=I, 'V' accumulate into Z
     * @param n      order of H and T
     * @param ilo    lower index of active submatrix (0-based)
     * @param ihi    upper index of active submatrix (0-based inclusive)
     * @param H      upper Hessenberg matrix (n×n, row-major), overwritten
     * @param hOff   offset into H
     * @param ldh    leading dimension of H
     * @param T      upper triangular matrix (n×n, row-major), overwritten
     * @param tOff   offset into T
     * @param ldt    leading dimension of T
     * @param alphar real parts of alpha (output, length n)
     * @param alphai imaginary parts of alpha (output, length n)
     * @param beta   beta values (output, length n)
     * @param Q      orthogonal Q (n×n, row-major); used if compq != 'N'
     * @param qOff   offset into Q
     * @param ldq    leading dimension of Q
     * @param Z      orthogonal Z (n×n, row-major); used if compz != 'N'
     * @param zOff   offset into Z
     * @param ldz    leading dimension of Z
     * @param work   workspace
     * @param workOff offset into work
     * @param lwork  workspace size; -1 for query
     * @return 0 on success; positive = number of eigenvalues not computed
     */
    static int dhgeqz(char job, char compq, char compz, int n, int ilo, int ihi,
                      double[] H, int hOff, int ldh,
                      double[] T, int tOff, int ldt,
                      double[] alphar, double[] alphai, double[] beta,
                      double[] Q, int qOff, int ldq,
                      double[] Z, int zOff, int ldz,
                      double[] work, int workOff, int lwork) {
        return Dhgeqz.dhgeqz(job, compq, compz, n, ilo, ihi,
                              H, hOff, ldh, T, tOff, ldt,
                              alphar, alphai, beta,
                              Q, qOff, ldq, Z, zOff, ldz,
                              work, workOff, lwork);
    }

    // ==================== Dtgevc (Generalized eigenvectors) ====================

    /**
     * Computes right and/or left generalized eigenvectors of (S, P) (LAPACK DTGEVC).
     *
     * @param side   'R' right only, 'L' left only, 'B' both
     * @param howmny 'A' all, 'B' backtransform, 'S' selected
     * @param select boolean array (length n); used if howmny='S'
     * @param n      order of S and P
     * @param S      quasi-upper-triangular matrix (n×n, row-major)
     * @param sOff   offset into S
     * @param lds    leading dimension of S
     * @param P      upper triangular matrix (n×n, row-major)
     * @param pOff   offset into P
     * @param ldp    leading dimension of P
     * @param VL     left eigenvectors (n×mm, row-major); used if side='L' or 'B'
     * @param vlOff  offset into VL
     * @param ldvl   leading dimension of VL
     * @param VR     right eigenvectors (n×mm, row-major); used if side='R' or 'B'
     * @param vrOff  offset into VR
     * @param ldvr   leading dimension of VR
     * @param mm     number of columns in VL/VR
     * @param work   workspace (length 6*n)
     * @param workOff offset into work
     * @return number of eigenvectors computed
     */
    static int dtgevc(char side, char howmny, boolean[] select, int n,
                      double[] S, int sOff, int lds,
                      double[] P, int pOff, int ldp,
                      double[] VL, int vlOff, int ldvl,
                      double[] VR, int vrOff, int ldvr,
                      int mm, double[] work, int workOff) {
        return Dtgevc.dtgevc(side, howmny, select, n, S, sOff, lds, P, pOff, ldp,
                             VL, vlOff, ldvl, VR, vrOff, ldvr, mm, work, workOff);
    }

    // ==================== Dggev (Non-symmetric generalized eigenvalue) ====================

    /**
     * Computes eigenvalues and optionally eigenvectors of A*x = lambda*B*x (LAPACK DGGEV).
     *
     * @param wantVL  true to compute left eigenvectors
     * @param wantVR  true to compute right eigenvectors
     * @param n       order of A and B
     * @param A       matrix A (n×n, row-major), overwritten
     * @param lda     leading dimension of A
     * @param B       matrix B (n×n, row-major), overwritten
     * @param ldb     leading dimension of B
     * @param alphar  real parts of alpha (output, length n)
     * @param alphai  imaginary parts of alpha (output, length n)
     * @param beta    beta values (output, length n)
     * @param VL      left eigenvectors (n×n, row-major); used if wantVL=true
     * @param ldvl    leading dimension of VL
     * @param VR      right eigenvectors (n×n, row-major); used if wantVR=true
     * @param ldvr    leading dimension of VR
     * @param work    workspace
     * @param workOff offset into work
     * @param lwork   workspace size; -1 for query
     * @return 0 on success; positive = number of eigenvalues not computed
     */
    static int dggev(boolean wantVL, boolean wantVR, int n,
                     double[] A, int lda, double[] B, int ldb,
                     double[] alphar, double[] alphai, double[] beta,
                     double[] VL, int ldvl, double[] VR, int ldvr,
                     double[] work, int workOff, int lwork) {
        return Dggev.dggev(wantVL, wantVR, n, A, lda, B, ldb,
                           alphar, alphai, beta, VL, ldvl, VR, ldvr,
                           work, workOff, lwork);
    }

    // ==================== Symmetric Eigenvalue ====================

    /**
     * Computes all eigenvalues and optionally eigenvectors of a real symmetric matrix.
     * LAPACK DSYEV.
     *
     * @param jobz    'N' eigenvalues only, 'V' eigenvalues and eigenvectors
     * @param uplo    'U' upper triangle, 'L' lower triangle
     * @param n       order of A
     * @param A       symmetric matrix (n×n, row-major), overwritten on exit
     * @param lda     leading dimension of A
     * @param w       eigenvalues output (length n)
     * @param wOff    offset into w
     * @param work    workspace (length lwork)
     * @param workOff offset into work
     * @param lwork   workspace size (>= 3*n-1, or -1 for query)
     * @return 0 on success
     */
    static int dsyev(char jobz, char uplo, int n, double[] A, int lda,
                     double[] w, int wOff, double[] work, int workOff, int lwork) {
        return Dsyev.dsyev(jobz, uplo, n, A, lda, w, wOff, work, workOff, lwork);
    }

    // ==================== Matrix Norms ====================

    /**
     * Returns the value of the specified norm of a real symmetric matrix.
     * LAPACK DLANSY.
     *
     * @param norm  'M' max abs, '1'/'O' 1-norm, 'I' inf-norm, 'F'/'E' Frobenius
     * @param uplo  upper or lower triangle
     * @param n     order of A
     * @param A     symmetric matrix (n×n, row-major)
     * @param lda   leading dimension of A
     * @param work  workspace (required for 1-norm or inf-norm)
     * @return computed norm
     */
    static double dlansy(Norm norm, Uplo uplo, int n, double[] A, int lda, double[] work) {
        return Dlansy.dlansy(norm.code, uplo, n, A, lda, work);
    }

    /**
     * Returns the value of the specified norm of a real triangular matrix.
     * LAPACK DLANTR.
     *
     * @param norm  'M' max abs, '1'/'O' 1-norm, 'I' inf-norm, 'F'/'E' Frobenius
     * @param uplo  upper or lower triangle
     * @param diag  unit or non-unit diagonal
     * @param m     number of rows
     * @param n     number of columns
     * @param A     triangular matrix (row-major)
     * @param lda   leading dimension of A
     * @param work  workspace (required for 1-norm or inf-norm)
     * @return computed norm
     */
    static double dlantr(Norm norm, Uplo uplo, Diag diag, int m, int n,
                         double[] A, int lda, double[] work) {
        return Dlantr.dlantr(norm.code, uplo, diag, m, n, A, lda, work);
    }

    /**
     * Returns the value of the specified norm of a real symmetric tridiagonal matrix.
     * LAPACK DLANST.
     *
     * @param norm  'M' max abs, '1'/'O' 1-norm, 'F'/'E' Frobenius
     * @param n     order of matrix
     * @param d     diagonal elements
     * @param dOff  offset into d
     * @param e     off-diagonal elements
     * @param eOff  offset into e
     * @return computed norm
     */
    static double dlanst(Norm norm, int n, double[] d, int dOff, double[] e, int eOff) {
        return Dlanst.dlanst(norm.code, n, d, dOff, e, eOff);
    }

    // ==================== Tridiagonal Eigenvalue ====================

    /**
     * Computes all eigenvalues of a real symmetric tridiagonal matrix (no eigenvectors).
     * LAPACK DSTERF.
     *
     * @param n    order of matrix
     * @param d    diagonal elements (overwritten with eigenvalues on exit)
     * @param dOff offset into d
     * @param e    off-diagonal elements (overwritten on exit)
     * @param eOff offset into e
     * @return true on success
     */
    static boolean dsterf(int n, double[] d, int dOff, double[] e, int eOff) {
        return Dsterf.dsterf(n, d, dOff, e, eOff);
    }

    // ==================== Hessenberg Reduction ====================

    /**
     * Reduces a general matrix to upper Hessenberg form (unblocked).
     * LAPACK DGEHD2.
     *
     * @param n       order of A
     * @param ilo     lower bound of active submatrix (0-based)
     * @param ihi     upper bound of active submatrix (0-based, inclusive)
     * @param A       n×n matrix (row-major), overwritten on exit
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     scalar factors of reflectors (length n-1)
     * @param tauOff  offset into tau
     * @param work    workspace (length n)
     * @param workOff offset into work
     */
    static void dgehd2(int n, int ilo, int ihi, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        Dgehd2.dgehd2(n, ilo, ihi, A, aOff, lda, tau, tauOff, work, workOff);
    }

    // ==================== Orthogonal Matrix Generation ====================

    /**
     * Generates the m×n matrix Q from a QL factorization (last n columns of a product of k reflectors).
     * LAPACK DORGQL.
     *
     * @param m       number of rows
     * @param n       number of columns
     * @param k       number of reflectors
     * @param A       matrix containing reflectors, overwritten with Q on exit
     * @param lda     leading dimension of A
     * @param aOff    offset into A
     * @param tau     scalar factors of reflectors
     * @param tauOff  offset into tau
     * @param work    workspace
     * @param workOff offset into work
     * @param lwork   workspace size (-1 for query)
     */
    static void dorgql(int m, int n, int k, double[] A, int lda, int aOff,
                       double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        Dorgql.dorgql(m, n, k, A, lda, aOff, tau, tauOff, work, workOff, lwork);
    }

    // ==================== Orthogonal Transformation ====================

    /**
     * Applies orthogonal matrix Q or P from a bidiagonal reduction (DGEBRD) to a matrix C.
     * LAPACK DORMBR.
     *
     * @param vect    'Q' to apply Q, 'P' to apply P
     * @param side    apply from left or right
     * @param trans   apply Q/P or Q^T/P^T
     * @param m       rows of C
     * @param n       columns of C
     * @param k       number of reflectors used in DGEBRD
     * @param A       matrix from DGEBRD
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param tau     scalar factors from DGEBRD
     * @param tauOff  offset into tau
     * @param C       matrix to transform (overwritten on exit)
     * @param cOff    offset into C
     * @param ldc     leading dimension of C
     * @param work    workspace
     * @param workOff offset into work
     * @param lwork   workspace size (-1 for query)
     * @return 0 on success
     */
    static int dormbr(char vect, Side side, Trans trans, int m, int n, int k,
                      double[] A, int aOff, int lda,
                      double[] tau, int tauOff,
                      double[] C, int cOff, int ldc,
                      double[] work, int workOff, int lwork) {
        return Dormbr.dormbr(vect, side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff, lwork);
    }

    // ==================== Triangular Solve with Scaling ====================

    /**
     * Solves a triangular system with scaling to prevent overflow.
     * LAPACK DLATRS.
     *
     * @param uplo    upper or lower triangular
     * @param trans   solve T*x=b or T^T*x=b
     * @param diag    unit or non-unit diagonal
     * @param normin  true if cnorm already contains column norms
     * @param n       order of T
     * @param A       triangular matrix (row-major)
     * @param lda     leading dimension of A
     * @param x       right-hand side on entry, solution on exit
     * @param xOff    offset into x
     * @param cnorm   column norms (input if normin=true, output otherwise)
     * @param cnormOff offset into cnorm
     * @return scale factor applied to x
     */
    static double dlatrs(Uplo uplo, Trans trans, Diag diag, boolean normin,
                         int n, double[] A, int lda, double[] x, int xOff,
                         double[] cnorm, int cnormOff) {
        return Dlatrs.dlatrs(uplo, trans, diag, normin, n, A, lda, x, xOff, cnorm, cnormOff);
    }

    // ==================== Symmetric Indefinite Factorization (blocked) ====================

    /**
     * Computes a partial blocked factorization of a real symmetric matrix (Bunch-Kaufman).
     * LAPACK DLASYF.
     *
     * @param uplo    upper or lower triangle
     * @param n       order of A
     * @param nb      block size
     * @param A       symmetric matrix (row-major), partially factored on exit
     * @param aOff    offset into A
     * @param lda     leading dimension of A
     * @param ipiv    pivot indices (output)
     * @param ipivOff offset into ipiv
     * @param W       workspace matrix
     * @param wOff    offset into W
     * @param ldw     leading dimension of W
     * @return number of columns factored (kb)
     */
    static int dlasyf(Uplo uplo, int n, int nb, double[] A, int aOff, int lda,
                      int[] ipiv, int ipivOff, double[] W, int wOff, int ldw) {
        return Dlasyf.dlasyf(uplo, n, nb, A, aOff, lda, ipiv, ipivOff, W, wOff, ldw);
    }

    // ==================== Dlasrt ====================

    /**
     * Sorts an array of doubles in increasing or decreasing order (LAPACK DLASRT).
     * Typically used to sort eigenvalues.
     *
     * @param id   'I' for increasing order, 'D' for decreasing order
     * @param n    number of elements to sort
     * @param d    array to sort (modified in place)
     * @param dOff offset into d
     */
    static void dlasrt(char id, int n, double[] d, int dOff) {
        Dlasrt.dlasrt(id, n, d, dOff);
    }

}

