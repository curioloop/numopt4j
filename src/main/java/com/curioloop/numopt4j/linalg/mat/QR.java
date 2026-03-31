/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;

import static java.lang.Math.abs;

public final class QR implements Decomposition {

    private static final double EPSILON = BLAS.dlamch('E');
    private static final double CONDITION_TOLERANCE = 1e16;

    /** Decomposition options for QR. */
    public enum Opts {
        /** Enable column pivoting (rank-revealing QR). */
        PIVOTING
    }

    public static final class Pool extends Decomposition.Workspace {
        /** Householder reflection factors */
        public double[] tau;

        /**
         * Ensure all buffers are allocated for QR decomposition of an m×n matrix.
         */
        public Pool ensure(int m, int n, boolean pivoting) {
            if (m <= 0 || n <= 0) {
                throw new IllegalArgumentException("Dimensions must be positive");
            }
            if (m < n) {
                throw new IllegalArgumentException("For QR decomposition, m must be >= n");
            }

            int k = Math.min(m, n);
            if (tau == null || tau.length < k) {
                tau = new double[k];
            }

            double[] tmp = new double[1];
            BLAS.dorgqr(m, m, k, null, 0, m, null, 0, tmp, 0, -1);
            int qWork = (int) tmp[0];

            if (pivoting) {
                BLAS.dgeqp3(m, n, null, 0, n, null, null, tmp, 0, -1);
                ensureWork(Math.max(Math.max((int) tmp[0], qWork), 3 * n));
            } else {
                BLAS.dgeqrf(m, n, null, 0, n, null, 0, tmp, 0, -1);
                ensureWork(Math.max(Math.max((int) tmp[0], qWork), 3 * n));
            }
            ensureIwork(n);
            return this;
        }
    }

    private Pool pool;
    private double[] QR;
    private int m;
    private int n;
    private int rank;
    private boolean pivoting;
    private boolean ok;
    private double condition;

    private QR() {}

    /**
     * Creates an empty workspace pool for QR decomposition.
     */
    public static Pool workspace() {
        return new Pool();
    }

    /**
     * Factorizes a square or tall matrix using the standard non-pivoted QR path.
     */
    public static QR decompose(double[] A, int m, int n) {
        return decompose(A, m, n, false, null);
    }

    /**
     * Factorizes a square or tall matrix using the standard non-pivoted QR path with workspace reuse.
     */
    public static QR decompose(double[] A, int m, int n, Pool ws) {
        return decompose(A, m, n, false, ws);
    }

    /**
     * Factorizes a square or tall matrix with optional column pivoting.
     */
    public static QR decompose(double[] A, int m, int n, boolean pivoting, Pool ws) {
        QR qr = new QR();
        qr.doDecompose(A, m, n, pivoting, ws);
        return qr;
    }

    private void doDecompose(double[] A, int m, int n, boolean pivoting, Pool ws) {
        if (m <= 0 || n <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive");
        }
        if (m < n) {
            throw new IllegalArgumentException("For QR decomposition, m must be >= n");
        }
        if (A == null || A.length < m * n) {
            throw new IllegalArgumentException("Matrix A must have length >= m*n");
        }

        this.QR = A;
        this.m = m;
        this.n = n;
        this.pivoting = pivoting;
        this.ok = false;
        this.condition = Double.NaN;
        this.pool = ws == null ? new Pool() : ws;
        pool.ensure(m, n, pivoting);

        if (pivoting) {
            int[] jpvt = pool.iwork();
            java.util.Arrays.fill(jpvt, 0, n, 0);
            double[] work = pool.work();
            int result = BLAS.dgeqp3(m, n, QR, 0, n, jpvt, pool.tau, work, 0, work.length);
            if (result != 0) {
                throw new IllegalStateException("QR factorization failed with dgeqp3 status=" + result);
            }
            this.rank = computeRank();
        } else {
            int result = BLAS.dgeqrf(m, n, QR, 0, n, pool.tau, 0, pool.work(), 0, pool.work().length);
            if (result != 0) {
                throw new IllegalStateException("QR factorization failed with dgeqrf status=" + result);
            }
            this.rank = n;
        }

        this.ok = true;
    }

    private int computeRank() {
        int k = Math.min(m, n);
        if (k == 0) return 0;
        double tol = EPSILON * Math.max(m, n) * abs(QR[0]);
        int r = 0;
        for (int i = 0; i < k; i++) {
            if (abs(QR[i * n + i]) > tol) r++;
            else break;
        }
        return r;
    }

    @Override
    public Pool pool() {
        return pool;
    }

    public int rank() {
        return pivoting ? rank : Math.min(m, n);
    }

    /**
     * Computes the numerical rank from the diagonal of {@code R} using a caller-provided tolerance.
     */
    public int rank(double tol) {
        int k = Math.min(m, n);
        if (k == 0) return 0;
        int r = 0;
        for (int i = 0; i < k; i++) {
            if (abs(QR[i * n + i]) > tol) r++;
        }
        return r;
    }

    /**
     * Returns the column permutation indices from pivoted QR, or {@code null} for non-pivoted QR.
     */
    public int[] piv() {
        return pivoting ? pool.iwork() : null;
    }

    /**
     * Returns whether this decomposition used column pivoting.
     */
    public boolean isPivoting() {
        return pivoting;
    }

    /**
     * Solves the square linear system {@code A*x = b} using the QR factorization.
     *
     * <p>This method is only defined for square factorizations ({@code m == n}).
     * For tall matrices, use {@link #leastSquares(double[], double[])} instead.</p>
     *
    * <p>To minimize allocations, this method uses {@code b} as the working buffer and may
    * overwrite it. Pass a distinct {@code x} buffer if the original right-hand side must be
    * preserved. When {@code x} is {@code null}, the solved vector is returned in {@code b}.</p>
     *
     * @param b right-hand side vector of length at least {@code m}
     * @param x destination vector of length at least {@code n}; may be {@code null}
     * @return the solution vector {@code x}
     * @throws IllegalStateException if the decomposition is unavailable or not square
     * @throws ArithmeticException if the factor is singular, rank deficient, or ill-conditioned
     */
    public double[] solve(double[] b, double[] x) {
        requireDecomposition();
        if (b == null || b.length < m) {
            throw new IllegalArgumentException("Vector b must have length >= " + m);
        }
        if (m != n) {
            throw new IllegalStateException("Exact solve requires a square QR factorization; use leastSquares for tall matrices");
        }
        double[] rhs = b;
        applyQt(rhs, n, pool.work(), 0);
        if (pivoting) {
            requireFullRank();
            if (!backSubstituteRank(rhs, n)) {
                throw new ArithmeticException("QR factor is singular");
            }
            unpermute(rhs);
        } else {
            requireWellConditioned();
            if (!BLAS.dtrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                    n, 1, QR, 0, this.n, rhs, 0, 1)) {
                throw new ArithmeticException("QR factor is singular");
            }
        }
        if (x == null) {
            return rhs;
        }
        if (x.length < n) {
            x = new double[n];
        }
        if (x != rhs) {
            BLAS.dcopy(n, rhs, 0, 1, x, 0, 1);
        }
        return x;
    }

    /**
     * Solves the least-squares problem {@code min ||A*x - b||_2} for a tall or square matrix.
     *
     * <p>The input right-hand side {@code b} is not modified. When pivoting is enabled, the
     * returned solution follows the rank-revealing QR path {@code A*P = Q*R} and is unpermuted
     * before being returned.</p>
     *
     * @param b right-hand side vector of length at least {@code m}
     * @param x destination vector of length at least {@code n}; may be {@code null}
     * @return the least-squares solution vector {@code x}
     * @throws IllegalStateException if the decomposition is unavailable
     * @throws ArithmeticException if the effective triangular factor is singular or ill-conditioned
     */
    public double[] leastSquares(double[] b, double[] x) {
        requireDecomposition();
        if (b == null || b.length < m) {
            throw new IllegalArgumentException("Vector b must have length >= " + m);
        }
        if (x == null || x.length < n) {
            x = new double[n];
        }
        double[] rhs = new double[m];
        System.arraycopy(b, 0, rhs, 0, m);
        int k = Math.min(m, n);
        applyQt(rhs, k, pool.work(), 0);
        if (pivoting) {
            BLAS.dset(n, 0.0, x, 0, 1);
            BLAS.dcopy(rank, rhs, 0, 1, x, 0, 1);
            if (!backSubstituteRank(x, rank)) {
                throw new ArithmeticException("QR factor is singular");
            }
            unpermute(x);
        } else {
            requireWellConditioned();
            BLAS.dcopy(n, rhs, 0, 1, x, 0, 1);
            if (!BLAS.dtrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                    n, 1, QR, 0, this.n, x, 0, 1)) {
                throw new ArithmeticException("QR factor is singular");
            }
        }
        return x;
    }

    /**
     * Computes the inverse of a square, non-pivoted matrix from its QR factorization.
     *
     * @param Ainv destination array of length at least {@code n*n}; may be {@code null}
     * @return the inverse matrix in row-major order
     * @throws IllegalStateException if the decomposition is unavailable or not square
     * @throws UnsupportedOperationException if pivoting is enabled
     * @throws ArithmeticException if the factor is singular or ill-conditioned
     */
    public double[] inverse(double[] Ainv) {
        requireDecomposition();
        if (pivoting) {
            throw new UnsupportedOperationException("Inverse not supported for pivoting QR");
        }
        if (m != n) {
            throw new IllegalStateException("Inverse requires a square QR factorization");
        }
        requireWellConditioned();
        if (Ainv == null || Ainv.length < n * n) {
            Ainv = new double[n * n];
        }
        double[] rhs = new double[n];
        double[] work = pool.work();
        for (int col = 0; col < n; col++) {
            BLAS.dset(n, 0.0, rhs, 0, 1);
            rhs[col] = 1.0;
            applyQt(rhs, n, work, 0);
            if (!BLAS.dtrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                    n, 1, QR, 0, this.n, rhs, 0, 1)) {
                throw new ArithmeticException("QR factor is singular");
            }
            for (int i = 0; i < n; i++) {
                Ainv[i * n + col] = rhs[i];
            }
        }
        return Ainv;
    }

    /**
     * Returns the full orthogonal factor {@code Q} as an {@code m x m} matrix.
     *
     * <p>This is the full orthogonal matrix, not the reduced {@code m x n} form.</p>
     */
    public Matrix toQ() {
        requireDecomposition();
        int k = Math.min(m, n);
        double[] dst = new double[m * m];
        for (int i = 0; i < m; i++) {
            if (n > 0) {
                System.arraycopy(QR, i * n, dst, i * m, n);
            }
        }
        for (int j = n; j < m; j++) {
            dst[j * m + j] = 1.0;
        }
        double[] tmp = new double[1];
        BLAS.dorgqr(m, m, k, null, 0, m, null, 0, tmp, 0, -1);
        pool.ensureWork(Math.max(pool.work().length, (int) tmp[0]));
        int result = BLAS.dorgqr(m, m, k, dst, 0, m, pool.tau, 0, pool.work(), 0, pool.work().length);
        if (result != 0) {
            throw new IllegalStateException("Failed to form Q with dorgqr status=" + result);
        }
        return new Matrix(m, m, false, dst);
    }

    /**
     * Returns the full upper-trapezoidal factor {@code R} as an {@code m x n} matrix.
     *
     * <p>Entries below the leading upper-triangular block are zero-filled.</p>
     */
    public Matrix toR() {
        requireDecomposition();
        int k = Math.min(m, n);
        double[] dst = new double[m * n];
        for (int i = 0; i < k; i++) {
            BLAS.dcopy(n - i, QR, i * n + i, 1, dst, i * n + i, 1);
        }
        return new Matrix(m, n, false, dst);
    }

    /**
     * Returns the column permutation matrix P (n×n) such that A*P = Q*R.
     *
     * @throws IllegalStateException if pivoting was not enabled for this decomposition
     */
    public Matrix toP() {
        requireDecomposition();
        if (!pivoting) {
            throw new IllegalStateException("Column permutation matrix is only available for pivoting QR");
        }
        double[] dst = new double[n * n];
        int[] jpvt = pool.iwork();
        java.util.Arrays.fill(dst, 0.0);
        for (int j = 0; j < n; j++) {
            dst[jpvt[j] * n + j] = 1.0;
        }
        return new Matrix(n, n, false, dst);
    }

    /**
     * Solves a square system with multiple right-hand sides in place.
     *
     * <p>This method is only available for non-pivoted square QR factorizations. The matrix
     * {@code B} is interpreted as {@code m x nrhs} in row-major layout and is overwritten with
     * the solution matrix.</p>
     *
     * @param B right-hand side matrix in row-major order, length at least {@code m*nrhs}
     * @param nrhs number of right-hand sides
     * @return the overwritten solution matrix {@code B}
     * @throws IllegalStateException if the decomposition is unavailable or not square
     * @throws UnsupportedOperationException if pivoting is enabled
     * @throws ArithmeticException if the factor is singular or ill-conditioned
     */
    public double[] solveMultiple(double[] B, int nrhs) {
        requireDecomposition();
        if (B == null || B.length < m * nrhs) {
            throw new IllegalArgumentException("Matrix B must have length >= " + (m * nrhs));
        }
        if (pivoting) {
            throw new UnsupportedOperationException("Multi-RHS solve not yet supported for pivoting QR");
        }
        if (m != n) {
            throw new IllegalStateException("Multi-RHS solve requires a square QR factorization");
        }
        requireWellConditioned();
        for (int j = 0; j < nrhs; j++) {
            applyQtCol(B, n, pool.work(), 0, j, nrhs);
        }
        if (!BLAS.dtrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, QR, 0, this.n, B, 0, nrhs)) {
            throw new ArithmeticException("QR factor is singular");
        }
        return B;
    }

    /**
     * Returns the condition number estimate associated with the triangular QR factor.
     *
      * <p>The value is computed lazily on first request from {@code R}. For pivoted QR, the
      * estimate is based on the leading rank-sized triangular block.</p>
     */
    public double cond() {
        requireDecomposition();
          return ensureCondition();
    }

    @Override
    public boolean ok() {
        return ok;
    }

    /** Returns the number of rows in the factorized matrix. */
    public int m() { return m; }

    /** Returns the number of columns in the factorized matrix. */
    public int n() { return n; }

    /** Returns the stored Householder scalar factors. */
    public double[] tau() { return pool.tau; }

    private double computeCondition() {
        if (n == 0) {
            return Double.NaN;
        }
        double[] work = pool.work();
        double rcond;
        if (pivoting) {
            if (rank == 0) {
                return Double.POSITIVE_INFINITY;
            }
            int[] iwork = new int[rank];
            rcond = BLAS.dtrcon(BLAS.Norm.One, BLAS.Uplo.Upper, BLAS.Diag.NonUnit, rank, QR, n, work, iwork);
        } else {
            int[] iwork = pool.iwork();
            rcond = BLAS.dtrcon(BLAS.Norm.One, BLAS.Uplo.Upper, BLAS.Diag.NonUnit, n, QR, n, work, iwork);
        }
        if (rcond == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return 1.0 / rcond;
    }

    private void requireDecomposition() {
        if (!ok) {
            throw new IllegalStateException("QR decomposition not completed or failed");
        }
    }

    private void requireFullRank() {
        if (rank < n) {
            throw new ArithmeticException("QR factor is rank deficient");
        }
    }

    private void requireWellConditioned() {
        double condition = ensureCondition();
        if (!Double.isFinite(condition) || condition > CONDITION_TOLERANCE) {
            throw new ArithmeticException("QR factor is singular or ill-conditioned");
        }
    }

        private double ensureCondition() {
            if (Double.isNaN(condition)) {
                condition = computeCondition();
            }
            return condition;
        }

    private void applyQt(double[] x, int k, double[] work, int workOff) {
        double[] tau = pool.tau;
        for (int i = 0; i < k; i++) {
            if (tau[i] == 0.0) continue;
            double aii = QR[i * n + i];
            QR[i * n + i] = 1.0;
            BLAS.dlarf(BLAS.Side.Left, m - i, 1, QR, i * n + i, n, tau[i], x, i, 1, work, workOff);
            QR[i * n + i] = aii;
        }
    }

    private void applyQtCol(double[] B, int k, double[] work, int workOff, int col, int ldb) {
        double[] tau = pool.tau;
        for (int i = 0; i < k; i++) {
            if (tau[i] == 0.0) continue;
            double aii = QR[i * n + i];
            QR[i * n + i] = 1.0;
            BLAS.dlarf(BLAS.Side.Left, m - i, 1, QR, i * n + i, n, tau[i], B, i * ldb + col, ldb, work, workOff);
            QR[i * n + i] = aii;
        }
    }

    private boolean backSubstituteRank(double[] b, int rank) {
        for (int i = 0; i < rank; i++) {
            if (abs(QR[i * n + i]) < EPSILON) return false;
        }
        for (int i = rank - 1; i >= 0; i--) {
            b[i] /= QR[i * n + i];
            for (int j = i - 1; j >= 0; j--) {
                b[j] -= QR[j * n + i] * b[i];
            }
        }
        return true;
    }

    private void unpermute(double[] x) {
        int[] jpvt = pool.iwork();
        double[] temp = pool.work();
        for (int i = 0; i < n; i++) {
            temp[jpvt[i]] = x[i];
        }
        BLAS.dcopy(n, temp, 0, 1, x, 0, 1);
    }
}
