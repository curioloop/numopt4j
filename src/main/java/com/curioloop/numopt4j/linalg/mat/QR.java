/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;

import static java.lang.Math.abs;

public final class QR implements Decomposition {

    private static final double EPSILON = BLAS.dlamch('E');

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
            int k = Math.min(m, n);
            if (tau == null || tau.length < k) {
                tau = new double[k];
            }
            if (pivoting) {
                // Query dgeqp3 for optimal work size
                double[] tmp = new double[1];
                BLAS.dgeqp3(m, n, null, 0, n, null, null, tmp, 0, -1);
                ensureWork((int) tmp[0]);
                ensureIwork(n);
            } else {
                // Query dgeqrf for optimal work size
                double[] tmp = new double[1];
                BLAS.dgeqrf(m, n, null, 0, n, null, 0, tmp, 0, -1);
                ensureWork(Math.max(n, (int) tmp[0]));
                ensureIwork(n);
            }
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

    private QR() {}

    public static Pool workspace(int m, int n) {
        return new Pool().ensure(m, n, false);
    }

    public static Pool workspace(int m, int n, boolean pivoting) {
        return new Pool().ensure(m, n, pivoting);
    }

    public static QR decompose(double[] A, int m, int n) {
        return decompose(A, m, n, (Pool) null);
    }

    public static QR decompose(double[] A, int m, int n, Pool ws) {
        QR qr = new QR();
        qr.doDecompose(A, m, n, false, ws);
        return qr;
    }

    public static QR decompose(double[] A, int m, int n, boolean pivoting, Pool ws) {
        QR qr = new QR();
        qr.doDecompose(A, m, n, pivoting, ws);
        return qr;
    }

    private void doDecompose(double[] A, int m, int n, boolean pivoting, Pool ws) {
        if (A == null || A.length < m * n) {
            throw new IllegalArgumentException("Matrix A must have length >= m*n");
        }
        if (m <= 0 || n <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive");
        }

        this.QR = A;
        this.m = m;
        this.n = n;
        this.pivoting = pivoting;
        this.ok = false;

        if (ws == null) {
            ws = new Pool();
        }
        this.pool = ws;
        pool.ensure(m, n, pivoting);

        if (pivoting) {
            int[] jpvt = pool.iwork();
            for (int j = 0; j < n; j++) {
                jpvt[j] = 0;
            }
            double[] work = pool.work();
            int result = BLAS.dgeqp3(m, n, A, 0, n, jpvt, pool.tau, work, 0, work.length);
            if (result != 0) {
                return;
            }
            this.rank = computeRank();
        } else {
            BLAS.dgeqr2(m, n, A, 0, n, pool.tau, 0, pool.work(), 0);
            this.rank = Math.min(m, n);
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

    public int rank(double tol) {
        int k = Math.min(m, n);
        if (k == 0) return 0;
        int r = 0;
        for (int i = 0; i < k; i++) {
            if (abs(QR[i * n + i]) > tol) r++;
        }
        return r;
    }

    public int[] piv() {
        return pivoting ? pool.iwork() : null;
    }

    public boolean isPivoting() {
        return pivoting;
    }

    public double[] solve(double[] b, double[] x) {
        if (!ok) return null;
        if (b == null || b.length < m) {
            throw new IllegalArgumentException("Vector b must have length >= m");
        }
        if (x == null || x.length < n) {
            x = new double[n];
        }
        int k = Math.min(m, n);
        double[] work = pool.work();
        if (x != b) {
            System.arraycopy(b, 0, x, 0, Math.min(b.length, x.length));
        }
        applyQt(x, k, work, 0);
        if (pivoting) {
            if (!backSubstituteRank(x, rank)) return null;
            unpermute(x);
        } else {
            if (!backSubstitute(x, n)) return null;
        }
        return x;
    }

    public double[] leastSquares(double[] b, double[] x) {
        if (!ok) return null;
        if (b == null || b.length < m) {
            throw new IllegalArgumentException("Vector b must have length >= m");
        }
        if (x == null || x.length < n) {
            x = new double[n];
        }
        int k = Math.min(m, n);
        double[] work = pool.work();
        applyQt(b, k, work, 0);
        if (pivoting) {
            BLAS.dset(n, 0.0, x, 0, 1);
            BLAS.dcopy(rank, b, 0, 1, x, 0, 1);
            if (!backSubstituteRank(x, rank)) return null;
            unpermute(x);
        } else {
            BLAS.dcopy(k, b, 0, 1, x, 0, 1);
            if (!backSubstitute(x, k)) return null;
        }
        return x;
    }

    public double[] inverse(double[] Ainv) {
        if (!ok) return null;
        if (pivoting) {
            throw new UnsupportedOperationException("Inverse not supported for pivoting QR");
        }
        if (Ainv == null || Ainv.length < n * n) {
            Ainv = new double[n * n];
        }
        double[] work = pool.work();
        for (int col = 0; col < n; col++) {
            BLAS.dset(n, 0.0, work, 0, 1);
            work[col] = 1.0;
            applyQt(work, n, work, n);
            if (!backSubstitute(work, n)) return null;
            for (int i = 0; i < n; i++) {
                Ainv[i * n + col] = work[i];
            }
        }
        return Ainv;
    }

    /** Returns the orthogonal matrix Q (m×n). Returns null if decomposition failed. */
    public Matrix toQ() {
        if (!ok) return null;
        int k = Math.min(m, n);
        double[] dst = new double[m * n];
        BLAS.dlacpy('A', m, n, QR, 0, n, dst, 0, n);
        BLAS.dorg2r(m, n, k, dst, 0, n, pool.tau, 0, pool.work(), 0);
        return new Matrix(m, n, false, dst);
    }

    /** Returns the upper triangular matrix R (min(m,n)×n). Returns null if decomposition failed. */
    public Matrix toR() {
        if (!ok) return null;
        int k = Math.min(m, n);
        double[] dst = new double[k * n];
        for (int i = 0; i < k; i++) {
            BLAS.dcopy(n - i, QR, i * n + i, 1, dst, i * n + i, 1);
        }
        return new Matrix(k, n, false, dst);
    }

    /**
     * Returns the column permutation matrix P (n×n) such that A*P = Q*R.
     * Returns null if not in pivoting mode or if decomposition failed.
     */
    public Matrix toP() {
        if (!ok || !pivoting) return null;
        double[] dst = new double[n * n];
        int[] jpvt = pool.iwork();
        java.util.Arrays.fill(dst, 0.0);
        for (int j = 0; j < n; j++) {
            dst[jpvt[j] * n + j] = 1.0;
        }
        return new Matrix(n, n, false, dst);
    }

    public double[] solveMultiple(double[] B, int nrhs) {
        if (!ok) return null;
        if (B == null || B.length < m * nrhs) {
            throw new IllegalArgumentException("Matrix B must have length >= m*nrhs");
        }
        if (pivoting) {
            throw new UnsupportedOperationException("Multi-RHS solve not yet supported for pivoting QR");
        }
        int k = Math.min(m, n);
        for (int j = 0; j < nrhs; j++) {
            applyQtCol(B, k, pool.work(), 0, j, nrhs);
        }
        if (!backSubstituteMultiple(B, n, nrhs)) return null;
        return B;
    }

    public double cond() {
        if (!ok || n == 0) return Double.NaN;
        double[] work = pool.work();
        double rcond;
        if (pivoting) {
            if (rank == 0) return Double.NaN;
            rcond = BLAS.dtrcon('1', BLAS.Uplo.Upper, BLAS.Diag.NonUnit, rank, QR, n, work, pool.iwork());
        } else {
            rcond = BLAS.dtrcon('1', BLAS.Uplo.Upper, BLAS.Diag.NonUnit, n, QR, n, work, pool.iwork());
        }
        if (rcond == 0) return Double.POSITIVE_INFINITY;
        return 1.0 / rcond;
    }

    @Override
    public boolean ok() {
        return ok;
    }

    public int m() { return m; }
    public int n() { return n; }
    public double[] tau() { return pool.tau; }

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

    private boolean backSubstitute(double[] b, int n) {
        for (int i = 0; i < n; i++) {
            if (Math.abs(QR[i * n + i]) < EPSILON) return false;
        }
        BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, n, 1, 1.0, QR, 0, n, b, 0, 1);
        return true;
    }

    private boolean backSubstituteMultiple(double[] B, int n, int nrhs) {
        for (int i = 0; i < n; i++) {
            if (Math.abs(QR[i * n + i]) < EPSILON) return false;
        }
        BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, n, nrhs, 1.0, QR, 0, n, B, 0, nrhs);
        return true;
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
