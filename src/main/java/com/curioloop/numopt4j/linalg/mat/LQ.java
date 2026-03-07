/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;

import static java.lang.Math.abs;
import static java.lang.Math.max;

public final class LQ implements Decomposition {

    private static final double EPSILON = BLAS.dlamch('E');

    /** Configuration options for LQ decomposition (currently no variants). */
    public enum Opts {}

    public static final class Pool extends Decomposition.Workspace {
        /**
         * Ensure all buffers are allocated for LQ decomposition of an m×n matrix.
         * Layout: work[0..m) = tau, work[m..) = scratch.
         */
        public Pool ensure(int m, int n) {
            // Query dgelqf for optimal scratch size; tau occupies work[0..m)
            // solveTranspose also needs work[m..m+n) as a temp buffer for b
            double[] tmp = new double[1];
            BLAS.dgelqf(m, n, null, 0, n, null, 0, tmp, 0, -1);
            ensureWork(m + n + (int) tmp[0]);
            return this;
        }
    }

    private Pool pool;
    private double[] LQ;
    private int m;
    private int n;
    private boolean ok;

    private LQ() {}

    public static Pool workspace(int m, int n) {
        return new Pool().ensure(m, n);
    }

    public static LQ decompose(double[] A, int m, int n) {
        return decompose(A, m, n, (Pool) null);
    }

    public static LQ decompose(double[] A, int m, int n, Pool ws) {
        LQ lq = new LQ();
        lq.doDecompose(A, m, n, ws);
        return lq;
    }

    private void doDecompose(double[] A, int m, int n, Pool ws) {
        if (A == null || A.length < m * n) {
            throw new IllegalArgumentException("Matrix A must have length >= m*n");
        }
        if (m <= 0 || n <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive");
        }
        if (m > n) {
            throw new IllegalArgumentException("For LQ decomposition, m must be <= n");
        }

        this.LQ = A;
        this.m = m;
        this.n = n;
        this.ok = false;

        if (ws == null) {
            ws = new Pool();
        }
        this.pool = ws;

        pool.ensure(m, n);

        BLAS.dgelqf(m, n, A, 0, n, pool.work(), 0, pool.work(), m, pool.work().length - m);
        this.ok = true;
    }

    public double[] solve(double[] b, double[] x) {
        if (!ok) return null;
        if (b == null || b.length < m) {
            throw new IllegalArgumentException("Vector b must have length >= m");
        }
        if (x == null || x.length < n) {
            x = new double[n];
        }

        if (x != b) {
            System.arraycopy(b, 0, x, 0, m);
        }

        for (int i = 0; i < m; i++) {
            if (abs(LQ[i * n + i]) < EPSILON) {
                return null;
            }
            double sum = x[i];
            for (int j = 0; j < i; j++) {
                sum -= LQ[i * n + j] * x[j];
            }
            x[i] = sum / LQ[i * n + i];
        }

        for (int i = m; i < n; i++) {
            x[i] = 0;
        }

        BLAS.dormlq(BLAS.Side.Left, BLAS.Transpose.Trans, n, 1, m, LQ, 0, n,
                pool.work(), 0, x, 0, 1,
                pool.work(), m, pool.work().length - m);

        return x;
    }

    public double[] solveTranspose(double[] b, double[] x) {
        if (!ok) return null;
        if (b == null || b.length < n) {
            throw new IllegalArgumentException("Vector b must have length >= n");
        }
        if (x == null || x.length < m) {
            x = new double[m];
        }

        double[] work = pool.work();
        int bOff = m;
        int scrOff = m + n;
        System.arraycopy(b, 0, work, bOff, n);

        BLAS.dormlq(BLAS.Side.Left, BLAS.Transpose.NoTrans, n, 1, m, LQ, 0, n,
                work, 0, work, bOff, 1,
                work, scrOff, work.length - scrOff);

        for (int i = m - 1; i >= 0; i--) {
            if (abs(LQ[i * n + i]) < EPSILON) {
                return null;
            }
            double sum = work[bOff + i];
            for (int j = i + 1; j < m; j++) {
                sum -= LQ[j * n + i] * x[j];
            }
            x[i] = sum / LQ[i * n + i];
        }

        return x;
    }

    public double[] leastSquares(double[] b, double[] x) {
        if (!ok) return null;
        if (b == null || b.length < n) {
            throw new IllegalArgumentException("Vector b must have length >= n");
        }
        if (x == null || x.length < n) {
            x = new double[n];
        }

        for (int i = 0; i < n; i++) {
            x[i] = b[i];
        }

        BLAS.dormlq(BLAS.Side.Left, BLAS.Transpose.Trans, n, 1, m, LQ, 0, n,
                pool.work(), 0, x, 0, 1,
                pool.work(), m, pool.work().length - m);

        for (int i = 0; i < m; i++) {
            if (abs(LQ[i * n + i]) < EPSILON) {
                return null;
            }
            double sum = x[i];
            for (int j = 0; j < i; j++) {
                sum -= LQ[i * n + j] * x[j];
            }
            x[i] = sum / LQ[i * n + i];
        }

        for (int i = m; i < n; i++) {
            x[i] = 0;
        }

        return x;
    }

    /** Returns the L factor as an m×m lower triangular matrix, or null if failed. */
    public Matrix toL() {
        if (!ok) return null;
        double[] dst = new double[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                dst[i * m + j] = (j <= i) ? LQ[i * n + j] : 0.0;
            }
        }
        return new Matrix(m, m, false, dst);
    }

    /** Returns the Q factor as an n×n orthogonal matrix, or null if failed. */
    public Matrix toQ() {
        if (!ok) return null;
        double[] dst = new double[n * n];
        java.util.Arrays.fill(dst, 0, n * n, 0.0);
        for (int i = 0; i < m; i++) {
            System.arraycopy(LQ, i * n, dst, i * n, n);
        }
        BLAS.dorglq(n, n, m, dst, 0, n, pool.work(), 0, pool.work(), m);
        return new Matrix(n, n, false, dst);
    }

    public double cond() {
        if (!ok || m == 0) return Double.NaN;

        double anorm = 0;
        for (int j = 0; j < m; j++) {
            double colSum = 0;
            for (int i = j; i < m; i++) {
                colSum += abs(LQ[i * n + j]);
            }
            anorm = max(anorm, colSum);
        }
        if (anorm == 0) return Double.POSITIVE_INFINITY;

        double[] v = pool.work();
        int vOff = m;
        double est = 0;
        int kase = 0;

        for (int i = 0; i < m; i++) {
            v[vOff + i] = 1.0 / (m - i);
        }

        while (true) {
            if (kase == 0) {
                double t = 0;
                for (int i = 0; i < m; i++) {
                    double sum = 0;
                    for (int j = i; j < m; j++) {
                        sum += abs(LQ[j * n + i]) * v[vOff + j];
                    }
                    v[vOff + i] = sum;
                    if (abs(v[vOff + i]) > t) t = abs(v[vOff + i]);
                }
                for (int i = 0; i < m; i++) {
                    v[vOff + i] /= t;
                }

                for (int i = m - 1; i >= 0; i--) {
                    double sum = v[vOff + i];
                    for (int j = 0; j < i; j++) {
                        sum -= LQ[i * n + j] * v[vOff + j];
                    }
                    if (abs(LQ[i * n + i]) > EPSILON) {
                        v[vOff + i] = sum / LQ[i * n + i];
                    } else {
                        v[vOff + i] = 0;
                    }
                }

                est = 0;
                for (int i = 0; i < m; i++) {
                    est = max(est, abs(v[vOff + i]));
                }
                kase = 1;
            } else if (kase == 1) {
                for (int i = 0; i < m; i++) {
                    double sum = v[vOff + i];
                    for (int j = 0; j < i; j++) {
                        sum -= LQ[i * n + j] * v[vOff + j];
                    }
                    if (abs(LQ[i * n + i]) > EPSILON) {
                        v[vOff + i] = sum / LQ[i * n + i];
                    } else {
                        v[vOff + i] = 0;
                    }
                }

                for (int i = m - 1; i >= 0; i--) {
                    double sum = 0;
                    for (int j = i; j < m; j++) {
                        sum += abs(LQ[j * n + i]) * v[vOff + j];
                    }
                    v[vOff + i] = sum;
                }

                double newEst = 0;
                for (int i = 0; i < m; i++) {
                    newEst = max(newEst, abs(v[vOff + i]));
                }

                if (newEst <= est) {
                    break;
                }
                est = newEst;
                kase = 2;
            } else {
                break;
            }
        }

        if (est == 0) return Double.POSITIVE_INFINITY;
        return anorm * est;
    }

    @Override
    public boolean ok() {
        return ok;
    }

    @Override
    public Pool pool() {
        return pool;
    }

    public int m() {
        return m;
    }

    public int n() {
        return n;
    }
}
