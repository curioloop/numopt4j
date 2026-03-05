/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.blas.Dlange;

import static java.lang.Math.abs;

public final class LU implements Decomposition {

    private static final int BLOCK_SIZE = 64;

    public static final class Pool extends Decomposition.Workspace {
        /** Allocate/expand work and iwork on demand; ipiv and piv both reuse iwork (first n for ipiv, next n for piv) */
        public Pool ensure(int n) {
            ensureWork(Math.max(n, n * BLOCK_SIZE));
            ensureIwork(2 * n);
            return this;
        }
    }

    private Pool pool;
    private double[] LU;
    private int n;
    private boolean ok;
    private double anorm;

    private LU() {}

    public static Decomposition.Workspace workspace(int n) {
        return new Pool().ensure(n);
    }

    public static boolean inverseInPlace(double[] A, int[] ipiv, double[] work, int n) {
        if (BLAS.dgetrf(n, n, A, 0, n, ipiv, 0) != 0) return false;
        return BLAS.dgetri(n, A, 0, n, ipiv, 0, work, 0, work != null ? work.length : n);
    }

    public static LU decompose(double[] A, int n) {
        return decompose(A, n, null);
    }

    public static LU decompose(double[] A, int n, Workspace ws) {
        LU lu = new LU();
        lu.doDecompose(A, n, ws);
        return lu;
    }

    private void doDecompose(double[] A, int n, Workspace ws) {
        if (A == null || A.length < n * n) {
            throw new IllegalArgumentException("Matrix A must have length >= n*n");
        }
        if (n <= 0) {
            throw new IllegalArgumentException("Matrix dimension must be positive");
        }

        this.LU = A;
        this.n = n;
        this.ok = false;
        this.anorm = Dlange.dlange('1', n, n, A, 0, n);

        if (ws != null && !(ws instanceof Pool)) {
            throw new IllegalArgumentException("Workspace must be an instance of LU.Pool");
        }
        if (ws == null) {
            ws = new Pool();
        }
        this.pool = (Pool) ws;

        pool.ensure(n);

        this.ok = BLAS.dgetrf(n, n, A, 0, n, pool.iwork(), n) == 0;

        updatePivots();
    }

    private void updatePivots() {
        // piv (permutation vector) stored in iwork[0..n), ipiv in iwork[n..2n)
        int[] iwork = pool.iwork();
        for (int i = 0; i < n; i++) {
            iwork[i] = i;
        }
        for (int i = n - 1; i >= 0; i--) {
            int v = iwork[n + i];
            int tmp = iwork[i];
            iwork[i] = iwork[v];
            iwork[v] = tmp;
        }
    }

    public double[] solve(double[] b, double[] x) {
        if (!ok) return null;
        if (b == null || b.length < n) {
            throw new IllegalArgumentException("Vector b must have length >= n");
        }
        if (x == null || x.length < n) {
            x = new double[n];
        }
        if (x != b) {
            System.arraycopy(b, 0, x, 0, n);
        }
        BLAS.dgetrs(BLAS.Transpose.NoTrans, n, 1, LU, 0, n, pool.iwork(), n, x, 0, 1);
        return x;
    }

    public double[] inverse(double[] Ainv) {
        if (!ok) return null;
        if (Ainv == null || Ainv.length < n * n) {
            Ainv = new double[n * n];
        }
        if (Ainv != LU) {
            System.arraycopy(LU, 0, Ainv, 0, n * n);
        }
        double[] work = pool.work();
        if (!BLAS.dgetri(n, Ainv, 0, n, pool.iwork(), n, work, 0, work.length)) {
            return null;
        }
        return Ainv;
    }

    public double determinant() {
        if (!ok) return Double.NaN;
        int[] iwork = pool.iwork();
        double det = 1.0;
        for (int i = 0; i < n; i++) {
            det *= LU[i * n + i];
            if (iwork[n + i] != i) {
                det = -det;
            }
        }
        return det;
    }

    public double[] logDet() {
        if (!ok) return new double[]{Double.NaN, Double.NaN};
        int[] iwork = pool.iwork();
        double logDet = 0.0;
        int sign = 1;
        for (int i = 0; i < n; i++) {
            double diag = LU[i * n + i];
            if (diag < 0) {
                sign = -sign;
                logDet += Math.log(-diag);
            } else {
                logDet += Math.log(diag);
            }
            if (iwork[n + i] != i) {
                sign = -sign;
            }
        }
        return new double[]{logDet, sign};
    }

    public double cond() {
        if (!ok || n == 0) return Double.NaN;
        double rcond = BLAS.dgecon('1', n, LU, n, anorm, pool.work(), pool.iwork());
        if (rcond == 0) return Double.POSITIVE_INFINITY;
        return 1.0 / rcond;
    }

    @Override
    public boolean ok() {
        return ok;
    }

    @Override
    public Pool work() {
        return pool;
    }

    @Override
    public int rows(Part part) {
        if (!ok) throw new IllegalStateException("Decomposition failed");
        switch (part) {
            case L:
            case U:
            case P:
                return n;
            default:
                throw new UnsupportedOperationException("Part " + part + " not supported");
        }
    }

    @Override
    public int cols(Part part) {
        if (!ok) throw new IllegalStateException("Decomposition failed");
        switch (part) {
            case L:
            case U:
            case P:
                return n;
            default:
                throw new UnsupportedOperationException("Part " + part + " not supported");
        }
    }

    public double[] LU() {
        return LU;
    }

    public int[] piv() {
        return pool.iwork();
    }

    @Override
    public Matrix extract(Part part) {
        return extract(part, null);
    }

    @Override
    public Matrix extract(Part part, double[] dst) {
        if (!ok) return null;
        switch (part) {
            case L:
            case U:
            case P:
                break;
            default:
                return null;
        }
        int size = size(part);
        if (dst == null || dst.length < size) dst = new double[size];
        switch (part) {
            case L: {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if (i > j) {
                            dst[i * n + j] = LU[i * n + j];
                        } else if (i == j) {
                            dst[i * n + j] = 1.0;
                        } else {
                            dst[i * n + j] = 0.0;
                        }
                    }
                }
                return new Matrix(n, n, false, dst);
            }
            case U: {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        dst[i * n + j] = (i <= j) ? LU[i * n + j] : 0.0;
                    }
                }
                return new Matrix(n, n, false, dst);
            }
            case P: {
                // piv stored in iwork[0..n) (0-indexed), computed by updatePivots()
                int[] piv = pool.iwork();
                for (int i = 0; i < n; i++) {
                    dst[i * n + piv[i]] = 1.0;
                }
                return new Matrix(n, n, false, dst);
            }
            default:
                return null;
        }
    }

    public int n() {
        return n;
    }
}
