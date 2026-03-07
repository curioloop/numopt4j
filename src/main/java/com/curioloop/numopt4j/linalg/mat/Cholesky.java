/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.blas.Dlansy;

import static java.lang.Math.abs;

public final class Cholesky implements Decomposition {

    /** Decomposition options for Cholesky / LDLᵀ. */
    public enum Opts {
        /** Use lower triangle of the input matrix (default). */
        LOWER,
        /** Use upper triangle of the input matrix. */
        UPPER,
        /** Use pivoted LDLᵀ decomposition instead of standard Cholesky. */
        PIVOTING
    }

    private Pool pool;
    private double[] LDL;
    private int n;
    private BLAS.Uplo uplo;
    private boolean pivoting;
    private boolean ok;
    private double anorm;

    /**
     * Pool for both Cholesky and LDL (pivoting) modes.
     *
     * <p>Work layout:
     * <ul>
     *   <li>pivoting=true (LDL): work >= max(n*nb, 2n) for dsytrf + dsycon; ipiv (separate field) for pivot indices; iwork = isgn scratch for dsycon</li>
     *   <li>pivoting=false (Cholesky): work >= 3n for dlansy + dpocon; iwork >= n for dpocon</li>
     * </ul>
     */
    public static final class Pool extends Workspace {
        /** Pivot indices from dsytrf (length n), separate from iwork scratch. */
        public int[] ipiv;

        private Pool() {}

        public Pool ensure(int n, boolean pivoting) {
            if (pivoting) {
                double[] tmp = new double[1];
                BLAS.dsytrf(BLAS.Uplo.Lower, n, new double[1], 0, n, new int[1], 0, tmp, -1);
                ensureWork(Math.max((int) tmp[0], 2 * n));
                if (ipiv == null || ipiv.length < n) ipiv = new int[n];
            } else {
                ensureWork(3 * n);
            }
            // iwork used as isgn scratch for dlacn2 inside dsycon/dpocon (length n)
            ensureIwork(n);
            return this;
        }
    }

    private Cholesky() {}

    public static Workspace workspace(int n) {
        return workspace(n, false);
    }

    public static Workspace workspace(int n, boolean pivoting) {
        return new Pool().ensure(n, pivoting);
    }

    public static Cholesky decompose(double[] A, int n, BLAS.Uplo uplo) {
        return decompose(A, n, uplo, false, null);
    }

    public static Cholesky decompose(double[] A, int n, BLAS.Uplo uplo, boolean pivoting, Pool ws) {
        Cholesky c = new Cholesky();
        c.doDecompose(A, n, uplo, pivoting, ws);
        return c;
    }

    private void doDecompose(double[] A, int n, BLAS.Uplo uplo, boolean pivoting, Pool ws) {
        if (A == null || A.length < n * n) {
            throw new IllegalArgumentException("Matrix A must have length >= n*n");
        }
        if (n <= 0) {
            throw new IllegalArgumentException("Matrix dimension must be positive");
        }
        if (uplo == null) {
            throw new NullPointerException("uplo must not be null");
        }

        this.LDL = A;
        this.n = n;
        this.uplo = uplo;
        this.pivoting = pivoting;
        this.ok = false;

        if (ws == null) ws = new Pool();
        this.pool = ws;
        this.pool.ensure(n, pivoting);

        this.anorm = Dlansy.dlansy('1', uplo, n, A, n, pool.work());

        if (pivoting) {
            decomposeLDL();
        } else {
            decomposeCholesky();
        }
    }

    private void decomposeCholesky() {
        boolean success = BLAS.dpotrf(uplo, n, LDL, 0, n) == 0;
        if (success) {
            clearOppositeTriangle(LDL, n, uplo == BLAS.Uplo.Lower);
            this.ok = true;
        }
    }

    private void decomposeLDL() {
        int info = BLAS.dsytrf(uplo, n, LDL, 0, n, pool.ipiv, 0, pool.work(), pool.work().length);
        if (info == 0) this.ok = true;
    }

    @Override
    public Pool work() {
        return pool;
    }

    public double[] solve(double[] b, double[] x) {
        if (!ok) return null;
        if (b == null || b.length < n) {
            throw new IllegalArgumentException("Vector b must have length >= n");
        }
        if (x == null || x.length < n) x = new double[n];
        if (x != b) System.arraycopy(b, 0, x, 0, n);
        if (pivoting) {
            BLAS.dsytrs(uplo, n, 1, LDL, 0, n, pool.ipiv, 0, x, 0, 1);
        } else {
            BLAS.dpotrs(uplo, n, 1, LDL, 0, n, x, 0, 1);
        }
        return x;
    }

    public double[] inverse(double[] Ainv) {
        if (!ok) return null;
        if (pivoting) {
            throw new UnsupportedOperationException("Inverse not supported for pivoting (LDL) decomposition");
        }
        if (Ainv == null || Ainv.length < n * n) Ainv = new double[n * n];
        if (Ainv != LDL) System.arraycopy(LDL, 0, Ainv, 0, n * n);
        if (!BLAS.dpotri(uplo, n, Ainv, 0, n)) return null;
        return Ainv;
    }

    public double determinant() {
        if (!ok) return Double.NaN;
        if (pivoting) {
            double det = 1.0;
            int[] piv = pool.ipiv;
            for (int i = 0; i < n; ) {
                if (piv[i] > 0) {
                    det *= LDL[i * n + i];
                    i++;
                } else {
                    int kp = -piv[i] - 1;
                    if (kp > i) {
                        double d11 = LDL[i * n + i];
                        double d12 = uplo == BLAS.Uplo.Lower ? LDL[kp * n + i] : LDL[i * n + kp];
                        double d22 = LDL[kp * n + kp];
                        det *= (d11 * d22 - d12 * d12);
                        i = kp + 1;
                    } else {
                        i++;
                    }
                }
            }
            return det;
        }
        double det = 1.0;
        for (int i = 0; i < n; i++) det *= LDL[i * n + i];
        return det * det;
    }

    public double logDet() {
        if (!ok) return Double.NaN;
        if (pivoting) {
            double logDet = 0.0;
            int[] piv = pool.ipiv;
            for (int i = 0; i < n; ) {
                if (piv[i] > 0) {
                    logDet += Math.log(abs(LDL[i * n + i]));
                    i++;
                } else {
                    int kp = -piv[i] - 1;
                    if (kp > i) {
                        double d11 = LDL[i * n + i];
                        double d12 = uplo == BLAS.Uplo.Lower ? LDL[kp * n + i] : LDL[i * n + kp];
                        double d22 = LDL[kp * n + kp];
                        logDet += Math.log(abs(d11 * d22 - d12 * d12));
                        i = kp + 1;
                    } else {
                        i++;
                    }
                }
            }
            return logDet;
        }
        double logDet = 0.0;
        for (int i = 0; i < n; i++) logDet += Math.log(LDL[i * n + i]);
        return 2 * logDet;
    }

    public double cond() {
        if (!ok || n == 0) return Double.NaN;
        double rcond;
        if (pivoting) {
            rcond = BLAS.dsycon(uplo, n, LDL, 0, n, pool.ipiv, 0, anorm, pool.work(), pool.iwork());
        } else {
            rcond = BLAS.dpocon(uplo, n, LDL, n, anorm, pool.work(), pool.iwork());
        }
        if (rcond == 0) return Double.POSITIVE_INFINITY;
        return 1.0 / rcond;
    }

    @Override
    public boolean ok() {
        return ok;
    }

    /** Returns the lower triangular matrix L (n×n). Returns null if decomposition failed. */
    public Matrix toL() {
        if (!ok) return null;
        return new Matrix(n, n, false, LDL);
    }

    /**
     * Returns the block-diagonal matrix D (n×n) from LDLᵀ decomposition.
     * Returns null if not in pivoting mode or if decomposition failed.
     */
    public Matrix toD() {
        if (!ok || !pivoting) return null;
        double[] D = new double[n * n];
        extractD(D);
        return new Matrix(n, n, false, D);
    }

    private void extractD(double[] D) {
        boolean lower = uplo == BLAS.Uplo.Lower;
        int[] piv = pool.ipiv;
        for (int i = 0; i < n; ) {
            if (piv[i] > 0) {
                D[i * n + i] = LDL[i * n + i];
                i++;
            } else {
                int kp = -piv[i] - 1;
                if (kp > i) {
                    double d11 = LDL[i * n + i];
                    double d12 = lower ? LDL[kp * n + i] : LDL[i * n + kp];
                    double d22 = LDL[kp * n + kp];
                    D[i * n + i] = d11;
                    D[i * n + kp] = d12;
                    D[kp * n + i] = d12;
                    D[kp * n + kp] = d22;
                    i = kp + 1;
                } else {
                    i++;
                }
            }
        }
    }

    public int[] piv() {
        return pivoting ? pool.ipiv : null;
    }

    public int n() { return n; }

    public BLAS.Uplo uplo() { return uplo; }

    public boolean isPivoting() { return pivoting; }

    private static void clearOppositeTriangle(double[] A, int n, boolean lower) {
        if (lower) {
            for (int i = 0, rowOff = 0; i < n; i++, rowOff += n)
                for (int j = i + 1; j < n; j++) A[rowOff + j] = 0;
        } else {
            for (int i = 0, rowOff = 0; i < n; i++, rowOff += n)
                for (int j = 0; j < i; j++) A[rowOff + j] = 0;
        }
    }
}
