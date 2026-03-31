/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.blas.Dlansy;

import java.util.Arrays;

import static java.lang.Math.abs;

public final class Cholesky implements Decomposition {

    private static final double CONDITION_TOLERANCE = 1e16;

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
    private double condition;

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

    public static Workspace workspace() {
        return new Pool();
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
        this.condition = Double.NaN;

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
    public Pool pool() {
        return pool;
    }

    /**
     * Solves {@code A*x = b} using the stored Cholesky or pivoted LDL factorization.
     *
     * <p>To minimize allocations, this method may use {@code b} as the destination buffer.
     * When {@code x} is {@code null}, the solution is written back into {@code b} and that
     * same array is returned.</p>
     *
     * <p>Factorization success remains available through {@link #ok()}, but operational failure
     * is reported with exceptions: failed factorizations, singular or indefinite factors, and
     * factors with estimated condition number above {@code 1e16} are rejected.</p>
     *
     * @param b right-hand side vector of length at least {@code n}
     * @param x destination vector of length at least {@code n}; may be {@code null}
     * @return the solution vector
     * @throws IllegalArgumentException if {@code b} is too short
     * @throws ArithmeticException if the factorization failed, is singular, indefinite, or ill-conditioned
     */
    public double[] solve(double[] b, double[] x) {
        if (b == null || b.length < n) {
            throw new IllegalArgumentException("Vector b must have length >= n");
        }
        if (!ok) {
            throw new ArithmeticException(pivoting ? "LDL factor is singular or indefinite" : "Cholesky factor is singular or indefinite");
        }
        double condition = ensureCondition();
        if (!Double.isFinite(condition) || condition > CONDITION_TOLERANCE) {
            throw new ArithmeticException(pivoting ? "LDL factor is singular or ill-conditioned" : "Cholesky factor is singular or ill-conditioned");
        }
        if (x == null) {
            x = b;
        } else if (x.length < n) {
            x = new double[n];
        }
        if (x != b) System.arraycopy(b, 0, x, 0, n);
        if (pivoting) {
            BLAS.dsytrs(uplo, n, 1, LDL, 0, n, pool.ipiv, 0, x, 0, 1);
        } else {
            BLAS.dpotrs(uplo, n, 1, LDL, 0, n, x, 0, 1);
        }
        return x;
    }

    /**
     * Forms the inverse from a non-pivoted Cholesky factorization.
     *
     * <p>This operation is only available for the standard Cholesky path. Pivoted LDL
     * factorizations continue to reject inversion with {@link UnsupportedOperationException}.</p>
     *
     * @param Ainv destination array of length at least {@code n*n}; may be {@code null}
     * @return the inverse matrix in row-major order
     * @throws UnsupportedOperationException if this decomposition uses pivoted LDL mode
     * @throws ArithmeticException if the factorization failed, is singular, indefinite, or ill-conditioned
     */
    public double[] inverse(double[] Ainv) {
        if (pivoting) {
            throw new UnsupportedOperationException("Inverse not supported for pivoting (LDL) decomposition");
        }
        if (!ok) {
            throw new ArithmeticException("Cholesky factor is singular or indefinite");
        }
        double condition = ensureCondition();
        if (!Double.isFinite(condition) || condition > CONDITION_TOLERANCE) {
            throw new ArithmeticException("Cholesky factor is singular or ill-conditioned");
        }
        if (Ainv == null || Ainv.length < n * n) Ainv = new double[n * n];
        if (Ainv != LDL) System.arraycopy(LDL, 0, Ainv, 0, n * n);
        if (!BLAS.dpotri(uplo, n, Ainv, 0, n)) {
            throw new ArithmeticException("Cholesky factor is singular");
        }
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
        return ensureCondition();
    }

    private double ensureCondition() {
        if (Double.isNaN(condition)) {
            condition = ok ? computeCondition() : Double.POSITIVE_INFINITY;
        }
        return condition;
    }

    private double computeCondition() {
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

    /**
     * Returns the cached condition number estimate for the factorized matrix.
     *
     * <p>The estimate is computed once during factorization. Failed factorizations report
     * {@link Double#POSITIVE_INFINITY}.</p>
     */
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
        if (n < 2) {
            return;
        }
        if (lower) {
            for (int i = 0, rowOff = 0; i < n - 1; i++, rowOff += n) {
                Arrays.fill(A, rowOff + i + 1, rowOff + n, 0.0);
            }
        } else {
            for (int i = 1, rowOff = n; i < n; i++, rowOff += n) {
                Arrays.fill(A, rowOff, rowOff + i, 0.0);
            }
        }
    }
}
