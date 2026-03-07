/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;

import static java.lang.Math.max;
import static java.lang.Math.min;

public final class SVD implements Decomposition {

    public static final int SVD_NONE   = 0;
    public static final int SVD_WANT_U = 1;
    public static final int SVD_WANT_V = 2;
    public static final int SVD_FULL_U = 4;
    public static final int SVD_FULL_V = 8;
    public static final int SVD_ALL    = SVD_WANT_U | SVD_WANT_V;

    /** Decomposition options for SVD. */
    public enum Opts {
        /** Compute thin U (m × min(m,n)). Default when no opts given. */
        WANT_U,
        /** Compute full U (m × m). */
        FULL_U,
        /** Compute thin Vᵀ (min(m,n) × n). Default when no opts given. */
        WANT_V,
        /** Compute full Vᵀ (n × n). */
        FULL_V
    }

    public static final class Pool extends Decomposition.Workspace {
        public double[] s;
        public double[] UV;

        public Pool ensureS(int minMN) {
            if (s == null || s.length < minMN) s = new double[minMN];
            return this;
        }

        public Pool ensureUV(int size) {
            if (size > 0 && (UV == null || UV.length < size)) UV = new double[size];
            return this;
        }
    }

    private Pool pool;
    private double[] s;
    private double[] U;
    private double[] VT;
    private int m;
    private int n;
    private int kind;
    private boolean ok;

    private SVD() {}

    public static Pool workspace(int m, int n) {
        return workspace(m, n, SVD_ALL);
    }

    public static Pool workspace(int m, int n, int kind) {
        int minMN = min(m, n);
        int maxMN = max(m, n);
        Pool pool = new Pool();
        pool.ensureS(minMN);

        boolean wantU  = (kind & SVD_WANT_U) != 0;
        boolean wantVT = (kind & SVD_WANT_V) != 0;
        boolean fullU  = (kind & SVD_FULL_U) != 0;
        boolean fullVT = (kind & SVD_FULL_V) != 0;

        // Work layout:
        //   work[0..minMN)         = e  (off-diagonal from dgebd2)
        //   work[minMN..2*minMN)   = tauQ
        //   work[2*minMN..3*minMN) = tauP
        //   work[3*minMN..)        = scratch for dorgbr / dbdsqr
        // dbdsqr needs 4*minMN+16 scratch; dgebd2 needs maxMN scratch
        double[] tmp = new double[1];
        int scratch = max(maxMN, 4 * minMN + 16); // dbdsqr needs 4n+16 (dlasq1 requirement)
        if (wantU) {
            int uCols = fullU ? m : minMN;
            BLAS.dorgbr('Q', m, uCols, n, null, 0, uCols, null, 0, tmp, 0, -1);
            scratch = Math.max(scratch, (int) tmp[0]);
        }
        if (wantVT) {
            int vtRows = fullVT ? n : minMN;
            BLAS.dorgbr('P', vtRows, n, m, null, 0, n, null, 0, tmp, 0, -1);
            scratch = Math.max(scratch, (int) tmp[0]);
        }
        pool.ensureWork(3 * minMN + scratch);

        int uSize  = wantU  ? (fullU  ? m * m : m * minMN) : 0;
        int vtSize = wantVT ? (fullVT ? n * n : minMN * n) : 0;
        pool.ensureUV(max(uSize, vtSize));
        return pool;
    }

    /**
     * Decompose with default kind = SVD_ALL.
     */
    public static SVD decompose(double[] A, int m, int n) {
        return decompose(A, m, n, SVD_ALL, (Pool) null);
    }

    /**
     * Decompose with explicit kindMask and optional workspace.
     */
    public static SVD decompose(double[] A, int m, int n, int kindMask, Pool ws) {
        SVD svd = new SVD();
        svd.m = m;
        svd.n = n;
        svd.kind = kindMask;
        svd.doDecompose(A, m, n, kindMask, ws);
        return svd;
    }

    private void doDecompose(double[] A, int m, int n, int kind, Pool wsIn) {
        int minMN = min(m, n);
        int maxMN = max(m, n);

        boolean wantU     = (kind & SVD_WANT_U) != 0;
        boolean wantVT    = (kind & SVD_WANT_V) != 0;
        boolean wantFullU = (kind & SVD_FULL_U) != 0;
        boolean wantFullV = (kind & SVD_FULL_V) != 0;

        if (wsIn == null) wsIn = workspace(m, n, kind);
        this.pool = wsIn;

        // S is always needed
        pool.ensureS(minMN);

        // Work layout:
        //   work[0..minMN)         = e  (off-diagonal from dgebd2)
        //   work[minMN..2*minMN)   = tauQ
        //   work[2*minMN..3*minMN) = tauP
        //   work[3*minMN..)        = scratch for dorgbr / dbdsqr
        // dbdsqr needs 4*minMN+16 scratch (dlasq1 requires 4n+16)
        {
            double[] tmp = new double[1];
            int scratch = max(maxMN, 4 * minMN + 16);
            if (wantU) {
                int uCols = wantFullU ? m : minMN;
                BLAS.dorgbr('Q', m, uCols, n, null, 0, uCols, null, 0, tmp, 0, -1);
                scratch = Math.max(scratch, (int) tmp[0]);
            }
            if (wantVT) {
                int vtRows = wantFullV ? n : minMN;
                BLAS.dorgbr('P', vtRows, n, m, null, 0, n, null, 0, tmp, 0, -1);
                scratch = Math.max(scratch, (int) tmp[0]);
            }
            pool.ensureWork(3 * minMN + scratch);
        }

        // Reuse A array where possible (A contents are no longer needed after dgebd2):
        // A size = m*n; let the larger output reuse A, allocate the smaller one from pool
        final int aLen = m * n;
        final double[] outS = pool.s;
        final double[] outU;
        final double[] outVT;

        // S is much smaller than A, use pool directly
        if (wantU && wantVT) {
            int uSize  = wantFullU ? m * m : m * minMN;
            int vtSize = wantFullV ? n * n : minMN * n;
            if (uSize >= vtSize && uSize <= aLen) {
                // U reuses A, VT from pool
                outU = A; 
                pool.ensureUV(vtSize); 
                outVT = pool.UV;
            } else if (vtSize <= aLen) {
                // VT reuses A, U from pool
                pool.ensureUV(uSize); 
                outU = pool.UV; 
                outVT = A;
            } else {
                // Theoretically unreachable for thin SVD (both uSize and vtSize <= aLen)
                // Defensive fallback: pool gets the larger one, smaller one is allocated separately
                if (uSize >= vtSize) {
                    pool.ensureUV(uSize); 
                    outU = pool.UV; 
                    outVT = new double[vtSize];
                } else {
                    pool.ensureUV(vtSize); 
                    outU = new double[uSize]; 
                    outVT = pool.UV;
                }
            }
        } else if (wantU) {
            int uSize = wantFullU ? m * m : m * minMN;
            if (uSize <= aLen) {
                outU = A;
            } else {
                pool.ensureUV(uSize);
                outU = pool.UV;
            }
            outVT = null;
        } else if (wantVT) {
            int vtSize = wantFullV ? n * n : minMN * n;
            outU = null;
            if (vtSize <= aLen) {
                outVT = A;
            } else {
                pool.ensureUV(vtSize);
                outVT = pool.UV;
            }
        } else {
            outU  = null;
            outVT = null;
        }

        if (minMN == 0) {
            this.s = outS; this.U = outU; this.VT = outVT;
            ok = true;
            return;
        }

        double[] work = pool.work();
        // Gonum-style layout: e at [0..minMN), tauQ at [minMN..2*minMN), tauP at [2*minMN..3*minMN), scratch at [3*minMN..)
        int eOff    = 0;
        int tauQOff = minMN;
        int tauPOff = 2 * minMN;
        int scratchOff = 3 * minMN;
        int lda = n;

        BLAS.dgebd2(m, n, A, 0, lda, outS, 0, work, eOff, work, tauQOff, work, tauPOff, work, scratchOff);

        if (m >= n) {
            if (wantVT) {
                dlacpyUpper(n, n, A, 0, lda, outVT, 0, n);
                BLAS.dorgbr('P', n, n, n, outVT, 0, n, work, tauPOff, work, scratchOff, work.length - scratchOff);
            }
            if (wantU) {
                if (wantFullU) {
                    dlacpyLower(m, n, A, 0, lda, outU, 0, m);
                    for (int i = 0; i < m; i++) for (int j = n; j < m; j++) outU[i * m + j] = 0.0;
                    BLAS.dorgbr('Q', m, m, n, outU, 0, m, work, tauQOff, work, scratchOff, work.length - scratchOff);
                } else {
                    // thin U
                    dlacpyLower(m, minMN, A, 0, lda, outU, 0, minMN);
                    BLAS.dorgbr('Q', m, minMN, n, outU, 0, minMN, work, tauQOff, work, scratchOff, work.length - scratchOff);
                }
            }
            int ncvt = wantVT ? n : 0, nru = wantU ? m : 0;
            int ldvt = wantVT ? n : 1, ldu = wantU ? (wantFullU ? m : minMN) : 1;
            ok = BLAS.dbdsqr(BLAS.Uplo.Upper, minMN, ncvt, nru, 0, outS, 0, work, eOff, outVT, 0, ldvt, outU, 0, ldu, null, 0, 0, work, scratchOff);
        } else {
            if (wantVT) {
                dlacpyUpper(m, n, A, 0, lda, outVT, 0, n);
                if (wantFullV && n > m) for (int i = m; i < n; i++) for (int j = 0; j < n; j++) outVT[i * n + j] = 0.0;
                int vtRows = wantFullV ? n : minMN;
                BLAS.dorgbr('P', vtRows, n, m, outVT, 0, n, work, tauPOff, work, scratchOff, work.length - scratchOff);
            }
            if (wantU) {
                int uCols = wantFullU ? m : minMN;
                dlacpyLower(m, m, A, 0, lda, outU, 0, uCols);
                BLAS.dorgbr('Q', m, uCols, n, outU, 0, uCols, work, tauQOff, work, scratchOff, work.length - scratchOff);
            }
            int ncvt = wantVT ? n : 0, nru = wantU ? m : 0;
            int ldvt = wantVT ? n : 1, ldu = wantU ? (wantFullU ? m : minMN) : 1;
            ok = BLAS.dbdsqr(BLAS.Uplo.Lower, minMN, ncvt, nru, 0, outS, 0, work, eOff, outVT, 0, ldvt, outU, 0, ldu, null, 0, 0, work, scratchOff);
        }

        // Record the arrays actually written (user array or pool array)
        this.s = outS; this.U = outU; this.VT = outVT;
    }

    private static void dlacpyUpper(int m, int n, double[] src, int srcOff, int lda, double[] dst, int dstOff, int ldb) {
        for (int i = 0; i < m; i++)
            for (int j = i; j < n; j++)
                dst[dstOff + i * ldb + j] = src[srcOff + i * lda + j];
    }

    private static void dlacpyLower(int m, int n, double[] src, int srcOff, int lda, double[] dst, int dstOff, int ldb) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < min(i + 1, n); j++)
                dst[dstOff + i * ldb + j] = src[srcOff + i * lda + j];
    }

    @Override
    public Pool work() {
        return pool;
    }

    @Override
    public boolean ok() {
        return ok;
    }

    public double[] singularValues() { return ok ? s : null; }

    public double[] U()  { return ok ? U : null; }
    public double[] VT() { return ok ? VT : null; }

    /** Returns the left singular vectors matrix U. Returns null if decomposition failed or U was not requested. */
    public Matrix toU() {
        if (!ok || U == null) return null;
        int cols = uCols();
        double[] dst = new double[m * cols];
        System.arraycopy(U, 0, dst, 0, m * cols);
        return new Matrix(m, cols, false, dst);
    }

    /** Returns the right singular vectors matrix Vᵀ. Returns null if decomposition failed or V was not requested. */
    public Matrix toVT() {
        if (!ok || VT == null) return null;
        int rows = vtRows();
        double[] dst = new double[rows * n];
        System.arraycopy(VT, 0, dst, 0, rows * n);
        return new Matrix(rows, n, false, dst);
    }

    public int uCols() {
        if (U == null) return 0;
        return (kind & SVD_FULL_U) != 0 ? m : min(m, n);
    }

    public int vtRows() {
        if (VT == null) return 0;
        return (kind & SVD_FULL_V) != 0 ? n : min(m, n);
    }

    public int m() { return m; }
    public int n() { return n; }
    public int kind() { return ok ? kind : -1; }

    public int rank() {
        if (s == null || s.length == 0) return 0;
        double tol = Math.max(m, n) * Math.ulp(s[0]);
        return rank(tol);
    }

    public int rank(double tol) {
        if (s == null) return 0;
        int r = 0;
        for (double s : s) if (s > tol) r++;
        return r;
    }

    public double cond() {
        if (s == null || s.length == 0) return Double.NaN;
        double sMin = s[s.length - 1];
        if (sMin <= 0) return Double.POSITIVE_INFINITY;
        return s[0] / sMin;
    }

    public double norm2() {
        if (s == null || s.length == 0) return Double.NaN;
        return s[0];
    }

    public double[] solve(double[] b, double[] x) {
        return solve(b, x, null);
    }

    public double[] solve(double[] b, double[] x, double[] tmp) {
        if (s == null || U == null || VT == null) return null;
        int minMN = min(m, n);
        int uCols = uCols();
        double tol = EPSILON * s[0] * max(m, n);
        if (tmp == null || tmp.length < minMN) tmp = new double[minMN];
        for (int i = 0; i < minMN; i++) {
            if (s[i] > tol) {
                double sum = 0;
                for (int j = 0; j < m; j++) sum += U[j * uCols + i] * b[j];
                tmp[i] = sum / s[i];
            } else {
                tmp[i] = 0;
            }
        }
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < minMN; j++) sum += VT[j * n + i] * tmp[j];
            x[i] = sum;
        }
        return x;
    }

    private static final double EPSILON = 0x1.0p-52;
}
