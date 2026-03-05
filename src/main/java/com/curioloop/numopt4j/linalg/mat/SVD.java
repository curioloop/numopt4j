/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;

import static java.lang.Math.max;
import static java.lang.Math.min;

public final class SVD implements Decomposition {

    public static final int SVD_NONE = 0;
    public static final int SVD_WANT_U = 1;
    public static final int SVD_WANT_V = 2;
    public static final int SVD_FULL_U = 4;
    public static final int SVD_FULL_V = 8;
    public static final int SVD_ALL = SVD_WANT_U | SVD_WANT_V;

    public static final class Pool extends Decomposition.Workspace {
        public double[] S;
        public double[] UV;  // shared buffer for U and VT — they are never written simultaneously, allocated as max(uSize, vtSize)

        public Pool ensureS(int minMN) {
            if (S == null || S.length < minMN) {
                S = new double[minMN];
            }
            return this;
        }

        /** Allocate/expand UV buffer on demand; size = max(uSize, vtSize) */
        public Pool ensureUV(int size) {
            if (size > 0 && (UV == null || UV.length < size)) {
                UV = new double[size];
            }
            return this;
        }
    }

    private SVD.Pool pool;
    private double[] S;   // points to actual output (pool.S)
    private double[] U;   // points to actual output (A reuse or pool.U)
    private double[] VT;  // points to actual output (A reuse or pool.VT)
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
        pool.ensureWork(maxMN * max(1, 4 * minMN) + minMN + 11 + 2 * minMN);
        pool.ensureS(minMN);
        boolean wantU  = (kind & SVD_WANT_U) != 0;
        boolean wantVT = (kind & SVD_WANT_V) != 0;
        boolean fullU  = (kind & SVD_FULL_U) != 0;
        boolean fullVT = (kind & SVD_FULL_V) != 0;
        int uSize  = wantU  ? (fullU  ? m * m : m * minMN) : 0;
        int vtSize = wantVT ? (fullVT ? n * n : minMN * n) : 0;
        pool.ensureUV(max(uSize, vtSize));
        return pool;
    }

    public static SVD decompose(double[] A, int m, int n) {
        return decompose(A, m, n, SVD_ALL, null);
    }

    public static SVD decompose(double[] A, int m, int n, int kind) {
        return decompose(A, m, n, kind, null);
    }

    public static SVD decompose(double[] A, int m, int n, int kind, Workspace ws) {
        SVD svd = new SVD();
        svd.m = m;
        svd.n = n;
        svd.kind = kind;
        svd.doDecompose(A, m, n, kind, ws);
        return svd;
    }

    private void doDecompose(double[] A, int m, int n, int kind, Workspace wsIn) {
        int minMN = min(m, n);
        int maxMN = max(m, n);

        boolean wantU = (kind & SVD_WANT_U) != 0;
        boolean wantVT = (kind & SVD_WANT_V) != 0;
        boolean wantFullU = (kind & SVD_FULL_U) != 0;
        boolean wantFullV = (kind & SVD_FULL_V) != 0;

        if (wsIn != null && !(wsIn instanceof Pool)) {
            throw new IllegalArgumentException("Workspace must be an instance of SVD.Pool");
        }

        if (wsIn == null) {
            wsIn = workspace(m, n, kind);
        }
        this.pool = (Pool) wsIn;

        // Reserve 2*minMN at the end of work for tau offsets (tauQ and tauP each occupy minMN)
        pool.ensureWork(maxMN * max(1, 4 * minMN) + minMN + 11 + 2 * minMN);
        // S is always needed
        pool.ensureS(minMN);

        // Reuse A array where possible (A contents are no longer needed after dgebd2):
        // A size = m*n; let the larger output reuse A, allocate the smaller one from pool
        final int aLen = m * n;
        final double[] outS;
        final double[] outU;
        final double[] outVT;

        // S is much smaller than A, use pool directly
        outS = pool.S;
        if (wantU && wantVT) {
            int uSize  = wantFullU ? m * m : m * minMN;
            int vtSize = wantFullV ? n * n : minMN * n;
            if (uSize >= vtSize && uSize <= aLen) {
                // U reuses A, VT from pool
                outU  = A;
                pool.ensureUV(vtSize);
                outVT = pool.UV;
            } else if (vtSize <= aLen) {
                // VT reuses A, U from pool
                pool.ensureUV(uSize);
                outU  = pool.UV;
                outVT = A;
            } else {
                // Theoretically unreachable for thin SVD (both uSize and vtSize <= aLen)
                // Defensive fallback: pool gets the larger one, smaller one is allocated separately
                if (uSize >= vtSize) {
                    pool.ensureUV(uSize);
                    outU  = pool.UV;
                    outVT = new double[vtSize];
                } else {
                    pool.ensureUV(vtSize);
                    outU  = new double[uSize];
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
            this.S = outS; this.U = outU; this.VT = outVT;
            ok = true;
            return;
        }

        double[] work = pool.work();
        int tauOff = work.length - 2 * minMN;
        // tauQ at [tauOff, tauOff+minMN), tauP at [tauOff+minMN, tauOff+2*minMN)
        double[] tauQ = work;
        int tauQOff = tauOff;
        double[] tauP = work;
        int tauPOff = tauOff + minMN;
        double[] e = work;
        int eOff = work.length - minMN - 2 * minMN;

        int lda = n;
        
        BLAS.dgebd2(m, n, A, 0, lda, outS, 0, e, eOff, tauQ, tauQOff, tauP, tauPOff, work, 0);

        if (m >= n) {
            if (wantVT) {
                dlacpyUpper(n, n, A, 0, lda, outVT, 0, n);
                BLAS.dorgbr('P', n, n, n, outVT, 0, n, tauP, tauPOff, work, 0, work.length);
            }

            if (wantU) {
                if (wantFullU && m > n) {
                    dlacpyLower(m, n, A, 0, lda, outU, 0, m);
                    for (int i = 0; i < m; i++) {
                        for (int j = n; j < m; j++) {
                            outU[i * m + j] = 0.0;
                        }
                    }
                    BLAS.dorgbr('Q', m, m, n, outU, 0, m, tauQ, tauQOff, work, 0, work.length);
                } else if (wantFullU) {
                    // m == n case
                    dlacpyLower(m, n, A, 0, lda, outU, 0, m);
                    for (int i = 0; i < m; i++) {
                        for (int j = n; j < m; j++) {
                            outU[i * m + j] = 0.0;
                        }
                    }
                    BLAS.dorgbr('Q', m, m, n, outU, 0, m, tauQ, tauQOff, work, 0, work.length);
                } else {
                    // thin U
                    dlacpyLower(m, minMN, A, 0, lda, outU, 0, minMN);
                    BLAS.dorgbr('Q', m, minMN, n, outU, 0, minMN, tauQ, tauQOff, work, 0, work.length);
                }
            }

            int ncvt = wantVT ? n : 0;
            int nru = wantU ? m : 0;
            int ldvt = wantVT ? n : 1;
            int ldu = wantU ? (wantFullU ? m : minMN) : 1;
            ok = BLAS.dbdsqr(BLAS.Uplo.Upper, minMN, ncvt, nru, 0, outS, 0, e, eOff, outVT, 0, ldvt, outU, 0, ldu, null, 0, 0, work, 0);
        } else {
            if (wantVT) {
                dlacpyUpper(m, n, A, 0, lda, outVT, 0, n);
                if (wantFullV && n > m) {
                    for (int i = m; i < n; i++) {
                        for (int j = 0; j < n; j++) {
                            outVT[i * n + j] = 0.0;
                        }
                    }
                }
                int vtRows = wantFullV ? n : minMN;
                BLAS.dorgbr('P', vtRows, n, m, outVT, 0, n, tauP, tauPOff, work, 0, work.length);
            }

            if (wantU) {
                int uCols = wantFullU ? m : minMN;
                dlacpyLower(m, m, A, 0, lda, outU, 0, uCols);
                BLAS.dorgbr('Q', m, uCols, m, outU, 0, uCols, tauQ, tauQOff, work, 0, work.length);
            }

            int ncvt = wantVT ? n : 0;
            int nru = wantU ? m : 0;
            int ldvt = wantVT ? n : 1;
            int ldu = wantU ? (wantFullU ? m : minMN) : 1;
            ok = BLAS.dbdsqr(BLAS.Uplo.Lower, minMN, ncvt, nru, 0, outS, 0, e, eOff, outVT, 0, ldvt, outU, 0, ldu, null, 0, 0, work, 0);
        }

        // Record the arrays actually written (user array or pool array)
        this.S  = outS;
        this.U  = outU;
        this.VT = outVT;
    }

    private static void dlacpyUpper(int m, int n, double[] src, int srcOff, int lda, double[] dst, int dstOff, int ldb) {
        for (int i = 0; i < m; i++) {
            for (int j = i; j < n; j++) {
                dst[dstOff + i * ldb + j] = src[srcOff + i * lda + j];
            }
        }
    }

    private static void dlacpyLower(int m, int n, double[] src, int srcOff, int lda, double[] dst, int dstOff, int ldb) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < min(i + 1, n); j++) {
                dst[dstOff + i * ldb + j] = src[srcOff + i * lda + j];
            }
        }
    }

    public double[] singularValues() {
        return S;
    }

    public double[] U() {
        return U;
    }

    public double[] VT() {
        return VT;
    }

    @Override
    public Matrix extract(Part part) {
        return extract(part, null);
    }

    @Override
    public Matrix extract(Part part, double[] dst) {
        if (!ok) return null;
        switch (part) {
            case S: {
                int len = min(m, n);
                if (S == null) return null;
                if (dst == null) {
                    return new Matrix(len, 1, false, S);
                }
                if (dst.length < len) throw new IllegalArgumentException("dst.length < " + len);
                System.arraycopy(S, 0, dst, 0, len);
                return new Matrix(len, 1, false, dst);
            }
            case U: {
                if (U == null) return null;
                int cols = uCols();
                int size = m * cols;
                if (dst == null) dst = new double[size];
                else if (dst.length < size) throw new IllegalArgumentException("dst.length < " + size);
                System.arraycopy(U, 0, dst, 0, size);
                return new Matrix(m, cols, false, dst);
            }
            case V: {
                if (VT == null) return null;
                int rows = vtRows();
                int size = n * rows;
                if (dst == null) dst = new double[size];
                else if (dst.length < size) throw new IllegalArgumentException("dst.length < " + size);
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < n; j++) {
                        dst[j * rows + i] = VT[i * n + j];
                    }
                }
                return new Matrix(n, rows, false, dst);
            }
            default:
                return null;
        }
    }

    public int kind() {
        return ok ? kind : -1;
    }

    public int uCols() {
        if (U == null) return 0;
        boolean wantFullU = (kind & SVD_FULL_U) != 0;
        return wantFullU ? m : min(m, n);
    }

    public int vtRows() {
        if (VT == null) return 0;
        boolean wantFullV = (kind & SVD_FULL_V) != 0;
        return wantFullV ? n : min(m, n);
    }

    @Override
    public boolean ok() {
        return ok;
    }

    @Override
    public Workspace work() {
        return pool;
    }

    @Override
    public int rows(Part part) {
        if (!ok) throw new IllegalStateException("Decomposition failed");
        switch (part) {
            case U:
                return m;
            case S:
                return min(m, n);
            case V:
                return min(m, n);
            default:
                throw new UnsupportedOperationException("Part " + part + " not supported");
        }
    }

    @Override
    public int cols(Part part) {
        if (!ok) throw new IllegalStateException("Decomposition failed");
        switch (part) {
            case U:
                return uCols();
            case S:
                return 1;
            case V:
                return n;
            default:
                throw new UnsupportedOperationException("Part " + part + " not supported");
        }
    }

    public int m() {
        return m;
    }

    public int n() {
        return n;
    }

    public int rank() {
        if (S == null || S.length == 0) return 0;
        double tol = Math.max(m, n) * Math.ulp(S[0]);
        return rank(tol);
    }

    public int rank(double tol) {
        if (S == null) return 0;
        int r = 0;
        for (int i = 0; i < S.length; i++) {
            if (S[i] > tol) r++;
        }
        return r;
    }

    public double cond() {
        if (S == null || S.length == 0) return Double.NaN;
        double sMax = S[0];
        double sMin = S[S.length - 1];
        if (sMin <= 0) return Double.POSITIVE_INFINITY;
        return sMax / sMin;
    }

    public double norm2() {
        if (S == null || S.length == 0) return Double.NaN;
        return S[0];
    }

    public double[] solve(double[] b, double[] x) {
        return solve(b, x, null);
    }

    public double[] solve(double[] b, double[] x, double[] tmp) {
        if (S == null || U == null || VT == null) {
            return null;
        }

        int minMN = min(m, n);
        int uCols = uCols();
        double tol = EPSILON * S[0] * max(m, n);

        if (tmp == null || tmp.length < minMN) {
            tmp = new double[minMN];
        }

        for (int i = 0; i < minMN; i++) {
            if (S[i] > tol) {
                double sum = 0;
                for (int j = 0; j < m; j++) {
                    sum += U[j * uCols + i] * b[j];
                }
                tmp[i] = sum / S[i];
            } else {
                tmp[i] = 0;
            }
        }

        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < minMN; j++) {
                sum += VT[j * n + i] * tmp[j];
            }
            x[i] = sum;
        }

        return x;
    }

    private static final double EPSILON = 0x1.0p-52;
}
