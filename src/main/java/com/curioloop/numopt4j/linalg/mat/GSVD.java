/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;

import static java.lang.Math.*;

public final class GSVD implements Decomposition {

    public static final int GSVD_NONE = 0;
    public static final int GSVD_U = 1;
    public static final int GSVD_V = 2;
    public static final int GSVD_Q = 4;
    public static final int GSVD_ALL = GSVD_U | GSVD_V | GSVD_Q;

    /**
     * Pool extends Workspace with result arrays for alpha, beta, sigma, U, V, Q.
     *
     * <p>Work layout: {@code work[0..n)} stores {@code tau} for {@code dggsvp3};
     * {@code work[n..)} is the algorithm scratch buffer.  The two regions never
     * overlap, so a single array suffices and no separate {@code tau} field is needed.
     * All callers pass {@code work} as the tau argument and {@code workOff=n} as the
     * scratch offset.
     */
    public static final class Pool extends Workspace {

        public double[] alpha;
        public double[] beta;
        public double[] sigma;
        public double[] U;
        public double[] V;
        public double[] Q;

        private Pool() {}

        public Pool ensure(int m, int p, int n, int kind) {
            if (alpha == null || alpha.length < n) alpha = new double[n];
            if (beta  == null || beta.length  < n) beta  = new double[n];

            boolean wantU = (kind & GSVD_U) != 0;
            boolean wantV = (kind & GSVD_V) != 0;
            boolean wantQ = (kind & GSVD_Q) != 0;

            if (wantU && (U == null || U.length < m * m)) U = new double[m * m];
            if (wantV && (V == null || V.length < p * p)) V = new double[p * p];
            if (wantQ && (Q == null || Q.length < n * n)) Q = new double[n * n];

            // work[0..n) = tau region, work[n..) = scratch; total = n + scratch
            ensureWork(n + 2 * n + max(max(n, m), p) + 3);
            ensureIwork(n);
            return this;
        }
    }

    // -------------------------------------------------------------------------

    private Pool pool;
    private int m;
    private int p;
    private int n;
    private int k;
    private int l;
    private int kind;
    private boolean ok;

    private GSVD() {}

    public static Workspace workspace(int m, int p, int n) {
        return new Pool().ensure(m, p, n, GSVD_ALL);
    }

    public static GSVD decompose(double[] A, int m, int n, double[] B, int p) {
        return decompose(A, m, n, B, p, GSVD_ALL, null);
    }

    public static GSVD decompose(double[] A, int m, int n, double[] B, int p, int kind) {
        return decompose(A, m, n, B, p, kind, null);
    }

    public static GSVD decompose(double[] A, int m, int n, double[] B, int p, int kind, Workspace ws) {
        GSVD gsvd = new GSVD();
        gsvd.m = m;
        gsvd.p = p;
        gsvd.n = n;
        gsvd.kind = kind;
        gsvd.doDecompose(A, m, n, B, p, kind, ws);
        return gsvd;
    }

    private void doDecompose(double[] A, int m, int n, double[] B, int p, int kind, Workspace ws) {

        boolean wantU = (kind & GSVD_U) != 0;
        boolean wantV = (kind & GSVD_V) != 0;
        boolean wantQ = (kind & GSVD_Q) != 0;

        BLAS.GsvdJob jobU = wantU ? BLAS.GsvdJob.Compute : BLAS.GsvdJob.None;
        BLAS.GsvdJob jobV = wantV ? BLAS.GsvdJob.Compute : BLAS.GsvdJob.None;
        BLAS.GsvdJob jobQ = wantQ ? BLAS.GsvdJob.Compute : BLAS.GsvdJob.None;

        if (ws == null) {
            ws = workspace(m, p, n);
        }
        if (!(ws instanceof Pool)) {
            throw new IllegalArgumentException("Workspace must be an instance of GSVD.Pool");
        }
        this.pool = (Pool) ws;
        this.pool.ensure(m, p, n, kind);

        double anorm = BLAS.dlange('F', m, n, A, 0, n);
        double bnorm = BLAS.dlange('F', p, n, B, 0, n);

        double eps    = BLAS.eps();
        double safmin = BLAS.safmin();
        double tola = max(m, n) * max(anorm, safmin) * eps;
        double tolb = max(p, n) * max(bnorm, safmin) * eps;

        // Resolve U/V/Q arrays (null when not requested)
        double[] U = wantU ? pool.U : null;
        double[] V = wantV ? pool.V : null;
        double[] Q = wantQ ? pool.Q : null;

        double[] work = pool.work();
        // work[0..n) = tau region; work[n..) = scratch
        int scratchOff = n;
        int lwork = work.length - scratchOff;

        int[] kl = BLAS.dggsvp3(jobU, jobV, jobQ, m, p, n,
                                 A, 0, n, B, 0, n,
                                 tola, tolb,
                                 U, 0, m, V, 0, p, Q, 0, n,
                                 pool.iwork(), work, work, scratchOff, lwork);

        k = kl[0];
        l = kl[1];

        ok = BLAS.dtgsja(jobU, jobV, jobQ, m, p, n, k, l,
                         A, 0, n, B, 0, n,
                         tola, tolb,
                         pool.alpha, 0, pool.beta, 0,
                         U, 0, m, V, 0, p, Q, 0, n,
                         work, scratchOff);

        if (ok) {
            int kl_sum = k + l;
            if (pool.sigma == null || pool.sigma.length < kl_sum) {
                pool.sigma = new double[kl_sum];
            }
            for (int i = 0; i < k; i++) pool.sigma[i] = 1.0;
            for (int i = k; i < kl_sum; i++) pool.sigma[i] = pool.alpha[i] / pool.beta[i];
        }
    }

    // -------------------------------------------------------------------------

    public double[] alpha()  { return pool != null ? pool.alpha : null; }
    public double[] beta()   { return pool != null ? pool.beta  : null; }
    public double[] sigma()  { return ok && pool != null ? pool.sigma : null; }
    public double[] U()      { return ok && pool != null ? pool.U : null; }
    public double[] V()      { return ok && pool != null ? pool.V : null; }
    public double[] Q()      { return ok && pool != null ? pool.Q : null; }
    public int k()           { return k; }
    public int l()           { return l; }
    public int m()           { return m; }
    public int p()           { return p; }
    public int n()           { return n; }
    public int kind()        { return ok ? kind : -1; }
    public boolean ok()      { return ok; }
    public int rank()        { return k + l; }

    public double cond() {
        if (pool == null || pool.sigma == null || pool.sigma.length == 0) return Double.NaN;
        double sMax = 0, sMin = Double.POSITIVE_INFINITY;
        for (int i = 0; i < pool.sigma.length; i++) {
            if (pool.sigma[i] > sMax) sMax = pool.sigma[i];
            if (pool.sigma[i] > 0 && pool.sigma[i] < sMin) sMin = pool.sigma[i];
        }
        if (sMin == Double.POSITIVE_INFINITY || sMin == 0) return Double.POSITIVE_INFINITY;
        return sMax / sMin;
    }

    public double[] generalizedSingularValues() {
        if (pool == null || pool.alpha == null || pool.beta == null) return null;
        double[] gsv = new double[k + l];
        for (int i = 0; i < k + l; i++) {
            if (pool.beta[i] != 0) gsv[i] = pool.alpha[i] / pool.beta[i];
            else if (pool.alpha[i] != 0) gsv[i] = Double.POSITIVE_INFINITY;
            else gsv[i] = 0;
        }
        return gsv;
    }

    @Override
    public Workspace work() { return pool; }

    @Override
    public Matrix extract(Part part) {
        return extract(part, null);
    }

    @Override
    public Matrix extract(Part part, double[] dst) {
        if (!ok) return null;
        switch (part) {
            case S: {
                if (pool == null || pool.sigma == null) return null;
                int len = k + l;
                if (dst == null) {
                    return new Matrix(len, 1, false, pool.sigma);
                }
                if (dst.length < len) throw new IllegalArgumentException("dst.length < " + len);
                System.arraycopy(pool.sigma, 0, dst, 0, len);
                return new Matrix(len, 1, false, dst);
            }
            case U: {
                if (pool == null || pool.U == null) return null;
                int size = m * m;
                if (dst == null) dst = new double[size];
                else if (dst.length < size) throw new IllegalArgumentException("dst.length < " + size);
                System.arraycopy(pool.U, 0, dst, 0, size);
                return new Matrix(m, m, false, dst);
            }
            case V: {
                if (pool == null || pool.V == null) return null;
                int size = p * p;
                if (dst == null) dst = new double[size];
                else if (dst.length < size) throw new IllegalArgumentException("dst.length < " + size);
                System.arraycopy(pool.V, 0, dst, 0, size);
                return new Matrix(p, p, false, dst);
            }
            case Q: {
                if (pool == null || pool.Q == null) return null;
                int size = n * n;
                if (dst == null) dst = new double[size];
                else if (dst.length < size) throw new IllegalArgumentException("dst.length < " + size);
                System.arraycopy(pool.Q, 0, dst, 0, size);
                return new Matrix(n, n, false, dst);
            }
            default:
                return null;
        }
    }

    @Override
    public int rows(Part part) {
        if (!ok) throw new IllegalStateException("Decomposition failed");
        switch (part) {
            case Q: return n;
            case U: return m;
            case V: return p;
            case S: return k + l;
            default: throw new UnsupportedOperationException("Part " + part + " not supported");
        }
    }

    @Override
    public int cols(Part part) {
        if (!ok) throw new IllegalStateException("Decomposition failed");
        switch (part) {
            case Q: return n;
            case U: return m;
            case V: return p;
            case S: return 1;
            default: throw new UnsupportedOperationException("Part " + part + " not supported");
        }
    }
}
