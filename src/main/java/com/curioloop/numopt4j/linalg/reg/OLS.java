/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.reg;

import com.curioloop.numopt4j.linalg.Regression;
import com.curioloop.numopt4j.linalg.blas.BLAS;

import static java.lang.Math.*;

/**
 * Ordinary Least Squares (OLS) linear regression.
 *
 * <p>Solves the linear model:
 * <pre>  y = Xβ + ε,  ε ~ N(0, σ²I)</pre>
 *
 * <p>Parameter estimation:
 * <pre>  β̂ = (XᵀX)⁻¹Xᵀy</pre>
 *
 * <p>Two solvers are supported:
 * <ul>
 *   <li>SVD — via pseudoinverse X⁺ = VΣ⁺Uᵀ (numerically robust, handles rank-deficient X)</li>
 *   <li>QR  — via QR factorization X = QR, then β̂ = R⁻¹Qᵀy (faster when X is full rank)</li>
 * </ul>
 *
 * <p>Data layout: X is row-major n×k, each row is one observation.
 * <b>X is overwritten in-place by the solver.</b> y is never modified.
 * Fitted values and residuals are computed inside the solver and cached in {@link Pool}.
 */
public class OLS extends Regression {

    final int nObs;
    final int nParams;
    int kConst;           // resolved lazily in fit(); -1 = not yet computed
    final double[] y;
    final double[] X;
    final boolean useQR;

    @Override public    int      nObs()       { return nObs; }
    @Override public    int      nParams()    { return nParams; }
    @Override public    int      kConst()     { return kConst; }
    @Override protected int      dfModel()    { return rank - kConst; }
    @Override protected int      dfResidual() { return nObs - rank; }
    @Override public    double[] endog()      { return y; }
    @Override public    double[] weights()    { return null; }

    // =========================================================================
    // Pool
    // =========================================================================

    /**
     * Reusable workspace for OLS computations.
     *
     * <p>All buffers grow on demand and are reused across fits.
     * No per-fit heap allocations after the first call.
     *
     * <ul>
     *   <li>{@code work}        — LAPACK floating-point scratch</li>
     *   <li>{@code ipiv}        — pivot indices (QR path)</li>
     *   <li>{@code beta}        — β̂ output, length k</li>
     *   <li>{@code unscaledCov} — (XᵀX)⁻¹, length k×k</li>
     *   <li>{@code fitted}      — ŷ = Xβ̂, length n (computed inside solver)</li>
     *   <li>{@code residual}    — e = y − ŷ, length n (computed inside solver)</li>
     * </ul>
     */
    public static class Pool {

        public double[] work;        // LAPACK floating-point scratch
        public int[]    ipiv;        // pivot indices for dgetrf/dgetri (QR path)
        public double[] beta;        // pre-allocated β̂ output (length k)
        public double[] unscaledCov; // pre-allocated (XᵀX)⁻¹ output (length k×k)
        public double[] fitted;      // ŷ = Xβ̂, length n
        public double[] residual;    // e = y − ŷ, length n

        public Pool() {}

        /**
         * Pre-allocates all buffers. Safe to call multiple times; buffers only grow.
         */
        public Pool ensure(int n, int k) {
            if (beta == null || beta.length < k)                  beta = new double[k];
            if (unscaledCov == null || unscaledCov.length < k*k)  unscaledCov = new double[k*k];
            if (fitted   == null || fitted.length   < n) fitted   = new double[n];
            if (residual == null || residual.length < n) residual = new double[n];
            return this;
        }

        Pool ensureWork(int size) {
            if (work == null || work.length < size) work = new double[size];
            return this;
        }

        Pool ensureIpiv(int size) {
            if (ipiv == null || ipiv.length < size) ipiv = new int[size];
            return this;
        }
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    public OLS(double[] y, double[] X, int n, int k, boolean useQR, boolean noConst) {
        if (n < 1 || k < 1) throw new IllegalArgumentException("n and k must be >= 1");
        if (y.length < n)   throw new IllegalArgumentException("y too short");
        if (X.length < n*k) throw new IllegalArgumentException("X too short");
        this.nObs    = n;
        this.nParams = k;
        this.y       = y;
        this.X       = X;
        this.useQR   = useQR;
        this.kConst  = noConst ? 0 : -1;
    }

    public OLS(double[] y, double[] X, int n, int k, boolean useQR) {
        this(y, X, n, k, useQR, false);
    }

    public OLS(double[] y, double[] X, int n, int k) {
        this(y, X, n, k, false, false);
    }

    // =========================================================================
    // fit
    // =========================================================================

    public OLS fit() {
        return fit(new Pool());
    }

    /**
     * Fits the model.
     *
     * <p><b>X is overwritten in-place.</b> The solver operates directly on X,
     * computes fitted values and residuals internally, and caches them in {@code ws}.
     */
    public OLS fit(Pool ws) {
        if (ws == null) return fit();
        ws.ensure(nObs, nParams);
        if (kConst < 0) kConst = detectConst(X, nObs, nParams, ws);
        if (useQR) solveQR(y, X, ws);
        else       solveSVD(y, X, ws);
        return this;
    }

    // =========================================================================
    // SVD solver
    // =========================================================================
    //
    // Decompose X = UΣVᵀ, then:
    //   β̂          = VΣ⁺Uᵀy
    //   fitted      = U·(Uᵀy)  (projection onto column space of X)
    //   unscaledCov = VΣ⁺²Vᵀ  (= (XᵀX)⁻¹ when full rank)
    //
    // Memory layout in ws.work:
    //   [lwork | U:n×r | VT:r×k | S:r | tmp:r]

    void solveSVD(double[] endo, double[] exog, Pool ws) {
        int n = nObs, k = nParams;
        int r = min(n, k);

        double[] wq = new double[1];
        BLAS.dgesvd('S', 'S', n, k, exog, 0, k, wq, 0, null, 0, r, null, 0, k, wq, 0, -1);
        int lwork = (int) wq[0];

        // work layout: [lwork | U:n×r | VT:r×k | S:r | tmp:r]
        int offU = lwork, offVT = lwork + n*r, offS = lwork + n*r + r*k, offTmp = lwork + n*r + r*k + r;
        ws.ensureWork(lwork + n*r + r*k + r + r);

        BLAS.dgesvd('S', 'S', n, k, exog, 0, k,
                    ws.work, offS, ws.work, offU, r, ws.work, offVT, k,
                    ws.work, 0, lwork);

        // rank / cond from Σ
        double tol = ws.work[offS] * 0x1p-53;
        rank = 0;
        for (int i = 0; i < r; i++) if (ws.work[offS + i] > tol) rank++;
        cond = sqrt((ws.work[offS] * ws.work[offS]) / (ws.work[offS + r - 1] * ws.work[offS + r - 1]));

        // tmp = Uᵀy  (used for both fitted and β̂)
        BLAS.dgemv(BLAS.Trans.Trans, n, r, 1.0, ws.work, offU, r, endo, 0, 1, 0.0, ws.work, offTmp, 1);

        // fitted = U·tmp,  residual = endo - fitted
        // zero out components beyond rank before projecting
        for (int i = rank; i < r; i++) ws.work[offTmp + i] = 0.0;
        BLAS.dgemv(BLAS.Trans.NoTrans, n, r, 1.0, ws.work, offU, r, ws.work, offTmp, 1, 0.0, ws.fitted, 0, 1);
        System.arraycopy(endo, 0, ws.residual, 0, n);
        BLAS.daxpy(n, -1.0, ws.fitted, 0, 1, ws.residual, 0, 1);
        setFittedResidualCache(ws);

        // Σ⁺: invert non-zero singular values (recompute after zeroing rank-truncated tmp)
        double cutoff = ws.work[offS] * 1e-15;
        for (int i = 0; i < r; i++)
            ws.work[offS + i] = (ws.work[offS + i] > cutoff) ? 1.0 / ws.work[offS + i] : 0.0;

        // β̂ = Vᵀᵀ(Σ⁺·(Uᵀy))
        // restore tmp = Uᵀy (re-read from endo since we zeroed rank-truncated entries)
        BLAS.dgemv(BLAS.Trans.Trans, n, r, 1.0, ws.work, offU, r, endo, 0, 1, 0.0, ws.work, offTmp, 1);
        for (int i = 0; i < r; i++) ws.work[offTmp + i] *= ws.work[offS + i];
        beta = ws.beta;
        BLAS.dgemv(BLAS.Trans.Trans, r, k, 1.0, ws.work, offVT, k, ws.work, offTmp, 1, 0.0, beta, 0, 1);

        // unscaledCov = (VΣ⁺)ᵀ(VΣ⁺) via dsyrk
        unscaledCov = ws.unscaledCov;
        for (int i = 0; i < r; i++) {
            double si = ws.work[offS + i];
            for (int j = 0; j < k; j++) ws.work[offVT + i*k + j] *= si;
        }
        BLAS.dsyrk(BLAS.Uplo.Upper, BLAS.Trans.Trans, k, r,
                   1.0, ws.work, offVT, k, 0.0, unscaledCov, 0, k);
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++) unscaledCov[j*k + i] = unscaledCov[i*k + j];
    }

    // =========================================================================
    // QR solver
    // =========================================================================
    //
    // Decompose X = QR, then:
    //   β̂          = R⁻¹Qᵀy
    //   fitted      = Q·(Qᵀy)  (projection onto column space of X)
    //   unscaledCov = (RᵀR)⁻¹
    //
    // Memory layout in ws.work:
    //   [lwork | R:k×k | tau:t]

    void solveQR(double[] endo, double[] exog, Pool ws) {
        int n = nObs, k = nParams;
        int t = min(n, k);

        double[] wq = new double[1];
        BLAS.dgeqrf(n, k, exog, 0, k, wq, 0, wq, 0, -1);
        int lwork = (int) wq[0];
        BLAS.dorgqr(n, k, k, exog, 0, k, wq, 0, wq, 0, -1);
        lwork = max(lwork, max((int) wq[0], k));

        // work layout: [lwork | R:k×k | tau:t]
        int offR = lwork, offTau = lwork + k*k;
        ws.ensureWork(lwork + k*k + t);
        ws.ensureIpiv(k);

        // Step 1: QR factorization
        BLAS.dgeqrf(n, k, exog, 0, k, ws.work, offTau, ws.work, 0, lwork);

        // Step 2: save R
        java.util.Arrays.fill(ws.work, offR, offR + k*k, 0.0);
        for (int i = 0; i < k; i++)
            System.arraycopy(exog, i*k + i, ws.work, offR + i*k + i, k - i);

        // Step 3: expand Q in-place (exog now holds Q)
        BLAS.dorgqr(n, k, k, exog, 0, k, ws.work, offTau, ws.work, 0, lwork);

        // Step 4: Qᵀy — write into beta as temp
        beta = ws.beta;
        BLAS.dgemv(BLAS.Trans.Trans, n, k, 1.0, exog, 0, k, endo, 0, 1, 0.0, beta, 0, 1);

        // Step 5: fitted = Q·(Qᵀy),  residual = endo - fitted
        BLAS.dgemv(BLAS.Trans.NoTrans, n, k, 1.0, exog, 0, k, beta, 0, 1, 0.0, ws.fitted, 0, 1);
        System.arraycopy(endo, 0, ws.residual, 0, n);
        BLAS.daxpy(n, -1.0, ws.fitted, 0, 1, ws.residual, 0, 1);
        setFittedResidualCache(ws);

        // Step 6: β̂ = R⁻¹(Qᵀy)
        BLAS.dtrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, k, 1,
                    ws.work, offR, k, beta, 0, 1);

        // Step 7: rank/cond via SVD of R — use ws.unscaledCov as temp
        unscaledCov = ws.unscaledCov;
        java.util.Arrays.fill(unscaledCov, 0, k*k, 0.0);
        for (int i = 0; i < k; i++)
            System.arraycopy(ws.work, offR + i*k + i, unscaledCov, i*k + i, k - i);
        BLAS.dgesvd('N', 'N', k, k, unscaledCov, 0, k, ws.work, 0,
                    null, 0, 1, null, 0, 1, ws.work, k, max(lwork - k, svdWorkSize(k, k)));
        rank = 0;
        for (int i = 0; i < k; i++) if (ws.work[i] > ws.work[0] * 0x1p-53) rank++;
        cond = sqrt((ws.work[0] * ws.work[0]) / (ws.work[k - 1] * ws.work[k - 1]));

        // Step 8: unscaledCov = (RᵀR)⁻¹ via LU
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, k, k, k,
                   1.0, ws.work, offR, k, ws.work, offR, k, 0.0, unscaledCov, 0, k);
        BLAS.dgetrf(k, k, unscaledCov, 0, k, ws.ipiv, 0);
        BLAS.dgetri(k, unscaledCov, 0, k, ws.ipiv, 0, ws.work, 0, lwork);
    }

    // =========================================================================
    // Constant detection
    // =========================================================================

    /**
     * Detects whether X contains a constant term (intercept column).
     */
    static int detectConst(double[] X, int n, int k, Pool ws) {
        // Fast path: single-pass O(nk) scan.
        // reuse work[0..k) as base — fast path does no LAPACK so work[0..k) is free
        ws.ensureWork(k);
        double[] base = ws.work;
        System.arraycopy(X, 0, base, 0, k);
        int remaining = k;
        for (int i = 1; i < n && remaining > 0; i++) {
            for (int j = 0; j < k; j++) {
                if (!Double.isNaN(base[j]) && X[i*k + j] != base[j]) {
                    base[j] = Double.NaN;
                    remaining--;
                }
            }
        }
        int constCount = 0;
        boolean hasExplicitOne = false;
        for (int j = 0; j < k; j++) {
            if (!Double.isNaN(base[j])) {
                constCount++;
                if (base[j] == 1.0) hasExplicitOne = true;
            }
        }
        if (constCount == 1) return 1;
        if (constCount > 1 && hasExplicitOne) return 1;

        // Slow path: compare rank(X) vs rank([1|X])
        // work layout: [ lwork | S: min(n,k1) | A: n×k1 ]
        int k1 = k + 1;
        int lwork = svdWorkSize(n, k1);
        int offS  = lwork;
        int offA  = lwork + min(n, k1);
        ws.ensureWork(offA + n * k1);

        // first call: rank(X) — copy X into work[offA], padded to k1 columns
        for (int i = 0; i < n; i++)
            System.arraycopy(X, i * k, ws.work, offA + i * k, k);
        int orgRank = numericalRank(ws.work, offA, n, k, offS, lwork, ws);

        // second call: rank([1|X]) — prepend ones column
        for (int i = 0; i < n; i++) {
            ws.work[offA + i * k1] = 1.0;
            System.arraycopy(X, i * k, ws.work, offA + i * k1 + 1, k);
        }
        int augRank = numericalRank(ws.work, offA, n, k1, offS, lwork, ws);

        return augRank == orgRank ? 1 : 0;
    }

    /** Returns the LAPACK lwork estimate for dgesvd('N','N', m, n). */
    static int svdWorkSize(int m, int n) {
        double[] wq = new double[1];
        BLAS.dgesvd('N', 'N', m, n, null, 0, n, wq, 0, null, 0, 1, null, 0, 1, wq, 0, -1);
        return (int) wq[0];
    }

    /** Computes numerical rank of A (m×n) stored in ws.work[offA..], using ws.work[offS..] for singular values. */
    private static int numericalRank(double[] work, int offA, int m, int n, int offS, int lwork, Pool ws) {
        int minmn = min(m, n);
        BLAS.dgesvd('N', 'N', m, n, work, offA, n,
                    work, offS, null, 0, 1, null, 0, 1,
                    work, 0, lwork);
        int rank = 0;
        for (int i = 0; i < minmn; i++) if (work[offS + i] > work[offS] * 0x1p-53) rank++;
        return rank;
    }
}
