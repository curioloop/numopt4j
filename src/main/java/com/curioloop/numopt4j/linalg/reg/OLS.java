/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.reg;

import com.curioloop.numopt4j.linalg.Regressor;
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
    @Override public    double[] endog()       { return y; }
    @Override public    double[] exog()       { return X; }
    @Override public    double[] weights()    { return null; }

    public OLS(double[] y, double[] X, int n, int k, boolean useQR, boolean noConst) {
        if (n < 1 || k < 1) throw new IllegalArgumentException("n and k must be >= 1");
        if (y.length < n)   throw new IllegalArgumentException("y too short");
        if (X.length < n*k) throw new IllegalArgumentException("X too short");
        this.nObs    = n;
        this.nParams    = k;
        this.y       = y;
        this.X       = X;
        this.useQR   = useQR;
        this.kConst  = noConst ? 0 : -1; // resolved lazily in fit()
    }

    public OLS(double[] y, double[] X, int n, int k, boolean useQR) {
        this(y, X, n, k, useQR, false);
    }

    public OLS(double[] y, double[] X, int n, int k) {
        this(y, X, n, k, false, false);
    }

    public OLS fit() {
        return fit(new Regressor.Pool());
    }

    public OLS fit(Regressor.Pool ws) {
        if (ws == null) return fit();
        int n = nObs, k = nParams;
        ws.ensureData(n, n * k);
        // Resolve kConst lazily, reusing ws.xCopy as scratch for SVD if needed.
        if (kConst < 0) kConst = detectConst(X, n, k, ws);
        System.arraycopy(y, 0, ws.yCopy, 0, n);
        System.arraycopy(X, 0, ws.xCopy, 0, n * k);
        if (useQR) solveQR(ws.yCopy, ws.xCopy, ws);
        else       solveSVD(ws.yCopy, ws.xCopy, ws);
        return this;
    }


    // ==================== SVD solver ====================
    //
    // Decompose X = UΣVᵀ, then:
    //   β̂         = X⁺y = VΣ⁺Uᵀy
    //   unscaledCov = X⁺X⁺ᵀ = (VΣ⁺)(VΣ⁺)ᵀ = VΣ⁺²Vᵀ  (= (XᵀX)⁻¹ when full rank)

    void solveSVD(double[] endo, double[] exog, Regressor.Pool ws) {
        int n = nObs, k = nParams;
        beta    = new double[k];
        unscaledCov = new double[k * k];
        int r = min(n, k);

        double[] wq = new double[1];
        BLAS.dgesvd('S', 'S', n, k, exog, 0, k, wq, 0, null, 0, r, null, 0, k, wq, 0, -1);
        int lwork = (int) wq[0];

        // work layout: [lwork | U:n×r | VT:r×k | S:r | tmp:r]
        int offU = lwork, offVT = lwork + n*r, offS = lwork + n*r + r*k, offTmp = lwork + n*r + r*k + r;
        ws.ensureWork(lwork + n*r + r*k + r + r);

        // Decompose X = UΣVᵀ
        BLAS.dgesvd('S', 'S', n, k, exog, 0, k,
                    ws.work, offS, ws.work, offU, r, ws.work, offVT, k,
                    ws.work, 0, lwork);

        // Rank from Σ, condition number σ_max/σ_min
        double tol = ws.work[offS] * 0x1p-53;
        rank = 0;
        for (int i = 0; i < r; i++) if (ws.work[offS + i] > tol) rank++;
        cond = sqrt((ws.work[offS] * ws.work[offS]) / (ws.work[offS + r - 1] * ws.work[offS + r - 1]));

        // Compute Σ⁺: invert non-zero singular values
        double cutoff = ws.work[offS] * 1e-15;
        for (int i = 0; i < r; i++)
            ws.work[offS + i] = (ws.work[offS + i] > cutoff) ? 1.0 / ws.work[offS + i] : 0.0;

        // β̂ = Vᵀᵀ(Σ⁺(Uᵀy))
        BLAS.dgemv(BLAS.Trans.Trans, n, r, 1.0, ws.work, offU, r, endo, 0, 1, 0.0, ws.work, offTmp, 1);
        for (int i = 0; i < r; i++) ws.work[offTmp + i] *= ws.work[offS + i];
        BLAS.dgemv(BLAS.Trans.Trans, r, k, 1.0, ws.work, offVT, k, ws.work, offTmp, 1, 0.0, beta, 0, 1);

        // normCov = (VΣ⁺)ᵀ(VΣ⁺)  via dsyrk on scaled Vᵀ rows
        for (int i = 0; i < r; i++) {
            double si = ws.work[offS + i];
            for (int j = 0; j < k; j++) ws.work[offVT + i*k + j] *= si;
        }
        BLAS.dsyrk(BLAS.Uplo.Upper, BLAS.Trans.Trans, k, r,
                   1.0, ws.work, offVT, k, 0.0, unscaledCov, 0, k);
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++) unscaledCov[j*k + i] = unscaledCov[i*k + j];
    }

    // ==================== QR solver ====================
    //
    // Decompose X = QR, then:
    //   β̂         = R⁻¹Qᵀy
    //   unscaledCov = (RᵀR)⁻¹  (= (XᵀX)⁻¹ when full rank)
    //
    // Steps:
    //   1. dgeqrf  → R in upper triangle of exog
    //   2. copy R  → work[offR]  (before dorgqr overwrites exog)
    //   3. dorgqr  → exog becomes Q;  work[offR] still holds R
    //   4. dtrtrs  → β̂ = R⁻¹(Qᵀy)
    //   5. SVD(R)  → rank/cond  (R copied into unscaledCov[], SVD overwrites unscaledCov[])
    //   6. re-copy R from work[offR], compute unscaledCov = (RᵀR)⁻¹ via LU
    //
    // work[offR] is never touched by dgeqrf/dorgqr/dgetri scratch (all use work[0..lwork),
    // and offR = lwork).

    void solveQR(double[] endo, double[] exog, Regressor.Pool ws) {
        int n = nObs, k = nParams;
        beta    = new double[k];
        unscaledCov = new double[k * k];
        int t = min(n, k); // tau length for Householder reflectors

        // Query lwork for dgeqrf and dorgqr; need >= k for dgetri
        double[] wq = new double[1];
        BLAS.dgeqrf(n, k, exog, 0, k, wq, 0, wq, 0, -1);
        int lwork = (int) wq[0];
        BLAS.dorgqr(n, k, k, exog, 0, k, wq, 0, wq, 0, -1);
        lwork = max(lwork, max((int) wq[0], k));

        // work layout: [lwork | R:k×k | tau:t]
        // offR = lwork, so scratch work[0..lwork) never overlaps R
        int offR = lwork, offTau = lwork + k*k;
        ws.ensureWork(lwork + k*k + t);
        ws.ensureIpiv(k);

        // Step 1: X = QR factorization
        BLAS.dgeqrf(n, k, exog, 0, k, ws.work, offTau, ws.work, 0, lwork);

        // Step 2: Save R into work[offR] before dorgqr overwrites exog
        java.util.Arrays.fill(ws.work, offR, offR + k*k, 0.0);
        for (int i = 0; i < k; i++)
            System.arraycopy(exog, i*k + i, ws.work, offR + i*k + i, k - i);

        // Step 3: Expand Q in-place (exog = Q now;  work[offR] = R intact)
        BLAS.dorgqr(n, k, k, exog, 0, k, ws.work, offTau, ws.work, 0, lwork);

        // Step 4: β̂ = R⁻¹(Qᵀy)
        BLAS.dgemv(BLAS.Trans.Trans, n, k, 1.0, exog, 0, k, endo, 0, 1, 0.0, beta, 0, 1);
        BLAS.dtrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, k, 1,
                    ws.work, offR, k, beta, 0, 1);

        // Step 5: rank/cond via SVD of R
        // Copy R into normCov[], run SVD — singular values go into work[0..k)
        // (safe to reuse scratch since offR = lwork >= k).
        // SVD overwrites normCov[] with garbage; recomputed in step 6.
        java.util.Arrays.fill(unscaledCov, 0.0);
        for (int i = 0; i < k; i++)
            System.arraycopy(ws.work, offR + i*k + i, unscaledCov, i*k + i, k - i);
        BLAS.dgesvd('N', 'N', k, k, unscaledCov, 0, k, ws.work, 0,
                    null, 0, 1, null, 0, 1, ws.work, k, lwork - k);
        rank = 0;
        for (int i = 0; i < k; i++) if (ws.work[i] > ws.work[0] * 0x1p-53) rank++;
        cond = sqrt((ws.work[0] * ws.work[0]) / (ws.work[k - 1] * ws.work[k - 1]));

        // Step 6: normCov = (RᵀR)⁻¹ via LU  (work[offR] still holds R)
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, k, k, k,
                   1.0, ws.work, offR, k, ws.work, offR, k, 0.0, unscaledCov, 0, k);
        BLAS.dgetrf(k, k, unscaledCov, 0, k, ws.ipiv, 0);
        BLAS.dgetri(k, unscaledCov, 0, k, ws.ipiv, 0, ws.work, 0, lwork);
    }


    // ==================== constant detection ====================

    /**
     * Detects whether X contains a constant term (intercept column).
     *
     * <p>Fast path: single-pass O(nk) scan with one array — each column's first
     * value is used as baseline; marked NaN if any later value differs.
     * Returns immediately when a single explicit constant column is found.
     *
     * <p>Slow path: when the scan is ambiguous (no constant column, or multiple
     * constant columns with no explicit 1), checks for an implicit constant by
     * comparing rank(X) vs rank([1|X]) via SVD.
     * Reuses ws.xCopy and ws.work as scratch (zero extra allocation).
     */
    static int detectConst(double[] X, int n, int k, Regressor.Pool ws) {
        // O(nk) single-pass scan — one array, NaN marks non-constant columns
        double[] base = new double[k];
        System.arraycopy(X, 0, base, 0, k); // first row as baseline
        int remaining = k; // columns not yet marked NaN
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

        // Ambiguous — compare rank(X) vs rank([1|X]) via SVD
        int k1 = k + 1;
        ws.ensureData(n, n * k1);
        double[] buf = ws.xCopy;

        // pass 1: rank of X (n×k)
        System.arraycopy(X, 0, buf, 0, n * k);
        int orgRank = numericalRank(buf, n, k, ws);

        // pass 2: rank of [1|X] (n×(k+1))
        for (int i = 0; i < n; i++) {
            buf[i * k1] = 1.0;
            System.arraycopy(X, i * k, buf, i * k1 + 1, k);
        }
        int augRank = numericalRank(buf, n, k1, ws);

        // rank([1|X]) == rank(X)  ⟹  1 is in col-span of X  ⟹  implicit constant
        return augRank == orgRank ? 1 : 0;
    }

    /**
     * Computes numerical rank of A (m×n) via SVD.
     * A is overwritten. Reuses ws.work as scratch.
     */
    private static int numericalRank(double[] A, int m, int n, Regressor.Pool ws) {
        int minmn = min(m, n);
        double[] wq = new double[1];
        BLAS.dgesvd('N', 'N', m, n, A, 0, n, wq, 0, null, 0, 1, null, 0, 1, wq, 0, -1);
        int lwork = (int) wq[0];
        // work layout: [lwork | S:minmn]
        int offS = lwork;
        ws.ensureWork(lwork + minmn);
        BLAS.dgesvd('N', 'N', m, n, A, 0, n,
                    ws.work, offS, null, 0, 1, null, 0, 1,
                    ws.work, 0, lwork);
        int rank = 0;
        for (int i = 0; i < minmn; i++) if (ws.work[offS + i] > ws.work[offS] * 0x1p-53) rank++;
        return rank;
    }
}
