/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Trust Region Reflective (TRF) - Bounded Nonlinear Least Squares
 *
 * Solves the problem min ½‖f(x)‖₂² subject to l ≤ x ≤ u.
 *   - f : ℝⁿ → ℝᵐ is a vector-valued residual function
 *   - J = ∂f/∂x ∈ ℝᵐˣⁿ is the Jacobian
 *   - l, u ∈ ℝⁿ are optional lower and upper bounds
 *
 * Algorithm Overview:
 * ------------------
 * At each iteration k, the quadratic model of ½‖f(x)‖₂² at xₖ is:
 *
 *   mₖ(p) = ½‖fₖ‖₂² + pᵀJₖᵀfₖ + ½pᵀJₖᵀJₖp
 *
 * The trust-region subproblem is:
 *
 *   min  mₖ(p)  subject to  ‖Dₖp‖₂ ≤ Δₖ  and  l ≤ xₖ + p ≤ u
 *
 * where Dₖ is the Coleman-Li diagonal scaling matrix.
 *
 * Coleman-Li Scaling:
 * ------------------
 * The diagonal scaling Dₖ = diag(vᵢ) is chosen gradient-aware:
 *
 *   vᵢ = uᵢ - xᵢ   if gᵢ < 0 (moving toward upper bound)
 *   vᵢ = xᵢ - lᵢ   if gᵢ > 0 (moving toward lower bound)
 *   vᵢ = min(xᵢ - lᵢ, uᵢ - xᵢ)  otherwise (at bound or no gradient)
 *
 * This keeps the scaled step ‖Dₖp‖₂ proportional to the feasible range,
 * preventing trust-region collapse near bounds.
 *
 * Reflective Step:
 * ---------------
 * When the unconstrained step p hits a bound at tHit ∈ (0,1], three candidates
 * are evaluated and the one minimizing mₖ is chosen:
 *
 *   1. Trust-region step:  p_tr = θ × tHit × p  (stepped back from bound)
 *   2. Reflective step:    p_ref = p + (tOpt - tHit) × p̃  (reflect off bound)
 *   3. Cauchy step:        p_c = tCauchy × (-Qᵀf)  (1-D minimization along anti-gradient)
 *
 * where θ = max(0.995, 1 - ‖g‖) keeps the step strictly interior.
 *
 * Key Differences from MINPACK lmder (LMCore):
 * --------------------------------------------
 *   1. Coleman-Li scaling: Dᵢ = min(xᵢ - lᵢ, uᵢ - xᵢ) keeps steps away from bounds.
 *   2. Reflective step: when a step hits a bound, reflect off it and minimize the 1-D
 *      quadratic model along the reflected direction.
 *   3. pnorm and prered are computed from the actual (post-reflection) step, so the
 *      trust-region update is self-consistent.
 *   4. Unbounded variables use Dᵢ = 1 (same as MINPACK scaling).
 *
 * References:
 * ----------
 *   Branch, M.A., Coleman, T.F., Li, Y. (1999). "A subspace, interior, and conjugate
 *   gradient method for large-scale bound-constrained minimization problems."
 *   SIAM Journal on Scientific Computing, 21(1), 1–23.
 *
 *   Mayorov, N. (2015). SciPy bounded-lsq GSoC implementation.
 *   https://github.com/nmayorov/bounded-lsq
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Multivariate;
import com.curioloop.numopt4j.optim.Optimization;


import java.util.Arrays;

import static com.curioloop.numopt4j.optim.trf.TRFConstants.*;

/**
 * Core algorithm implementation for Trust Region Reflective (TRF) optimization.
 * All methods are package-private static. Fully independent of {@code LMCore}.
 * The solution is written back into the caller-supplied {@code x} array in-place.
 */
final class TRFCore {

    private TRFCore() {}

    // ── Linear algebra helpers ────────────────────────────────────────────────

    /** Euclidean norm of {@code a[off..off+n-1]}. */
    static double enorm(int n, double[] a, int off) {
        return BLAS.dnrm2(n, a, off, 1);
    }

    /** Clips {@code x[0..n-1]} to box constraints. No-op if bounds is null. */
    static void applyBounds(double[] x, Bound[] bounds, int n) {
        if (bounds == null) return;
        for (int i = 0; i < n; i++) {
            Bound b = bounds[i];
            if (b == null || b.isUnbounded()) continue;
            if (b.hasLower()) x[i] = Math.max(x[i], b.getLower());
            if (b.hasUpper()) x[i] = Math.min(x[i], b.getUpper());
        }
    }

    /**
     * Column-pivoted QR factorization: A·P = Q·R (unblocked Householder).
     *
     * <p>Given m × n matrix A, computes the factorization:</p>
     * <pre>
     *   A·P = Q·R
     * </pre>
     * <p>where:</p>
     * <ul>
     *   <li>P is an n × n column permutation matrix (stored as ipvt)</li>
     *   <li>Q is an m × m orthogonal matrix (stored implicitly in lower triangle of A)</li>
     *   <li>R is an m × n upper triangular matrix (stored in upper triangle of A)</li>
     * </ul>
     *
     * <p>Column pivoting ensures |R₁₁| ≥ |R₂₂| ≥ ··· ≥ |Rₖₖ| which improves
     * numerical stability and enables rank determination.</p>
     *
     * <p>Identical algorithm to LMCore.qrfac but operates on TRF arrays.</p>
     *
     * @param m      number of rows in A
     * @param n      number of columns in A
     * @param a      matrix A (row-major m×n), modified in place to store Q and R
     * @param ipvt   output: column permutation P (length n)
     * @param rdiag  output: diagonal of R (length n)
     * @param acnorm output: column norms of original A (length n)
     * @param wa     scratch array (length n)
     * @param tmp    scratch array of length n (reuses ws.qtf before applyQtToVec)
     * @param tmpOff offset into tmp[]
     */
    static void qrfac(int m, int n, double[] a, int[] ipvt,
                      double[] rdiag, int rdiagOff,
                      double[] acnorm, int acnormOff,
                      double[] wa, int waOff,
                      double[] tmp, int tmpOff) {
        for (int j = 0; j < n; j++) { acnorm[acnormOff+j] = 0.0; rdiag[rdiagOff+j] = 0.0; }
        for (int i = 0; i < m; i++) {
            int rowBase = i * n;
            for (int j = 0; j < n; j++) { double v = a[rowBase + j]; acnorm[acnormOff+j] += v * v; }
        }
        for (int j = 0; j < n; j++) {
            acnorm[acnormOff+j] = Math.sqrt(acnorm[acnormOff+j]);
            rdiag[rdiagOff+j]   = acnorm[acnormOff+j];
            wa[waOff+j]         = rdiag[rdiagOff+j];
            ipvt[j]             = j;
        }

        int minmn = Math.min(m, n);
        for (int j = 0; j < minmn; j++) {
            int kmax = j;
            for (int k = j + 1; k < n; k++) if (rdiag[rdiagOff+k] > rdiag[rdiagOff+kmax]) kmax = k;
            if (kmax != j) {
                for (int i = 0; i < m; i++) {
                    int base = i * n;
                    double t = a[base + j]; a[base + j] = a[base + kmax]; a[base + kmax] = t;
                }
                rdiag[rdiagOff+kmax] = rdiag[rdiagOff+j]; wa[waOff+kmax] = wa[waOff+j];
                int itmp = ipvt[j]; ipvt[j] = ipvt[kmax]; ipvt[kmax] = itmp;
            }

            double ajnorm = 0.0;
            for (int i = j; i < m; i++) { double v = a[i * n + j]; ajnorm += v * v; }
            ajnorm = Math.sqrt(ajnorm);
            if (ajnorm == 0.0) { rdiag[rdiagOff+j] = 0.0; continue; }
            if (a[j * n + j] < 0.0) ajnorm = -ajnorm;
            for (int i = j; i < m; i++) a[i * n + j] /= ajnorm;
            a[j * n + j] += 1.0;

            int nk = n - j - 1;
            if (nk > 0) {
                for (int k = j + 1; k < n; k++) tmp[tmpOff+k] = 0.0;
                for (int i = j; i < m; i++) {
                    double vij = a[i * n + j];
                    int rowBase = i * n + j + 1;
                    for (int k = 0; k < nk; k++) tmp[tmpOff+j+1+k] += vij * a[rowBase + k];
                }
                double ajj = a[j * n + j];
                for (int k = j + 1; k < n; k++) tmp[tmpOff+k] /= ajj;
                for (int i = j; i < m; i++) {
                    double vij = a[i * n + j];
                    int rowBase = i * n + j + 1;
                    for (int k = 0; k < nk; k++) a[rowBase + k] -= tmp[tmpOff+j+1+k] * vij;
                }
                for (int k = j + 1; k < n; k++) {
                    if (rdiag[rdiagOff+k] != 0.0) {
                        double t = a[j * n + k] / rdiag[rdiagOff+k];
                        rdiag[rdiagOff+k] *= Math.sqrt(Math.max(0.0, 1.0 - t * t));
                        if (0.05 * (rdiag[rdiagOff+k] / wa[waOff+k]) * (rdiag[rdiagOff+k] / wa[waOff+k]) <= EPSMCH) {
                            double s2 = 0.0;
                            for (int i = j + 1; i < m; i++) { double v = a[i * n + k]; s2 += v * v; }
                            rdiag[rdiagOff+k] = Math.sqrt(s2); wa[waOff+k] = rdiag[rdiagOff+k];
                        }
                    }
                }
            }
            rdiag[rdiagOff+j] = -ajnorm;
        }
    }

    /**
     * Applies Qᵀ to vector b in-place, then restores the R diagonal.
     *
     * <p>Given the Householder factors stored in the lower triangle of fjac
     * (as produced by qrfac), applies the accumulated orthogonal transformation:</p>
     * <pre>
     *   b ← Qᵀb
     * </pre>
     * <p>After application, restores the diagonal of R from rdiag so that
     * fjac contains the upper triangular R in its upper triangle.</p>
     *
     * @param fjac  Jacobian factor (row-major m×n), lower triangle holds Householder vectors
     * @param m     number of rows
     * @param n     number of columns
     * @param b     vector to transform in place (length m)
     * @param rdiag diagonal of R (length n), restored into fjac diagonal on exit
     */
    static void applyQtToVec(double[] fjac, int m, int n, double[] b, double[] rdiag, int rdiagOff) {
        int minmn = Math.min(m, n);
        for (int j = 0; j < minmn; j++) {
            if (fjac[j*n+j] == 0.0) { fjac[j*n+j] = rdiag[rdiagOff+j]; continue; }
            double sum = 0.0;
            for (int i = j; i < m; i++) sum += fjac[i*n+j] * b[i];
            double tmp = -sum / fjac[j*n+j];
            for (int i = j; i < m; i++) b[i] += fjac[i*n+j] * tmp;
            fjac[j*n+j] = rdiag[rdiagOff+j];
        }
    }

    /**
     * Solves the augmented least-squares system via Givens rotations (qrsolv).
     *
     * <p>Given the QR factorization A·P = Q·R, solves the augmented system:</p>
     * <pre>
     *   min ‖[ R  ] x - [ Qᵀb ]‖₂
     *       ‖[ D  ]     [  0  ]‖
     * </pre>
     * <p>where D = diag(d₁, ..., dₙ) is a diagonal matrix. This is equivalent to
     * solving the normal equations (RᵀR + DᵀD)x = Rᵀ(Qᵀb).</p>
     *
     * <p>The algorithm uses Givens rotations to zero out the diagonal elements of D
     * one at a time, updating R and Qᵀb accordingly. The resulting upper triangular
     * system is then solved by back-substitution.</p>
     *
     * @param n      number of variables
     * @param r      upper triangular R (n×n, row-major), modified in place
     * @param ipvt   column permutation from qrfac (length n)
     * @param diag   diagonal scaling D (length n)
     * @param qtb    Qᵀb vector (length n)
     * @param x      output: solution vector (length n)
     * @param sdiag  output: diagonal of the modified R after Givens rotations (length n)
     * @param wa     scratch array (length n)
     */
    static void qrsolv(int n, double[] r, int rOff, int[] ipvt, double[] diag,
                       double[] qtb, int qtbOff, double[] x, int xOff, double[] sdiag, int sdiagOff,
                       double[] wa) {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) r[rOff+i*n+j] = r[rOff+j*n+i];
            x[xOff+j] = r[rOff+j*n+j];
            wa[j]     = qtb[qtbOff+j];
        }
        for (int j = 0; j < n; j++) {
            int l = ipvt[j];
            if (diag[l] == 0.0) { sdiag[sdiagOff+j] = r[rOff+j*n+j]; r[rOff+j*n+j] = x[xOff+j]; continue; }
            for (int k = j; k < n; k++) sdiag[sdiagOff+k] = 0.0;
            sdiag[sdiagOff+j] = diag[l];
            double qtbpj = 0.0;
            for (int k = j; k < n; k++) {
                if (sdiag[sdiagOff+k] == 0.0) continue;
                double cos, sin;
                if (Math.abs(r[rOff+k*n+k]) >= Math.abs(sdiag[sdiagOff+k])) {
                    double tan = sdiag[sdiagOff+k] / r[rOff+k*n+k];
                    cos = 0.5 / Math.sqrt(0.25 + 0.25*tan*tan);
                    sin = cos * tan;
                } else {
                    double cotan = r[rOff+k*n+k] / sdiag[sdiagOff+k];
                    sin = 0.5 / Math.sqrt(0.25 + 0.25*cotan*cotan);
                    cos = sin * cotan;
                }
                r[rOff+k*n+k] = cos*r[rOff+k*n+k] + sin*sdiag[sdiagOff+k];
                double temp = cos*wa[k] + sin*qtbpj;
                qtbpj = -sin*wa[k] + cos*qtbpj;
                wa[k] = temp;
                for (int i = k+1; i < n; i++) {
                    temp              =  cos*r[rOff+i*n+k] + sin*sdiag[sdiagOff+i];
                    sdiag[sdiagOff+i] = -sin*r[rOff+i*n+k] + cos*sdiag[sdiagOff+i];
                    r[rOff+i*n+k]     = temp;
                }
            }
            sdiag[sdiagOff+j] = r[rOff+j*n+j];
            r[rOff+j*n+j] = x[xOff+j];
        }
        int nsing = n;
        for (int j = 0; j < n; j++) {
            if (sdiag[sdiagOff+j] == 0.0 && nsing == n) nsing = j;
            if (nsing < n) wa[j] = 0.0;
        }
        for (int k = 0; k < nsing; k++) {
            int j = nsing - 1 - k;
            double sum = 0.0;
            for (int i = j+1; i < nsing; i++) sum += r[rOff+i*n+j] * wa[i];
            wa[j] = (wa[j] - sum) / sdiag[sdiagOff+j];
        }
        for (int j = 0; j < n; j++) x[xOff+ipvt[j]] = wa[j];
    }

    /**
     * Finds the Levenberg-Marquardt parameter λ such that ‖D·p(λ)‖₂ ≈ Δ (lmpar).
     *
     * <p>Given the QR factorization A·P = Q·R, finds λ ≥ 0 such that the solution p
     * to the augmented system:</p>
     * <pre>
     *   (RᵀR + λDᵀD)p = -Rᵀ(Qᵀf)
     * </pre>
     * <p>satisfies ‖D·p‖₂ ≈ Δ (the trust-region radius).</p>
     *
     * <p>The algorithm uses a Newton iteration on the secular equation:</p>
     * <pre>
     *   φ(λ) = 1/‖D·p(λ)‖₂ - 1/Δ = 0
     * </pre>
     * <p>starting from a bracket [λₗ, λᵤ] determined by the Gershgorin bounds.</p>
     *
     * <p>Special cases:</p>
     * <ul>
     *   <li>If ‖D·p(0)‖₂ ≤ Δ (unconstrained solution is feasible), returns λ = 0</li>
     *   <li>If rank(R) &lt; n (singular), uses the minimum-norm solution</li>
     * </ul>
     *
     * @param n      number of variables
     * @param r      upper triangular R (n×n, row-major), modified in place by qrsolv
     * @param ipvt   column permutation from qrfac (length n)
     * @param diag   diagonal scaling D (length n)
     * @param qtb    Qᵀf vector (length n)
     * @param delta  trust-region radius Δ
     * @param par    initial estimate of λ (updated on return)
     * @param x      output: solution p(λ) (length n, starting at xOff)
     * @param xOff   offset into x[]
     * @param sdiag  scratch: modified diagonal from qrsolv (length n)
     * @param wa1    scratch array (length n)
     * @param wa2    scratch array (length n)
     * @return       updated Levenberg-Marquardt parameter λ
     */
    static double lmpar(int n, double[] r, int rOff, int[] ipvt,
                        double[] diag, int diagOff,
                        double[] qtb, int qtbOff,
                        double delta, double par,
                        double[] x, int xOff,
                        double[] sdiag, int sdiagOff,
                        double[] wa1,
                        double[] wa2) {
        final double dwarf = Double.MIN_VALUE;
        final double p1 = 0.1, p001 = 0.001;

        int nsing = n;
        System.arraycopy(qtb, qtbOff, wa1, 0, n);
        for (int j = 0; j < n; j++) {
            if (r[rOff+j*n+j] == 0.0 && nsing == n) nsing = j;
            if (nsing < n) wa1[j] = 0.0;
        }
        for (int k = 0; k < nsing; k++) {
            int j = nsing - 1 - k;
            wa1[j] /= r[rOff+j*n+j];
            double tmp = wa1[j];
            for (int i = 0; i < j; i++) wa1[i] -= r[rOff+i*n+j] * tmp;
        }
        for (int j = 0; j < n; j++) x[xOff+ipvt[j]] = wa1[j];

        int iter = 0;
        for (int j = 0; j < n; j++) wa2[j] = diag[diagOff+j] * x[xOff+j];
        double dxnorm = enorm(n, wa2, 0);
        double fp = dxnorm - delta;
        if (fp <= p1 * delta) { if (iter == 0) par = 0.0; return par; }

        double parl = 0.0;
        if (nsing >= n) {
            for (int j = 0; j < n; j++) {
                int l = ipvt[j];
                wa1[j] = diag[diagOff+l] * (wa2[l] / dxnorm);
            }
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < j; i++) sum += r[rOff+i*n+j] * wa1[i];
                wa1[j] = (wa1[j] - sum) / r[rOff+j*n+j];
            }
            double tmp = enorm(n, wa1, 0);
            parl = ((fp / delta) / tmp) / tmp;
        }

        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i <= j; i++) sum += r[rOff+i*n+j] * qtb[qtbOff+i];
            int l = ipvt[j];
            wa1[j] = sum / diag[diagOff+l];
        }
        double gnorm = enorm(n, wa1, 0);
        double paru = gnorm / delta;
        if (paru == 0.0) paru = dwarf / Math.min(delta, p1);

        par = Math.max(par, parl);
        par = Math.min(par, paru);
        if (par == 0.0) par = gnorm / dxnorm;

        for (iter = 1; iter <= 10; iter++) {
            if (par == 0.0) par = Math.max(dwarf, p001 * paru);
            double sqrtPar = Math.sqrt(par);
            for (int j = 0; j < n; j++) wa1[j] = sqrtPar * diag[diagOff+j];
            qrsolv(n, r, rOff, ipvt, wa1, qtb, qtbOff, x, xOff, sdiag, sdiagOff, wa2);
            for (int j = 0; j < n; j++) wa2[j] = diag[diagOff+j] * x[xOff+j];
            dxnorm = enorm(n, wa2, 0);
            double fpOld = fp;
            fp = dxnorm - delta;

            if (Math.abs(fp) <= p1 * delta
                || (parl == 0.0 && fp <= fpOld && fpOld < 0.0)
                || iter == 10) break;

            for (int j = 0; j < n; j++) {
                int l = ipvt[j];
                wa1[j] = diag[diagOff+l] * (wa2[l] / dxnorm);
            }
            for (int j = 0; j < n; j++) {
                wa1[j] /= sdiag[sdiagOff+j];
                double tmp = wa1[j];
                for (int i = j+1; i < n; i++) wa1[i] -= r[rOff+i*n+j] * tmp;
            }
            double tmp = enorm(n, wa1, 0);
            double parc = ((fp / delta) / tmp) / tmp;
            if (fp > 0.0) parl = Math.max(parl, par);
            if (fp < 0.0) paru = Math.min(paru, par);
            par = Math.max(parl, par + parc);
        }
        if (iter == 0) par = 0.0;
        return par;
    }

    // ── Coleman-Li scaling ────────────────────────────────────────────────────

    /**
     * Computes the Coleman-Li diagonal scaling vector v ∈ ℝⁿ.
     *
     * <p>The scaling is gradient-aware: the distance is chosen based on the gradient
     * direction so the scaling reflects how far x can move before hitting a bound
     * in the descent direction:</p>
     * <pre>
     *   vᵢ = uᵢ - xᵢ              if gᵢ &lt; 0 and uᵢ - xᵢ &gt; 0  (moving toward upper bound)
     *   vᵢ = xᵢ - lᵢ              if gᵢ &gt; 0 and xᵢ - lᵢ &gt; 0  (moving toward lower bound)
     *   vᵢ = min(xᵢ - lᵢ, uᵢ - xᵢ)  otherwise                  (at bound or no gradient)
     * </pre>
     *
     * <p>When x is at or near a bound in the gradient direction (active constraint),
     * falls back to the geometric distance to the opposite bound (or 1.0) to prevent
     * trust-region collapse.</p>
     *
     * <p>Reference: SciPy {@code CL_scaling_vector} in {@code _lsq/trf.py}</p>
     *
     * @param x      current point xₖ (length n)
     * @param qtf    gradient proxy Qᵀf (length n); may be null on first call
     * @param bounds box constraints; null means unbounded (vᵢ = 1 for all i)
     * @param n      problem dimension
     * @param v      output: scaling vector (length n)
     */
    static void clScaling(double[] x, double[] qtf, int qtfOff, Bound[] bounds, int n, double[] v, int vOff) {
        if (bounds == null) {
            Arrays.fill(v, vOff, vOff + n, 1.0);
            return;
        }
        for (int i = 0; i < n; i++) {
            Bound b = bounds[i];
            if (b == null || b.isUnbounded()) {
                v[vOff+i] = 1.0;
                continue;
            }
            double dLo = b.hasLower() ? (x[i] - b.getLower()) : Double.MAX_VALUE;
            double dHi = b.hasUpper() ? (b.getUpper() - x[i]) : Double.MAX_VALUE;

            double g = (qtf != null) ? qtf[qtfOff+i] : 0.0;
            double dist;
            if (g < 0 && b.hasUpper() && dHi > 1e-8) {
                // moving toward upper bound — use distance to upper
                dist = dHi;
            } else if (g > 0 && b.hasLower() && dLo > 1e-8) {
                // moving toward lower bound — use distance to lower
                dist = dLo;
            } else {
                // scipy CL_scaling_vector returns 1 for the "otherwise" case:
                // gradient blocked at bound (dHi/dLo ≈ 0), no bound in gradient
                // direction, or gradient is zero. Using 1.0 prevents effDiag from
                // collapsing to zero when x is at a bound, which would cause lmpar
                // to produce an unbounded step that gets truncated to zero by
                // reflectiveStep, stalling convergence of unconstrained variables.
                dist = 1.0;
            }
            v[vOff+i] = Math.max(1e-10, dist);
        }
    }

    // ── Strictly feasible projection ─────────────────────────────────────────

    /**
     * Pushes any component of x that sits exactly on a bound one ULP into the interior.
     *
     * <p>For each variable at a bound:</p>
     * <pre>
     *   xᵢ = 𝚗𝚎𝚡𝚝𝚊𝚏𝚝𝚎𝚛(lᵢ, +∞)   if xᵢ ≤ lᵢ
     *   xᵢ = 𝚗𝚎𝚡𝚝𝚊𝚏𝚝𝚎𝚛(uᵢ, -∞)   if xᵢ ≥ uᵢ
     * </pre>
     *
     * <p>This prevents the gradient-aware Coleman-Li scaling from collapsing to zero
     * on the next iteration when x is exactly at a bound.</p>
     *
     * <p>Mirrors SciPy's {@code make_strictly_feasible} with {@code rstep=0} (nextafter).</p>
     *
     * @param x      current point (length n), modified in place
     * @param bounds box constraints; null means no-op
     * @param n      problem dimension
     */
    static void makeStrictlyFeasible(double[] x, Bound[] bounds, int n) {
        if (bounds == null) return;
        for (int i = 0; i < n; i++) {
            Bound b = bounds[i];
            if (b == null || b.isUnbounded()) continue;
            if (b.hasLower() && x[i] <= b.getLower())
                x[i] = Math.nextAfter(b.getLower(), Double.POSITIVE_INFINITY);
            if (b.hasUpper() && x[i] >= b.getUpper())
                x[i] = Math.nextAfter(b.getUpper(), Double.NEGATIVE_INFINITY);
        }
    }

    // ── Reflective step ───────────────────────────────────────────────────────

    /**
     * Computes the best feasible step from x using three candidates (select_step).
     *
     * <p>Given the unconstrained step p from lmpar, evaluates three candidates and
     * selects the one minimizing the quadratic model mₖ(p) = ‖Qᵀf/‖f‖ + R·Pᵀ·p‖₂²:</p>
     *
     * <ol>
     *   <li>Trust-region step:  p_tr = θ × tHit × p  (stepped back from first bound hit)</li>
     *   <li>Reflective step:    p_ref = tHit × p + tOpt × p̃  (reflect off bound, minimize 1-D model)</li>
     *   <li>Cauchy step:        p_c = tCauchy × (-Qᵀf)  (1-D minimization along anti-gradient)</li>
     * </ol>
     *
     * <p>where θ = max(0.995, 1 - ‖g‖) keeps the step strictly interior.</p>
     *
     * <p><b>Deviation from SciPy select_step:</b> SciPy evaluates candidates using the
     * full quadratic model {@code evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)}, which
     * includes the Coleman-Li curvature term C = diag(g·dv·scale). This implementation
     * uses the simpler model ‖Qᵀf/‖f‖ + R·Pᵀ·p‖₂² (no diag_h), consistent with the
     * lmpar-based subproblem solver. The step selection is still correct but may choose
     * a slightly suboptimal candidate when variables are near bounds.</p>
     *
     * <p>First bound hit along p:</p>
     * <pre>
     *   tHit = min { (lᵢ - xᵢ)/pᵢ : pᵢ &lt; 0 } ∪ { (uᵢ - xᵢ)/pᵢ : pᵢ &gt; 0 }
     * </pre>
     *
     * <p>Reflected direction p̃ is obtained by flipping the component that hit the bound:</p>
     * <pre>
     *   p̃ᵢ = -pᵢ   if i = hitIdx
     *   p̃ᵢ =  pᵢ   otherwise
     * </pre>
     *
     * <p>Optimal reflected stride tOpt minimizes the 1-D quadratic model along p̃:</p>
     * <pre>
     *   tOpt = 𝚌𝚕𝚒𝚙(-bRef / (2 × aRef), rStrideL, rStrideU)
     * </pre>
     *
     * <p>Mirrors SciPy's {@code select_step} in {@code _lsq/trf.py}.</p>
     *
     * @param x      current point xₖ (length n)
     * @param p      unconstrained step direction from lmpar (length n, starting at pOff)
     * @param pOff   offset into p[]
     * @param bounds box constraints; null means unbounded (full step used)
     * @param n      problem dimension
     * @param theta  step-back fraction θ (0.995 ≤ θ ≤ 1); keeps step strictly interior
     * @param xNew   output: xₖ + actual_step (length n)
     * @param step   output: actual_step = xNew - xₖ (length n)
     * @param Jp     scratch: R·Pᵀ·(tHit·p) for quadratic model (length n)
     * @param Jpr    scratch: R·Pᵀ·p̃ for quadratic model (length n)
     * @param xHit   scratch: point where step hits a bound (length n)
     * @param pRef   scratch: reflected direction p̃ (length n)
     * @param ag     scratch: Cauchy direction -Qᵀf (length n)
     * @param m      number of residuals
     * @param fjac   Jacobian factor (row-major m×n, upper-tri = R after qrfac)
     * @param ipvt   column permutation P from qrfac (length n)
     * @param qtf    Qᵀf (length n) — used as gradient proxy for Cauchy direction
     * @param fnorm  current ‖f‖₂
     * @param delta  current trust-region radius Δ — used to bound Cauchy step length
     * @param delta  current trust-region radius Δ — used to bound Cauchy step length
     */
    static void reflectiveStep(double[] x, double[] p, int pOff, Bound[] bounds, int n,
                                double theta,
                                double[] xNew, double[] step, int stepOff,
                                double[] Jp, int JpOff, double[] Jpr, int JprOff,
                                double[] xHit, int xHitOff, double[] pRef, int pRefOff,
                                double[] ag, int agOff,
                                int m,
                                double[] fjac, int[] ipvt,
                                double[] qtf, int qtfOff,
                                double fnorm, double delta) {
        if (bounds == null) {
            for (int i = 0; i < n; i++) { xNew[i] = x[i] + p[pOff+i]; step[stepOff+i] = p[pOff+i]; }
            return;
        }

        // ── Find first bound hit along p ──────────────────────────────────────
        double tHit = 1.0;
        int hitIdx = -1;
        for (int i = 0; i < n; i++) {
            Bound b = bounds[i];
            if (b == null || b.isUnbounded()) continue;
            if (p[pOff+i] < 0 && b.hasLower()) {
                double t = (b.getLower() - x[i]) / p[pOff+i];
                if (t > 0 && t < tHit) { tHit = t; hitIdx = i; }
            } else if (p[pOff+i] > 0 && b.hasUpper()) {
                double t = (b.getUpper() - x[i]) / p[pOff+i];
                if (t > 0 && t < tHit) { tHit = t; hitIdx = i; }
            }
        }

        if (hitIdx < 0) {
            // No bound hit: use full step (clipped for safety)
            for (int i = 0; i < n; i++) { xNew[i] = x[i] + p[pOff+i]; step[stepOff+i] = p[pOff+i]; }
            applyBounds(xNew, bounds, n);
            for (int i = 0; i < n; i++) step[stepOff+i] = xNew[i] - x[i];
            return;
        }

        // ── Candidate 1: trust-region step, stepped back by theta ────────────
        // p_tr = theta * tHit * p  (stop just before the bound)
        double tTR = theta * tHit;

        // ── Candidate 2: reflective step ──────────────────────────────────────
        // Point where we hit the bound
        for (int i = 0; i < n; i++) xHit[xHitOff+i] = x[i] + tHit * p[pOff+i];

        // Reflected direction: flip the component that hit the bound
        System.arraycopy(p, pOff, pRef, pRefOff, n);
        pRef[pRefOff+hitIdx] = -pRef[pRefOff+hitIdx];
        double tRemain = 1.0 - tHit;

        // Clip reflected direction to stay feasible
        double tRef = tRemain;
        for (int i = 0; i < n; i++) {
            if (pRef[pRefOff+i] == 0) continue;
            Bound b = bounds[i];
            if (b == null || b.isUnbounded()) continue;
            if (pRef[pRefOff+i] < 0 && b.hasLower()) {
                double t = (b.getLower() - xHit[xHitOff+i]) / pRef[pRefOff+i];
                if (t >= 0 && t < tRef) tRef = t;
            } else if (pRef[pRefOff+i] > 0 && b.hasUpper()) {
                double t = (b.getUpper() - xHit[xHitOff+i]) / pRef[pRefOff+i];
                if (t >= 0 && t < tRef) tRef = t;
            }
        }
        // Step back reflected stride by theta as well
        double rStrideL = (1.0 - theta) * tHit / Math.max(tRef, 1e-300);
        double rStrideU = theta * tRef;

        // ── Candidate 3: Cauchy step along −g (true anti-gradient) ───────────
        // ag[i] = −g[i] = −(Jᵀf)[i], the steepest descent direction.
        // g = Jᵀf = P·Rᵀ·(Qᵀf) = P·Rᵀ·qtf, so g[ipvt[j]] = Σᵢ≤ⱼ R[i,j]*qtf[i].

        // Project out components blocked at a bound: if ag[i] > 0 and x[i] is at its
        // upper bound (dHi < threshold), or ag[i] < 0 and x[i] is at its lower bound
        // (dLo < threshold), set ag[i] = 0.  This mirrors scipy's hat-space projection
        // (d_h[i] = 0 when v[i] ≈ 0) and prevents the Cauchy step from being collapsed
        // to near-zero by a single variable that is already at a bound.
        double agNorm = 0.0;
        for (int i = 0; i < n; i++) ag[agOff+i] = 0.0;
        for (int j = 0; j < n; j++) {
            int l = ipvt[j];
            double gj = 0.0;
            for (int i = 0; i <= j; i++) gj += fjac[i * n + j] * qtf[qtfOff+i];
            ag[agOff+l] = -gj;
        }
        for (int i = 0; i < n; i++) {
            double ai = ag[agOff+i];
            if (bounds != null && ai != 0.0) {
                Bound b = bounds[i];
                if (b != null && !b.isUnbounded()) {
                    double dLo = b.hasLower() ? (x[i] - b.getLower()) : Double.MAX_VALUE;
                    double dHi = b.hasUpper() ? (b.getUpper() - x[i]) : Double.MAX_VALUE;
                    // Suppress component if it pushes into an already-active bound:
                    //   ai > 0 → wants to increase x[i] → blocked if dHi ≈ 0 (at upper bound)
                    //   ai < 0 → wants to decrease x[i] → blocked if dLo ≈ 0 (at lower bound)
                    if (ai > 0 && dHi < 1e-8) ai = 0.0;
                    else if (ai < 0 && dLo < 1e-8) ai = 0.0;
                }
            }
            ag[agOff+i] = ai;
            agNorm += ai * ai;
        }
        agNorm = Math.sqrt(agNorm);
        // Clip Cauchy direction to trust-region ball radius (Δ) and bounds.
        // scipy uses to_tr = Delta / norm(ag_h), NOT pNorm / agNorm.
        // Using pNorm would collapse the Cauchy step to zero when tHit≈0 (x at bound),
        // preventing convergence of unconstrained variables in the same iteration.
        double tCauchy = (agNorm > 0) ? 1.0 : 0.0;
        if (agNorm > 0) {
            // Scale so that ||ag * tCauchy|| ≈ Delta (trust-region radius)
            tCauchy = delta / agNorm;
            // Clip to bounds
            for (int i = 0; i < n; i++) {
                if (ag[agOff+i] == 0) continue;
                Bound b = bounds[i];
                if (b == null || b.isUnbounded()) continue;
                if (ag[agOff+i] < 0 && b.hasLower()) {
                    double t = (b.getLower() - x[i]) / ag[agOff+i];
                    if (t >= 0 && t < tCauchy) tCauchy = t;
                } else if (ag[agOff+i] > 0 && b.hasUpper()) {
                    double t = (b.getUpper() - x[i]) / ag[agOff+i];
                    if (t >= 0 && t < tCauchy) tCauchy = t;
                }
            }
            tCauchy *= theta;
        }

        // ── Compute R·Pᵀ vectors for quadratic model evaluation ──────────────
        // Jp  = R·Pᵀ·(tHit·p)
        // Jpr = R·Pᵀ·pRef
        // Jag = R·Pᵀ·ag  (reuse xHit as scratch for Jag)
        Arrays.fill(Jp,   JpOff,   JpOff   + n, 0.0);
        Arrays.fill(Jpr,  JprOff,  JprOff  + n, 0.0);
        Arrays.fill(xHit, xHitOff, xHitOff + n, 0.0); // reuse as Jag scratch
        for (int j = 0; j < n; j++) {
            int l = ipvt[j];
            double vp   = tHit * p[pOff+l];
            double vpr  = pRef[pRefOff+l];
            double vag  = ag[agOff+l];
            for (int i = 0; i <= j; i++) {
                double fij = fjac[i * n + j];
                Jp[JpOff+i]   += fij * vp;
                Jpr[JprOff+i] += fij * vpr;
                xHit[xHitOff+i] += fij * vag;
            }
        }

        // Quadratic model: ‖Qᵀ·f/‖f‖ + R·Pᵀ·s‖²
        // Evaluate at three candidates and pick the best
        double invFnorm = (fnorm > 0) ? 1.0 / fnorm : 1.0;

        // Candidate 1: TR step at tTR (= theta*tHit along p)
        double valTR = 0;
        for (int i = 0; i < n; i++) {
            double qi = qtf[qtfOff+i] * invFnorm;
            double d  = qi + (tTR / Math.max(tHit, 1e-300)) * Jp[JpOff+i];
            valTR += d * d;
        }

        // Candidate 2: reflective — find optimal t along reflected path
        double aRef = 0, bRef = 0;
        for (int i = 0; i < n; i++) {
            double qi = qtf[qtfOff+i] * invFnorm;
            aRef += Jpr[JprOff+i] * Jpr[JprOff+i];
            bRef += (Jp[JpOff+i] + qi) * Jpr[JprOff+i];
        }
        double tOptRef = (aRef > 0) ? Math.max(rStrideL, Math.min(rStrideU, -bRef / (2 * aRef))) : rStrideL;
        double valRef = 0;
        for (int i = 0; i < n; i++) {
            double qi = qtf[qtfOff+i] * invFnorm;
            double d  = qi + Jp[JpOff+i] + Jpr[JprOff+i] * tOptRef;
            valRef += d * d;
        }

        // Candidate 3: Cauchy step
        double valCauchy = 0;
        if (agNorm > 0) {
            for (int i = 0; i < n; i++) {
                double qi = qtf[qtfOff+i] * invFnorm;
                double d  = qi + xHit[xHitOff+i] * tCauchy; // xHit reused as Jag
                valCauchy += d * d;
            }
        } else {
            valCauchy = Double.MAX_VALUE;
        }

        // ── Restore xHit to actual bound-hit point ────────────────────────────
        for (int i = 0; i < n; i++) xHit[xHitOff+i] = x[i] + tHit * p[pOff+i];

        // ── Pick best candidate ───────────────────────────────────────────────
        if (valTR <= valRef && valTR <= valCauchy) {
            // Candidate 1: step back along p
            for (int i = 0; i < n; i++) {
                xNew[i] = x[i] + tTR * p[pOff+i];
                step[stepOff+i] = tTR * p[pOff+i];
            }
        } else if (valRef <= valCauchy) {
            // Candidate 2: reflective
            for (int i = 0; i < n; i++) {
                xNew[i] = xHit[xHitOff+i] + tOptRef * pRef[pRefOff+i];
                step[stepOff+i] = xNew[i] - x[i];
            }
        } else {
            // Candidate 3: Cauchy
            for (int i = 0; i < n; i++) {
                xNew[i] = x[i] + tCauchy * ag[agOff+i];
                step[stepOff+i] = tCauchy * ag[agOff+i];
            }
        }
        applyBounds(xNew, bounds, n);
        for (int i = 0; i < n; i++) step[stepOff+i] = xNew[i] - x[i];
    }

    // ── Main optimization loop ────────────────────────────────────────────────

    /**
     * Runs the Trust Region Reflective iteration.
     *
     * <p>Solves min ½‖f(x)‖₂² subject to l ≤ x ≤ u using the TRF algorithm.</p>
     *
     * <h3>Outer Loop (Jacobian evaluation)</h3>
     * <ol>
     *   <li>Evaluate residuals f(x) and Jacobian J(x)</li>
     *   <li>Factor J·P = Q·R via column-pivoted QR (qrfac)</li>
     *   <li>Compute Coleman-Li scaling v = CL_scaling(x, Qᵀf)</li>
     *   <li>Compute Qᵀf and gradient norm ‖Dₖ⁻¹Jᵀf‖∞</li>
     *   <li>Check gradient convergence: ‖Dₖ⁻¹Jᵀf‖∞ ≤ gtol</li>
     * </ol>
     *
     * <h3>Inner Loop (trust-region step)</h3>
     * <ol>
     *   <li>Solve augmented system (JᵀJ + λDᵀD)p = -Jᵀf via lmpar with effective diagonal Dₖ = diag × v</li>
     *   <li>Apply reflective step: select best of TR / reflective / Cauchy candidates</li>
     *   <li>Evaluate trial point f(xₖ + p)</li>
     *   <li>Compute actual reduction: ρₐ = 1 - (‖f(xₖ+p)‖/‖f(xₖ)‖)²</li>
     *   <li>Compute predicted reduction: ρₚ = (‖Rp‖² + 2λ‖Dp‖²) / ‖f(xₖ)‖²</li>
     *   <li>Update trust-region radius Δ based on ratio ρₐ/ρₚ</li>
     *   <li>Accept step if ρₐ/ρₚ ≥ 0.0001</li>
     * </ol>
     *
     * <h3>Deviation from SciPy trf_bounds</h3>
     * <p>SciPy solves a richer trust-region subproblem in "hat space":</p>
     * <pre>
     *   B_h = J_h^T J_h + C,   C = diag(g · dv · scale)
     * </pre>
     * <p>where {@code dv[i] ∈ {-1, 0, +1}} is the derivative of the Coleman-Li scaling
     * vector v with respect to xᵢ. The extra term C accounts for the curvature of the
     * CL transformation near bounds and is non-zero only when a variable is actively
     * approaching a bound. SciPy uses SVD on the augmented matrix {@code [J_h; diag(C^0.5)]}
     * followed by a More-Sorensen solver to handle the potentially non-positive-definite B_h.</p>
     *
     * <p>This implementation omits C and uses MINPACK-style QR + lmpar instead, which
     * assumes a positive-definite Hessian approximation (J^T J only). The trade-off:</p>
     * <ul>
     *   <li>Unbounded problems: no difference (dv = 0 everywhere, C = 0)</li>
     *   <li>Bounded, far from bounds: negligible difference (C ≈ 0)</li>
     *   <li>Bounded, near bounds: may require more outer iterations to converge,
     *       but each outer iteration is ~3-5× faster (QR vs SVD)</li>
     * </ul>
     *
     * <h3>Convergence Criteria</h3>
     * <ul>
     *   <li>ftol: |ρₐ| ≤ ftol and ρₚ ≤ ftol and ½ρₐ/ρₚ ≤ 1 (chi-squared reduction)</li>
     *   <li>xtol: Δ ≤ xtol × ‖Dₖx‖₂ (parameter change)</li>
     *   <li>gtol: ‖Dₖ⁻¹Jᵀf‖∞ ≤ gtol (gradient norm)</li>
     * </ul>
     *
     * @param m        number of residuals
     * @param n        number of parameters
     * @param fcn      objective: evaluates residuals f(x) and optionally Jacobian J(x)
     * @param ftol     relative tolerance on chi-squared reduction
     * @param xtol     relative tolerance on parameter change
     * @param gtol     absolute tolerance on gradient norm
     * @param maxfev   maximum number of function evaluations
     * @param factor   initial trust-region scale factor
     * @param userDiag caller-supplied scaling diagonal (null = auto)
     * @param bounds   optional box constraints l ≤ x ≤ u
     * @param loss     robust loss function (LINEAR = standard least-squares)
     * @param lossScale soft margin C for loss scaling (f_scale in scipy)
     * @param x        initial guess on entry; solution on exit
     * @param ws       pre-allocated workspace
     */
    static Optimization optimize(int m, int n,
                                 Multivariate fcn,
                                 double ftol, double xtol, double gtol,
                                 int maxfev, double factor,
                                 double[] userDiag, Bound[] bounds,
                                 RobustLoss loss, double lossScale,
                                 double[] x, TRFWorkspace ws) {

        // ── eval-facing standalone arrays ─────────────────────────────────────
        double[] fvec = ws.fvec;   // f(xₖ)
        double[] fjac = ws.fjac;   // J(xₖ) row-major
        double[] wa2  = ws.wa2;    // acnorm (qrfac output) / trial point xₖ+p (eval input) / xnorm scratch
        double[] wa4  = ws.wa4;    // f(xₖ+p) — trial residuals
        int[]    ipvt = ws.ipvt;

        // ── merged scratch buffer + offsets ───────────────────────────────────
        double[] work      = ws.work;
        final int rworkOff   = ws.rworkOff;
        final int diagOff    = ws.diagOff;
        final int qtfOff     = ws.qtfOff;
        final int wa1Off     = ws.wa1Off;
        final int wa3Off     = ws.wa3Off;
        final int clScaleOff = ws.clScaleOff;
        final int effDiagOff = ws.effDiagOff;
        final int JpOff      = ws.JpOff;
        final int JprOff     = ws.JprOff;
        final int stepOff    = ws.stepOff;
        final int xHitOff    = ws.xHitOff;
        final int pRefOff    = ws.pRefOff;
        final int agOff      = ws.agOff;

        applyBounds(x, bounds, n);

        fcn.evaluate(x, fvec, null);
        int nfev = 1;
        double fnorm = enorm(m, fvec, 0);

        double par  = 0.0;
        int    iter = 1;
        Optimization.Status status = null;

        // Tracks cost at current x; updated at outer loop top and on each accepted step.
        // For LINEAR: cost = 0.5*‖f‖²; for robust: cost = 0.5*C²*Σρ(f²/C²).
        double cost = 0.0;

        outer:
        while (true) {
            // ── Evaluate residuals and Jacobian ───────────────────────────────
            fcn.evaluate(x, fvec, fjac);

            // ── Compute robust cost BEFORE scaling (uses original residuals) ──
            // Must be computed here because scaleJF modifies fvec in-place.
            cost = loss.cost(fvec, m, lossScale);

            // ── Apply robust loss scaling to J and f BEFORE QR factorization ──
            // scipy applies scale_for_robust_loss_function before QR, so all
            // subsequent steps (qrfac, lmpar, gradient) operate on the scaled system.
            // No-op for LINEAR.
            loss.scaleJF(fvec, fjac, m, n, lossScale);

            // ── Factor J·P = Q·R (on the loss-scaled Jacobian) ───────────────
            // wa1 → rdiag, wa2 → acnorm (offset 0, standalone), wa3 → wa scratch, qtf → tmp scratch
            // work[qtfOff] is passed as tmp[] scratch — safe because qrfac runs before applyQtToVec
            qrfac(m, n, fjac, ipvt,
                  work, wa1Off,   // rdiag
                  wa2,  0,        // acnorm — wa2 is standalone, offset 0
                  work, wa3Off,   // wa scratch
                  work, qtfOff);  // dot scratch

            // For robust loss, use cost-based effective norm so that actred and prered
            // are computed on the same scale: fnorm² ≈ 2*cost, fnorm = sqrt(2*cost).
            // For LINEAR, cost = 0.5*‖f‖², so sqrt(2*cost) = ‖f‖ — identical.
            fnorm = Math.sqrt(2.0 * cost);

            // ── Coleman-Li scaling from current x ────────────────────────────
            // qtf not yet computed on first call; pass null to fall back to geometric distance
            clScaling(x, (iter == 1) ? null : work, qtfOff, bounds, n, work, clScaleOff);

            // ── Initialize scaling diagonal on first iteration ────────────────
            if (iter == 1) {
                if (userDiag == null) {
                    for (int j = 0; j < n; j++)
                        work[diagOff+j] = (wa2[j] != 0.0) ? wa2[j] : 1.0;  // wa2 = acnorm here
                } else {
                    System.arraycopy(userDiag, 0, work, diagOff, n);
                }
                for (int j = 0; j < n; j++)
                    work[wa3Off+j] = work[diagOff+j] * work[clScaleOff+j] * x[j];
                double xnorm = enorm(n, work, wa3Off);
                double delta = (xnorm != 0.0) ? factor * xnorm : factor;
                ws.delta = delta;
                ws.xnorm = xnorm;
            }

            // ── Compute Qᵀ·f ──────────────────────────────────────────────────
            System.arraycopy(fvec, 0, wa4, 0, m);
            applyQtToVec(fjac, m, n, wa4, work, wa1Off);  // wa1 = rdiag
            System.arraycopy(wa4, 0, work, qtfOff, n);

            // ── Gradient norm (Coleman-Li scaled) ─────────────────────────────
            // Also compute the true gradient g = Jᵀf = P·Rᵀ·(Qᵀf) = P·Rᵀ·qtf,
            // stored in work[wa3Off+ipvt[j]] = Σᵢ fjac[i*n+j] * qtf[i].
            // This is needed for clScaling (which requires the true gradient direction,
            // not qtf = Qᵀf whose sign may differ from g due to Q's sign convention).
            double gnorm = 0.0;
            if (fnorm != 0.0) {
                for (int j = 0; j < n; j++) {
                    int l = ipvt[j];
                    double sum = 0.0;
                    for (int i = 0; i <= j; i++) sum += fjac[i * n + j] * work[qtfOff+i];
                    work[wa3Off+l] = sum; // true gradient g[l] = (Rᵀ·qtf)[j] mapped back via P
                    if (wa2[l] == 0.0) continue;  // wa2 = acnorm here
                    gnorm = Math.max(gnorm, Math.abs((sum / fnorm) * work[clScaleOff+l] / wa2[l]));
                }
            }

            if (gnorm <= gtol) { status = Optimization.Status.GRADIENT_TOLERANCE_REACHED; break; }

            if (userDiag == null) {
                for (int j = 0; j < n; j++)
                    work[diagOff+j] = Math.max(work[diagOff+j], wa2[j]);  // wa2 = acnorm here
            }

            // ── Effective lmpar diagonal: diag·clScale ────────────────────────
            // SciPy uses effDiag = diag * sqrt(v) * x_scale (hat-space scaling),
            // and augments the trust-region Hessian with C = diag(g·dv·scale)
            // to account for CL-transformation curvature near bounds. We omit C
            // and use MINPACK-style lmpar (positive-definite JᵀJ only), which is
            // ~3-5× faster per outer iteration at the cost of potentially more
            // iterations when variables are actively constrained at bounds.
            for (int j = 0; j < n; j++)
                work[effDiagOff+j] = work[diagOff+j] * work[clScaleOff+j];

            // ── Freeze variables at active bounds for lmpar ───────────────────
            // When a variable is at a bound in the gradient direction (dHi/dLo < 1e-8),
            // clScaling returns 1.0 to prevent effDiag collapse. But lmpar will still
            // try to step through the bound, producing tHit ≈ 0 and collapsing the
            // entire TR step. We mimic scipy's hat-space effect (d[i] = 0 → J_h col = 0
            // → p_h[i] = 0) by inflating effDiag[i] to a large value, forcing lmpar
            // to produce p[i] ≈ 0 for the frozen variable. The pnorm is computed with
            // the original effDiag (not inflated) so delta updates remain correct.
            //
            // lmpar gives p = −(JᵀJ + λDᵀD)⁻¹·Jᵀf ≈ −H⁻¹·g:
            //   g[j] < 0 → p[j] > 0 (wants to increase x[j]) → blocked if dHi ≈ 0
            //   g[j] > 0 → p[j] < 0 (wants to decrease x[j]) → blocked if dLo ≈ 0
            // Use true gradient g = Jᵀf (stored in work[wa3Off] from gnorm computation),
            // NOT qtf = Qᵀf (whose sign may differ from g due to Q's sign convention).
            if (bounds != null) {
                for (int j = 0; j < n; j++) {
                    Bound b = bounds[j];
                    if (b == null || b.isUnbounded()) continue;
                    double dLo = b.hasLower() ? (x[j] - b.getLower()) : Double.MAX_VALUE;
                    double dHi = b.hasUpper() ? (b.getUpper() - x[j]) : Double.MAX_VALUE;
                    double gj  = work[wa3Off+j]; // true gradient g[j] = (Jᵀf)[j]
                    if ((gj < 0 && dHi < 1e-8) || (gj > 0 && dLo < 1e-8)) {
                        work[effDiagOff+j] = work[diagOff+j] * 1e10; // inflate to freeze
                    }
                }
            }

            // ── theta: step-back fraction (SciPy: max(0.995, 1 − ‖g‖)) ───────
            double theta = Math.max(0.995, 1.0 - gnorm);

            // ════════════════════════════════════════════════════════════════
            // Inner loop: trust-region step selection
            // ════════════════════════════════════════════════════════════════
            while (true) {
                // Copy upper triangle of fjac into rwork for lmpar (lmpar modifies r in-place)
                for (int j = 0; j < n; j++)
                    for (int i = 0; i <= j; i++)
                        work[rworkOff+i*n+j] = fjac[i*n+j];

                par = lmpar(n, work, rworkOff, ipvt,
                            work, effDiagOff,
                            work, qtfOff,
                            ws.delta, par,
                            work, wa1Off,   // x output → work[wa1Off] (step p)
                            work, wa3Off,   // sdiag scratch
                            wa2,            // wa1 scratch (eval not yet called, safe to reuse)
                            wa4);           // wa2 scratch (eval not yet called, safe to reuse)

                // work[wa1Off] now holds the unconstrained step p; negate it
                for (int j = 0; j < n; j++) work[wa1Off+j] = -work[wa1Off+j];

                // ── Reflective step: find best feasible step ──────────────────
                // p = work[wa1Off] (negated step), xNew = wa2 (standalone), step → work[stepOff]
                reflectiveStep(x, work, wa1Off, bounds, n,
                               theta,
                               wa2, work, stepOff,
                               work, JpOff, work, JprOff,
                               work, xHitOff, work, pRefOff, work, agOff,
                               m, fjac, ipvt, work, qtfOff, fnorm,
                               ws.delta);

                // pnorm based on actual step (Coleman-Li scaled, using effDiag = diag·clScale).
                // effDiag is computed before lmpar and not modified by it, so it's stable here.
                for (int j = 0; j < n; j++)
                    work[wa3Off+j] = work[effDiagOff+j] * work[stepOff+j];
                double pnorm = enorm(n, work, wa3Off);

                if (iter == 1 && pnorm > 0) ws.delta = Math.min(ws.delta, pnorm);

                // Push trial point strictly into interior before evaluating
                // (mirrors scipy's make_strictly_feasible with rstep=0).
                // Prevents CL scaling from collapsing to zero on the next outer iteration.
                makeStrictlyFeasible(wa2, bounds, n);
                fcn.evaluate(wa2, wa4, null);
                nfev++;
                // For LINEAR: cost1 = 0.5*‖f_new‖², fnorm1 = ‖f_new‖ (same as enorm).
                // For robust: cost1 = 0.5*C²*Σρ(f²/C²), fnorm1 = sqrt(2*cost1).
                double cost1  = loss.cost(wa4, m, lossScale);
                double fnorm1 = Math.sqrt(2.0 * cost1);

                double actred = -1.0;
                if (P1 * fnorm1 < fnorm) {
                    double r = fnorm1 / fnorm;
                    actred = 1.0 - r * r;
                }

                // Predicted reduction: wa3 = R·Pᵀ·step
                Arrays.fill(work, wa3Off, wa3Off + n, 0.0);
                for (int j = 0; j < n; j++) {
                    int l = ipvt[j];
                    double tmp = work[stepOff+l];
                    for (int i = 0; i <= j; i++) work[wa3Off+i] += fjac[i * n + j] * tmp;
                }
                double temp1  = enorm(n, work, wa3Off) / fnorm;
                double temp2  = (Math.sqrt(par) * pnorm) / fnorm;
                double prered = temp1 * temp1 + temp2 * temp2 / P5;
                double dirder = -(temp1 * temp1 + temp2 * temp2);

                double ratio = (prered != 0.0) ? actred / prered : 0.0;

                // ── Trust-region update ───────────────────────────────────────
                if (ratio <= P25) {
                    double tmp;
                    if (actred >= 0.0) tmp = P5;
                    else tmp = P5 * dirder / (dirder + P5 * actred);
                    if (P1 * fnorm1 >= fnorm || tmp < P1) tmp = P1;
                    ws.delta = tmp * Math.min(ws.delta, pnorm / P1);
                    par /= tmp;
                } else {
                    if (par == 0.0 || ratio >= P75) {
                        ws.delta = pnorm / P5;
                        par *= P5;
                    }
                }

                // ── Accept step ───────────────────────────────────────────────
                if (ratio >= P0001) {
                    System.arraycopy(wa2, 0, x, 0, n);
                    System.arraycopy(wa4, 0, fvec, 0, m);
                    cost = cost1;
                    clScaling(x, work, qtfOff, bounds, n, work, clScaleOff);
                    for (int j = 0; j < n; j++) work[effDiagOff+j] = work[diagOff+j] * work[clScaleOff+j];
                    for (int j = 0; j < n; j++) wa2[j] = work[effDiagOff+j] * x[j];  // xnorm scratch
                    ws.xnorm = enorm(n, wa2, 0);
                    fnorm = fnorm1;
                    iter++;
                }

                // ── Convergence checks ────────────────────────────────────────
                if (Math.abs(actred) <= ftol && prered <= ftol && P5 * ratio <= 1.0)
                    status = Optimization.Status.CHI_SQUARED_TOLERANCE_REACHED;
                if (ws.delta <= xtol * ws.xnorm)
                    status = Optimization.Status.COEFFICIENT_TOLERANCE_REACHED;
                if (status != null) break outer;

                if (nfev >= maxfev) { status = Optimization.Status.MAX_EVALUATIONS_REACHED; break outer; }
                if (Math.abs(actred) <= EPSMCH && prered <= EPSMCH && P5 * ratio <= 1.0)
                    { status = Optimization.Status.CHI_SQUARED_TOLERANCE_REACHED; break outer; }
                if (ws.delta <= EPSMCH * ws.xnorm)
                    { status = Optimization.Status.COEFFICIENT_TOLERANCE_REACHED; break outer; }
                if (gnorm <= EPSMCH)
                    { status = Optimization.Status.GRADIENT_TOLERANCE_REACHED; break outer; }

                if (ratio >= P0001) break;
            }
        }

        // For LINEAR: return ‖f‖² (RSS), consistent with original behavior.
        // For robust: return scipy cost = 0.5*C²*Σρ(f²/C²).
        double finalCost = (loss == RobustLoss.LINEAR) ? fnorm * fnorm : cost;
        return new Optimization(Double.NaN, null, finalCost, status, iter - 1, nfev);
    }
}
