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
import com.curioloop.numopt4j.optim.OptimizationResult;
import com.curioloop.numopt4j.optim.OptimizationStatus;

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
     * @param dot    scratch array of length n (reuses ws.qtf before applyQtToVec)
     */
    static void qrfac(int m, int n, double[] a, int[] ipvt,
                      double[] rdiag, double[] acnorm, double[] wa, double[] dot) {
        for (int j = 0; j < n; j++) { acnorm[j] = 0.0; rdiag[j] = 0.0; }
        for (int i = 0; i < m; i++) {
            int rowBase = i * n;
            for (int j = 0; j < n; j++) { double v = a[rowBase + j]; acnorm[j] += v * v; }
        }
        for (int j = 0; j < n; j++) {
            acnorm[j] = Math.sqrt(acnorm[j]);
            rdiag[j]  = acnorm[j];
            wa[j]     = rdiag[j];
            ipvt[j]   = j;
        }

        int minmn = Math.min(m, n);
        for (int j = 0; j < minmn; j++) {
            int kmax = j;
            for (int k = j + 1; k < n; k++) if (rdiag[k] > rdiag[kmax]) kmax = k;
            if (kmax != j) {
                for (int i = 0; i < m; i++) {
                    int base = i * n;
                    double tmp = a[base + j]; a[base + j] = a[base + kmax]; a[base + kmax] = tmp;
                }
                rdiag[kmax] = rdiag[j]; wa[kmax] = wa[j];
                int itmp = ipvt[j]; ipvt[j] = ipvt[kmax]; ipvt[kmax] = itmp;
            }

            double ajnorm = 0.0;
            for (int i = j; i < m; i++) { double v = a[i * n + j]; ajnorm += v * v; }
            ajnorm = Math.sqrt(ajnorm);
            if (ajnorm == 0.0) { rdiag[j] = 0.0; continue; }
            if (a[j * n + j] < 0.0) ajnorm = -ajnorm;
            for (int i = j; i < m; i++) a[i * n + j] /= ajnorm;
            a[j * n + j] += 1.0;

            int nk = n - j - 1;
            if (nk > 0) {
                for (int k = j + 1; k < n; k++) dot[k] = 0.0;
                for (int i = j; i < m; i++) {
                    double vij = a[i * n + j];
                    int rowBase = i * n + j + 1;
                    for (int k = 0; k < nk; k++) dot[j + 1 + k] += vij * a[rowBase + k];
                }
                double ajj = a[j * n + j];
                for (int k = j + 1; k < n; k++) dot[k] /= ajj;
                for (int i = j; i < m; i++) {
                    double vij = a[i * n + j];
                    int rowBase = i * n + j + 1;
                    for (int k = 0; k < nk; k++) a[rowBase + k] -= dot[j + 1 + k] * vij;
                }
                for (int k = j + 1; k < n; k++) {
                    if (rdiag[k] != 0.0) {
                        double t = a[j * n + k] / rdiag[k];
                        rdiag[k] *= Math.sqrt(Math.max(0.0, 1.0 - t * t));
                        if (0.05 * (rdiag[k] / wa[k]) * (rdiag[k] / wa[k]) <= EPSMCH) {
                            double s2 = 0.0;
                            for (int i = j + 1; i < m; i++) { double v = a[i * n + k]; s2 += v * v; }
                            rdiag[k] = Math.sqrt(s2); wa[k] = rdiag[k];
                        }
                    }
                }
            }
            rdiag[j] = -ajnorm;
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
    static void applyQtToVec(double[] fjac, int m, int n, double[] b, double[] rdiag) {
        int minmn = Math.min(m, n);
        for (int j = 0; j < minmn; j++) {
            if (fjac[j*n+j] == 0.0) { fjac[j*n+j] = rdiag[j]; continue; }
            double sum = 0.0;
            for (int i = j; i < m; i++) sum += fjac[i*n+j] * b[i];
            double tmp = -sum / fjac[j*n+j];
            for (int i = j; i < m; i++) b[i] += fjac[i*n+j] * tmp;
            fjac[j*n+j] = rdiag[j];
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
    static void qrsolv(int n, double[] r, int[] ipvt, double[] diag,
                       double[] qtb, double[] x, double[] sdiag, double[] wa) {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) r[i*n+j] = r[j*n+i];
            x[j]  = r[j*n+j];
            wa[j] = qtb[j];
        }
        for (int j = 0; j < n; j++) {
            int l = ipvt[j];
            if (diag[l] == 0.0) { sdiag[j] = r[j*n+j]; r[j*n+j] = x[j]; continue; }
            for (int k = j; k < n; k++) sdiag[k] = 0.0;
            sdiag[j] = diag[l];
            double qtbpj = 0.0;
            for (int k = j; k < n; k++) {
                if (sdiag[k] == 0.0) continue;
                double cos, sin;
                if (Math.abs(r[k*n+k]) >= Math.abs(sdiag[k])) {
                    double tan = sdiag[k] / r[k*n+k];
                    cos = 0.5 / Math.sqrt(0.25 + 0.25*tan*tan);
                    sin = cos * tan;
                } else {
                    double cotan = r[k*n+k] / sdiag[k];
                    sin = 0.5 / Math.sqrt(0.25 + 0.25*cotan*cotan);
                    cos = sin * cotan;
                }
                r[k*n+k] = cos*r[k*n+k] + sin*sdiag[k];
                double temp = cos*wa[k] + sin*qtbpj;
                qtbpj = -sin*wa[k] + cos*qtbpj;
                wa[k] = temp;
                for (int i = k+1; i < n; i++) {
                    temp      =  cos*r[i*n+k] + sin*sdiag[i];
                    sdiag[i]  = -sin*r[i*n+k] + cos*sdiag[i];
                    r[i*n+k]  = temp;
                }
            }
            sdiag[j] = r[j*n+j];
            r[j*n+j] = x[j];
        }
        int nsing = n;
        for (int j = 0; j < n; j++) {
            if (sdiag[j] == 0.0 && nsing == n) nsing = j;
            if (nsing < n) wa[j] = 0.0;
        }
        for (int k = 0; k < nsing; k++) {
            int j = nsing - 1 - k;
            double sum = 0.0;
            for (int i = j+1; i < nsing; i++) sum += r[i*n+j] * wa[i];
            wa[j] = (wa[j] - sum) / sdiag[j];
        }
        for (int j = 0; j < n; j++) x[ipvt[j]] = wa[j];
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
     * @param x      output: solution p(λ) (length n)
     * @param sdiag  scratch: modified diagonal from qrsolv (length n)
     * @param wa1    scratch array (length n)
     * @param wa2    scratch array (length n)
     * @return       updated Levenberg-Marquardt parameter λ
     */
    static double lmpar(int n, double[] r, int[] ipvt, double[] diag,
                        double[] qtb, double delta, double par,
                        double[] x, double[] sdiag, double[] wa1, double[] wa2) {
        final double dwarf = Double.MIN_VALUE;
        final double p1 = 0.1, p001 = 0.001;

        int nsing = n;
        System.arraycopy(qtb, 0, wa1, 0, n);
        for (int j = 0; j < n; j++) {
            if (r[j*n+j] == 0.0 && nsing == n) nsing = j;
            if (nsing < n) wa1[j] = 0.0;
        }
        for (int k = 0; k < nsing; k++) {
            int j = nsing - 1 - k;
            wa1[j] /= r[j*n+j];
            double tmp = wa1[j];
            for (int i = 0; i < j; i++) wa1[i] -= r[i*n+j] * tmp;
        }
        for (int j = 0; j < n; j++) x[ipvt[j]] = wa1[j];

        int iter = 0;
        for (int j = 0; j < n; j++) wa2[j] = diag[j] * x[j];
        double dxnorm = enorm(n, wa2, 0);
        double fp = dxnorm - delta;
        if (fp <= p1 * delta) { if (iter == 0) par = 0.0; return par; }

        double parl = 0.0;
        if (nsing >= n) {
            for (int j = 0; j < n; j++) {
                int l = ipvt[j];
                wa1[j] = diag[l] * (wa2[l] / dxnorm);
            }
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < j; i++) sum += r[i*n+j] * wa1[i];
                wa1[j] = (wa1[j] - sum) / r[j*n+j];
            }
            double tmp = enorm(n, wa1, 0);
            parl = ((fp / delta) / tmp) / tmp;
        }

        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i <= j; i++) sum += r[i*n+j] * qtb[i];
            int l = ipvt[j];
            wa1[j] = sum / diag[l];
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
            for (int j = 0; j < n; j++) wa1[j] = sqrtPar * diag[j];
            qrsolv(n, r, ipvt, wa1, qtb, x, sdiag, wa2);
            for (int j = 0; j < n; j++) wa2[j] = diag[j] * x[j];
            dxnorm = enorm(n, wa2, 0);
            double fpOld = fp;
            fp = dxnorm - delta;

            if (Math.abs(fp) <= p1 * delta
                || (parl == 0.0 && fp <= fpOld && fpOld < 0.0)
                || iter == 10) break;

            for (int j = 0; j < n; j++) {
                int l = ipvt[j];
                wa1[j] = diag[l] * (wa2[l] / dxnorm);
            }
            for (int j = 0; j < n; j++) {
                wa1[j] /= sdiag[j];
                double tmp = wa1[j];
                for (int i = j+1; i < n; i++) wa1[i] -= r[i*n+j] * tmp;  // lower triangle (S factor from qrsolv)
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
    static void clScaling(double[] x, double[] qtf, Bound[] bounds, int n, double[] v) {
        if (bounds == null) {
            Arrays.fill(v, 0, n, 1.0);
            return;
        }
        for (int i = 0; i < n; i++) {
            Bound b = bounds[i];
            if (b == null || b.isUnbounded()) {
                v[i] = 1.0;
                continue;
            }
            double dLo = b.hasLower() ? (x[i] - b.getLower()) : Double.MAX_VALUE;
            double dHi = b.hasUpper() ? (b.getUpper() - x[i]) : Double.MAX_VALUE;

            double g = (qtf != null) ? qtf[i] : 0.0;
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
            v[i] = Math.max(1e-10, dist);
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
     * @param p      unconstrained step direction from lmpar (length n)
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
    static void reflectiveStep(double[] x, double[] p, Bound[] bounds, int n,
                                double theta,
                                double[] xNew, double[] step,
                                double[] Jp, double[] Jpr,
                                double[] xHit, double[] pRef, double[] ag,
                                int m,
                                double[] fjac, int[] ipvt, double[] qtf, double fnorm,
                                double delta) {
        if (bounds == null) {
            for (int i = 0; i < n; i++) { xNew[i] = x[i] + p[i]; step[i] = p[i]; }
            return;
        }

        // ── Find first bound hit along p ──────────────────────────────────────
        double tHit = 1.0;
        int hitIdx = -1;
        for (int i = 0; i < n; i++) {
            Bound b = bounds[i];
            if (b == null || b.isUnbounded()) continue;
            if (p[i] < 0 && b.hasLower()) {
                double t = (b.getLower() - x[i]) / p[i];
                if (t > 0 && t < tHit) { tHit = t; hitIdx = i; }
            } else if (p[i] > 0 && b.hasUpper()) {
                double t = (b.getUpper() - x[i]) / p[i];
                if (t > 0 && t < tHit) { tHit = t; hitIdx = i; }
            }
        }

        if (hitIdx < 0) {
            // No bound hit: use full step (clipped for safety)
            for (int i = 0; i < n; i++) { xNew[i] = x[i] + p[i]; step[i] = p[i]; }
            applyBounds(xNew, bounds, n);
            for (int i = 0; i < n; i++) step[i] = xNew[i] - x[i];
            return;
        }

        // ── Candidate 1: trust-region step, stepped back by theta ────────────
        // p_tr = theta * tHit * p  (stop just before the bound)
        double tTR = theta * tHit;

        // ── Candidate 2: reflective step ──────────────────────────────────────
        // Point where we hit the bound
        for (int i = 0; i < n; i++) xHit[i] = x[i] + tHit * p[i];

        // Reflected direction: flip the component that hit the bound
        System.arraycopy(p, 0, pRef, 0, n);
        pRef[hitIdx] = -pRef[hitIdx];
        double tRemain = 1.0 - tHit;

        // Clip reflected direction to stay feasible
        double tRef = tRemain;
        for (int i = 0; i < n; i++) {
            if (pRef[i] == 0) continue;
            Bound b = bounds[i];
            if (b == null || b.isUnbounded()) continue;
            if (pRef[i] < 0 && b.hasLower()) {
                double t = (b.getLower() - xHit[i]) / pRef[i];
                if (t >= 0 && t < tRef) tRef = t;
            } else if (pRef[i] > 0 && b.hasUpper()) {
                double t = (b.getUpper() - xHit[i]) / pRef[i];
                if (t >= 0 && t < tRef) tRef = t;
            }
        }
        // Step back reflected stride by theta as well
        double rStrideL = (1.0 - theta) * tHit / Math.max(tRef, 1e-300);
        double rStrideU = theta * tRef;
        // tOpt for reflected path is in [rStrideL, rStrideU]

        // ── Candidate 3: Cauchy step along -g (true anti-gradient) ──────────
        // ag[i] = -g[i] = -(J^T f)[i], the steepest descent direction.
        // g = J^T f = P R^T (Q^T f) = P R^T qtf, so g[ipvt[j]] = sum_{i<=j} R[i,j]*qtf[i].
        // Project out components blocked at a bound: if ag[i] > 0 and x[i] is at its
        // upper bound (dHi < threshold), or ag[i] < 0 and x[i] is at its lower bound
        // (dLo < threshold), set ag[i] = 0.  This mirrors scipy's hat-space projection
        // (d_h[i] = 0 when v[i] ≈ 0) and prevents the Cauchy step from being collapsed
        // to near-zero by a single variable that is already at a bound.
        double agNorm = 0.0;
        // First compute g = J^T f = P R^T qtf, then negate to get ag = -g
        for (int i = 0; i < n; i++) ag[i] = 0.0;
        for (int j = 0; j < n; j++) {
            int l = ipvt[j];
            double gj = 0.0;
            for (int i = 0; i <= j; i++) gj += fjac[i * n + j] * qtf[i];
            ag[l] = -gj; // ag[l] = -g[l]
        }
        for (int i = 0; i < n; i++) {
            double ai = ag[i];
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
            ag[i] = ai;
            agNorm += ai * ai;
        }
        agNorm = Math.sqrt(agNorm);
        // Clip Cauchy direction to trust-region ball radius (Delta) and bounds.
        // scipy uses to_tr = Delta / norm(ag_h), NOT pNorm / agNorm.
        // Using pNorm would collapse the Cauchy step to zero when tHit≈0 (x at bound),
        // preventing convergence of unconstrained variables in the same iteration.
        double tCauchy = (agNorm > 0) ? 1.0 : 0.0;
        if (agNorm > 0) {
            // Scale so that ||ag * tCauchy|| ≈ Delta (trust-region radius)
            tCauchy = delta / agNorm;
            // Clip to bounds
            for (int i = 0; i < n; i++) {
                if (ag[i] == 0) continue;
                Bound b = bounds[i];
                if (b == null || b.isUnbounded()) continue;
                if (ag[i] < 0 && b.hasLower()) {
                    double t = (b.getLower() - x[i]) / ag[i];
                    if (t >= 0 && t < tCauchy) tCauchy = t;
                } else if (ag[i] > 0 && b.hasUpper()) {
                    double t = (b.getUpper() - x[i]) / ag[i];
                    if (t >= 0 && t < tCauchy) tCauchy = t;
                }
            }
            tCauchy *= theta;
        }

        // ── Compute R·P^T vectors for quadratic model evaluation ─────────────
        // Jp  = R·P^T·(tHit·p)
        // Jpr = R·P^T·pRef
        // Jag = R·P^T·ag  (reuse xHit as scratch for Jag)
        Arrays.fill(Jp,   0, n, 0.0);
        Arrays.fill(Jpr,  0, n, 0.0);
        Arrays.fill(xHit, 0, n, 0.0); // reuse as Jag scratch
        for (int j = 0; j < n; j++) {
            int l = ipvt[j];
            double vp   = tHit * p[l];
            double vpr  = pRef[l];
            double vag  = ag[l];
            for (int i = 0; i <= j; i++) {
                double fij = fjac[i * n + j];
                Jp[i]   += fij * vp;
                Jpr[i]  += fij * vpr;
                xHit[i] += fij * vag;
            }
        }

        // Quadratic model: ||Q^T·f/fnorm + R·P^T·s||^2
        // Evaluate at three candidates and pick the best
        double invFnorm = (fnorm > 0) ? 1.0 / fnorm : 1.0;

        // Candidate 1: TR step at tTR (= theta*tHit along p, so Jp scaled by tTR/tHit = theta)
        double valTR = 0;
        for (int i = 0; i < n; i++) {
            double qi = qtf[i] * invFnorm;
            double d  = qi + (tTR / Math.max(tHit, 1e-300)) * Jp[i];
            valTR += d * d;
        }

        // Candidate 2: reflective — find optimal t along reflected path
        double aRef = 0, bRef = 0;
        for (int i = 0; i < n; i++) {
            double qi = qtf[i] * invFnorm;
            aRef += Jpr[i] * Jpr[i];
            bRef += (Jp[i] + qi) * Jpr[i];
        }
        double tOptRef = (aRef > 0) ? Math.max(rStrideL, Math.min(rStrideU, -bRef / (2 * aRef))) : rStrideL;
        double valRef = 0;
        for (int i = 0; i < n; i++) {
            double qi = qtf[i] * invFnorm;
            double d  = qi + Jp[i] + Jpr[i] * tOptRef;
            valRef += d * d;
        }

        // Candidate 3: Cauchy step
        double valCauchy = 0;
        if (agNorm > 0) {
            for (int i = 0; i < n; i++) {
                double qi = qtf[i] * invFnorm;
                double d  = qi + xHit[i] * tCauchy; // xHit reused as Jag
                valCauchy += d * d;
            }
        } else {
            valCauchy = Double.MAX_VALUE;
        }

        // ── Restore xHit to actual bound-hit point ────────────────────────────
        for (int i = 0; i < n; i++) xHit[i] = x[i] + tHit * p[i];

        // ── Pick best candidate ───────────────────────────────────────────────
        if (valTR <= valRef && valTR <= valCauchy) {
            // Candidate 1: step back along p
            for (int i = 0; i < n; i++) {
                xNew[i] = x[i] + tTR * p[i];
                step[i] = tTR * p[i];
            }
        } else if (valRef <= valCauchy) {
            // Candidate 2: reflective
            for (int i = 0; i < n; i++) {
                xNew[i] = xHit[i] + tOptRef * pRef[i];
                step[i] = xNew[i] - x[i];
            }
        } else {
            // Candidate 3: Cauchy
            for (int i = 0; i < n; i++) {
                xNew[i] = x[i] + tCauchy * ag[i];
                step[i] = tCauchy * ag[i];
            }
        }
        applyBounds(xNew, bounds, n);
        for (int i = 0; i < n; i++) step[i] = xNew[i] - x[i];
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
    static OptimizationResult optimize(int m, int n,
                                       Multivariate fcn,
                                       double ftol, double xtol, double gtol,
                                       int maxfev, double factor,
                                       double[] userDiag, Bound[] bounds,
                                       RobustLoss loss, double lossScale,
                                       double[] x, TRFWorkspace ws) {

        double[] fvec    = ws.fvec;
        double[] fjac    = ws.fjac;
        double[] rwork   = ws.rwork;
        double[] diag    = ws.diag;
        double[] qtf     = ws.qtf;
        double[] wa1     = ws.wa1;
        double[] wa2     = ws.wa2;
        double[] wa3     = ws.wa3;
        double[] wa4     = ws.wa4;
        int[]    ipvt    = ws.ipvt;
        double[] clScale = ws.clScale;
        double[] effDiag = ws.effDiag;
        double[] Jp      = ws.Jp;
        double[] Jpr     = ws.Jpr;
        double[] step    = ws.step;
        double[] xHit    = ws.xHit;
        double[] pRef    = ws.pRef;
        double[] ag      = ws.ag;

        applyBounds(x, bounds, n);

        fcn.evaluate(x, fvec, null);
        int nfev = 1;
        double fnorm = enorm(m, fvec, 0);

        double par  = 0.0;
        int    iter = 1;
        OptimizationStatus status = null;

        // Tracks cost at current x; updated at outer loop top and on each accepted step.
        // For LINEAR: cost = 0.5*||f||²; for robust: cost = 0.5*C²*Σρ(f²/C²).
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
            // qtf is passed as dot[] scratch — safe because qrfac runs before applyQtToVec
            qrfac(m, n, fjac, ipvt, wa1, wa2, wa3, qtf);

            // For robust loss, use cost-based effective norm so that actred and prered
            // are computed on the same scale: fnorm² ≈ 2*cost, fnorm = sqrt(2*cost).
            // For LINEAR, cost = 0.5*||f||², so sqrt(2*cost) = ||f|| — identical.
            fnorm = Math.sqrt(2.0 * cost);

            // ── Coleman-Li scaling from current x ────────────────────────────
            // qtf not yet computed on first call; pass null to fall back to geometric distance
            clScaling(x, (iter == 1) ? null : qtf, bounds, n, clScale);

            // ── Initialize scaling diagonal on first iteration ────────────────
            if (iter == 1) {
                if (userDiag == null) {
                    for (int j = 0; j < n; j++) diag[j] = (wa2[j] != 0.0) ? wa2[j] : 1.0;
                } else {
                    System.arraycopy(userDiag, 0, diag, 0, n);
                }
                for (int j = 0; j < n; j++) wa3[j] = diag[j] * clScale[j] * x[j];
                double xnorm = enorm(n, wa3, 0);
                double delta = (xnorm != 0.0) ? factor * xnorm : factor;
                ws.delta = delta;
                ws.xnorm = xnorm;
            }

            // ── Compute Q^T·f ─────────────────────────────────────────────────
            System.arraycopy(fvec, 0, wa4, 0, m);
            applyQtToVec(fjac, m, n, wa4, wa1);
            System.arraycopy(wa4, 0, qtf, 0, n);

            // ── Gradient norm (Coleman-Li scaled) ─────────────────────────────
            // Also compute the true gradient g = J^T f = P R^T (Q^T f) = P R^T qtf,
            // stored in wa3[ipvt[j]] = sum_i fjac[i*n+j] * qtf[i].
            // This is needed for clScaling (which requires the true gradient direction,
            // not qtf = Q^T f whose sign may differ from g due to Q's sign convention).
            double gnorm = 0.0;
            if (fnorm != 0.0) {
                for (int j = 0; j < n; j++) {
                    int l = ipvt[j];
                    double sum = 0.0;
                    for (int i = 0; i <= j; i++) sum += fjac[i * n + j] * qtf[i];
                    wa3[l] = sum; // true gradient g[l] = (R^T qtf)[j] mapped back via P
                    if (wa2[l] == 0.0) continue;
                    gnorm = Math.max(gnorm, Math.abs((sum / fnorm) * clScale[l] / wa2[l]));
                }
            }

            if (gnorm <= gtol) { status = OptimizationStatus.GRADIENT_TOLERANCE_REACHED; break; }

            if (userDiag == null) {
                for (int j = 0; j < n; j++) diag[j] = Math.max(diag[j], wa2[j]);
            }

            // ── Effective lmpar diagonal: diag * clScale ──────────────────────
            // SciPy uses effDiag = diag * sqrt(v) * x_scale (hat-space scaling),
            // and augments the trust-region Hessian with C = diag(g * dv * scale)
            // to account for CL-transformation curvature near bounds. We omit C
            // and use MINPACK-style lmpar (positive-definite J^T J only), which is
            // ~3-5× faster per outer iteration at the cost of potentially more
            // iterations when variables are actively constrained at bounds.
            for (int j = 0; j < n; j++) effDiag[j] = diag[j] * clScale[j];

            // ── Freeze variables at active bounds for lmpar ───────────────────
            // When a variable is at a bound in the gradient direction (dHi/dLo < 1e-8),
            // clScaling returns 1.0 to prevent effDiag collapse. But lmpar will still
            // try to step through the bound, producing tHit ≈ 0 and collapsing the
            // entire TR step. We mimic scipy's hat-space effect (d[i] = 0 → J_h col = 0
            // → p_h[i] = 0) by inflating effDiag[i] to a large value, forcing lmpar
            // to produce p[i] ≈ 0 for the frozen variable. The pnorm is computed with
            // the original effDiag (not inflated) so delta updates remain correct.
            //
            // lmpar gives p = -(J^T J + λD^T D)^{-1} J^T f ≈ -H^{-1} g:
            //   g[j] < 0 → p[j] > 0 (wants to increase x[j]) → blocked if dHi ≈ 0
            //   g[j] > 0 → p[j] < 0 (wants to decrease x[j]) → blocked if dLo ≈ 0
            // Use true gradient g = J^T f (stored in wa3 from gnorm computation),
            // NOT qtf = Q^T f (whose sign may differ from g due to Q's sign convention).
            if (bounds != null) {
                for (int j = 0; j < n; j++) {
                    Bound b = bounds[j];
                    if (b == null || b.isUnbounded()) continue;
                    double dLo = b.hasLower() ? (x[j] - b.getLower()) : Double.MAX_VALUE;
                    double dHi = b.hasUpper() ? (b.getUpper() - x[j]) : Double.MAX_VALUE;
                    double gj  = wa3[j]; // true gradient g[j] = (J^T f)[j]
                    if ((gj < 0 && dHi < 1e-8) || (gj > 0 && dLo < 1e-8)) {
                        effDiag[j] = diag[j] * 1e10; // inflate to freeze
                    }
                }
            }

            // ── theta: step-back fraction (SciPy: max(0.995, 1 - ||g||)) ──────
            double theta = Math.max(0.995, 1.0 - gnorm);

            // ════════════════════════════════════════════════════════════════
            // Inner loop: trust-region step selection
            // ════════════════════════════════════════════════════════════════
            while (true) {
                for (int j = 0; j < n; j++)
                    for (int i = 0; i <= j; i++)
                        rwork[i*n+j] = fjac[i*n+j];

                par = lmpar(n, rwork, ipvt, effDiag, qtf, ws.delta, par, wa1, wa2, wa3, wa4);

                // wa1 = unconstrained step p (sign-flipped)
                for (int j = 0; j < n; j++) wa1[j] = -wa1[j];

                // ── Reflective step: find best feasible step ──────────────────
                reflectiveStep(x, wa1, bounds, n,
                               theta,
                               wa2, step,
                               Jp, Jpr, xHit, pRef, ag, m, fjac, ipvt, qtf, fnorm,
                               ws.delta);

                // pnorm based on actual step (Coleman-Li scaled, using base effDiag without
                // the freeze inflation applied above). Frozen variables have step ≈ 0 anyway,
                // but using diag*clScale avoids any 1e10 * ~0 numerical noise.
                for (int j = 0; j < n; j++) wa3[j] = diag[j] * clScale[j] * step[j];
                double pnorm = enorm(n, wa3, 0);

                if (iter == 1 && pnorm > 0) ws.delta = Math.min(ws.delta, pnorm);

                // Push trial point strictly into interior before evaluating
                // (mirrors scipy's make_strictly_feasible with rstep=0).
                // Prevents CL scaling from collapsing to zero on the next outer iteration.
                makeStrictlyFeasible(wa2, bounds, n);
                fcn.evaluate(wa2, wa4, null);
                nfev++;
                // For LINEAR: cost1 = 0.5*||f_new||², fnorm1 = ||f_new|| (same as enorm).
                // For robust: cost1 = 0.5*C²*Σρ(f²/C²), fnorm1 = sqrt(2*cost1).
                double cost1  = loss.cost(wa4, m, lossScale);
                double fnorm1 = Math.sqrt(2.0 * cost1);

                double actred = -1.0;
                if (P1 * fnorm1 < fnorm) {
                    double r = fnorm1 / fnorm;
                    actred = 1.0 - r * r;
                }                Arrays.fill(wa3, 0, n, 0.0);
                for (int j = 0; j < n; j++) {
                    int l = ipvt[j];
                    double tmp = step[l];
                    for (int i = 0; i <= j; i++) wa3[i] += fjac[i * n + j] * tmp;
                }
                double temp1  = enorm(n, wa3, 0) / fnorm;
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
                    cost = cost1; // update to cost at new x
                    clScaling(x, qtf, bounds, n, clScale);
                    for (int j = 0; j < n; j++) effDiag[j] = diag[j] * clScale[j];
                    for (int j = 0; j < n; j++) wa2[j] = effDiag[j] * x[j];
                    ws.xnorm = enorm(n, wa2, 0);
                    fnorm = fnorm1;
                    iter++;
                }

                // ── Convergence checks ────────────────────────────────────────
                if (Math.abs(actred) <= ftol && prered <= ftol && P5 * ratio <= 1.0)
                    status = OptimizationStatus.CHI_SQUARED_TOLERANCE_REACHED;
                if (ws.delta <= xtol * ws.xnorm)
                    status = OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED;
                if (status != null) break outer;

                if (nfev >= maxfev) { status = OptimizationStatus.MAX_EVALUATIONS_REACHED; break outer; }
                if (Math.abs(actred) <= EPSMCH && prered <= EPSMCH && P5 * ratio <= 1.0)
                    { status = OptimizationStatus.CHI_SQUARED_TOLERANCE_REACHED; break outer; }
                if (ws.delta <= EPSMCH * ws.xnorm)
                    { status = OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED; break outer; }
                if (gnorm <= EPSMCH)
                    { status = OptimizationStatus.GRADIENT_TOLERANCE_REACHED; break outer; }

                if (ratio >= P0001) break;
            }
        }

        // For LINEAR: return ||f||² (RSS), consistent with original behavior.
        // For robust: return scipy cost = 0.5*C²*Σρ(f²/C²).
        double objectiveValue = (loss == RobustLoss.LINEAR) ? fnorm * fnorm : cost;
        return new OptimizationResult(objectiveValue, status, iter - 1, nfev);
    }
}
