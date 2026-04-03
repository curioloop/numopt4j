/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

import com.curioloop.numopt4j.linalg.blas.Dsyev;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Optimization;

import com.curioloop.numopt4j.optim.Univariate;
import java.util.Arrays;
import java.util.Random;

/**
 * Stateless CMA-ES core algorithm.
 *
 * <p>All mutable state lives in {@link CMAESWorkspace}; every method here is static.
 * Configuration is read from {@link CMAESProblem}.</p>
 *
 * <p>All matrix operations use flat {@code double[]} row-major layout (element (i,j)
 * at index {@code i*n+j}) and call BLAS directly.  The hot path allocates zero objects
 * per iteration — all scratch space is pre-allocated in the workspace arenas.</p>
 *
 * <h2>Per-iteration call order</h2>
 * <pre>
 *  1. sampleOffspring      — draw λ candidates x_k ~ 𝒩(m, σ²C)
 *  2. evaluateFitness      — evaluate f(x_k), apply boundary penalty
 *  3. sortIndices          — rank candidates by fitness
 *  4. updateMean           — m ← Σ wᵢ x_{i:λ}  (weighted recombination)
 *  5. updateEvolutionPaths — update p_σ and p_c
 *  6. updateCovariance     — update C (full) or diagC (sep)
 *  7. updateSigma          — CSA step-size control
 *  8. eigenDecompose       — C = B·diag(D²)·Bᵀ  (scheduled, full mode only)
 * </pre>
 *
 * <h2>Scratch-slot timing (arena reuse)</h2>
 * <pre>
 *  nVec[EVALS]  — step 1: y = D⊙z scratch
 *                 step 2: xbuf passed to fn  (via ws.xbuf alias)
 *                 step 5: C^{−½}·(m−m_old) scratch
 *                 step 8: dsyev eigenvalue output
 *                 → steps 1/5/8 never overlap with step 2 ✓
 *  nVec[DIAGC]  — step 6 (full mode): δ_k = (x_k−m_old)/σ scratch
 *                 diagC is unused in full mode, so the slot is free ✓
 *  lVec[PENALTY]— step 2: boundary-penalty scratch
 *                 does not overlap with FITNESS or WEIGHTS ✓
 * </pre>
 */
public final class CMAESCore {

    private CMAESCore() {}

    // ── Parameter initialization ──────────────────────────────────────────

    /**
     * Initialises all CMA-ES strategy parameters from scratch.
     * Called once at the start of each {@code optimize()} run.
     *
     * <h3>Positive weights (i = 0…μ−1)</h3>
     * <pre>
     *   w̃ᵢ = ln(μ + 0.5) − ln(i + 1)
     *   wᵢ  = w̃ᵢ / Σ w̃ⱼ          (normalised, sum = 1)
     *   μ_eff = (Σwᵢ)² / Σwᵢ²
     * </pre>
     *
     * <h3>Strategy parameters (pycma formulas, independent of maxIterations)</h3>
     * <pre>
     *   c_c    = (4 + μ_eff/n) / (n + 4 + 2μ_eff/n)
     *   c_s    = (μ_eff + 2) / (n + μ_eff + 5)
     *   c_1    = 2 / ((n+1.3)² + μ_eff)
     *   c_μ    = min(1−c_1, 2·(μ_eff−2+1/μ_eff) / ((n+2)²+μ_eff))
     *   d_eff  = max(1, 3·(1 − 0.5^{n/10}))
     *   d_amps = 0.5 + 2·d_eff·max(0, d_eff·√((μ_eff−1)/(n+1)) − 1) + c_s
     *   χ_n    ≈ √n·(1 − 1/(4n) + 1/(21n²))
     * </pre>
     *
     * <h3>Negative weights (Active CMA, i = μ…λ−1)</h3>
     * <p>Three-step finalisation (mirrors pycma {@code finalize_negative_weights})
     * to guarantee the zero-drift condition |c_1 + c_μ·Σwᵢ| &lt; 1e-10:</p>
     * <ol>
     *   <li>Scale so that |Σ neg| = 1 + c_1/c_μ  (zero-drift target)</li>
     *   <li>Clamp so that |Σ neg| ≤ (1−c_1−c_μ)/(c_μ·n)  (positive-definiteness)</li>
     *   <li>Clamp so that |Σ neg| ≤ 1 + 2μ_eff⁻/(μ_eff+2)  (μ_eff constraint)</li>
     * </ol>
     *
     * @param ws            workspace to initialise
     * @param maxIterations kept for API compatibility; NOT used in any formula
     */
    static void initParams(CMAESWorkspace ws, int maxIterations) {
        final int n = ws.n;
        final int lambda = ws.lambda;
        final double[] lVec = ws.lVec;
        final int WEIGHTS = ws.WEIGHTS;

        // mu = floor(lambda/2)
        ws.mu = lambda / 2;
        final int mu = ws.mu;

        // ── Step 1: Positive weights ──────────────────────────────────────
        // w_i = ln(mu+0.5) - ln(i+1), normalized so sum = 1
        double sumw = 0, sumwq = 0;
        for (int i = 0; i < mu; i++) {
            double w = Math.log(mu + 0.5) - Math.log(i + 1);
            lVec[WEIGHTS + i] = w;
            sumw  += w;
            sumwq += w * w;
        }
        for (int i = 0; i < mu; i++) lVec[WEIGHTS + i] /= sumw;
        ws.mueff = sumw * sumw / sumwq;

        // ── Step 2: Strategy parameters (pycma formulas, no maxIterations) ─
        ws.cc      = (4.0 + ws.mueff / n) / (n + 4.0 + 2.0 * ws.mueff / n);
        ws.cs      = (ws.mueff + 2.0) / (n + ws.mueff + 5.0);
        ws.ccov1   = 2.0 / ((n + 1.3) * (n + 1.3) + ws.mueff);
        ws.ccovmu  = Math.min(1.0 - ws.ccov1,
                       2.0 * (ws.mueff - 2.0 + 1.0 / ws.mueff) / ((n + 2.0) * (n + 2.0) + ws.mueff));
        double dampInEff = Math.max(1.0, 3.0 * (1.0 - Math.pow(0.5, n / 10.0)));
        ws.damps   = 0.5 + 2.0 * dampInEff
                   * Math.max(0.0, dampInEff * Math.sqrt((ws.mueff - 1.0) / (n + 1.0)) - 1.0)
                   + ws.cs;
        ws.chiN    = Math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));

        final double c1  = ws.ccov1;
        final double cmu = ws.ccovmu;

        // ── Step 3: Raw negative weights (i = mu..lambda-1) ──────────────
        double sumNeg = 0, sumNegSq = 0;
        for (int i = mu; i < lambda; i++) {
            double w = Math.log(mu + 0.5) - Math.log(i + 1);
            lVec[WEIGHTS + i] = w;
            sumNeg  += w;
            sumNegSq += w * w;
        }
        ws.mueffNeg = (sumNeg != 0) ? sumNeg * sumNeg / sumNegSq : 0.0;
        final double mueffMinus = ws.mueffNeg;

        if (sumNeg < 0) {
            for (int i = mu; i < lambda; i++) lVec[WEIGHTS + i] /= (-sumNeg);
        }

        // ── Step 4: Three-step finalization ───────────────────────────────
        if (cmu > 0) {
            double negScale = 1.0 + c1 / cmu;
            for (int i = mu; i < lambda; i++) lVec[WEIGHTS + i] *= negScale;

            double posDefLimit = (1.0 - c1 - cmu) / cmu / n;
            double curAbsSum = 0;
            for (int i = mu; i < lambda; i++) curAbsSum -= lVec[WEIGHTS + i];
            if (curAbsSum > posDefLimit) {
                double factor = posDefLimit / curAbsSum;
                for (int i = mu; i < lambda; i++) lVec[WEIGHTS + i] *= factor;
            }
        }
        double mueffLimit = 1.0 + 2.0 * mueffMinus / (ws.mueff + 2.0);
        double curAbsSum2 = 0;
        for (int i = mu; i < lambda; i++) curAbsSum2 -= lVec[WEIGHTS + i];
        if (curAbsSum2 > mueffLimit) {
            double factor = mueffLimit / curAbsSum2;
            for (int i = mu; i < lambda; i++) lVec[WEIGHTS + i] *= factor;
        }

        if (cmu == 0) {
            for (int i = mu; i < lambda; i++) lVec[WEIGHTS + i] = 0.0;
        }

        // ── Step 5: Diagonal mode learning rates ──────────────────────────
        ws.ccov1Sep  = Math.min(1.0, c1  * (n + 1.5) / 3.0);
        ws.ccovmuSep = Math.min(1.0 - ws.ccov1Sep, cmu * (n + 1.5) / 3.0);

        // ── Step 6: Eigendecomposition interval ───────────────────────────
        ws.eigenInterval = Math.max(1,
            (int) Math.floor(lambda / (c1 + cmu + 1e-5) / n / 10.0));
    }


    // ── Offspring sampling ────────────────────────────────────────────────

    /**
     * Samples λ offspring from the current distribution 𝒩(m, σ²C).
     *
     * <p><b>Full-matrix mode</b> ({@code diagonalOnly=false}):</p>
     * <pre>
     *   z_k ~ 𝒩(0, I)
     *   y_k  = D ⊙ z_k                    (stored in nVec[EVALS] scratch)
     *   x_k  = m + σ · B · y_k
     * </pre>
     *
     * <p><b>Sep mode</b> ({@code diagonalOnly=true}):</p>
     * <pre>
     *   z_k ~ 𝒩(0, I)
     *   x_k  = m + σ · (diagD ⊙ z_k)
     * </pre>
     *
     * <p>If {@code resampleThreshold > 0} and bounds are provided, infeasible samples
     * are redrawn up to {@code resampleThreshold} times; the last sample is kept even
     * if still infeasible (boundary penalty handles it in {@code evaluateFitness}).</p>
     *
     * @param ws                workspace
     * @param diagonalOnly      {@code true} for sep-CMA-ES
     * @param rng               random number generator
     * @param bounds            variable bounds, or {@code null} for unconstrained
     * @param checkFeasible number of resampling attempts per infeasible individual
     */
    static void sampleOffspring(CMAESWorkspace ws, boolean diagonalOnly,
                                 Random rng, Bound[] bounds, int checkFeasible) {
        final int n = ws.n;
        final int lambda = ws.lambda;
        final double[] nVec = ws.nVec;
        final int XMEAN = ws.XMEAN, D_OFF = ws.D_OFF, DIAGD = ws.DIAGD, EVALS = ws.EVALS;
        final int B_OFF = ws.B_OFF;

        for (int k = 0; k < lambda; k++) {
            int attempts = checkFeasible + 1;

            while (attempts-- > 0) {
                for (int i = 0; i < n; i++) {
                    ws.arz[i * lambda + k] = rng.nextGaussian();
                }

                if (diagonalOnly) {
                    for (int i = 0; i < n; i++) {
                        ws.arx[i * lambda + k] = nVec[XMEAN + i]
                            + ws.sigma * nVec[DIAGD + i] * ws.arz[i * lambda + k];
                    }
                } else {
                    // y = D ⊙ z_k stored in EVALS scratch slot
                    for (int i = 0; i < n; i++) {
                        nVec[EVALS + i] = nVec[D_OFF + i] * ws.arz[i * lambda + k];
                    }
                    // arx[:,k] = xmean + sigma * B * y  via Dgemv: y = alpha*A*x + beta*y
                    // Copy xmean into arx column k (stride=lambda) via Dcopy, then dgemv beta=1
                    BLAS.dcopy(n, nVec, XMEAN, 1, ws.arx, k, lambda);
                    BLAS.dgemv(BLAS.Trans.NoTrans, n, n, ws.sigma,
                               ws.mat, B_OFF, n,
                               nVec, EVALS, 1, 1.0,
                               ws.arx, k, lambda);
                }

                if (bounds == null || checkFeasible == 0) break;
                if (isFeasible(ws.arx, k, lambda, n, bounds)) break;
            }
        }
    }

    private static boolean isFeasible(double[] arx, int k, int lambda, int n, Bound[] bounds) {
        for (int i = 0; i < n; i++) {
            double xi = arx[i * lambda + k];
            Bound b = Bound.of(bounds, i, Bound.unbounded());
            if (b.hasLower() && xi < b.getLower()) return false;
            if (b.hasUpper() && xi > b.getUpper()) return false;
        }
        return true;
    }


    // ── Mean update ───────────────────────────────────────────────────────

    /**
     * Updates the distribution mean m via weighted recombination of the top μ individuals.
     *
     * <pre>
     *   m_old ← m
     *   m     ← Σᵢ₌₀^{μ−1} wᵢ · x_{i:λ}
     * </pre>
     *
     * <p>where x_{i:λ} denotes the i-th ranked candidate (ascending fitness).
     * The old mean m_old is saved in {@code nVec[XOLD]} for use by
     * {@code updateEvolutionPaths} and {@code updateCovariance}.</p>
     */
    static void updateMean(CMAESWorkspace ws) {
        final int n = ws.n;
        final int lambda = ws.lambda;
        final int mu = ws.mu;
        final double[] nVec = ws.nVec;
        final double[] lVec = ws.lVec;
        final int XMEAN = ws.XMEAN, XOLD = ws.XOLD, WEIGHTS = ws.WEIGHTS;

        // Save old mean
        System.arraycopy(nVec, XMEAN, nVec, XOLD, n);

        // m ← Σᵢ₌₀^{μ−1} wᵢ · x_{i:λ}  via Daxpy: y += α·x
        Arrays.fill(nVec, XMEAN, XMEAN + n, 0.0);
        for (int i = 0; i < mu; i++) {
            int idx = ws.arindex[i];
            double w = lVec[WEIGHTS + i];
            // arx column idx has stride lambda; daxpy: nVec[XMEAN] += w * arx[:,idx]
            BLAS.daxpy(n, w, ws.arx, idx, lambda, nVec, XMEAN, 1);
        }
    }


    // ── Evolution paths ───────────────────────────────────────────────────

    static boolean updateEvolutionPaths(CMAESWorkspace ws) {
        return updateEvolutionPaths(ws, false);
    }

    /**
     * Updates the step-size evolution path p_σ and the covariance evolution path p_c.
     *
     * <h3>Step-size path (CSA)</h3>
     * <pre>
     *   z    = C^{−½} · (m − m_old) / σ
     *   p_σ  ← (1 − c_s)·p_σ + √(c_s·(2−c_s)·μ_eff) · z
     *   ‖p_σ‖ is stored in ws.normps
     * </pre>
     *
     * <p>C^{−½}·v is computed as B·(D^{−1}⊙(Bᵀ·v)) in full mode,
     * or diagD^{−1}⊙v in sep mode.  The intermediate vector u = Bᵀ·(m−m_old)
     * is written into {@code ctmpBuf[0…n−1]}; the result z goes into
     * {@code nVec[EVALS]}.</p>
     *
     * <h3>Stagnation indicator (hsig)</h3>
     * <pre>
     *   hsig = (‖p_σ‖² / n) / (1 − (1−c_s)^{2t}) &lt; 2 + 4/(n+1)
     * </pre>
     * <p>When {@code hsig=false} the rank-1 update is slightly reduced to compensate
     * for the stalled step-size path.</p>
     *
     * <h3>Covariance path</h3>
     * <pre>
     *   p_c ← (1 − c_c)·p_c  [+ √(c_c·(2−c_c)·μ_eff)/σ · (m−m_old)  if hsig]
     * </pre>
     *
     * @param diagonalOnly {@code true} for sep-CMA-ES
     * @return {@code hsig} flag
     */
    static boolean updateEvolutionPaths(CMAESWorkspace ws, boolean diagonalOnly) {
        final int n = ws.n;
        final int iter = ws.iterations;
        final double[] nVec = ws.nVec;
        final double[] mat = ws.mat;
        final int XMEAN = ws.XMEAN, XOLD = ws.XOLD, PS = ws.PS, PC = ws.PC;
        final int D_OFF = ws.D_OFF, DIAGD = ws.DIAGD, EVALS = ws.EVALS;
        final int B_OFF = ws.B_OFF;
        final double[] ctmpBuf = ws.ctmpBuf; // scratch for B^T*(xmean-xold)

        // Compute z = C^{-1/2} * (xmean - xold) / sigma into EVALS scratch
        if (diagonalOnly) {
            for (int i = 0; i < n; i++) {
                nVec[EVALS + i] = (nVec[DIAGD + i] > 0)
                    ? (nVec[XMEAN + i] - nVec[XOLD + i]) / (nVec[DIAGD + i] * ws.sigma)
                    : 0.0;
            }
        } else {
            // Step 1: diff = m − m_old into ctmpBuf[0…n−1]
            for (int i = 0; i < n; i++) ctmpBuf[i] = nVec[XMEAN + i] - nVec[XOLD + i];
            // Step 2: u = Bᵀ·diff into nVec[EVALS]  via Dgemv Trans
            BLAS.dgemv(BLAS.Trans.Trans, n, n, 1.0,
                       mat, B_OFF, n,
                       ctmpBuf, 0, 1, 0.0,
                       nVec, EVALS, 1);
            // Step 3: v = D^{−1} ⊙ u into ctmpBuf (frees EVALS for output)
            for (int j = 0; j < n; j++) {
                ctmpBuf[j] = (nVec[D_OFF + j] > 0) ? nVec[EVALS + j] / nVec[D_OFF + j] : 0.0;
            }
            // Step 4: z = B·v / σ directly into nVec[EVALS]  via Dgemv NoTrans
            BLAS.dgemv(BLAS.Trans.NoTrans, n, n, 1.0 / ws.sigma,
                       mat, B_OFF, n,
                       ctmpBuf, 0, 1, 0.0,
                       nVec, EVALS, 1);
        }

        // ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * z
        double csCoeff = Math.sqrt(ws.cs * (2.0 - ws.cs) * ws.mueff);
        for (int i = 0; i < n; i++) {
            nVec[PS + i] = (1.0 - ws.cs) * nVec[PS + i] + csCoeff * nVec[EVALS + i];
        }

        // ‖p_σ‖ via Dnrm2
        ws.normps = BLAS.dnrm2(n, nVec, PS, 1);

        // hsig
        double denom = 1.0 - Math.pow(1.0 - ws.cs, 2.0 * iter);
        boolean hsig = (denom > 0)
            ? (ws.normps * ws.normps / n) / denom < 2.0 + 4.0 / (n + 1.0)
            : true;

        // pc update
        double ccFac = 1.0 - ws.cc;
        for (int i = 0; i < n; i++) nVec[PC + i] *= ccFac;
        if (hsig) {
            double pcCoeff = Math.sqrt(ws.cc * (2.0 - ws.cc) * ws.mueff) / ws.sigma;
            for (int i = 0; i < n; i++) {
                nVec[PC + i] += pcCoeff * (nVec[XMEAN + i] - nVec[XOLD + i]);
            }
        }

        return hsig;
    }


    // ── Covariance matrix update ──────────────────────────────────────────

    /**
     * Updates the full covariance matrix C via rank-1 + rank-μ updates (Active CMA).
     *
     * <h3>Formula</h3>
     * <pre>
     *   C ← α·C  +  c_1·p_c·p_cᵀ  +  c_μ · Σₖ w̃ₖ · δₖ·δₖᵀ
     * </pre>
     * where:
     * <ul>
     *   <li>α = 1 − c_1a − c_μ·Σwᵢ  (zero-drift scale factor)</li>
     *   <li>c_1a = c_1 if hsig, else c_1·(1 − c_c·(2−c_c))  (hsig correction)</li>
     *   <li>δₖ = (x_k − m_old) / σ  (computed into {@code nVec[DIAGC]} scratch)</li>
     * </ul>
     *
     * <h3>Active CMA negative-weight scaling</h3>
     * <p>For negative weights (Active CMA), the raw weight is scaled by the
     * Mahalanobis norm to guarantee positive-definiteness:</p>
     * <pre>
     *   mah² = ‖C^{−½}·δₖ‖² = ‖D^{−1}·Bᵀ·δₖ‖²
     *   w̃ₖ   = wₖ · n / (√mah² + 1e-9)²
     * </pre>
     *
     * <p>The {@code nVec[DIAGC]} slot is used as scratch for δₖ because diagC is
     * unused in full-matrix mode — the two uses never overlap.</p>
     *
     * @param hsig        stagnation flag from {@code updateEvolutionPaths}
     * @param isActiveCMA {@code true} to apply negative-weight (Active CMA) updates
     */
    static void updateCovariance(CMAESWorkspace ws, boolean hsig, boolean isActiveCMA, double sumW) {
        final int n = ws.n;
        final int lambda = ws.lambda;
        final double[] nVec = ws.nVec;
        final double[] lVec = ws.lVec;
        final double[] mat = ws.mat;
        final int XOLD = ws.XOLD, PC = ws.PC, D_OFF = ws.D_OFF;
        final int DIAGC = ws.DIAGC;  // scratch for delta in full-matrix mode
        final int C_OFF = ws.C_OFF, B_OFF = ws.B_OFF;
        final int WEIGHTS = ws.WEIGHTS;

        final double c1  = ws.ccov1;
        final double cmu = ws.ccovmu;

        double c1a = hsig ? c1 : c1 * (1.0 - ws.cc * (2.0 - ws.cc));

        double scaleFactor = 1.0 - c1a - cmu * sumW;

        // Step 1: C *= scaleFactor  via Dscal on lower triangle rows
        // Since C is stored as full symmetric matrix, scale all n² elements
        BLAS.dscal(n * n, scaleFactor, mat, C_OFF, 1);

        // Step 2: rank-1 update C += c_1·p_c·p_cᵀ  (symmetric, both triangles)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double val = c1 * nVec[PC + i] * nVec[PC + j];
                mat[C_OFF + i * n + j] += val;
                if (i != j) mat[C_OFF + j * n + i] += val;
            }
        }

        // Step 3: rank-μ update — unified over all λ individuals
        for (int k = 0; k < lambda; k++) {
            int idx = ws.arindex[k];
            double wk = lVec[WEIGHTS + k];

            if (wk == 0.0) continue;

            // δ_k = (x_k − m_old) / σ  into DIAGC scratch  via Daxpy
            // δ = -m_old/σ + x_k/σ: first fill with -m_old/σ, then add x_k/σ
            for (int d = 0; d < n; d++) {
                nVec[DIAGC + d] = (ws.arx[d * lambda + idx] - nVec[XOLD + d]) / ws.sigma;
            }

            double effectiveW;
            if (wk >= 0) {
                effectiveW = cmu * wk;
            } else if (isActiveCMA) {
                // mah² = ‖D^{−1}·Bᵀ·δ‖²  via Dgemv Trans then Ddot
                // tmp = Bᵀ·δ into ctmpBuf (first n elements)
                BLAS.dgemv(BLAS.Trans.Trans, n, n, 1.0,
                           mat, B_OFF, n,
                           nVec, DIAGC, 1, 0.0,
                           ws.ctmpBuf, 0, 1);
                // scale by D^{−1} in-place
                for (int j = 0; j < n; j++) {
                    ws.ctmpBuf[j] = (nVec[D_OFF + j] > 0) ? ws.ctmpBuf[j] / nVec[D_OFF + j] : 0.0;
                }
                // mah² = ‖tmp‖²  via Ddot(tmp, tmp)
                double mahSq = BLAS.ddot(n, ws.ctmpBuf, 0, 1, ws.ctmpBuf, 0, 1);
                double mahNorm = Math.sqrt(mahSq);
                double wScaled = wk * n / ((mahNorm + 1e-9) * (mahNorm + 1e-9));
                effectiveW = cmu * wScaled;
            } else {
                continue;
            }

            // C += effectiveW · δ·δᵀ  (symmetric, both triangles)
            for (int r = 0; r < n; r++) {
                for (int c = 0; c <= r; c++) {
                    double val = effectiveW * nVec[DIAGC + r] * nVec[DIAGC + c];
                    mat[C_OFF + r * n + c] += val;
                    if (r != c) mat[C_OFF + c * n + r] += val;
                }
            }
        }
    }

    /**
     * Updates the diagonal covariance in sep-CMA-ES mode.
     *
     * <pre>
     *   diagC ← α·diagC  +  c_1^sep·(p_c ⊙ p_c)  +  c_μ^sep · Σᵢ₌₀^{μ−1} wᵢ·(z_{i:λ} ⊙ z_{i:λ})
     *   diagD ← √max(1e-20, diagC)
     * </pre>
     * where α = (1 − c_1^sep − c_μ^sep) + (0 if hsig else c_1^sep·c_c·(2−c_c)).
     *
     * <p>Note: the rank-μ term uses the raw noise vectors z (from {@code arz}),
     * not the normalised steps δ, because in sep mode C^{−½} = diag(diagD^{−1})
     * and the z vectors are already in the isotropic basis.</p>
     *
     * @param hsig stagnation flag from {@code updateEvolutionPaths}
     */
    static void updateCovarianceDiag(CMAESWorkspace ws, boolean hsig) {
        final int n = ws.n;
        final int lambda = ws.lambda;
        final int mu = ws.mu;
        final double[] nVec = ws.nVec;
        final double[] lVec = ws.lVec;
        final int PC = ws.PC, DIAGC = ws.DIAGC, DIAGD = ws.DIAGD, WEIGHTS = ws.WEIGHTS;

        double oldFac = (hsig ? 0.0 : ws.ccov1Sep * ws.cc * (2.0 - ws.cc))
                      + (1.0 - ws.ccov1Sep - ws.ccovmuSep);

        for (int i = 0; i < n; i++) {
            nVec[DIAGC + i] *= oldFac;
            nVec[DIAGC + i] += ws.ccov1Sep * nVec[PC + i] * nVec[PC + i];
        }

        for (int i = 0; i < mu; i++) {
            int idx = ws.arindex[i];
            double w = lVec[WEIGHTS + i];
            for (int d = 0; d < n; d++) {
                double z = ws.arz[d * lambda + idx];
                nVec[DIAGC + d] += ws.ccovmuSep * w * z * z;
            }
        }

        for (int i = 0; i < n; i++) {
            nVec[DIAGD + i] = Math.sqrt(Math.max(1e-20, nVec[DIAGC + i]));
        }
    }


    // ── Sigma update ──────────────────────────────────────────────────────

    /**
     * Updates σ using the Cumulative Step-size Adaptation (CSA) rule.
     *
     * <pre>
     *   σ ← σ · exp(min(1, (‖p_σ‖/χ_n − 1) · c_s / d_amps))
     * </pre>
     *
     * <p>The exponent is clamped to 1 to prevent catastrophic step-size explosion
     * in early generations when p_σ may be far from its expected length χ_n.</p>
     */
    static void updateSigma(CMAESWorkspace ws) {
        double exponent = Math.min(1.0, (ws.normps / ws.chiN - 1.0) * ws.cs / ws.damps);
        ws.sigma *= Math.exp(exponent);
    }

    // ── Eigendecomposition ────────────────────────────────────────────────

    /**
     * Performs eigendecomposition of C via LAPACK dsyev, updating B and D.
     *
     * <pre>
     *   C = B · diag(D²) · Bᵀ
     *   D_i = √max(1e-20, λ_i)   (λ_i = i-th eigenvalue of C)
     * </pre>
     *
     * <p>C is copied to {@code ctmpBuf} before the call because dsyev overwrites
     * its input matrix.  On success, {@code ctmpBuf} contains the eigenvectors
     * (columns), which are then copied to {@code mat[B_OFF]}.</p>
     *
     * <h3>Condition-number repair</h3>
     * <p>If max(D)/min(D) &gt; 1e7 (condition number &gt; 1e14), a regularisation
     * term τ = max(D)²/1e14 − min(D)² is added to the diagonal of C and D is
     * recomputed.  This mirrors the hipparchus repair strategy and avoids
     * numerical breakdown without aborting the run.</p>
     *
     * <p>If dsyev returns a non-zero info code the decomposition is silently
     * skipped and the previous B/D are retained.</p>
     */
    static void eigenDecompose(CMAESWorkspace ws) {
        final int n = ws.n;
        final double[] nVec = ws.nVec;
        final double[] mat = ws.mat;
        final int D_OFF = ws.D_OFF, EVALS = ws.EVALS;
        final int C_OFF = ws.C_OFF, B_OFF = ws.B_OFF;

        // Copy C to ctmpBuf (dsyev overwrites input; API has no offset support for A)
        System.arraycopy(mat, C_OFF, ws.ctmpBuf, 0, n * n);

        int info = Dsyev.dsyev('V', 'L', n, ws.ctmpBuf, n,
                               nVec, EVALS,
                               ws.eigenWork, 0, ws.lwork);

        if (info == 0) {
            for (int i = 0; i < n; i++) {
                nVec[D_OFF + i] = Math.sqrt(Math.max(1e-20, nVec[EVALS + i]));
            }
            // ctmpBuf now contains eigenvectors; copy to B
            System.arraycopy(ws.ctmpBuf, 0, mat, B_OFF, n * n);

            // Condition number repair
            double maxD = nVec[D_OFF], minD = nVec[D_OFF];
            for (int i = 1; i < n; i++) {
                if (nVec[D_OFF + i] > maxD) maxD = nVec[D_OFF + i];
                if (nVec[D_OFF + i] < minD) minD = nVec[D_OFF + i];
            }
            if (minD > 0 && maxD / minD > 1e7) {
                double tfac = maxD * maxD / 1e14 - minD * minD;
                for (int i = 0; i < n; i++) mat[C_OFF + i * n + i] += tfac;
                for (int i = 0; i < n; i++) {
                    nVec[D_OFF + i] = Math.sqrt(Math.max(1e-20, nVec[EVALS + i] + tfac));
                }
            }
        }

        ws.lastEigenIter = ws.iterations;
    }


    // ── Stop conditions ───────────────────────────────────────────────────

    /**
     * Checks all stop conditions in priority order and returns the first triggered status,
     * or {@code null} to continue.
     *
     * <ol>
     *   <li><b>MaxEval</b>  — evaluations ≥ maxEvaluations</li>
     *   <li><b>StopFitness</b> — bestFitness &lt; stopFitness</li>
     *   <li><b>TolUpSigma</b> — σ·D_i &gt; tolUpSigma·σ₀ for any i  (divergence)</li>
     *   <li><b>ConditionCov</b> — max(D)/min(D) &gt; 1e7  (ill-conditioned C)</li>
     *   <li><b>TolX</b> — σ·max(|p_c_i|, √C_ii) &lt; tolX·σ₀ for all i  (step too small)</li>
     *   <li><b>TolFun</b> — range of fitness history &lt; tolFun  (flat landscape)</li>
     *   <li><b>SigmaMin</b> — σ &lt; 1e-20·σ₀</li>
     * </ol>
     */
    static Optimization.Status checkStopConditions(CMAESWorkspace ws, CMAESProblem cfg,
                                                    double bestFitness, boolean diagonalOnly) {
        final int n = ws.n;
        final double[] nVec = ws.nVec;
        final double[] mat = ws.mat;
        final int D_OFF = ws.D_OFF, DIAGC = ws.DIAGC, DIAGD = ws.DIAGD, PC = ws.PC;
        final int C_OFF = ws.C_OFF;

        if (ws.evaluations >= cfg.effectiveMaxEvaluations()) {
            return Optimization.Status.MAX_EVALUATIONS_REACHED;
        }

        if (bestFitness < cfg.stopFitness()) {
            return Optimization.Status.FUNCTION_TOLERANCE_REACHED;
        }

        double tolUpSigma = cfg.sigmaUpperBound();
        if (diagonalOnly) {
            for (int i = 0; i < n; i++) {
                if (ws.sigma * Math.sqrt(nVec[DIAGC + i]) > tolUpSigma * ws.sigma0) {
                    return Optimization.Status.ABNORMAL_TERMINATION;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                if (ws.sigma * nVec[D_OFF + i] > tolUpSigma * ws.sigma0) {
                    return Optimization.Status.ABNORMAL_TERMINATION;
                }
            }
        }

        if (!diagonalOnly && ws.iterations == ws.lastEigenIter) {
            double maxD = nVec[D_OFF], minD = nVec[D_OFF];
            for (int i = 1; i < n; i++) {
                if (nVec[D_OFF + i] > maxD) maxD = nVec[D_OFF + i];
                if (nVec[D_OFF + i] < minD) minD = nVec[D_OFF + i];
            }
            if (minD > 0 && maxD / minD > 1e7) {
                return Optimization.Status.ABNORMAL_TERMINATION;
            }
        } else if (diagonalOnly) {
            double maxD = nVec[DIAGD], minD = nVec[DIAGD];
            for (int i = 1; i < n; i++) {
                if (nVec[DIAGD + i] > maxD) maxD = nVec[DIAGD + i];
                if (nVec[DIAGD + i] < minD) minD = nVec[DIAGD + i];
            }
            if (minD > 0 && maxD / minD > 1e7) {
                return Optimization.Status.ABNORMAL_TERMINATION;
            }
        }

        double tolX = cfg.parameterTolerance();
        boolean tolXMet = true;
        for (int i = 0; i < n; i++) {
            double diagCi = diagonalOnly ? nVec[DIAGC + i] : mat[C_OFF + i * n + i];
            double threshold = Math.max(Math.abs(nVec[PC + i]), Math.sqrt(diagCi));
            if (ws.sigma * threshold >= tolX * ws.sigma0) {
                tolXMet = false;
                break;
            }
        }
        if (tolXMet) return Optimization.Status.COEFFICIENT_TOLERANCE_REACHED;

        double tolFun = cfg.functionTolerance();
        if (ws.iterations > ws.histSize) {
            double histMin = ws.fitnessHistory[0], histMax = ws.fitnessHistory[0];
            for (int i = 1; i < ws.histSize; i++) {
                if (ws.fitnessHistory[i] < histMin) histMin = ws.fitnessHistory[i];
                if (ws.fitnessHistory[i] > histMax) histMax = ws.fitnessHistory[i];
            }
            if (histMax - histMin < tolFun) {
                return Optimization.Status.FUNCTION_TOLERANCE_REACHED;
            }
        }

        if (ws.sigma < 1e-20 * ws.sigma0) {
            return Optimization.Status.COEFFICIENT_TOLERANCE_REACHED;
        }

        return null;
    }


    // ── Boundary penalty ──────────────────────────────────────────────────

    /**
     * Computes the squared boundary-violation distance for individual k.
     *
     * <pre>
     *   penalty = Σᵢ max(0, lb_i − x_i)² + max(0, x_i − ub_i)²
     * </pre>
     *
     * <p>The raw penalty is later scaled by the value range of feasible offspring
     * inside {@code evaluateFitness} so that it is commensurate with the objective.</p>
     *
     * @param arx    candidate matrix (n×λ, row-major)
     * @param k      individual index (column)
     * @param lambda population size λ
     * @param n      problem dimension
     * @param bounds variable bounds array
     * @return unscaled squared violation distance
     */
    static double computeRawPenalty(double[] arx, int k, int lambda, int n, Bound[] bounds) {
        double penalty = 0;
        for (int i = 0; i < n; i++) {
            double xi = arx[i * lambda + k];
            Bound b = Bound.of(bounds, i, Bound.unbounded());
            if (b.hasLower() && xi < b.getLower()) {
                double viol = b.getLower() - xi;
                penalty += viol * viol;
            }
            if (b.hasUpper() && xi > b.getUpper()) {
                double viol = xi - b.getUpper();
                penalty += viol * viol;
            }
        }
        return penalty;
    }

    /**
     * Evaluates all λ offspring, applies boundary penalties, and handles NaN/Inf values.
     *
     * <p>For each candidate k:</p>
     * <ol>
     *   <li>Copy x_k into {@code ws.xbuf} and call {@code fn.evaluate(xbuf, ws.n)}.</li>
     *   <li>Store raw fitness in {@code lVec[FITNESS+k]}.</li>
     *   <li>Compute boundary penalty into {@code lVec[PENALTY+k]} (no allocation).</li>
     * </ol>
     *
     * <p><b>NaN/Inf handling:</b> any non-finite fitness is replaced by the worst
     * finite value seen in this generation.  If all λ values are non-finite,
     * {@code false} is returned (triggers {@code CALLBACK_ERROR}).</p>
     *
     * <p><b>Penalty scaling:</b> the raw penalty is multiplied by
     * {@code max(1, feasMax − feasMin)} so that infeasible individuals are ranked
     * worse than all feasible ones regardless of the objective scale.</p>
     *
     * @return {@code true} if at least one finite fitness value was observed
     */
    static boolean evaluateFitness(CMAESWorkspace ws, Univariate.Objective fn,
                                    Bound[] bounds) {
        final int n = ws.n;
        final int lambda = ws.lambda;
        final double[] lVec = ws.lVec;
        final double[] xbuf = ws.xbuf;
        final int FITNESS = ws.FITNESS, PENALTY = ws.PENALTY;

        for (int k = 0; k < lambda; k++) {
            for (int i = 0; i < n; i++) xbuf[i] = ws.arx[i * lambda + k];
            double f = fn.evaluate(xbuf, ws.n);
            lVec[FITNESS + k] = f;
            lVec[PENALTY + k] = (bounds != null) ? computeRawPenalty(ws.arx, k, lambda, n, bounds) : 0.0;
        }
        ws.evaluations += lambda;

        double worstFinite = Double.NEGATIVE_INFINITY;
        boolean anyFinite = false;
        for (int k = 0; k < lambda; k++) {
            if (Double.isFinite(lVec[FITNESS + k])) {
                if (lVec[FITNESS + k] > worstFinite) worstFinite = lVec[FITNESS + k];
                anyFinite = true;
            }
        }
        if (!anyFinite) return false;

        for (int k = 0; k < lambda; k++) {
            if (!Double.isFinite(lVec[FITNESS + k])) {
                lVec[FITNESS + k] = worstFinite;
            }
        }

        if (bounds != null) {
            double feasMin = Double.MAX_VALUE, feasMax = Double.NEGATIVE_INFINITY;
            boolean anyFeasible = false;
            for (int k = 0; k < lambda; k++) {
                if (lVec[PENALTY + k] == 0.0) {
                    if (lVec[FITNESS + k] < feasMin) feasMin = lVec[FITNESS + k];
                    if (lVec[FITNESS + k] > feasMax) feasMax = lVec[FITNESS + k];
                    anyFeasible = true;
                }
            }
            double valueRange = anyFeasible ? Math.max(1.0, feasMax - feasMin) : 1.0;
            for (int k = 0; k < lambda; k++) {
                lVec[FITNESS + k] += lVec[PENALTY + k] * valueRange;
            }
        }

        return true;
    }


    // ── Main optimization loop ────────────────────────────────────────────

    /**
     * Runs a single CMA-ES optimisation from initial point {@code x0}.
     *
     * <p>The main loop executes the standard CMA-ES update sequence each generation
     * (see class-level javadoc for the call order) until a stop condition is met
     * or {@code maxIterations} is reached.</p>
     *
     * <p><b>bestX storage:</b> the current best solution is kept in {@code ws.xbuf}.
     * This is safe because {@code evaluateFitness} writes {@code xbuf} and returns
     * before the best-solution update overwrites it — the two uses are non-overlapping
     * within a single iteration.</p>
     *
     * @param x0   initial point (not modified)
     * @param fn   objective function to minimise
     * @param bounds variable bounds, or {@code null} for unconstrained
     * @param ws   pre-allocated workspace (must match problem dimension and λ)
     * @param cfg  problem configuration snapshot
     * @param rng  random number generator
     * @return optimisation result containing best solution, cost, status and counters
     */
    public static Optimization optimize(double[] x0, Univariate.Objective fn,
                                         Bound[] bounds, CMAESWorkspace ws,
                                         CMAESProblem cfg, Random rng) {
        final int n = ws.n;
        final int lambda = ws.lambda;
        final boolean diagonalOnly = cfg.updateMode().separable;
        final boolean isActiveCMA  = cfg.updateMode().activeCMA;
        final int maxIterations    = cfg.maxIterations();
        final double[] nVec = ws.nVec;
        final double[] lVec = ws.lVec;
        final double[] mat = ws.mat;
        final int XMEAN = ws.XMEAN, D_OFF = ws.D_OFF, DIAGC = ws.DIAGC, DIAGD = ws.DIAGD;
        final int PS = ws.PS, PC = ws.PC;
        final int C_OFF = ws.C_OFF, B_OFF = ws.B_OFF;
        final int FITNESS = ws.FITNESS;
        final double[] xbuf = ws.xbuf; // dual-purpose: fn scratch + bestX storage

        initParams(ws, maxIterations);

        // sumW = Σwᵢ (all λ weights) — fixed after initParams, computed once
        double sumW = 0;
        for (int i = 0; i < lambda; i++) sumW += lVec[ws.WEIGHTS + i];

        // Initialize state
        System.arraycopy(x0, 0, nVec, XMEAN, n);
        ws.sigma  = cfg.sigma0();
        ws.sigma0 = cfg.sigma0();

        // C = I, B = I, D = 1 (full-matrix mode only)
        if (mat != null) {
            Arrays.fill(mat, C_OFF, C_OFF + n * n, 0.0);
            Arrays.fill(mat, B_OFF, B_OFF + n * n, 0.0);
            for (int i = 0; i < n; i++) {
                mat[C_OFF + i * n + i] = 1.0;
                mat[B_OFF + i * n + i] = 1.0;
            }
        }
        for (int i = 0; i < n; i++) nVec[D_OFF + i] = 1.0;
        Arrays.fill(nVec, DIAGC, DIAGC + n, 1.0);
        Arrays.fill(nVec, DIAGD, DIAGD + n, 1.0);
        Arrays.fill(nVec, PS, PS + n, 0.0);
        Arrays.fill(nVec, PC, PC + n, 0.0);
        ws.normps = 0.0;
        ws.lastEigenIter = 0;
        Arrays.fill(ws.fitnessHistory, Double.MAX_VALUE);
        ws.histHead = 0;
        ws.iterations = 0;
        ws.evaluations = 0;

        double bestFitness = Double.MAX_VALUE;
        // bestX stored in ws.xbuf (safe: evaluateFitness writes xbuf then returns,
        // bestX update happens after — no overlap within a single iteration)
        System.arraycopy(x0, 0, xbuf, 0, n);

        for (int iter = 1; iter <= maxIterations; iter++) {
            ws.iterations = iter;

            sampleOffspring(ws, diagonalOnly, rng, bounds, cfg.maxResample());

            boolean valid = evaluateFitness(ws, fn, bounds);
            if (!valid) {
                return new Optimization(Double.NaN, xbuf.clone(), bestFitness,
                    Optimization.Status.CALLBACK_ERROR, iter, ws.evaluations);
            }

            if (ws.evaluations >= cfg.effectiveMaxEvaluations()) {
                return new Optimization(Double.NaN, xbuf.clone(), bestFitness,
                    Optimization.Status.MAX_EVALUATIONS_REACHED, iter, ws.evaluations);
            }

            sortIndices(lVec, FITNESS, ws.arindex, lambda);

            updateMean(ws);

            boolean hsig = updateEvolutionPaths(ws, diagonalOnly);

            if (diagonalOnly) {
                updateCovarianceDiag(ws, hsig);
            } else {
                updateCovariance(ws, hsig, isActiveCMA, sumW);
            }

            updateSigma(ws);

            if (!diagonalOnly && (iter - ws.lastEigenIter) >= ws.eigenInterval) {
                eigenDecompose(ws);
            }

            // Update best solution (xbuf is safe to overwrite here: evaluateFitness already returned)
            double iterBest = lVec[FITNESS + ws.arindex[0]];
            if (iterBest < bestFitness) {
                bestFitness = iterBest;
                for (int i = 0; i < n; i++) {
                    xbuf[i] = ws.arx[i * lambda + ws.arindex[0]];
                }
            }

            ws.fitnessHistory[ws.histHead] = iterBest;
            ws.histHead = (ws.histHead + 1) % ws.histSize;

            Optimization.Status status = checkStopConditions(ws, cfg, bestFitness, diagonalOnly);
            if (status != null) {
                return new Optimization(Double.NaN, xbuf.clone(), bestFitness,
                    status, iter, ws.evaluations);
            }
        }

        return new Optimization(Double.NaN, xbuf.clone(), bestFitness,
            Optimization.Status.MAX_ITERATIONS_REACHED, maxIterations, ws.evaluations);
    }

    /** Sorts {@code arindex[0…λ−1]} by ascending {@code fitness} values (insertion sort). */
    static void sortIndices(double[] fitness, int[] arindex, int lambda) {
        for (int i = 0; i < lambda; i++) arindex[i] = i;
        for (int i = 1; i < lambda; i++) {
            int key = arindex[i];
            double keyVal = fitness[key];
            int j = i - 1;
            while (j >= 0 && fitness[arindex[j]] > keyVal) {
                arindex[j + 1] = arindex[j];
                j--;
            }
            arindex[j + 1] = key;
        }
    }

    /** Sorts {@code arindex[0…λ−1]} by ascending {@code lVec[fitnessOffset+k]} values (insertion sort). */
    static void sortIndices(double[] lVec, int fitnessOffset, int[] arindex, int lambda) {
        for (int i = 0; i < lambda; i++) arindex[i] = i;
        for (int i = 1; i < lambda; i++) {
            int key = arindex[i];
            double keyVal = lVec[fitnessOffset + key];
            int j = i - 1;
            while (j >= 0 && lVec[fitnessOffset + arindex[j]] > keyVal) {
                arindex[j + 1] = arindex[j];
                j--;
            }
            arindex[j + 1] = key;
        }
    }
}
