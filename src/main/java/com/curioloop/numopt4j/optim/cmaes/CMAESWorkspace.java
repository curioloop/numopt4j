/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

import com.curioloop.numopt4j.linalg.blas.Dsyev;

import java.util.Arrays;

/**
 * Pre-allocated workspace for the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.
 *
 * <p>All arrays are allocated once at construction time and reused across multiple {@code solve()}
 * calls via {@link #reset()}, eliminating per-iteration heap allocation in the hot path.</p>
 *
 * <h2>Algorithm Overview</h2>
 * <p>CMA-ES maintains a multivariate normal distribution 𝒩(m, σ²C) over ℝⁿ and adapts its
 * parameters each generation to minimise an objective function f : ℝⁿ → ℝ:</p>
 * <ul>
 *   <li><b>m</b>  — distribution mean (current best estimate of the optimum)</li>
 *   <li><b>σ</b>  — global step size (scalar)</li>
 *   <li><b>C</b>  — covariance matrix (n×n, symmetric positive-definite)</li>
 * </ul>
 *
 * <h2>Memory Layout</h2>
 * <p>To minimise GC pressure all n-dimensional vectors share a single contiguous arena
 * ({@code nVec}), all λ-dimensional vectors share {@code lVec}, and the two n×n matrices
 * share {@code mat}.  Scratch buffers reuse arena slots whose lifetimes do not overlap
 * (see timing analysis in {@link CMAESCore}).</p>
 *
 * <pre>
 *  nVec  [8n]  slot 0  XMEAN  — mean vector m
 *              slot 1  XOLD   — previous mean m_old
 *              slot 2  PS     — step-size evolution path p_σ
 *              slot 3  PC     — covariance evolution path p_c
 *              slot 4  D_OFF  — eigenvalue square-roots D = diag(√λ_i)
 *              slot 5  DIAGC  — diagonal covariance (sep mode) / delta scratch (full mode)
 *              slot 6  DIAGD  — √diagC (sep mode)
 *              slot 7  EVALS  — LAPACK eigenvalue output / B·D·z scratch
 *
 *  lVec  [3λ]  slot 0  FITNESS  — f(x_k) + penalty for each offspring
 *              slot 1  WEIGHTS  — recombination weights w_i
 *              slot 2  PENALTY  — boundary penalty scratch (replaces per-iter allocation)
 *
 *  mat   [2n²] slot 0  C_OFF  — covariance matrix C  (full mode only)
 *              slot 1  B_OFF  — eigenvector matrix B  (full mode only)
 * </pre>
 *
 * <p>In sep-CMA-ES mode ({@code diagonalOnly=true}) {@code mat}, {@code ctmpBuf} and
 * {@code eigenWork} are {@code null} — the O(n²) allocations are skipped entirely.</p>
 *
 * <h2>Matrix Storage</h2>
 * <p>All matrices use flat row-major layout: element (i,j) is at index {@code i*n + j}.
 * For symmetric matrices only the lower triangle is written; both triangles are kept
 * consistent after each update.</p>
 */
public final class CMAESWorkspace {

    // ── Problem dimension ─────────────────────────────────────────────────

    /** Problem dimension n. */
    public final int n;

    /** Population size λ (offspring count per generation). */
    public final int lambda;

    // ── Arena fields ──────────────────────────────────────────────────────

    /**
     * n-dimensional vector arena of length 8n.
     * Slots (each of size n): XMEAN · XOLD · PS · PC · D_OFF · DIAGC · DIAGD · EVALS.
     * See class-level javadoc for the slot map and scratch-reuse rules.
     */
    public final double[] nVec;

    /**
     * λ-dimensional vector arena of length 3λ.
     * Slots (each of size λ): FITNESS · WEIGHTS · PENALTY.
     */
    public final double[] lVec;

    /**
     * n×n matrix arena of length 2n².
     * Slots (each of size n²): C_OFF · B_OFF.
     * {@code null} in sep-CMA-ES mode ({@code diagonalOnly=true}).
     */
    public final double[] mat;

    // ── Arena offset constants (nVec) ─────────────────────────────────────

    /** nVec offset for mean vector m.  Slot 0, offset = 0. */
    public final int XMEAN;
    /** nVec offset for previous mean m_old.  Slot 1, offset = n. */
    public final int XOLD;
    /** nVec offset for step-size evolution path p_σ.  Slot 2, offset = 2n. */
    public final int PS;
    /** nVec offset for covariance evolution path p_c.  Slot 3, offset = 3n. */
    public final int PC;
    /** nVec offset for eigenvalue square-roots D_i = √λ_i.  Slot 4, offset = 4n. */
    public final int D_OFF;
    /**
     * nVec offset for diagonal covariance diag(C) in sep mode.  Slot 5, offset = 5n.
     * In full-matrix mode this slot is repurposed as a scratch buffer for δ_k
     * = (x_k − m_old) / σ during the rank-μ covariance update (diagC is unused
     * when diagonalOnly=false, so the slot is free).
     */
    public final int DIAGC;
    /** nVec offset for √diagC_i (sep mode).  Slot 6, offset = 6n. */
    public final int DIAGD;
    /**
     * nVec offset for LAPACK dsyev eigenvalue output.  Slot 7, offset = 7n.
     * Also used as scratch for the intermediate vector y = D⊙z during offspring
     * sampling and for C^{−½}·(m−m_old) during evolution-path update.
     * Timing analysis guarantees these uses never overlap within one iteration.
     */
    public final int EVALS;

    // ── Arena offset constants (lVec) ─────────────────────────────────────

    /** lVec offset for fitness values f(x_k).  Slot 0, offset = 0. */
    public final int FITNESS;
    /**
     * lVec offset for recombination weights w_i.
     * Positive weights (i = 0…μ−1) sum to 1; negative weights (i = μ…λ−1)
     * are used by Active CMA for the rank-μ update.  Slot 1, offset = λ.
     */
    public final int WEIGHTS;
    /**
     * lVec offset for boundary-penalty scratch.  Slot 2, offset = 2λ.
     * Replaces the per-iteration {@code new double[lambda]} that the original
     * implementation allocated inside {@code evaluateFitness}.
     */
    public final int PENALTY;

    // ── Arena offset constants (mat) ──────────────────────────────────────

    /**
     * mat offset for covariance matrix C (n×n, row-major).  Slot 0, offset = 0.
     * C is symmetric positive-definite; only the lower triangle is written,
     * but both triangles are kept consistent.
     */
    public final int C_OFF;
    /**
     * mat offset for eigenvector matrix B (n×n, row-major).  Slot 1, offset = n².
     * Columns of B are the eigenvectors of C: C = B · diag(D²) · Bᵀ.
     * Updated by {@code eigenDecompose} via LAPACK dsyev.
     */
    public final int B_OFF;

    // ── Distribution state (updated each generation) ──────────────────────

    /** Global step size σ.  Updated by CSA: σ ← σ · exp(min(1, (‖p_σ‖/χ_n − 1)·c_s/d_amps)). */
    public double sigma;

    /** ‖p_σ‖₂, cached after each evolution-path update to avoid recomputation. */
    public double normps;

    /**
     * Effective selection mass μ_eff = (Σᵢ wᵢ)² / Σᵢ wᵢ².
     * Measures the effective number of parents contributing to the mean update.
     */
    public double mueff;

    // ── Strategy parameters (fixed after initParams) ──────────────────────

    /** Parent count μ = ⌊λ/2⌋. */
    public int mu;

    /**
     * Covariance path cumulation constant c_c.
     * Controls the decay of the rank-1 evolution path p_c.
     * Formula: c_c = (4 + μ_eff/n) / (n + 4 + 2μ_eff/n).
     */
    public double cc;

    /**
     * Step-size path cumulation constant c_s (also written c_σ).
     * Controls the decay of the CSA path p_σ.
     * Formula: c_s = (μ_eff + 2) / (n + μ_eff + 5).
     */
    public double cs;

    /**
     * Step-size damping coefficient d_amps.
     * Scales the CSA update: σ ← σ · exp(min(1, (‖p_σ‖/χ_n − 1)·c_s/d_amps)).
     * Larger d_amps → slower σ adaptation.
     */
    public double damps;

    /**
     * Rank-1 covariance update learning rate c_1.
     * Formula: c_1 = 2 / ((n + 1.3)² + μ_eff).
     */
    public double ccov1;

    /**
     * Rank-μ covariance update learning rate c_μ.
     * Formula: c_μ = min(1 − c_1, 2·(μ_eff − 2 + 1/μ_eff) / ((n+2)² + μ_eff)).
     */
    public double ccovmu;

    /** Rank-1 learning rate for sep-CMA-ES: c_1^sep = min(1, c_1·(n+1.5)/3). */
    public double ccov1Sep;

    /** Rank-μ learning rate for sep-CMA-ES: c_μ^sep = min(1−c_1^sep, c_μ·(n+1.5)/3). */
    public double ccovmuSep;

    /**
     * Expected length of a standard normal vector: χ_n = E[‖𝒩(0,I)‖].
     * Approximation: χ_n ≈ √n · (1 − 1/(4n) + 1/(21n²)).
     */
    public double chiN;

    /**
     * Effective mass of negative weights μ_eff⁻ = (Σᵢ<0 wᵢ)² / Σᵢ<0 wᵢ².
     * Used by Active CMA to bound the negative-weight scaling.
     */
    public double mueffNeg;

    // ── Eigendecomposition scheduling ─────────────────────────────────────

    /**
     * Perform eigendecomposition every {@code eigenInterval} generations.
     * Computed as ⌊λ / ((c_1 + c_μ) · n · 10)⌋, clamped to ≥ 1.
     * Skipping decompositions amortises the O(n³) cost over multiple iterations.
     */
    public int eigenInterval;

    /** Generation index of the most recent eigendecomposition. */
    public int lastEigenIter;

    // ── Population buffers (independent allocations) ──────────────────────

    /**
     * Standard-normal random matrix z_k ~ 𝒩(0, I), shape n×λ, row-major.
     * Element (i, k) is at index {@code i*lambda + k}.
     * Generated fresh each generation by {@code sampleOffspring}.
     */
    public double[] arz;

    /**
     * Candidate solution matrix x_k, shape n×λ, row-major.
     * Full mode:  x_k = m + σ · B · (D ⊙ z_k)
     * Sep mode:   x_k = m + σ · (diagD ⊙ z_k)
     * Element (i, k) is at index {@code i*lambda + k}.
     */
    public double[] arx;

    /** Indices 0…λ−1 sorted by ascending fitness after each generation. */
    public int[] arindex;

    /**
     * Dual-purpose scratch buffer of length n.
     * <ol>
     *   <li>During {@code evaluateFitness}: holds a copy of candidate x_k passed to the
     *       objective function (avoids exposing the internal arx layout to callers).</li>
     *   <li>Between {@code evaluateFitness} and the best-solution update: stores the
     *       current best-known solution x* (replaces a separate bestX allocation).
     *       The two uses are non-overlapping within a single iteration.</li>
     * </ol>
     */
    public final double[] xbuf;

    // ── LAPACK workspace (full mode only, null in sep mode) ───────────────

    /**
     * LAPACK dsyev optimal work array.  Length determined by a workspace-size query
     * ({@code lwork = -1}) at construction time.  {@code null} in sep mode.
     */
    public double[] eigenWork;

    /**
     * n×n scratch buffer for dsyev (which overwrites its input matrix A).
     * Also used as a length-n scratch for the intermediate vector
     * u = Bᵀ·(m − m_old) inside {@code updateEvolutionPaths}.
     * {@code null} in sep mode.
     */
    public final double[] ctmpBuf;

    // ── Fitness history (stop conditions) ─────────────────────────────────

    /**
     * Ring buffer storing the best fitness value of each recent generation.
     * Length {@code histSize = 10 + ⌊30n/λ⌋}.  Used by the TolFun stop condition:
     * stop when max(history) − min(history) &lt; tolFun.
     */
    public double[] fitnessHistory;

    /** Write pointer into the {@code fitnessHistory} ring buffer. */
    public int histHead;

    /** Ring-buffer capacity: {@code 10 + (int)(3·10·n / λ)}. */
    public int histSize;

    // ── Iteration counters ────────────────────────────────────────────────

    /** Current generation index (1-based, incremented at the start of each iteration). */
    public int iterations;

    /** Cumulative number of objective-function evaluations (incremented by λ each generation). */
    public int evaluations;

    /** Initial step size σ₀, stored for the TolUpSigma divergence check. */
    public double sigma0;

    /** Optimal lwork for dsyev; 0 in sep mode. */
    final int lwork;

    // ── Constructors ──────────────────────────────────────────────────────

    /**
     * Allocates a full-matrix workspace for dimension {@code n} and population size {@code lambda}.
     * Equivalent to {@code new CMAESWorkspace(n, lambda, false)}.
     *
     * @param n      problem dimension (must be &gt; 0)
     * @param lambda population size λ (must be &gt; 0)
     */
    public CMAESWorkspace(int n, int lambda) {
        this(n, lambda, false);
    }

    /**
     * Allocates a workspace for dimension {@code n} and population size {@code lambda}.
     *
     * <p>When {@code diagonalOnly=true} (sep-CMA-ES), the O(n²) allocations
     * ({@code mat}, {@code ctmpBuf}, {@code eigenWork}) are skipped, reducing memory
     * from O(n²) to O(nλ).  The {@code mat} field will be {@code null}.</p>
     *
     * @param n            problem dimension (must be &gt; 0)
     * @param lambda       population size λ (must be &gt; 0)
     * @param diagonalOnly {@code true} for sep-CMA-ES (diagonal covariance only)
     */
    public CMAESWorkspace(int n, int lambda, boolean diagonalOnly) {
        if (n <= 0) throw new IllegalArgumentException("n must be positive, got " + n);
        if (lambda <= 0) throw new IllegalArgumentException("lambda must be positive, got " + lambda);

        this.n = n;
        this.lambda = lambda;

        // ── Arena offset constants ────────────────────────────────────────
        XMEAN = 0;   XOLD  = n;    PS    = 2*n;  PC    = 3*n;
        D_OFF = 4*n; DIAGC = 5*n;  DIAGD = 6*n;  EVALS = 7*n;

        FITNESS = 0; WEIGHTS = lambda; PENALTY = 2*lambda;

        C_OFF = 0; B_OFF = n*n;

        // ── Allocate arenas ───────────────────────────────────────────────
        nVec = new double[8*n];
        lVec = new double[3*lambda];
        // sep mode: C and B are not maintained — skip the O(n²) allocation
        mat  = diagonalOnly ? null : new double[2*n*n];

        // ── Independent allocations ───────────────────────────────────────
        arz     = new double[n * lambda];
        arx     = new double[n * lambda];
        arindex = new int[lambda];
        xbuf    = new double[n];

        // sep mode: no eigendecomposition → skip eigenWork and ctmpBuf
        if (diagonalOnly) {
            lwork     = 0;
            eigenWork = null;
            ctmpBuf   = null;
        } else {
            // Query optimal lwork via dsyev workspace-size query (lwork = -1)
            Dsyev.dsyev('V', 'L', n, null, n, null, 0, xbuf, 0, -1);
            lwork = Math.max(1, (int) xbuf[0]);
            eigenWork = new double[lwork];
            ctmpBuf   = new double[n * n];
        }

        // History size: 10 + ⌊30n/λ⌋  (pycma default)
        histSize = 10 + (int)(3.0 * 10 * n / lambda);
        fitnessHistory = new double[histSize];

        reset();
    }

    // ── Reset ─────────────────────────────────────────────────────────────

    /**
     * Resets all mutable state to the post-construction initial values without
     * reallocating memory.  Must be called before each {@code solve()} invocation
     * when reusing a workspace across multiple runs.
     *
     * <p>Initial values:</p>
     * <ul>
     *   <li>m, m_old, p_σ, p_c, eigenVals scratch → 0</li>
     *   <li>D_i = 1 (identity scaling), diagC_i = 1, diagD_i = 1</li>
     *   <li>C = I, B = I (full mode only)</li>
     *   <li>fitness history → +∞ (so TolFun does not trigger prematurely)</li>
     *   <li>arindex[i] = i, all scalars → 0</li>
     * </ul>
     */
    public void reset() {
        Arrays.fill(nVec, 0.0);
        Arrays.fill(lVec, 0.0);

        // D = 1 (slot 4), diagC = 1 (slot 5), diagD = 1 (slot 6)
        Arrays.fill(nVec, D_OFF, D_OFF + n, 1.0);
        Arrays.fill(nVec, DIAGC, DIAGC + n, 1.0);
        Arrays.fill(nVec, DIAGD, DIAGD + n, 1.0);

        // C = I, B = I (full mode only)
        if (mat != null) {
            Arrays.fill(mat, 0.0);
            for (int i = 0; i < n; i++) mat[C_OFF + i*n + i] = 1.0;
            for (int i = 0; i < n; i++) mat[B_OFF + i*n + i] = 1.0;
        }

        for (int i = 0; i < lambda; i++) arindex[i] = i;

        Arrays.fill(arz,  0.0);
        Arrays.fill(arx,  0.0);
        Arrays.fill(xbuf, 0.0);

        if (eigenWork != null) Arrays.fill(eigenWork, 0.0);
        Arrays.fill(fitnessHistory, Double.MAX_VALUE);
        histHead = 0;

        sigma = 0.0; sigma0 = 0.0; normps = 0.0;
        mueff = 0.0; mueffNeg = 0.0;

        mu = 0; cc = 0.0; cs = 0.0; damps = 0.0;
        ccov1 = 0.0; ccovmu = 0.0;
        ccov1Sep = 0.0; ccovmuSep = 0.0; chiN = 0.0;

        eigenInterval = 1; lastEigenIter = 0;
        iterations = 0; evaluations = 0;
    }
}
