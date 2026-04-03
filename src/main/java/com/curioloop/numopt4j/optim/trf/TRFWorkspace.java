/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import java.util.Arrays;

/**
 * Pre-allocated workspace for the Trust Region Reflective (TRF) algorithm.
 *
 * <p>Memory is split into two categories:</p>
 *
 * <p><b>Standalone eval-facing arrays</b> — passed directly to the user callback
 * {@code fcn.evaluate(x, fvec, fjac)} or {@code fcn.evaluate(wa2, wa4, null)}.
 * These must be independent {@code double[]} starting at index 0 because the
 * callback signature carries no offset parameter.</p>
 *
 * <p><b>Merged scratch buffer</b> — all internal work arrays that are never handed
 * to the callback are packed into a single {@code double[] work} allocation.
 * Each logical array is identified by a companion {@code *Off} offset field.</p>
 *
 * <pre>
 *  Standalone (eval-facing)   Size      Role
 *  ─────────────────────────────────────────────────────────────────────
 *  fvec                       m         residuals f(x)          (output from eval)
 *  fjac                       m×n       Jacobian J (row-major)  (output from eval, then overwritten by qrfac)
 *  wa2                        n         trial point xₖ + p      (input to eval)
 *  wa4                        m         f(xₖ + p)               (output from eval)
 *
 *  Merged work buffer         Offset    Size        Role
 *  ─────────────────────────────────────────────────────────────────────
 *  rwork                      0         n²          working copy of upper-triangular R for lmpar
 *  diag                       n²        n           base scaling diagonal D (column norms or user-supplied)
 *  qtf                        n²+n      n           Qᵀ·f (first n elements); also tmp[] scratch in qrfac
 *  wa1                        n²+2n     n           rdiag from qrfac / step p from lmpar
 *  wa3                        n²+3n     n           wa scratch in qrfac / D·step scratch
 *  clScale                    n²+4n     n           Coleman-Li scaling: distance to nearest bound
 *  effDiag                    n²+5n     n           effective diagonal = diag·clScale
 *  Jp                         n²+6n     n           R·Pᵀ·(tHit·p) for quadratic model
 *  Jpr                        n²+7n     n           R·Pᵀ·p̃ for quadratic model
 *  step                       n²+8n     n           actual step after reflection / clipping
 *  xHit                       n²+9n     n           point where unconstrained step hits a bound
 *  pRef                       n²+10n    n           reflected direction p̃ after bound hit
 *  ag                         n²+11n    n           Cauchy direction (−g, scaled anti-gradient)
 * </pre>
 *
 * <p>Note: {@code wa2[n]} doubles as acnorm output from qrfac (offset 0) and trial point
 * xₖ + p passed to eval — these roles are non-overlapping within each iteration.</p>
 *
 * <p>Total allocation: 4 standalone arrays (fvec[m], fjac[m*n], wa2[n], wa4[m]) +
 * one merged buffer of size n² + 12n + one int[] ipvt[n].</p>
 *
 * @see TRFCore
 */
public final class TRFWorkspace {

    // ── eval-facing: must remain standalone double[] from index 0 ────────────
    double[] fvec;   // f(xₖ)          — residuals filled by eval
    double[] fjac;   // J(xₖ) row-major — Jacobian filled by eval, then overwritten by qrfac
    double[] wa2;    // xₖ + p          — trial point passed to eval
    double[] wa4;    // f(xₖ + p)       — trial residuals filled by eval

    // ── integer pivot array (separate type, cannot merge into double[]) ───────
    int[] ipvt;      // column pivot permutation from qrfac  [n]

    // ── merged scratch buffer ─────────────────────────────────────────────────
    double[] work;   // [ rwork | diag | qtf | wa1 | wa3 | clScale | effDiag | Jp | Jpr | step | xHit | pRef | ag ]

    // ── offsets into work[] ───────────────────────────────────────────────────
    int rworkOff;    // rwork    [n²]   working copy of R for lmpar
    int diagOff;     // diag     [n]    base scaling diagonal D
    int qtfOff;      // qtf      [n]    Qᵀ·f / tmp[] scratch in qrfac
    int wa1Off;      // wa1      [n]    step p from lmpar / rdiag from qrfac
    int wa3Off;      // wa3      [n]    D·step / scratch
    int clScaleOff;  // clScale  [n]    Coleman-Li scaling vector
    int effDiagOff;  // effDiag  [n]    diag·clScale
    int JpOff;       // Jp       [n]    R·Pᵀ·(tHit·p)
    int JprOff;      // Jpr      [n]    R·Pᵀ·p̃
    int stepOff;     // step     [n]    actual step after reflection
    int xHitOff;     // xHit     [n]    bound-hit point
    int pRefOff;     // pRef     [n]    reflected direction p̃
    int agOff;       // ag       [n]    Cauchy direction −g

    // ── mutable state carried between inner/outer loops ───────────────────────
    double delta;
    double xnorm;

    /**
     * Creates an uninitialized TRF workspace. Call {@link #ensure(int, int)} before use.
     */
    private void init(int m, int n) {
        fvec = new double[m];
        fjac = new double[m * n];
        wa2  = new double[n];
        wa4  = new double[m];
        ipvt = new int[n];

        rworkOff   = 0;
        diagOff    = n * n;
        qtfOff     = n * n +  n;
        wa1Off     = n * n + 2 * n;
        wa3Off     = n * n + 3 * n;
        clScaleOff = n * n + 4 * n;
        effDiagOff = n * n + 5 * n;
        JpOff      = n * n + 6 * n;
        JprOff     = n * n + 7 * n;
        stepOff    = n * n + 8 * n;
        xHitOff    = n * n + 9 * n;
        pRefOff    = n * n + 10 * n;
        agOff      = n * n + 11 * n;

        work  = new double[n * n + 12 * n];
        delta = 0.0;
        xnorm = 0.0;
    }

    /** Returns the number of residuals this workspace was allocated for (0 before first {@link #ensure}). */
    public int getM() { return fvec != null ? fvec.length : 0; }

    /** Returns the number of parameters this workspace was allocated for (0 before first {@link #ensure}). */
    public int getN() { return ipvt != null ? ipvt.length : 0; }

    /** Returns true if this workspace is compatible with the given problem dimensions. */
    public boolean isCompatible(int m, int n) {
        return getM() == m && getN() == n;
    }

    /**
     * Ensures this workspace can handle dimensions {@code m} × {@code n}.
     * Reallocates all arrays only when capacity is exceeded; always resets to zero.
     */
    public void ensure(int m, int n) {
        if (m > getM() || n > getN()) {
            init(m, n);  // reallocate
        } else {
            reset();     // reuse existing buffers
        }
    }

    /** Resets all buffers to zero and clears mutable state. */
    public void reset() {
        if (fvec != null) Arrays.fill(fvec, 0.0);
        if (fjac != null) Arrays.fill(fjac, 0.0);
        if (wa2  != null) Arrays.fill(wa2,  0.0);
        if (wa4  != null) Arrays.fill(wa4,  0.0);
        if (ipvt != null) Arrays.fill(ipvt, 0);
        if (work != null) Arrays.fill(work, 0.0);
        delta = 0.0;
        xnorm = 0.0;
    }
}
