/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

/**
 * Workspace for the Trust Region Reflective (TRF) optimization algorithm.
 *
 * <p>Pre-allocates all working arrays required by TRF to avoid repeated
 * memory allocation during optimization. A workspace instance can be reused
 * across multiple calls to {@link TRFCore#optimize}.</p>
 *
 * <h2>Memory layout</h2>
 * <pre>
 *   fvec    [m]   — residual vector f(x)
 *   fjac    [m×n] — Jacobian J (row-major); overwritten by qrfac with Q/R
 *   rwork   [n×n] — working copy of upper triangle of R for lmpar
 *   diag    [n]   — base scaling diagonal D (column norms or user-supplied)
 *   qtf     [n]   — Q^T·f (first n elements); also used as dot[] scratch in qrfac
 *   wa1     [n]   — work: step p from lmpar / rdiag from qrfac
 *   wa2     [n]   — work: trial point x+p / acnorm from qrfac
 *   wa3     [n]   — work: D·step / scratch
 *   wa4     [m]   — work: f(x+p) / Q^T·f during applyQtToVec
 *   ipvt    [n]   — column pivot permutation from qrfac
 *   clScale [n]   — Coleman-Li scaling: distance to nearest bound
 *   effDiag [n]   — effective diagonal = diag * clScale
 *   Jp      [n]   — R·P^T·(tHit·p)  for quadratic model
 *   Jpr     [n]   — R·P^T·p_ref     for quadratic model
 *   step    [n]   — actual step after reflection / clipping
 *   xHit    [n]   — point where unconstrained step hits a bound
 *   pRef    [n]   — reflected direction after bound hit
 *   ag      [n]   — Cauchy direction (-g_h, scaled anti-gradient) for 3-way step selection
 * </pre>
 *
 * @see TRFCore
 */
public final class TRFWorkspace {

    final double[] fvec;    // residuals          [m]
    final double[] fjac;    // Jacobian            [m*n]
    final double[] rwork;   // R copy for lmpar    [n*n]
    final double[] diag;    // base scaling        [n]
    final double[] qtf;     // Q^T·f / dot scratch [n]
    final double[] wa1;     // work                [n]
    final double[] wa2;     // work                [n]
    final double[] wa3;     // work                [n]
    final double[] wa4;     // work                [m]
    final int[]    ipvt;    // pivots              [n]

    final double[] clScale; // Coleman-Li scaling  [n]
    final double[] effDiag; // diag * clScale      [n]
    final double[] Jp;      // R·P^T·(tHit·p)      [n]
    final double[] Jpr;     // R·P^T·p_ref         [n]
    final double[] step;    // actual step         [n]
    final double[] xHit;    // bound-hit point     [n]
    final double[] pRef;    // reflected direction [n]
    final double[] ag;      // Cauchy direction (-g_h scaled gradient) [n]

    // mutable state carried between inner/outer loops
    double delta;
    double xnorm;

    /**
     * Allocates a TRF workspace for a problem with {@code m} residuals and {@code n} parameters.
     *
     * @param m number of residuals (m &ge; n &gt; 0)
     * @param n number of parameters
     */
    public TRFWorkspace(int m, int n) {
        fvec    = new double[m];
        fjac    = new double[m * n];
        rwork   = new double[n * n];
        diag    = new double[n];
        qtf     = new double[n];
        wa1     = new double[n];
        wa2     = new double[n];
        wa3     = new double[n];
        wa4     = new double[m];
        ipvt    = new int[n];
        clScale = new double[n];
        effDiag = new double[n];
        Jp      = new double[n];
        Jpr     = new double[n];
        step    = new double[n];
        xHit    = new double[n];
        pRef    = new double[n];
        ag      = new double[n];
    }

    /** Returns the number of residuals this workspace was allocated for. */
    public int getM() { return fvec.length; }

    /** Returns the number of parameters this workspace was allocated for. */
    public int getN() { return diag.length; }

    /** Returns true if this workspace is compatible with the given problem dimensions. */
    public boolean isCompatible(int m, int n) {
        return fvec.length == m && diag.length == n;
    }
}
