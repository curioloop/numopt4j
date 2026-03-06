package com.curioloop.numopt4j.optim.root;

import java.util.Arrays;

/**
 * Pre-allocated workspace for {@link BroydenSolver}.
 *
 * <pre>
 *  Standalone (eval-facing)        Size   Role
 *  ──────────────────────────────────────────────────────
 *  x                               n      current iterate xₖ  (input to fn)
 *  fx                              n      F(xₖ)               (output from fn)
 *  xNew                            n      candidate xₖ + s·dx (input to fn)
 *  fNew                            n      F(xNew)             (output from fn)
 *
 *  Merged work buffer              Size   Role
 *  ──────────────────────────────────────────────────────
 *  work[0 .. n²-1]                 n²     H: inverse-Jacobian approx (row-major)
 *  work[n²       .. n²+n-1]        n      dx:  Newton step = H·F
 *  work[n²+n     .. n²+2n-1]       n      Hdf: H·dF scratch (reused as c = dx − H·dF)
 *  work[n²+2n    .. n²+3n-1]       n      dxH: dxᵀ·H (row vector stored as column)
 *  work[n²+3n    .. n²+4n-1]       n      dF:  F(xNew) − F(xₖ)
 * </pre>
 *
 * <p>H is zero-initialised; {@link BroydenSolver} writes the diagonal
 * H₀ = −α·I before the first iteration.</p>
 */
public final class BroydenWorkspace {

    // ── eval-facing: must remain standalone double[] from index 0 ────────────
    final double[] x;       // current iterate
    final double[] fx;      // F(x)
    final double[] xNew;    // candidate next iterate
    final double[] fNew;    // F(xNew)

    // ── merged scratch buffer ─────────────────────────────────────────────────
    final double[] work;    // H[n²] | dx[n] | Hdf[n] | dxH[n] | dF[n]

    // ── offsets into work[] ───────────────────────────────────────────────────
    final int hOff;    // H      [n*n]
    final int dxOff;   // dx     [n]
    final int HdfOff;  // Hdf    [n]
    final int dxHOff;  // dxH    [n]
    final int dFOff;   // dF     [n]

    public BroydenWorkspace(int n) {
        if (n < 1) throw new IllegalArgumentException("Workspace dimension must be >= 1, got: " + n);
        this.x    = new double[n];
        this.fx   = new double[n];
        this.xNew = new double[n];
        this.fNew = new double[n];

        this.hOff   = 0;
        this.dxOff  = n * n;
        this.HdfOff = n * n + n;
        this.dxHOff = n * n + 2 * n;
        this.dFOff  = n * n + 3 * n;

        this.work = new double[n * n + 4 * n];
    }

    public boolean isCompatible(int n) { return x.length == n; }

    public void reset() {
        Arrays.fill(x,    0.0);
        Arrays.fill(fx,   0.0);
        Arrays.fill(xNew, 0.0);
        Arrays.fill(fNew, 0.0);
        Arrays.fill(work, 0.0);
    }
}
