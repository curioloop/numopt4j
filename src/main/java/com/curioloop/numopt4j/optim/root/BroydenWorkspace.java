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
    double[] x;       // current iterate
    double[] fx;      // F(x)
    double[] xNew;    // candidate next iterate
    double[] fNew;    // F(xNew)

    // ── merged scratch buffer ─────────────────────────────────────────────────
    double[] work;    // H[n²] | dx[n] | Hdf[n] | dxH[n] | dF[n]

    // ── offsets into work[] ───────────────────────────────────────────────────
    int hOff;    // H      [n*n]
    int dxOff;   // dx     [n]
    int HdfOff;  // Hdf    [n]
    int dxHOff;  // dxH    [n]
    int dFOff;   // dF     [n]

    /** Returns the current dimension, or 0 if not yet allocated. */
    public int getN() { return x != null ? x.length : 0; }

    /**
     * Ensures the workspace is allocated for dimension {@code n}.
     * Reallocates if {@code n > getN()}, then always calls {@link #reset()}.
     */
    public void ensure(int n) {
        if (n > getN()) {
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
            reset();
        } else {
            reset();
        }
    }

    public boolean isCompatible(int n) { return x != null && x.length == n; }

    public void reset() {
        if (x    != null) Arrays.fill(x,    0.0);
        if (fx   != null) Arrays.fill(fx,   0.0);
        if (xNew != null) Arrays.fill(xNew, 0.0);
        if (fNew != null) Arrays.fill(fNew, 0.0);
        if (work != null) Arrays.fill(work, 0.0);
    }
}
