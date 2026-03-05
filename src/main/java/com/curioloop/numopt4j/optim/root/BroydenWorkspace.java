package com.curioloop.numopt4j.optim.root;

import java.util.Arrays;

/**
 * Pre-allocated workspace for {@link BroydenSolver}.
 *
 * <pre>
 *  Field    Size   Role
 *  ──────────────────────────────────────────────────────
 *  x        n      current iterate
 *  fx       n      F(x)
 *  H        n²     inverse-Jacobian approximation (row-major)
 *  dx       n      Newton step = H·F
 *  Hdf      n      H·dF scratch
 *  dxH      n      dxᵀ·H (row vector stored as column)
 *  xNew     n      candidate x + s·dx
 *  fNew     n      F(xNew)
 *  dF       n      F(xNew) - F(x)
 * </pre>
 *
 * <p>{@code H} is zero-initialised; {@link BroydenSolver} writes the diagonal
 * {@code -alpha·I} before the first iteration.</p>
 */
public final class BroydenWorkspace {

    final double[] x;
    final double[] fx;
    final double[] H;      // n×n inverse-Jacobian, row-major  (≡ fjac in old RootWorkspace)
    final double[] dx;     // Newton step                       (≡ wa1)
    final double[] Hdf;    // H·dF scratch                     (≡ wa2)
    final double[] dxH;    // dxᵀ·H                            (≡ wa3)
    final double[] xNew;   // candidate next iterate            (≡ wa4)
    final double[] fNew;   // F(xNew)
    final double[] dF;     // F(xNew) - F(x)

    public BroydenWorkspace(int n) {
        if (n < 1) throw new IllegalArgumentException("Workspace dimension must be >= 1, got: " + n);
        this.x    = new double[n];
        this.fx   = new double[n];
        this.H    = new double[n * n];
        this.dx   = new double[n];
        this.Hdf  = new double[n];
        this.dxH  = new double[n];
        this.xNew = new double[n];
        this.fNew = new double[n];
        this.dF   = new double[n];
    }

    public boolean isCompatible(int n) { return x.length == n; }

    public void reset() {
        Arrays.fill(x,    0.0);
        Arrays.fill(fx,   0.0);
        Arrays.fill(H,    0.0);
        Arrays.fill(dx,   0.0);
        Arrays.fill(Hdf,  0.0);
        Arrays.fill(dxH,  0.0);
        Arrays.fill(xNew, 0.0);
        Arrays.fill(fNew, 0.0);
        Arrays.fill(dF,   0.0);
    }
}
