package com.curioloop.numopt4j.optim.root;

import java.util.Arrays;

/**
 * Pre-allocated workspace for {@link HYBRSolver}.
 *
 * <pre>
 *  Field    Size          Role
 *  ──────────────────────────────────────────────────────
 *  x        n             current iterate
 *  fx       n             F(x)
 *  fjac     n²            col-major Jacobian J
 *  wa1–wa4  n each        scratch buffers
 *  r        n(n+1)/2      packed upper-triangular R
 *  rdiag    n             QR diagonal (written by qrfac, packed into r, then dead)
 *  acnorm   n             column norms (written by qrfac, used to update diag, then dead)
 *  diag     n             scaling vector
 *  qtf      n             Qᵀ·fvec
 * </pre>
 */
public final class HYBRWorkspace {

    final double[] x;
    final double[] fx;
    final double[] fjac;
    final double[] wa1;
    final double[] wa2;
    final double[] wa3;
    final double[] wa4;
    final double[] r;
    final double[] rdiag;
    final double[] acnorm;
    final double[] diag;
    final double[] qtf;

    public HYBRWorkspace(int n) {
        if (n < 1) throw new IllegalArgumentException("Workspace dimension must be >= 1, got: " + n);
        this.x      = new double[n];
        this.fx     = new double[n];
        this.fjac   = new double[n * n];
        this.wa1    = new double[n];
        this.wa2    = new double[n];
        this.wa3    = new double[n];
        this.wa4    = new double[n];
        this.r      = new double[n * (n + 1) / 2];
        this.rdiag  = new double[n];
        this.acnorm = new double[n];
        this.diag   = new double[n];
        this.qtf    = new double[n];
    }

    public boolean isCompatible(int n) { return x.length == n; }

    public void reset() {
        Arrays.fill(x,      0.0);
        Arrays.fill(fx,     0.0);
        Arrays.fill(fjac,   0.0);
        Arrays.fill(wa1,    0.0);
        Arrays.fill(wa2,    0.0);
        Arrays.fill(wa3,    0.0);
        Arrays.fill(wa4,    0.0);
        Arrays.fill(r,      0.0);
        Arrays.fill(rdiag,  0.0);
        Arrays.fill(acnorm, 0.0);
        Arrays.fill(diag,   0.0);
        Arrays.fill(qtf,    0.0);
    }
}
