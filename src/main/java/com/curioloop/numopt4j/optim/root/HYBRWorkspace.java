package com.curioloop.numopt4j.optim.root;

import java.util.Arrays;

/**
 * Pre-allocated workspace for {@link HYBRSolver}.
 *
 * <p>Memory is split into two categories:</p>
 *
 * <p><b>Standalone eval-facing arrays</b> — passed directly to the user callback
 * {@code eval.evaluate(x, fx, fjac)} or {@code eval.evaluate(wa2, wa4, null)}.
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
 *  x                          n         current iterate xₖ  (input to eval)
 *  fx                         n         F(xₖ)               (output from eval)
 *  fjac                       n²        Jacobian J (col-major, output from eval)
 *  wa2                        n         trial point x_trial = xₖ + p  (input to eval)
 *  wa4                        n         F(x_trial)          (output from eval)
 *
 *  Merged work buffer         Offset    Size        Role
 *  ─────────────────────────────────────────────────────────────────────
 *  wa1                        0         n           dogleg step p / qform scratch
 *  wa3                        n         n           D·p / predicted-reduction scratch
 *  rdiag                      2n        n           diagonal of R  (written by qrfac)
 *  acnorm                     3n        n           ‖A[:,j]‖₂ column norms (written by qrfac)
 *  diag                       4n        n           scaling diagonal D
 *  qtf                        5n        n           Qᵀ·F(xₖ)
 *  r                          6n        n(n+1)/2    packed upper-triangular R
 * </pre>
 *
 * <p>Total allocation: 5 standalone arrays of size n + n² + one merged buffer of size
 * 6n + n(n+1)/2.</p>
 */
public final class HYBRWorkspace {

    // ── eval-facing: must remain standalone double[] from index 0 ────────────
    double[] x;      // xₖ — current iterate passed to eval
    double[] fx;     // F(xₖ) — equation residuals filled by eval
    double[] fjac;   // J(xₖ) — col-major Jacobian filled by eval  (fjac[i + n*j] = J[i,j])
    double[] wa2;    // x_trial = xₖ + p — trial point passed to eval
    double[] wa4;    // F(x_trial) — trial residuals filled by eval

    // ── merged scratch buffer ─────────────────────────────────────────────────
    double[] work;   // [ wa1 | wa3 | rdiag | acnorm | diag | qtf | r ]

    // ── offsets into work[] ───────────────────────────────────────────────────
    int wa1Off;    // wa1    [n]   dogleg step p / qform scratch
    int wa3Off;    // wa3    [n]   D·p and predicted-reduction scratch
    int rdiagOff;  // rdiag  [n]   diagonal of R  (R[j,j] = rdiag[j])
    int acnormOff; // acnorm [n]   ‖J[:,j]‖₂ column norms before pivoting
    int diagOff;   // diag   [n]   scaling diagonal D
    int qtfOff;    // qtf    [n]   Qᵀ·F(xₖ)
    int rOff;      // r      [n*(n+1)/2]  packed upper-triangular R

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
            this.fjac = new double[n * n];
            this.wa2  = new double[n];
            this.wa4  = new double[n];
            this.wa1Off    = 0;
            this.wa3Off    = n;
            this.rdiagOff  = 2 * n;
            this.acnormOff = 3 * n;
            this.diagOff   = 4 * n;
            this.qtfOff    = 5 * n;
            this.rOff      = 6 * n;
            this.work = new double[6 * n + n * (n + 1) / 2];
            reset();
        } else {
            reset();
        }
    }

    public boolean isCompatible(int n) { return x != null && x.length == n; }

    public void reset() {
        if (x    != null) Arrays.fill(x,    0.0);
        if (fx   != null) Arrays.fill(fx,   0.0);
        if (fjac != null) Arrays.fill(fjac, 0.0);
        if (wa2  != null) Arrays.fill(wa2,  0.0);
        if (wa4  != null) Arrays.fill(wa4,  0.0);
        if (work != null) Arrays.fill(work, 0.0);
    }
}
