package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.Multivariate;
import com.curioloop.numopt4j.optim.NumericalJacobian;
import com.curioloop.numopt4j.optim.Optimization;


import java.util.function.BiConsumer;

/**
 * Powell Hybrid (HYBR) solver for nonlinear systems F(x) = 0 — column-major variant.
 *
 * <h3>Algorithm</h3>
 * <p>Implements the Powell hybrid method (MINPACK {@code hybrd}) with a trust-region
 * dogleg step.  Each outer iteration recomputes the full Jacobian J = ∂F/∂x and
 * performs a QR factorisation J·P = Q·R.  The inner loop then iterates using
 * rank-1 Broyden-style updates of R (via {@link Minpack#r1updt}) until the Jacobian
 * becomes stale (two consecutive step failures), at which point a fresh Jacobian is
 * requested.</p>
 *
 * <p>Step selection follows the dogleg strategy:</p>
 * <ol>
 *   <li>Compute Gauss-Newton step p_gn = −R⁻¹·Qᵀ·F.</li>
 *   <li>If ‖D·p_gn‖₂ &le; Δ, accept p_gn directly.</li>
 *   <li>Otherwise blend p_gn with the scaled steepest-descent direction p_sd,
 *       clipping to the trust-region boundary ‖D·p‖₂ = Δ.</li>
 * </ol>
 *
 * <p>The trust-region radius Δ is updated based on the ratio
 * ρ = actual_reduction / predicted_reduction.</p>
 *
 * <h3>Storage conventions</h3>
 * <ul>
 *   <li>Jacobian {@code fjac}: col-major, {@code fjac[i + n*j]} = J[i,j].</li>
 *   <li>Packed R {@code r}: row-packed upper triangular,
 *       {@code r[i*(2n-i-1)/2 + j]} = R[i,j] for i &le; j.</li>
 * </ul>
 *
 * <h3>Callback protocol</h3>
 * <ul>
 *   <li>{@code eval.evaluate(x, fvec, null)} — compute F(x) only.</li>
 *   <li>{@code eval.evaluate(x, fvec, fjac)} — compute F(x) and col-major J(x).</li>
 * </ul>
 * When no analytical Jacobian is available, wrap with
 * {@link NumericalJacobian#wrap(BiConsumer, int, int, boolean) NumericalJacobian.FORWARD.wrap(fn, n, n, true)}
 * to obtain a forward-difference col-major Jacobian automatically.
 */
public final class HYBRSolver {

    private HYBRSolver() {}

    public static final double DEFAULT_FTOL          = 1.49e-8;
    public static final int    DEFAULT_MAXFEV_FACTOR = 200;
    public static final double DEFAULT_FACTOR        = 100.0;

    /**
     * Solve F(x) = 0 using a {@link Multivariate} that provides F and optionally its Jacobian.
     *
     * <h4>Callback protocol</h4>
     * <ul>
     *   <li>{@code eval.evaluate(x, fvec, null)} — compute F(x) only</li>
     *   <li>{@code eval.evaluate(x, fvec, fjac)} — compute F(x) and col-major J(x)</li>
     * </ul>
     * When no analytical Jacobian is available, wrap the residual function with
     * {@link NumericalJacobian#wrap(BiConsumer, int, int, boolean) NumericalJacobian.FORWARD.wrap(fn, n, n, true)}
     * to obtain a forward-difference col-major Jacobian automatically.
     *
     * <h4>Convergence criteria</h4>
     * <p>The solver terminates with the following {@code info} codes:</p>
     * <ol>
     *   <li>{@code info=1} — step tolerance satisfied: ‖D·p‖₂ &le; xtol·‖D·xₖ‖₂
     *       (or ‖F(xₖ)‖₂ = 0); maps to {@code COEFFICIENT_TOLERANCE_REACHED}.</li>
     *   <li>{@code info=2} — function evaluation budget exhausted ({@code nfev &ge; maxfev});
     *       maps to {@code MAX_ITERATIONS_REACHED}.</li>
     *   <li>{@code info=3} — step too small relative to machine precision:
     *       0.1·max(0.1·Δ, ‖p‖₂) &le; ε·‖D·xₖ‖₂; maps to {@code MAX_ITERATIONS_REACHED}.</li>
     *   <li>{@code info=4} — slow progress: 5 consecutive outer iterations with
     *       actual reduction &lt; 10 %; maps to {@code MAX_ITERATIONS_REACHED}.</li>
     *   <li>{@code info=5} — very slow progress: 10 consecutive inner iterations with
     *       actual reduction &lt; 0.1 %; maps to {@code MAX_ITERATIONS_REACHED}.</li>
     *   <li>{@code info=-1} — NaN/Infinity detected in F(x) or J(x);
     *       maps to {@code ABNORMAL_TERMINATION}.</li>
     * </ol>
     *
     * <h4>Trust-region update</h4>
     * <p>After each trial step p the ratio ρ = actual_reduction / predicted_reduction
     * governs the trust-region radius Δ:</p>
     * <ul>
     *   <li>ρ &lt; 0.1 → Δ ← Δ/2 (step rejected)</li>
     *   <li>ρ &ge; 0.5 or two consecutive successes → Δ ← max(Δ, ‖p‖₂/0.5)</li>
     *   <li>|ρ − 1| &le; 0.1 → Δ ← ‖p‖₂/0.5</li>
     *   <li>ρ &ge; 10⁻⁴ → step accepted, xₖ₊₁ = xₖ + p</li>
     * </ul>
     *
     * @param eval   equation system + optional Jacobian (col-major when fjac != null)
     * @param x0     initial point x₀
     * @param xtol   convergence tolerance on relative step size ‖D·p‖₂ / ‖D·xₖ‖₂
     * @param maxfev maximum number of function evaluations
     * @param ws     pre-allocated workspace (must satisfy {@code ws.isCompatible(x0.length)})
     * @return root-finding result containing the final iterate, ‖F‖₂, status, and nfev
     */
    public static Optimization solve(
            Multivariate eval,
            double[] x0, double xtol, int maxfev, HYBRWorkspace ws) {

        final int n = x0.length;
        for (double v : x0) {
            if (Double.isNaN(v) || Double.isInfinite(v))
                throw new IllegalArgumentException("Initial point x0 contains NaN or Infinity");
        }

        final double epsmch = Minpack.dpmpar(1);

        // ── eval-facing arrays (standalone) ──────────────────────────────────
        final double[] x    = ws.x;
        final double[] fvec = ws.fx;
        final double[] fjac = ws.fjac;
        final double[] wa2  = ws.wa2;   // trial point x_trial  (input to eval)
        final double[] wa4  = ws.wa4;   // F(x_trial)           (output from eval)

        // ── merged scratch buffer + offsets ──────────────────────────────────
        final double[] work      = ws.work;
        final int      wa1Off    = ws.wa1Off;
        final int      wa3Off    = ws.wa3Off;
        final int      rdiagOff  = ws.rdiagOff;
        final int      acnormOff = ws.acnormOff;
        final int      diagOff   = ws.diagOff;
        final int      qtfOff    = ws.qtfOff;
        final int      rOff      = ws.rOff;
        final int      lr        = n * (n + 1) / 2;

        System.arraycopy(x0, 0, x, 0, n);
        eval.evaluate(x, n, fvec, n, null);
        int nfev = 1;
        for (double v : fvec) {
            if (Double.isNaN(v) || Double.isInfinite(v))
                return new Optimization(Double.NaN, x.clone(), Double.NaN, Optimization.Status.INVALID_INPUT, nfev, nfev);
        }

        double fnorm = Minpack.enorm(n, fvec);
        if (fnorm == 0.0)
            return new Optimization(Double.NaN, x.clone(), fnorm, Optimization.Status.COEFFICIENT_TOLERANCE_REACHED, nfev, nfev);

        int    iter   = 1;
        double delta  = 0.0;
        double xnorm  = 0.0;
        int    ncfail = 0, ncsuc = 0;
        int    nslow1 = 0, nslow2 = 0;
        int    info   = 0;

        outer:
        while (true) {

            // ── Compute Jacobian (col-major) ──────────────────────────────────
            eval.evaluate(x, n, fvec, n, fjac);
            nfev++;
            for (double v : fjac) {
                if (Double.isNaN(v) || Double.isInfinite(v)) { info = -1; break outer; }
            }

            // ── QR factorization (col-major, no pivoting) ─────────────────────
            // rdiag → work[rdiagOff], acnorm → work[acnormOff], wa → work[wa1Off]
            Minpack.qrfac(n, n, fjac, n, false, null, 1,
                          work, rdiagOff,
                          work, acnormOff,
                          work, wa1Off);

            if (iter == 1) {
                for (int j = 0; j < n; j++)
                    work[diagOff + j] = (work[acnormOff + j] != 0.0) ? work[acnormOff + j] : 1.0;
                double s = 0.0;
                for (int j = 0; j < n; j++) { double v = work[diagOff + j] * x[j]; s += v * v; }
                xnorm = Math.sqrt(s);
                delta = DEFAULT_FACTOR * xnorm;
                if (delta == 0.0) delta = DEFAULT_FACTOR;
            }

            // ── Pack upper triangle into r (row-packed) ───────────────────────
            for (int j = 0; j < n; j++) {
                int ll = j;
                for (int i = 0; i < j; i++) { work[rOff + ll] = fjac[i + n * j]; ll += n - i - 1; }
                work[rOff + ll] = work[rdiagOff + j];
            }

            // ── Compute Qᵀ·fvec → qtf ────────────────────────────────────────
            System.arraycopy(fvec, 0, work, qtfOff, n);
            for (int j = 0; j < n; j++) {
                double diag_jj = fjac[j + n * j];
                if (diag_jj != 0.0) {
                    double sum = 0.0;
                    for (int i = j; i < n; i++) sum += fjac[i + n * j] * work[qtfOff + i];
                    double temp = -sum / diag_jj;
                    for (int i = j; i < n; i++) work[qtfOff + i] += fjac[i + n * j] * temp;
                }
            }

            // ── Accumulate Q in fjac (col-major, in-place) ────────────────────
            // wa → work[wa1Off]
            Minpack.qform(n, n, fjac, n, work, wa1Off);

            if (iter > 1) {
                for (int j = 0; j < n; j++)
                    work[diagOff + j] = Math.max(work[diagOff + j], work[acnormOff + j]);
            }

            // ── Inner loop (dogleg + rank-1 update) ───────────────────────────
            boolean jeval = true;
            inner:
            while (true) {
                // r → work[rOff], diag → work[diagOff], qtf → work[qtfOff]
                // wa1 → work[wa1Off], wa2 → standalone
                Minpack.dogleg(n,
                               work, rOff, lr,
                               work, diagOff,
                               work, qtfOff,
                               delta,
                               work, wa1Off,   // output step → work[wa1Off]
                               work, wa3Off,   // internal work → work[wa3Off]
                               wa2);           // internal work → standalone wa2

                // wa1 holds the step; negate and form trial point in wa2
                double pnorm = 0.0;
                for (int j = 0; j < n; j++) {
                    work[wa1Off + j] = -work[wa1Off + j];
                    wa2[j] = x[j] + work[wa1Off + j];
                    work[wa3Off + j] = work[diagOff + j] * work[wa1Off + j];
                    pnorm += work[wa3Off + j] * work[wa3Off + j];
                }
                pnorm = Math.sqrt(pnorm);
                if (iter == 1) delta = Math.min(delta, pnorm);

                eval.evaluate(wa2, n, wa4, n, null);
                nfev++;
                for (double v : wa4) {
                    if (Double.isNaN(v) || Double.isInfinite(v)) { info = -1; break outer; }
                }
                double fnorm1 = Minpack.enorm(n, wa4);

                double actred = -1.0;
                if (fnorm1 < fnorm) actred = 1.0 - (fnorm1 / fnorm) * (fnorm1 / fnorm);

                // Predicted reduction: wa3 = Qᵀ·F + R·p
                int l = 0;
                for (int i = 0; i < n; i++) {
                    double sum = 0.0;
                    for (int j = i; j < n; j++) { sum += work[rOff + l] * work[wa1Off + j]; l++; }
                    work[wa3Off + i] = work[qtfOff + i] + sum;
                }
                double temp = Minpack.enorm(n, work, wa3Off);
                double prered = (temp < fnorm) ? 1.0 - (temp / fnorm) * (temp / fnorm) : 0.0;
                double ratio  = (prered > 0.0) ? actred / prered : 0.0;

                // Update trust region
                if (ratio < 0.1) {
                    ncsuc = 0; ncfail++;
                    delta *= 0.5;
                } else {
                    ncfail = 0; ncsuc++;
                    if (ratio >= 0.5 || ncsuc > 1) delta = Math.max(delta, pnorm / 0.5);
                    if (Math.abs(ratio - 1.0) <= 0.1) delta = pnorm / 0.5;
                }

                // Accept step
                if (ratio >= 1e-4) {
                    System.arraycopy(wa2, 0, x, 0, n);
                    for (int j = 0; j < n; j++) {
                        wa2[j] = work[diagOff + j] * x[j];
                        fvec[j] = wa4[j];
                    }
                    xnorm = Minpack.enorm(n, wa2);
                    fnorm = fnorm1;
                    iter++;
                }

                nslow1++;
                if (actred >= 0.001) nslow1 = 0;
                if (jeval) nslow2++;
                if (actred >= 0.1)   nslow2 = 0;

                if (delta <= xtol * xnorm || fnorm == 0.0) { info = 1; break outer; }
                if (nfev >= maxfev)                         { info = 2; break outer; }
                if (0.1 * Math.max(0.1 * delta, pnorm) <= epsmch * xnorm) { info = 3; break outer; }
                if (nslow2 == 5)                            { info = 4; break outer; }
                if (nslow1 == 10)                           { info = 5; break outer; }

                if (ncfail == 2) break inner;  // recalculate Jacobian

                // ── Rank-1 update (col-major fjac) ────────────────────────────
                // wa2 (standalone) = (Jᵀ·wa4 − wa3) / ‖p‖₂  (reuse wa2 as v for r1updt)
                // wa1 (work[wa1Off]) = D·(D·p) / ‖p‖₂
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < n; i++) sum += fjac[i + n * j] * wa4[i];
                    wa2[j] = (sum - work[wa3Off + j]) / pnorm;
                    work[wa1Off + j] = work[diagOff + j] * ((work[diagOff + j] * work[wa1Off + j]) / pnorm);
                    if (ratio >= 1e-4) work[qtfOff + j] = sum;
                }
                boolean[] sing = {false};
                // s → work[rOff], u → work[wa1Off], v → wa2 (standalone), w → work[wa3Off]
                Minpack.r1updt(n, n, work, rOff, lr, work, wa1Off, wa2, 0, work, wa3Off, sing);
                // fjac updated by A*Q: a → fjac (standalone, aOff=0), v → wa2, w → work[wa3Off]
                Minpack.r1mpyq(n, n, fjac, 0, n, wa2, work, wa3Off);
                // qtf updated as 1×n matrix: a → work[qtfOff], v → wa2, w → work[wa3Off]
                Minpack.r1mpyq(1, n, work, qtfOff, 1, wa2, work, wa3Off);

                jeval = false;
            }
        }

        Optimization.Status status;
        if      (info == 1)  status = Optimization.Status.COEFFICIENT_TOLERANCE_REACHED;
        else if (info == -1) status = Optimization.Status.ABNORMAL_TERMINATION;
        else                 status = Optimization.Status.MAX_ITERATIONS_REACHED;

        return new Optimization(Double.NaN, x.clone(), fnorm, status, nfev, nfev);
    }
}
