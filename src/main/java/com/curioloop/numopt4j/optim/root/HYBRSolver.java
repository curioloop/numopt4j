package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.Multivariate;
import com.curioloop.numopt4j.optim.NumericalJacobian;
import com.curioloop.numopt4j.optim.OptimizationResult;
import com.curioloop.numopt4j.optim.OptimizationStatus;

import java.util.function.BiConsumer;

/**
 * Powell Hybrid (HYBR) solver — column-major variant.
 *
 * <p>Accepts a {@link Multivariate} that evaluates F(x) and optionally its Jacobian.
 * The Jacobian passed to {@link Multivariate#evaluate} is always col-major
 * ({@code fjac[i + n*j]}); use {@link NumericalJacobian#wrap(BiConsumer, int, int, boolean)}
 * with {@code transpose=true} when wrapping a numerical Jacobian, or supply an analytical
 * implementation that writes col-major directly.</p>
 *
 * <p>Storage conventions:</p>
 * <ul>
 *   <li>Jacobian {@code fjac}: col-major, {@code fjac[i + n*j]} = J[i,j]</li>
 *   <li>Packed R {@code r}: row-packed upper triangular,
 *       {@code r[i*(2n-i-1)/2 + j]} = R[i,j] for i &le; j</li>
 * </ul>
 *
 * <p>Key properties:</p>
 * <ul>
 *   <li>Col-major Jacobian — cache-friendly {@code qrfac}/{@code qform}</li>
 *   <li>In-place Householder accumulation — no {@code fjac.clone()} in {@code qform}</li>
 *   <li>{@link Minpack#dpmpar} for portable machine epsilon</li>
 *   <li>Unified {@link Multivariate} interface — fn and Jacobian in one callback</li>
 * </ul>
 */
public final class HYBRSolver {

    private HYBRSolver() {}

    static final double DEFAULT_FTOL          = 1.49e-8;
    static final int    DEFAULT_MAXFEV_FACTOR = 200;
    static final double DEFAULT_FACTOR        = 100.0;

    /**
     * Solve F(x)=0 using a {@link Multivariate} that provides F and optionally its Jacobian.
     *
     * <p>The {@code eval} callback is invoked as:</p>
     * <ul>
     *   <li>{@code eval.evaluate(x, fvec, null)} — compute F(x) only</li>
     *   <li>{@code eval.evaluate(x, fvec, fjac)} — compute F(x) and col-major Jacobian</li>
     * </ul>
     *
     * <p>When no analytical Jacobian is available, wrap the residual function with
     * {@link NumericalJacobian#wrap(BiConsumer, int, int, boolean) NumericalJacobian.FORWARD.wrap(fn, n, n, true)}
     * to get forward-difference col-major Jacobian automatically.</p>
     *
     * @param eval   equation system + optional Jacobian (col-major when jacobian != null)
     * @param x0     initial point
     * @param xtol   convergence tolerance on step size
     * @param maxfev maximum function evaluations
     * @param ws     pre-allocated workspace
     * @return root-finding result
     */
    public static OptimizationResult solve(
            Multivariate eval,
            double[] x0, double xtol, int maxfev, HYBRWorkspace ws) {

        final int n = x0.length;
        for (double v : x0) {
            if (Double.isNaN(v) || Double.isInfinite(v))
                throw new IllegalArgumentException("Initial point x0 contains NaN or Infinity");
        }

        final double epsmch = Minpack.dpmpar(1);
        final double[] x    = ws.x;
        final double[] fvec = ws.fx;
        final double[] fjac = ws.fjac;
        final double[] r    = ws.r;
        final double[] wa1  = ws.wa1;
        final double[] wa2  = ws.wa2;
        final double[] wa3  = ws.wa3;
        final double[] wa4  = ws.wa4;
        final int lr = n * (n + 1) / 2;

        final double[] rdiag  = ws.rdiag;
        final double[] acnorm = ws.acnorm;
        final double[] diag   = ws.diag;
        final double[] qtf    = ws.qtf;

        System.arraycopy(x0, 0, x, 0, n);
        eval.evaluate(x, fvec, null);
        int nfev = 1;
        for (double v : fvec) {
            if (Double.isNaN(v) || Double.isInfinite(v))
                return new OptimizationResult(Double.NaN, x.clone(), Double.NaN, OptimizationStatus.INVALID_INPUT, nfev);
        }

        double fnorm = Minpack.enorm(n, fvec);
        if (fnorm == 0.0)
            return new OptimizationResult(Double.NaN, x.clone(), fnorm, OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED, nfev);

        int    iter   = 1;
        double delta  = 0.0;
        double xnorm  = 0.0;
        int    ncfail = 0, ncsuc = 0;
        int    nslow1 = 0, nslow2 = 0;
        int    info   = 0;

        outer:
        while (true) {

            // ── Compute Jacobian (col-major) ──────────────────────────────────
            eval.evaluate(x, fvec, fjac);
            nfev++;
            for (double v : fjac) {
                if (Double.isNaN(v) || Double.isInfinite(v)) { info = -1; break outer; }
            }

            // ── QR factorization (col-major, no pivoting) ─────────────────────
            Minpack.qrfac(n, n, fjac, n, false, null, 1, rdiag, acnorm, wa1);

            if (iter == 1) {
                for (int j = 0; j < n; j++) diag[j] = (acnorm[j] != 0.0) ? acnorm[j] : 1.0;
                double s = 0.0;
                for (int j = 0; j < n; j++) { double v = diag[j] * x[j]; s += v * v; }
                xnorm = Math.sqrt(s);
                delta = DEFAULT_FACTOR * xnorm;
                if (delta == 0.0) delta = DEFAULT_FACTOR;
            }

            // ── Pack upper triangle into r (row-packed) ───────────────────────
            for (int j = 0; j < n; j++) {
                int ll = j;
                for (int i = 0; i < j; i++) { r[ll] = fjac[i + n * j]; ll += n - i - 1; }
                r[ll] = rdiag[j];
            }

            // ── Compute Q^T * fvec -> qtf ─────────────────────────────────────
            System.arraycopy(fvec, 0, qtf, 0, n);
            for (int j = 0; j < n; j++) {
                double diag_jj = fjac[j + n * j];
                if (diag_jj != 0.0) {
                    double sum = 0.0;
                    for (int i = j; i < n; i++) sum += fjac[i + n * j] * qtf[i];
                    double temp = -sum / diag_jj;
                    for (int i = j; i < n; i++) qtf[i] += fjac[i + n * j] * temp;
                }
            }

            // ── Accumulate Q in fjac (col-major, in-place) ────────────────────
            Minpack.qform(n, n, fjac, n, wa1);

            if (iter > 1) {
                for (int j = 0; j < n; j++) diag[j] = Math.max(diag[j], acnorm[j]);
            }

            // ── Inner loop (dogleg + rank-1 update) ───────────────────────────
            boolean jeval = true;
            inner:
            while (true) {
                Minpack.dogleg(n, r, lr, diag, qtf, delta, wa1, wa2, wa3);

                double pnorm = 0.0;
                for (int j = 0; j < n; j++) {
                    wa1[j] = -wa1[j];
                    wa2[j] = x[j] + wa1[j];
                    wa3[j] = diag[j] * wa1[j];
                    pnorm += wa3[j] * wa3[j];
                }
                pnorm = Math.sqrt(pnorm);
                if (iter == 1) delta = Math.min(delta, pnorm);

                eval.evaluate(wa2, wa4, null);
                nfev++;
                for (double v : wa4) {
                    if (Double.isNaN(v) || Double.isInfinite(v)) { info = -1; break outer; }
                }
                double fnorm1 = Minpack.enorm(n, wa4);

                double actred = -1.0;
                if (fnorm1 < fnorm) actred = 1.0 - (fnorm1 / fnorm) * (fnorm1 / fnorm);

                // Predicted reduction
                int l = 0;
                for (int i = 0; i < n; i++) {
                    double sum = 0.0;
                    for (int j = i; j < n; j++) { sum += r[l] * wa1[j]; l++; }
                    wa3[i] = qtf[i] + sum;
                }
                double temp = Minpack.enorm(n, wa3);
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
                    for (int j = 0; j < n; j++) { wa2[j] = diag[j] * x[j]; fvec[j] = wa4[j]; }
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
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < n; i++) sum += fjac[i + n * j] * wa4[i];
                    wa2[j] = (sum - wa3[j]) / pnorm;
                    wa1[j] = diag[j] * ((diag[j] * wa1[j]) / pnorm);
                    if (ratio >= 1e-4) qtf[j] = sum;
                }
                boolean[] sing = {false};
                Minpack.r1updt(n, n, r, lr, wa1, wa2, wa3, sing);
                Minpack.r1mpyq(n, n, fjac, n, wa2, wa3);
                Minpack.r1mpyq(1, n, qtf, 1, wa2, wa3);

                jeval = false;
            }
        }

        OptimizationStatus status;
        if      (info == 1)  status = OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED;
        else if (info == -1) status = OptimizationStatus.ABNORMAL_TERMINATION;
        else                 status = OptimizationStatus.MAX_ITERATIONS_REACHED;

        return new OptimizationResult(Double.NaN, x.clone(), fnorm, status, nfev);
    }
}
