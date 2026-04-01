package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.Optimization;

import com.curioloop.numopt4j.linalg.blas.BLAS;

import java.util.Arrays;
import java.util.function.BiConsumer;

/**
 * Good Broyden (broyden1) solver — strict port of scipy's {@code nonlin_solve + BroydenFirst}.
 *
 * <h3>Algorithm</h3>
 * <p>Maintains an explicit n×n dense approximation of the <em>inverse</em> Jacobian H,
 * initialised as {@code H₀ = -alpha·I} where
 * {@code alpha = 0.5·max(‖x₀‖₂, 1) / ‖F(x₀)‖₂}
 * (scipy {@code GenericBroyden.setup()}).</p>
 *
 * <p>Each iteration:</p>
 * <ol>
 *   <li>Newton step: {@code dx = H·F(x)}</li>
 *   <li>Armijo backtracking line search with quadratic/cubic interpolation
 *       ({@code scalar_search_armijo}, {@code amin=1e-2}, {@code c1=1e-4})</li>
 *   <li>Good Broyden rank-1 update of H (inverse Jacobian form):
 *       {@code H₊ = H + (dx - H·dF)·(dx^T·H) / (dx^T·H·dF)}</li>
 * </ol>
 *
 * <p>Convergence: {@code max|F(x)| ≤ f_tol} (max-norm, scipy default {@code tol_norm=maxnorm}).</p>
 *
 * <p>Singular H: reset to {@code H₀} and retry the step (scipy {@code BroydenFirst.solve()}).</p>
 */
public final class BroydenSolver {

    private BroydenSolver() {}

    /** Default f_tol: {@code eps^(1/3) ≈ 6e-6} — matches scipy's TerminationCondition default. */
    public static final double DEFAULT_FTOL = 6e-6;

    public static Optimization solve(
            BiConsumer<double[], double[]> fn,
            double[] x0,
            double ftol,
            int maxiter,
            BroydenWorkspace ws) {

        final int n = x0.length;

        for (double v : x0) {
            if (Double.isNaN(v) || Double.isInfinite(v))
                throw new IllegalArgumentException("Initial point x0 contains NaN or Infinity");
        }

        // ── workspace fields ─────────────────────────────────────────────────
        final double[] x    = ws.x;
        final double[] fx   = ws.fx;
        final double[] xNew = ws.xNew;
        final double[] fNew = ws.fNew;
        final double[] work = ws.work;
        final int hOff   = ws.hOff;
        final int dxOff  = ws.dxOff;
        final int HdfOff = ws.HdfOff;
        final int dxHOff = ws.dxHOff;
        final int dFOff  = ws.dFOff;

        System.arraycopy(x0, 0, x, 0, n);
        fn.accept(x, fx);
        int nfev = 1;

        for (double v : fx) {
            if (Double.isNaN(v) || Double.isInfinite(v))
                return new Optimization(Double.NaN, x.clone(), Double.NaN,
                        Optimization.Status.ABNORMAL_TERMINATION, nfev, nfev);
        }

        // ── initial convergence check ────────────────────────────────────────
        double fnorm = maxnorm(fx, n);
        if (fnorm <= ftol)
            return new Optimization(Double.NaN, x.clone(), fnorm,
                    Optimization.Status.COEFFICIENT_TOLERANCE_REACHED, 0, 0);

        // ── H₀ = -alpha·I  (scipy GenericBroyden.setup) ─────────────────────
        // alpha = 0.5 * max(‖x₀‖₂, 1) / ‖F₀‖₂
        // fjac is zero-initialised by RootWorkspace; only write diagonal.
        double alpha = 0.5 * Math.max(BLAS.dnrm2(n, x, 0, 1), 1.0)
                           / BLAS.dnrm2(n, fx, 0, 1);
        Arrays.fill(work, hOff, hOff + n * n, 0.0);
        BLAS.dset(n, -alpha, work, hOff, n + 1);  // stride n+1 → diagonal only

        for (int iter = 1; iter <= maxiter; iter++) {

            // ── Newton step: dx = -H · F ─────────────────────────────────────
            // scipy BroydenFirst.solve(): r = Gm.matvec(f); if not finite → reset H, return Gm.matvec(f)
            // so dx = -H·F, with singular-reset applied before the step is used.
            BLAS.dgemv(BLAS.Trans.NoTrans, n, n, 1.0, work, hOff, n, fx, 0, 1, 0.0, work, dxOff, 1);
            boolean finite = true;
            for (int i = 0; i < n; i++) if (!Double.isFinite(work[dxOff + i])) { finite = false; break; }
            if (!finite) {
                // reset H₀ = -resetAlpha·I  (scipy: self.setup(last_x, last_f, func))
                // H·fx = -resetAlpha·fx, so dx = -H·fx = resetAlpha·fx — no dgemv needed
                double resetAlpha = 0.5 * Math.max(BLAS.dnrm2(n, x, 0, 1), 1.0)
                                        / Math.max(fnorm, 1e-300);
                BLAS.dlaset(BLAS.Uplo.All, n, n, 0.0, -resetAlpha, work, hOff, n);  // full clear + diagonal
                BLAS.dcopy(n, fx, 0, 1, work, dxOff, 1);
                BLAS.dscal(n, -resetAlpha, work, dxOff, 1);
            } else {
                BLAS.dscal(n, -1.0, work, dxOff, 1);
            }

            // ── Armijo line search (scalar_search_armijo) ────────────────────
            // scipy _nonlin_line_search: phi(s) = ‖F(x+s·dx)‖₂²  (L2 norm squared)
            // derphi0 = -tmp_phi[0] = -‖F‖₂²  (scipy convention)
            // Convergence check uses max-norm, but line search phi uses L2 norm.
            // amin = 1e-2,  c1 = 1e-4
            final double fnormL2 = BLAS.dnrm2(n, fx, 0, 1);
            final double phi0    = fnormL2 * fnormL2;   // ‖F‖₂²
            final double derphi0 = -phi0;               // scipy: -tmp_phi[0]
            final double amin    = 1e-2;
            final double c1      = 1e-4;

            double step;
            double fnormNew;

            // evaluate phi(alpha0=1)
            BLAS.dcopy(n, x, 0, 1, xNew, 0, 1);
            BLAS.daxpy(n, 1.0, work, dxOff, 1, xNew, 0, 1);
            fn.accept(xNew, fNew);
            nfev++;
            double phi_a0 = l2normSq(fNew, n);

            if (phi_a0 <= phi0 + c1 * derphi0) {
                // full step accepted immediately
                step = 1.0;
                fnormNew = maxnorm(fNew, n);
            } else {
                // quadratic interpolation: alpha1 = -derphi0·alpha0² / (2·(phi_a0-phi0-derphi0·alpha0))
                double alpha0 = 1.0;
                double alpha1 = -derphi0 * alpha0 * alpha0
                        / (2.0 * (phi_a0 - phi0 - derphi0 * alpha0));
                BLAS.dcopy(n, x, 0, 1, xNew, 0, 1);
                BLAS.daxpy(n, alpha1, work, dxOff, 1, xNew, 0, 1);
                fn.accept(xNew, fNew);
                nfev++;
                double phi_a1 = l2normSq(fNew, n);

                if (phi_a1 <= phi0 + c1 * alpha1 * derphi0) {
                    step = alpha1;
                    fnormNew = maxnorm(fNew, n);
                } else {
                    // cubic interpolation loop — fev[] carries extra evaluation count back
                    int[] fev = {0};
                    step = cubicSearchArmijo(
                            fn, x, work, dxOff, n, phi0, derphi0, c1, amin,
                            alpha0, phi_a0, alpha1, phi_a1,
                            xNew, fNew, fev);
                    nfev += fev[0];
                    if (step < 0) {
                        // line search failed (returned sentinel -1.0):
                        // scipy returns None → s=1.0, x=x+dx, Fx=func(x+dx)
                        step = 1.0;
                        BLAS.dcopy(n, x, 0, 1, xNew, 0, 1);
                        BLAS.daxpy(n, 1.0, work, dxOff, 1, xNew, 0, 1);
                        fn.accept(xNew, fNew);
                        nfev++;
                    }
                    fnormNew = maxnorm(fNew, n);
                }
            }

            // ── NaN/Inf guard ────────────────────────────────────────────────
            for (double v : fNew) {
                if (Double.isNaN(v) || Double.isInfinite(v))
                    return new Optimization(Double.NaN, x.clone(), fnorm,
                            Optimization.Status.ABNORMAL_TERMINATION, nfev, nfev);
            }

            // ── convergence check ────────────────────────────────────────────
            if (fnormNew <= ftol) {
                BLAS.dcopy(n, xNew, 0, 1, x, 0, 1);
                return new Optimization(Double.NaN, x.clone(), fnormNew,
                        Optimization.Status.COEFFICIENT_TOLERANCE_REACHED, iter, iter);
            }

            // ── actual dx = s·dx, dF = F_new - F ────────────────────────────
            BLAS.dscal(n, step, work, dxOff, 1);
            BLAS.dcopy(n, fNew, 0, 1, work, dFOff, 1);
            BLAS.daxpy(n, -1.0, fx, 0, 1, work, dFOff, 1);

            // ── Good Broyden rank-1 update of H (inverse Jacobian) ───────────
            // scipy BroydenFirst._update:
            //   v  = Gm.rmatvec(dx)  = H^T · dx
            //   c  = dx - Gm.matvec(dF) = dx - H·dF
            //   d  = v / vdot(dF, v)
            //   Gm.append(c, d)  →  H += c·dᵀ
            //
            // Expanded: H₊ = H + (dx - H·dF)·(dxᵀ·H) / (dxᵀ·H·dF)

            // dxH = Hᵀ·dx  (rmatvec)
            BLAS.dgemv(BLAS.Trans.Trans, n, n, 1.0, work, hOff, n, work, dxOff, 1, 0.0, work, dxHOff, 1);
            // Hdf = H·dF  (matvec), then c = dx - Hdf  reusing Hdf array: Hdf = dx - H·dF
            BLAS.dgemv(BLAS.Trans.NoTrans, n, n, -1.0, work, hOff, n, work, dFOff, 1, 0.0, work, HdfOff, 1);
            BLAS.daxpy(n, 1.0, work, dxOff, 1, work, HdfOff, 1);  // Hdf = dx - H·dF  (= c)

            double denom = BLAS.ddot(n, work, dFOff, 1, work, dxHOff, 1);  // dFᵀ·(Hᵀ·dx) = dxᵀ·H·dF

            if (Math.abs(denom) > 1e-300) {
                // H += c·dᵀ / denom  →  dger(alpha=1/denom, x=Hdf(=c), y=dxH)
                BLAS.dger(n, n, 1.0 / denom, work, HdfOff, 1, work, dxHOff, 1, work, hOff, n);
            }
            // Note: singular H (denom≈0) is detected at the start of the next iteration
            // via the finite-check on dx = H·F, matching scipy BroydenFirst.solve().

            BLAS.dcopy(n, xNew, 0, 1, x, 0, 1);
            BLAS.dcopy(n, fNew, 0, 1, fx, 0, 1);
            fnorm = fnormNew;
        }

        return new Optimization(Double.NaN, x.clone(), fnorm,
                Optimization.Status.MAX_ITERATIONS_REACHED, maxiter, maxiter);
    }

    // ── Armijo cubic interpolation loop ──────────────────────────────────────

    /**
     * Cubic interpolation loop of {@code scalar_search_armijo}.
     * Writes the accepted {@code xNew = x + step·dx} and {@code fNew = F(xNew)} into the
     * provided arrays. Returns the accepted step size.
     * {@code fev[0]} is incremented by the number of additional function evaluations.
     */
    private static double cubicSearchArmijo(
            BiConsumer<double[], double[]> fn,
            double[] x, double[] work, int dxOff, int n,
            double phi0, double derphi0, double c1, double amin,
            double alpha0, double phi_a0,
            double alpha1, double phi_a1,
            double[] xNew, double[] fNew,
            int[] fev) {
        int extraFev = 0;

        double alpha2 = alpha1;   // will be overwritten; initialise to satisfy compiler

        while (alpha1 > amin) {
            // cubic interpolant through (alpha0, phi_a0) and (alpha1, phi_a1)
            // with derivative derphi0 at 0
            double factor = alpha0 * alpha0 * alpha1 * alpha1 * (alpha1 - alpha0);
            double a = (alpha0 * alpha0 * (phi_a1 - phi0 - derphi0 * alpha1)
                      - alpha1 * alpha1 * (phi_a0 - phi0 - derphi0 * alpha0))
                     / factor;
            double b = (-alpha0 * alpha0 * alpha0 * (phi_a1 - phi0 - derphi0 * alpha1)
                       + alpha1 * alpha1 * alpha1 * (phi_a0 - phi0 - derphi0 * alpha0))
                     / factor;

            alpha2 = (-b + Math.sqrt(Math.abs(b * b - 3.0 * a * derphi0))) / (3.0 * a);

            BLAS.dcopy(n, x, 0, 1, xNew, 0, 1);
            BLAS.daxpy(n, alpha2, work, dxOff, 1, xNew, 0, 1);
            fn.accept(xNew, fNew);
            extraFev++;
            double phi_a2 = l2normSq(fNew, n);

            if (phi_a2 <= phi0 + c1 * alpha2 * derphi0) {
                fev[0] = extraFev;
                return alpha2;
            }

            // guard: if alpha2 is not shrinking fast enough, halve alpha1
            if ((alpha1 - alpha2) > alpha1 / 2.0 || (1.0 - alpha2 / alpha1) < 0.96) {
                alpha2 = alpha1 / 2.0;
            }

            alpha0 = alpha1;
            alpha1 = alpha2;
            phi_a0 = phi_a1;
            phi_a1 = phi_a2;
        }

        // failed — scipy returns None → caller will use s=1.0
        fev[0] = extraFev;
        return -1.0;  // sentinel: line search failed
    }




    /** Max-norm: scipy default tol_norm = maxnorm. */
    private static double maxnorm(double[] v, int n) {
        return BLAS.damax(n, v, 0, 1);
    }

    /** L2 norm squared: used by Armijo line search phi (matches scipy _safe_norm). */
    private static double l2normSq(double[] v, int n) {
        double s = BLAS.dnrm2(n, v, 0, 1);
        return s * s;
    }
}
