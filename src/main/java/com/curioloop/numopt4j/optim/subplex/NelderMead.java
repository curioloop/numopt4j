/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Nelder-Mead Simplex — Derivative-Free Minimization
 *
 * Solves the problem  min f(x)  where f : ℝⁿ → ℝ, optionally subject to l ≤ x ≤ u.
 * No gradient information is required.
 *
 * Algorithm Overview:
 * ------------------
 * Maintains a simplex of N+1 vertices {v₀, v₁, …, vₙ} sorted by f(vᵢ).
 * Each iteration replaces the worst vertex vₙ via one of four operations
 * applied to the centroid x̄ = (1/N) Σᵢ₌₀ⁿ⁻¹ vᵢ (excluding worst):
 *
 *   1. Reflect:   xᵣ = (1+ρ)x̄ − ρvₙ           move away from worst
 *   2. Expand:    xₑ = (1+ρχ)x̄ − ρχvₙ          stretch further if reflection is best
 *   3. Contract:  xc = (1+ψρ)x̄ − ψρvₙ          outside: f(xᵣ) < f(vₙ)
 *                 xc = (1−ψ)x̄ + ψvₙ            inside:  f(xᵣ) ≥ f(vₙ)
 *   4. Shrink:    vᵢ = v₀ + σ(vᵢ − v₀)  ∀i>0   contract all toward best
 *
 * Decision tree per iteration:
 *   if f(xᵣ) < f(v₀):          try expand; accept xe if better, else accept xᵣ
 *   elif f(xᵣ) < f(vₙ₋₁):     accept xᵣ
 *   elif f(xᵣ) < f(vₙ):        outside contract; shrink if contract fails
 *   else:                       inside contract; shrink if contract fails
 *
 * Coefficients:
 * ------------
 * Uses the standard Nelder-Mead coefficients:
 *
 *   ρ = 1    (reflection)
 *   χ = 2    (expansion)
 *   ψ = 0.5  (contraction)
 *   σ = 0.5  (shrinkage)
 *
 * Convergence Criteria:
 * --------------------
 * Two modes (selected by psiDiam parameter):
 *
 *   Standard (psiDiam ≤ 0):
 *     max|vᵢ − v₀| ≤ xatol  AND  max|f(vᵢ) − f(v₀)| ≤ fatol  for i=1…N
 *
 *   Diameter-based (psiDiam > 0, used by Subplex):
 *     Σⱼ|v₀ⱼ − vₙⱼ| < psiDiam × Σⱼ|v₀ⱼ − vₙⱼ|₀  (L1 diameter shrunk by factor ψ)
 *
 * Bound Handling:
 * --------------
 * Trial points are clipped to [l, u] after each operation (Richardson-Kuester 1973).
 * If clipping causes the trial point to coincide with the centroid or the worst
 * vertex (degeneracy), iteration terminates — the simplex cannot make progress.
 *
 * Initial Simplex Construction:
 * ----------------------------
 * Vertex 0 = x₀. Vertex k+1 = x₀ with x₀[k] perturbed by step[k].
 * Default step: 5% of |x₀[k]| (or 0.00025 if x₀[k]=0).
 * Bound-aware: if perturbation exceeds a bound, tries opposite direction,
 * then the bound itself, then the midpoint toward the further bound.
 *
 * References:
 * ----------
 *   Nelder, J.A. and Mead, R. (1965). "A simplex method for function minimization."
 *   The Computer Journal, 7(4), 308–313.
 *
 *   Richardson, J.A. and Kuester, J.L. (1973). "The complex method for constrained
 *   optimization." Communications of the ACM, 16(8), 487–489.
 */
package com.curioloop.numopt4j.optim.subplex;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.OptimizationStatus;

import java.util.function.ToDoubleFunction;
final class NelderMead {

    private static final double NONZDELT = 0.05;
    private static final double ZDELT = 0.00025;

    private NelderMead() {}

    /**
     * Minimizes a scalar function using the Nelder-Mead simplex method.
     *
     * @param n              problem dimension
     * @param x              initial point (overwritten with solution on return)
     * @param func           objective function
     * @param bounds         variable bounds (may be null)
     * @param maxIter        maximum iterations
     * @param maxEval        maximum function evaluations
     * @param xatol          parameter convergence tolerance (used when psiDiam &le; 0)
     * @param fatol          function value convergence tolerance (used when psiDiam &le; 0)
     * @param psiDiam        if &gt; 0, use diameter-based convergence: stop when
     *                       simplex diameter &lt; psiDiam * initial_diameter (for Subplex)
     * @param initialStep    per-dimension initial step sizes (may be null for defaults)
     * @param initialF       if finite, used as f(vertex 0) without evaluation;
     *                       pass {@code Double.NaN} to evaluate all vertices
     * @param evalCounter    shared evaluation counter (may be null; if non-null, [0] is
     *                       incremented on each evaluation and checked against maxEval)
     * @param ws             pre-allocated workspace
     * @return termination status (cost available in {@code ws.fsim[0]},
     *         solution written to {@code x} in-place)
     */
    static OptimizationStatus optimize(
            int n,
            double[] x,
            ToDoubleFunction<double[]> func,
            Bound[] bounds,
            int maxIter,
            int maxEval,
            double xatol,
            double fatol,
            double psiDiam,
            double[] initialStep,
            double initialF,
            int[] evalCounter,
            SubplexWorkspace ws) {

        // Standard Nelder-Mead coefficients
        final double rho = 1, chi = 2, psi = 0.5, sigma = 0.5;

        final double[] sim = ws.sim;
        final double[] fsim = ws.fsim;
        final double[] xbar = ws.xbar;
        final double[] xr = ws.xr;
        final double[] xc = ws.xc;

        final boolean hasBounds = bounds != null;
        if (hasBounds) {
            clip(x, bounds, n);
        }

        // ── Build initial simplex ──────────────────────────────────────────────
        buildSimplex(sim, x, n, initialStep, bounds);

        // Shared or local eval counter
        final boolean sharedCounter = evalCounter != null;
        int nfev = sharedCounter ? evalCounter[0] : 0;

        // Evaluate vertices: reuse initialF for vertex 0 if known
        if (Double.isFinite(initialF)) {
            fsim[0] = initialF;
        } else {
            fsim[0] = safeEval(func, sim, 0, n, xr);
            nfev++;
        }
        for (int k = 1; k <= n; k++) {
            fsim[k] = safeEval(func, sim, k * n, n, xr);
            nfev++;
            if (nfev >= maxEval) break;
        }
        sortSimplex(sim, fsim, n, ws.xc);

        // Initial simplex diameter (for psi-convergence)
        double initDiam = 0;
        if (psiDiam > 0) {
            for (int j = 0; j < n; j++) {
                initDiam += Math.abs(sim[j] - sim[n * n + j]);
            }
        }

        // ── Main loop ──────────────────────────────────────────────────────────
        int iterations = 1;
        OptimizationStatus status = OptimizationStatus.MAX_ITERATIONS_REACHED;

        while (nfev < maxEval && iterations < maxIter) {

            // Convergence: psi-diameter mode or standard xatol+fatol
            if (psiDiam > 0) {
                double diam = 0;
                for (int j = 0; j < n; j++) {
                    diam += Math.abs(sim[j] - sim[n * n + j]);
                }
                if (diam < psiDiam * initDiam) {
                    status = OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED;
                    break;
                }
            } else if (converged(sim, fsim, n, xatol, fatol)) {
                status = OptimizationStatus.FUNCTION_TOLERANCE_REACHED;
                break;
            }

            computeCentroid(sim, xbar, n);
            final int worst = n * n;

            // Reflect
            reflect(xbar, sim, worst, 1 + rho, -rho, xr, n);
            if (hasBounds && !clipAndCheck(xr, xbar, sim, worst, bounds, n)) {
                status = OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED;
                break;
            }
            double fxr = safeEval(func, xr, n);
            nfev++;

            boolean doshrink = false;

            if (fxr < fsim[0]) {
                // Expand
                reflect(xbar, sim, worst, 1 + rho * chi, -rho * chi, xc, n);
                if (hasBounds) clip(xc, bounds, n);
                double fxc = safeEval(func, xc, n);
                nfev++;

                if (fxc < fxr) {
                    System.arraycopy(xc, 0, sim, worst, n);
                    fsim[n] = fxc;
                } else {
                    System.arraycopy(xr, 0, sim, worst, n);
                    fsim[n] = fxr;
                }
            } else {
                if (fxr < fsim[n - 1]) {
                    System.arraycopy(xr, 0, sim, worst, n);
                    fsim[n] = fxr;
                } else {
                    // Contract (outside if fxr < fsim[n], inside otherwise)
                    if (fxr < fsim[n]) {
                        reflect(xbar, sim, worst, 1 + psi * rho, -psi * rho, xc, n);
                    } else {
                        reflect(xbar, sim, worst, 1 - psi, psi, xc, n);
                    }
                    if (hasBounds && !clipAndCheck(xc, xbar, sim, worst, bounds, n)) {
                        status = OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED;
                        break;
                    }
                    double fxc = safeEval(func, xc, n);
                    nfev++;

                    if (fxr < fsim[n] ? fxc <= fxr : fxc < fsim[n]) {
                        System.arraycopy(xc, 0, sim, worst, n);
                        fsim[n] = fxc;
                    } else {
                        doshrink = true;
                    }

                    if (doshrink) {
                        for (int i = 1; i <= n; i++) {
                            int iOff = i * n;
                            for (int j = 0; j < n; j++) {
                                sim[iOff + j] = sim[j] + sigma * (sim[iOff + j] - sim[j]);
                            }
                            if (hasBounds) clipRow(sim, iOff, bounds, n);
                            fsim[i] = safeEval(func, sim, iOff, n, xr);
                            nfev++;
                            if (nfev >= maxEval) break;
                        }
                    }
                }
            }

            iterations++;

            if (doshrink) {
                sortSimplex(sim, fsim, n, ws.xc);
            } else {
                insertLast(sim, fsim, n, ws.xc);
            }
        }

        if (status != OptimizationStatus.FUNCTION_TOLERANCE_REACHED
                && status != OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED) {
            status = nfev >= maxEval
                    ? OptimizationStatus.MAX_EVALUATIONS_REACHED
                    : OptimizationStatus.MAX_ITERATIONS_REACHED;
        }

        System.arraycopy(sim, 0, x, 0, n);
        if (sharedCounter) evalCounter[0] = nfev;
        return status;
    }

    // ── Simplex construction ──────────────────────────────────────────────────

    /**
     * Builds a bound-aware simplex around point x.
     * When a perturbation violates bounds, tries opposite direction, then bound itself,
     * then midpoint toward the further bound (inspired by NLopt's Richardson-Kuester approach).
     */
    private static void buildSimplex(double[] sim, double[] x, int n,
                                     double[] initialStep, Bound[] bounds) {
        System.arraycopy(x, 0, sim, 0, n);
        for (int k = 0; k < n; k++) {
            System.arraycopy(x, 0, sim, (k + 1) * n, n);
            int idx = (k + 1) * n + k;
            double step = initialStep != null
                    ? initialStep[k]
                    : ((x[k] != 0) ? NONZDELT * x[k] : ZDELT);
            double pt = x[k] + step;

            if (bounds != null) {
                Bound b = bounds[k];
                if (b != null) {
                    double lb = b.getLower();
                    double ub = b.getUpper();
                    if (pt > ub) {
                        // Try bound itself if far enough from x; else opposite direction
                        pt = (ub - x[k] > Math.abs(step) * 0.1) ? ub : x[k] - Math.abs(step);
                    }
                    if (pt < lb) {
                        pt = (x[k] - lb > Math.abs(step) * 0.1) ? lb : x[k] + Math.abs(step);
                        if (pt > ub) {
                            // Both directions blocked; go toward the further bound
                            pt = 0.5 * (((ub - x[k] > x[k] - lb) ? ub : lb) + x[k]);
                        }
                    }
                }
            }
            sim[idx] = pt;
        }
    }

    // ── Reflection + degeneracy detection ─────────────────────────────────────

    /**
     * Computes xnew[j] = a * xbar[j] + b * sim[worstOff + j].
     */
    private static void reflect(double[] xbar, double[] sim, int worstOff,
                                double a, double b, double[] xnew, int n) {
        for (int j = 0; j < n; j++) {
            xnew[j] = a * xbar[j] + b * sim[worstOff + j];
        }
    }

    /**
     * Clips point to bounds and checks for degeneracy: returns false if the
     * clipped point coincides with the centroid or the original worst vertex.
     * Based on NLopt's reflectpt approach (Richardson-Kuester 1973).
     */
    private static boolean clipAndCheck(double[] pt, double[] centroid,
                                        double[] sim, int worstOff,
                                        Bound[] bounds, int n) {
        boolean equalCentroid = true, equalWorst = true;
        for (int j = 0; j < n; j++) {
            Bound b = bounds[j];
            if (b != null) {
                if (b.hasLower() && pt[j] < b.getLower()) pt[j] = b.getLower();
                if (b.hasUpper() && pt[j] > b.getUpper()) pt[j] = b.getUpper();
            }
            if (equalCentroid) equalCentroid = close(pt[j], centroid[j]);
            if (equalWorst) equalWorst = close(pt[j], sim[worstOff + j]);
        }
        return !(equalCentroid || equalWorst);
    }

    /**
     * Approximate floating-point equality (relative tolerance 1e-13).
     */
    private static boolean close(double a, double b) {
        return Math.abs(a - b) <= 1e-13 * (Math.abs(a) + Math.abs(b));
    }

    // ── Safe evaluation (NaN/Inf protection) ──────────────────────────────────

    /**
     * Evaluates function, returning +Inf if result is NaN or Inf.
     */
    private static double safeEval(ToDoubleFunction<double[]> func, double[] pt, int n) {
        double f = func.applyAsDouble(pt);
        return Double.isFinite(f) ? f : Double.MAX_VALUE;
    }

    /**
     * Evaluates function at sim[off..off+n), copying to tmp buffer first.
     */
    private static double safeEval(ToDoubleFunction<double[]> func, double[] sim,
                                   int off, int n, double[] tmp) {
        System.arraycopy(sim, off, tmp, 0, n);
        return safeEval(func, tmp, n);
    }

    // ── Sort helpers ──────────────────────────────────────────────────────────

    private static void insertLast(double[] sim, double[] fsim, int n, double[] tmp) {
        double fNew = fsim[n];
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (fsim[mid] <= fNew) lo = mid + 1;
            else hi = mid;
        }
        if (lo < n) {
            System.arraycopy(sim, n * n, tmp, 0, n);
            System.arraycopy(sim, lo * n, sim, (lo + 1) * n, (n - lo) * n);
            System.arraycopy(fsim, lo, fsim, lo + 1, n - lo);
            System.arraycopy(tmp, 0, sim, lo * n, n);
            fsim[lo] = fNew;
        }
    }

    private static void sortSimplex(double[] sim, double[] fsim, int n, double[] tmp) {
        int np1 = n + 1;
        for (int i = 1; i < np1; i++) {
            double keyF = fsim[i];
            System.arraycopy(sim, i * n, tmp, 0, n);
            int j = i - 1;
            while (j >= 0 && fsim[j] > keyF) {
                fsim[j + 1] = fsim[j];
                System.arraycopy(sim, j * n, sim, (j + 1) * n, n);
                j--;
            }
            fsim[j + 1] = keyF;
            System.arraycopy(tmp, 0, sim, (j + 1) * n, n);
        }
    }

    // ── Geometry helpers ──────────────────────────────────────────────────────

    private static void clip(double[] v, Bound[] bounds, int n) {
        for (int j = 0; j < n; j++) {
            Bound b = bounds[j];
            if (b == null) continue;
            if (b.hasLower() && v[j] < b.getLower()) v[j] = b.getLower();
            if (b.hasUpper() && v[j] > b.getUpper()) v[j] = b.getUpper();
        }
    }

    private static void clipRow(double[] sim, int off, Bound[] bounds, int n) {
        for (int j = 0; j < n; j++) {
            Bound b = bounds[j];
            if (b == null) continue;
            if (b.hasLower() && sim[off + j] < b.getLower()) sim[off + j] = b.getLower();
            if (b.hasUpper() && sim[off + j] > b.getUpper()) sim[off + j] = b.getUpper();
        }
    }

    private static void computeCentroid(double[] sim, double[] xbar, int n) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += sim[i * n + j];
            }
            xbar[j] = sum / n;
        }
    }

    private static boolean converged(double[] sim, double[] fsim, int n, double xatol, double fatol) {
        double maxXDiff = 0;
        for (int i = 1; i <= n; i++) {
            int off = i * n;
            for (int j = 0; j < n; j++) {
                double d = Math.abs(sim[off + j] - sim[j]);
                if (d > maxXDiff) maxXDiff = d;
            }
        }
        if (maxXDiff > xatol) return false;

        double maxFDiff = 0;
        for (int i = 1; i <= n; i++) {
            double d = Math.abs(fsim[0] - fsim[i]);
            if (d > maxFDiff) maxFDiff = d;
        }
        return maxFDiff <= fatol;
    }
}
