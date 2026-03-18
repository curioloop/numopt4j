/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Subplex — Subspace Simplex Method for Derivative-Free Minimization
 *
 * Solves  min f(x),  f : ℝⁿ → ℝ,  optionally subject to l ≤ x ≤ u.
 * Decomposes the N-dimensional problem into a sequence of low-dimensional
 * Nelder-Mead subproblems, making derivative-free optimization practical
 * for N > 10 where plain NM degrades.
 *
 * Algorithm Overview:
 * ------------------
 * repeat until converged:
 *   1. Compute progress vector:  dx = x − x_prev
 *   2. Sort dimension indices by decreasing |dx[i]|
 *   3. Partition sorted indices into subspaces S₁, S₂, …
 *      (each of dimension nsmin…NSMAX, chosen by Rowan's goodness criterion)
 *   4. For each subspace Sₖ of dimension nₖ:
 *      a. Extract subspace coordinates, step sizes, and bounds
 *      b. Run Nelder-Mead on the nₖ-dimensional subproblem with
 *         ψ-convergence: stop when simplex diameter < ψ × initial_diameter
 *      c. Copy solution back to full x
 *   5. Check outer convergence (xatol + step size check)
 *   6. Update step sizes for next iteration
 *
 * Subspace Partitioning (Rowan's Criterion):
 * -----------------------------------------
 * Dimensions are sorted by |dx| (most progress first). The partition size
 * ns ∈ [nsmin, NSMAX] is chosen to maximize:
 *
 *   goodness(k) = ‖dx[i..k]‖₁ / (k−i+1) − ‖dx[k+1..n]‖₁ / (n−k−1)
 *
 * This detects sudden drops in average progress, grouping dimensions
 * with similar activity into the same subspace.
 *
 * Step Size Update:
 * ----------------
 * After each outer iteration:
 *
 *   if nsubs = 1:  scale = ψ                            (single subspace)
 *   else:          scale = clamp(‖dx‖₁/‖xstep‖₁, ω, 1/ω)  (ratio of progress to step)
 *
 *   xstep[j] = sign(dx[j]) × |xstep[j]| × scale    (align with progress direction)
 *              −|xstep[j]| × scale                   (negate if dx[j] = 0)
 *
 * Strategy Constants:
 * ------------------
 *   ψ     = 0.25   simplex diameter reduction factor for inner NM
 *   ω     = 0.1    minimum step size scale factor
 *   nsmin = 2      minimum subspace dimension
 *   NSMAX = 5      maximum subspace dimension (NM works well for n ≤ 5)
 *
 * Outer Convergence:
 * -----------------
 *   |x[j] − x_prev[j]| ≤ xatol × (1 + |x[j]|)  ∀j   (x converged)
 *   AND  |xstep[j]| × ψ ≤ xatol × (1 + |x[j]|)  ∀j   (steps also small)
 *
 * References:
 * ----------
 *   Rowan, T. (1990). "Functional Stability Analysis of Numerical Algorithms."
 *   Ph.D. thesis, Department of Computer Sciences, University of Texas at Austin.
 *
 *   Johnson, S.G. NLopt sbplx.c implementation (MIT license).
 *   https://github.com/stevengj/nlopt
 */
package com.curioloop.numopt4j.optim.subplex;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.OptimizationResult;
import com.curioloop.numopt4j.optim.OptimizationStatus;

import java.util.function.ToDoubleFunction;
final class SubplexCore {

    private static final double PSI = 0.25;    // simplex diameter reduction factor
    private static final double OMEGA = 0.1;   // minimum step size scale factor
    private static final int NSMIN = 2;         // minimum subspace dimension
    private static final int NSMAX = 5;         // maximum subspace dimension

    private SubplexCore() {}

    /**
     * Minimizes a scalar function using the Subplex method.
     *
     * @param n       problem dimension
     * @param x       initial point (overwritten with solution on return)
     * @param func    objective function
     * @param bounds  variable bounds (may be null)
     * @param xstep   initial step sizes per dimension (must not be null)
     * @param maxEval maximum function evaluations
     * @param xatol   parameter convergence tolerance
     * @param fatol   function value convergence tolerance
     * @param nmWs    pre-allocated NM workspace (dimension &ge; NSMAX)
     * @return optimization result
     */
    static OptimizationResult optimize(
            int n,
            double[] x,
            ToDoubleFunction<double[]> func,
            Bound[] bounds,
            double[] xstep,
            int maxEval,
            double xatol,
            double fatol,
            SubplexWorkspace nmWs) {

        // Use workspace arrays (no per-call allocation)
        nmWs.ensureFullCapacity(n);
        nmWs.resetFull();
        double[] xprev = nmWs.xprev;
        double[] dx = nmWs.dx;
        int[] perm = nmWs.perm;
        double[] xs = nmWs.xs;
        double[] xsstep = nmWs.xsstep;
        Bound[] subBounds = bounds != null ? nmWs.subBounds : null;
        int[] evalCounter = {0};

        // Initial evaluation
        double minf = func.applyAsDouble(x);
        if (!Double.isFinite(minf)) minf = Double.MAX_VALUE;
        evalCounter[0] = 1;

        OptimizationStatus status = OptimizationStatus.MAX_EVALUATIONS_REACHED;
        int outerIter = 0;

        while (evalCounter[0] < maxEval) {
            outerIter++;
            System.arraycopy(x, 0, xprev, 0, n);

            // Sort dimension indices by decreasing |dx[i]|
            for (int i = 0; i < n; i++) perm[i] = i;
            sortPermByDx(perm, dx, n);

            // Compute L1 norm of dx
            double normdx = 0;
            for (int i = 0; i < n; i++) normdx += Math.abs(dx[i]);

            // Partition into subspaces and run NM on each
            double normi = 0;     // accumulated |dx| norm for goodness criterion
            double fdiffMax = 0;  // max (fh - fl) across all subproblems
            int nsubs = 0;
            int i = 0;

            while (i < n && evalCounter[0] < maxEval) {
                int remaining = n - i;
                int ns;

                if (remaining <= NSMAX) {
                    ns = remaining;
                } else if (remaining < NSMIN + NSMIN) {
                    ns = remaining;
                } else {
                    ns = chooseSubspaceSize(perm, dx, i, n, normi, normdx, NSMAX);
                }

                // Accumulate normi for this subspace (used by next goodness call)
                for (int k = i; k < i + ns; k++) {
                    normi += Math.abs(dx[perm[k]]);
                }

                // Extract subspace coordinates, step sizes, bounds
                for (int k = 0; k < ns; k++) {
                    int pi = perm[i + k];
                    xs[k] = x[pi];
                    xsstep[k] = xstep[pi];
                    if (subBounds != null) subBounds[k] = bounds[pi];
                }

                // Subspace objective: maps subspace coords back to full x
                final int subStart = i;
                final int subDim = ns;
                final int[] permRef = perm;
                ToDoubleFunction<double[]> subFunc = xsub -> {
                    for (int k = 0; k < subDim; k++) {
                        x[permRef[subStart + k]] = xsub[k];
                    }
                    return func.applyAsDouble(x);
                };

                // Run NM on this subspace with psi-convergence
                nmWs.ensureNmCapacity(ns);
                nmWs.resetNm();
                OptimizationStatus subStatus = NelderMead.optimize(
                        ns, xs, subFunc,
                        subBounds,
                        Integer.MAX_VALUE, maxEval,
                        xatol, fatol, PSI,
                        xsstep, minf,
                        evalCounter, nmWs);

                // Read fdiff (fh - fl) and cost from NM workspace (simplex still sorted)
                double fdiff = nmWs.fsim[ns] - nmWs.fsim[0];
                if (fdiff > fdiffMax) fdiffMax = fdiff;
                double subCost = nmWs.fsim[0];

                // Copy subspace solution back to full x
                for (int k = 0; k < ns; k++) {
                    x[perm[i + k]] = xs[k];
                }

                if (subCost < minf) minf = subCost;
                nsubs++;

                if (subStatus == OptimizationStatus.ABNORMAL_TERMINATION) {
                    status = OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED;
                    break;
                }
                if (subStatus != OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED
                        && subStatus != OptimizationStatus.FUNCTION_TOLERANCE_REACHED) {
                    status = subStatus;
                    break;
                }

                i += ns;
            }

            if (status != OptimizationStatus.MAX_EVALUATIONS_REACHED) break;

            // ── Termination tests ──────────────────────────────────────────────

            // Function tolerance: max simplex spread across all subproblems
            if (fdiffMax <= fatol) {
                status = OptimizationStatus.FUNCTION_TOLERANCE_REACHED;
                break;
            }

            boolean xConverged = true;
            for (int j = 0; j < n; j++) {
                if (Math.abs(x[j] - xprev[j]) > xatol + xatol * Math.abs(x[j])) {
                    xConverged = false;
                    break;
                }
            }
            if (xConverged) {
                // Also check that step sizes are small enough
                boolean stepsSmall = true;
                for (int j = 0; j < n; j++) {
                    if (Math.abs(xstep[j]) * PSI > xatol + xatol * Math.abs(x[j])) {
                        stepsSmall = false;
                        break;
                    }
                }
                if (stepsSmall) {
                    status = OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED;
                    break;
                }
            }

            // ── Update progress and step sizes ─────────────────────────────────

            for (int j = 0; j < n; j++) dx[j] = x[j] - xprev[j];

            // Scale step sizes
            double scale;
            if (nsubs == 1) {
                scale = PSI;
            } else {
                double stepnorm = 0, dxnorm = 0;
                for (int j = 0; j < n; j++) {
                    stepnorm += Math.abs(xstep[j]);
                    dxnorm += Math.abs(dx[j]);
                }
                scale = (stepnorm > 0) ? dxnorm / stepnorm : PSI;
                if (scale < OMEGA) scale = OMEGA;
                if (scale > 1.0 / OMEGA) scale = 1.0 / OMEGA;
            }

            for (int j = 0; j < n; j++) {
                xstep[j] = (dx[j] == 0)
                        ? -(xstep[j] * scale)  // flip direction: no progress → try opposite
                        : Math.copySign(xstep[j] * scale, dx[j]);
            }
        }

        int totalEvals = evalCounter[0];
        if (status == OptimizationStatus.MAX_EVALUATIONS_REACHED && evalCounter[0] < maxEval) {
            // Exited loop without hitting maxEval → must be convergence from inner check
            status = OptimizationStatus.FUNCTION_TOLERANCE_REACHED;
        }

        return new OptimizationResult(Double.NaN, null, minf, status, outerIter, totalEvals);
    }

    /**
     * Chooses the subspace size starting at index {@code i} in the permutation,
     * using Rowan's goodness criterion with global (cumulative) norm.
     *
     * @param normi  accumulated |dx| from previous subspaces (global offset)
     * @param normdx total L1 norm of dx across all dimensions
     */
    private static int chooseSubspaceSize(int[] perm, double[] dx, int i, int n,
                                          double normi, double normdx, int NSMAX) {
        int nk = Math.min(i + NSMAX, n);
        double norm = normi;
        // Accumulate norm for first nsmin-1 elements of this subspace
        for (int k = i; k < i + NSMIN - 1; k++) {
            norm += Math.abs(dx[perm[k]]);
        }
        int bestNs = NSMIN;
        double bestGoodness = Double.NEGATIVE_INFINITY;

        for (int k = i + NSMIN - 1; k < nk; k++) {
            norm += Math.abs(dx[perm[k]]);
            int remaining = n - (k + 1);
            if (remaining < NSMIN) continue;
            // Rowan's goodness: cumulative avg |dx| up to k vs avg of remaining
            double goodness = (k + 1 < n)
                    ? norm / (k + 1) - (normdx - norm) / remaining
                    : normdx / n;
            if (goodness > bestGoodness) {
                bestGoodness = goodness;
                bestNs = (k + 1) - i;
            }
        }
        return bestNs;
    }

    /**
     * Sorts permutation indices by decreasing |dx[perm[i]]|.
     * Uses insertion sort — N is typically moderate.
     */
    private static void sortPermByDx(int[] perm, double[] dx, int n) {
        for (int i = 1; i < n; i++) {
            int key = perm[i];
            double keyVal = Math.abs(dx[key]);
            int j = i - 1;
            while (j >= 0 && Math.abs(dx[perm[j]]) < keyVal) {
                perm[j + 1] = perm[j];
                j--;
            }
            perm[j + 1] = key;
        }
    }
}
