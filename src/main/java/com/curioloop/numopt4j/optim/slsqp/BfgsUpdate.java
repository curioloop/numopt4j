/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * SLSQP BFGS update — LDLᵀ rank-1 modification.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Algorithm Overview
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The Hessian approximation B is stored in LDLᵀ factorization form:
 *
 *   B = LDLᵀ
 *
 * where:
 *   L — lower triangular matrix with unit diagonal elements
 *   D — diagonal matrix with positive diagonal elements
 *
 * The modified BFGS update computes Bᵏ⁺¹ = Bᵏ + qqᵀ/qᵀs - Bᵏssᵀ Bᵏ/sᵀBᵏs
 * as two rank-1 modifications of the LDLᵀ factors:
 *
 *   compositeT(n, L, q, +1/h₁, null)   where h₁ = sᵀq  (positive update)
 *   compositeT(n, L, v, -1/h₂, u)      where h₂ = sᵀBᵏs (negative update)
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Composite Transformation (compositeT)
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Given A = LDLᵀ (n×n positive definite), computes the LDLᵀ factorization
 * of the rank-1 modified matrix A' = A + σzzᵀ = ∑ l'ᵢ d'ᵢ l'ᵢᵀ.
 *
 * For σ > 0, the update formulas are:
 *
 *   tᵢ₊₁ = tᵢ + vᵢ²/dᵢ           (running sum, t₁ = 1/σ)
 *   αᵢ   = tᵢ₊₁ / tᵢ             (scaling factor)
 *   d'ᵢ  = αᵢ · dᵢ               (updated diagonal)
 *   βᵢ   = (vᵢ / dᵢ) / tᵢ        (update coefficient)
 *   l'ᵢ  = lᵢ + βᵢ · z⁽ⁱ⁺¹⁾ᵢ    (when αᵢ ≤ 4)
 *   l'ᵢ  = (tᵢ/tᵢ₊₁)lᵢ + βᵢ · z⁽ⁱ⁾ᵢ  (when αᵢ > 4)
 *   z⁽ⁱ⁺¹⁾ = z⁽ⁱ⁾ - vᵢ · lᵢ     (updated z vector)
 *
 * For σ < 0, an auxiliary vector w is used:
 *   w = z - L⁻¹z
 *   tₙ = ε/σ if tₙ ≥ 0
 *
 * Reference: Dieter Kraft, 'A Software Package for Sequential Quadratic
 * Programming', DFVLR-FB 88-28, 1988. Chapter 2.32.
 */
package com.curioloop.numopt4j.optim.slsqp;

import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.EPS;

/**
 * BFGS update algorithms for the SLSQP optimization algorithm.
 *
 * <p>This class provides methods for updating the Hessian approximation stored
 * in LDLᵀ factorization form using the modified BFGS quasi-Newton update.</p>
 *
 * <h2>LDLᵀ Factorization</h2>
 * <p>The Hessian approximation B is stored as B = LDLᵀ where:</p>
 * <ul>
 *   <li>L — lower triangular matrix with unit diagonal elements</li>
 *   <li>D — diagonal matrix with positive diagonal elements</li>
 * </ul>
 * <p>L and D are stored together in a packed lower triangular array,
 * with D on the diagonal positions.</p>
 *
 * <h2>Modified BFGS Update</h2>
 * <p>The update Bᵏ⁺¹ = Bᵏ + qqᵀ/qᵀs - Bᵏssᵀ Bᵏ/sᵀBᵏs is computed as
 * two rank-1 modifications of the LDLᵀ factors via {@link #compositeT}:</p>
 * <pre>
 *   compositeT(n, L, q, +1/h₁, null)   where h₁ = sᵀq  (positive update: +qqᵀ/qᵀs)
 *   compositeT(n, L, v, -1/h₂, u)      where h₂ = sᵀBᵏs (negative update: -Bᵏssᵀ Bᵏ/sᵀBᵏs)
 * </pre>
 *
 * @see SLSQPConstants#EPS
 */
public final class BfgsUpdate {

    private BfgsUpdate() {
        // Prevent instantiation
    }

    // ========================================================================
    // Composite Transformation Update
    // ========================================================================

    /**
     * Compute LDLᵀ factorization for a rank-1 modified matrix A' = A + σzzᵀ.
     *
     * <p>Given A = LDLᵀ (n×n positive definite), computes the LDLᵀ factorization
     * of the rank-1 modified matrix:</p>
     * <pre>
     *   A' = A + σzzᵀ = ∑ l'ᵢ d'ᵢ l'ᵢᵀ
     * </pre>
     *
     * <h3>Update Formulas (σ &gt; 0)</h3>
     * <pre>
     *   tᵢ₊₁ = tᵢ + vᵢ²/dᵢ           (running sum, t₁ = 1/σ)
     *   αᵢ   = tᵢ₊₁ / tᵢ             (scaling factor)
     *   d'ᵢ  = αᵢ · dᵢ               (updated diagonal)
     *   βᵢ   = (vᵢ / dᵢ) / tᵢ        (update coefficient)
     *   l'ᵢ  = lᵢ + βᵢ · z⁽ⁱ⁺¹⁾ᵢ    (when αᵢ ≤ 4)
     *   l'ᵢ  = (tᵢ/tᵢ₊₁)lᵢ + βᵢ · z⁽ⁱ⁾ᵢ  (when αᵢ &gt; 4)
     *   z⁽ⁱ⁺¹⁾ = z⁽ⁱ⁾ - vᵢ · lᵢ     (updated z vector)
     * </pre>
     *
     * <h3>Negative Update (σ &lt; 0)</h3>
     * <p>An auxiliary vector w is used to handle the negative update:</p>
     * <pre>
     *   Solve Lv = z and compute tᵢ₊₁ = tᵢ + vᵢ²/dᵢ
     *   If tₙ ≥ 0: set tₙ = ε/σ  (safeguard to maintain positive definiteness)
     *   Recompute tᵢ₋₁ = tᵢ - vᵢ²/dᵢ  (backward pass)
     * </pre>
     *
     * <p>Reference: Go function compositeT() in tool.go</p>
     * <p>Reference: Dieter Kraft, 'A Software Package for Sequential Quadratic
     * Programming', DFVLR-FB 88-28, 1988. Chapter 2.32.</p>
     *
     * @param n      order of matrix
     * @param l      LDLᵀ factors in packed form (L and D stored together, modified in place)
     * @param lOff   offset into l array
     * @param z      vector z for rank-1 update σzzᵀ (modified during computation)
     * @param zOff   offset into z array
     * @param sigma  scalar σ for rank-1 update
     * @param w      working vector (required if σ ≤ 0, can be null if σ &gt; 0)
     * @param wOff   offset into w array (ignored if w is null)
     *
     * @see SLSQPConstants#EPS
     */
    public static void compositeT(int n, double[] l, int lOff,
                                   double[] z, int zOff,
                                   double sigma,
                                   double[] w, int wOff) {
        int i, j;
        int ij;
        double t, v, uVal, delta, tp, alpha, beta, gamma;

        /* if σ = 0 then terminate */
        if (sigma == 0.0) {
            return;
        }

        t = 1.0 / sigma;
        ij = 0;

        if (n <= 0) {
            return;
        }

        /* if σ < 0 construct w = z - L^{-1}z */
        if (sigma <= 0.0) {
            if (w == null) {
                return;
            }

            /* copy z to w */
            for (i = 0; i < n; i++) {
                w[wOff + i] = z[zOff + i];
            }

            /* solve Lv = z and update t_{i+1} = t_i + v_i²/d_i */
            for (i = 0; i < n; i++) {
                v = w[wOff + i];
                t += v * v / l[lOff + ij];
                for (j = i + 1; j < n; j++) {
                    ij++;
                    w[wOff + j] -= v * l[lOff + ij];
                }
                ij++;
            }

            /* if t_n ≥ 0 then set t_n = ε/σ */
            if (t >= 0.0) {
                t = EPS / sigma;
            }

            /* recompute t_{i-1} = t_i - v_i²/d_i */
            for (j = n - 1; j >= 0; j--) {
                uVal = w[wOff + j];
                w[wOff + j] = t;
                ij -= n - j;
                t -= uVal * uVal / l[lOff + ij];
            }
        }

        ij = 0;
        for (i = 0; i < n; i++) {
            v = z[zOff + i];
            delta = v / l[lOff + ij];

            if (sigma < 0.0) {
                tp = w[wOff + i];              /* t_{i+1} = w_{i+1} */
            } else {
                tp = t + delta * v;           /* t_{i+1} = t_i + v_i²/d_i */
            }

            alpha = tp / t;                   /* α_i = t_{i+1} / t_i */
            l[lOff + ij] *= alpha;            /* d'_i = α_i * d_i */

            if (i == n - 1) {
                break;
            }

            beta = delta / tp;                /* β_i = (v_i / d_i) / t_i */

            if (alpha > 4.0) {
                gamma = t / tp;
                for (j = i + 1; j < n; j++) {
                    ij++;
                    uVal = l[lOff + ij];                              /* l_i */
                    l[lOff + ij] = gamma * uVal + beta * z[zOff + j]; /* l'_i = (t_i / t_{i+1})l_i + β_i z^{(i)}_i */
                    z[zOff + j] -= v * uVal;                          /* z^{(i+1)} = z^{(i)} - v_i l_i */
                }
            } else {
                for (j = i + 1; j < n; j++) {
                    ij++;
                    z[zOff + j] -= v * l[lOff + ij];                  /* z^{(i+1)} = z^{(i)} - v_i l_i */
                    l[lOff + ij] += beta * z[zOff + j];               /* l'_i = l_i + β_i z^{(i+1)}_i */
                }
            }
            ij++;
            t = tp;
        }
    }

    // ========================================================================
    // BFGS Reset
    // ========================================================================

    /**
     * Reset the BFGS Hessian approximation to the identity matrix.
     * <p>
     * This method initializes the LDL^T factorization such that B = I (identity matrix).
     * In LDL^T form, this means:
     * </p>
     * <ul>
     *   <li>L = I (identity matrix, all off-diagonal elements are zero)</li>
     *   <li>D = I (identity matrix, all diagonal elements are one)</li>
     * </ul>
     * <p>
     * The packed storage format stores the diagonal elements of D at positions
     * 0, n, n+(n-1), n+(n-1)+(n-2), ... in the array.
     * </p>
     *
     * <h3>Packed Storage Layout</h3>
     * <p>
     * For n=3, the packed array stores:
     * </p>
     * <pre>
     *   [d₁, l₂₁, l₃₁, d₂, l₃₂, d₃]
     *   indices: 0, 1, 2, 3, 4, 5
     *   diagonal positions: 0, 3, 5 (j = 0, 3, 5)
     * </pre>
     *
     * @param n     order of matrix
     * @param l     lower triangular matrix in packed form (modified to identity)
     * @param lOff  offset into l array
     *
     */
    public static void reset(int n, double[] l, int lOff) {
        if (n <= 0) {
            return;
        }

        /* Compute size of packed storage: n*(n+1)/2 */
        int n2 = n * (n + 1) / 2;

        /* Zero out all elements */
        for (int i = 0; i < n2; i++) {
            l[lOff + i] = 0.0;
        }

        /* Set diagonal elements to 1.0 */
        /* Diagonal positions: 0, n, n+(n-1), n+(n-1)+(n-2), ... */
        /* Or equivalently: j = 0, then j += (n-i) for i = 0, 1, 2, ... */
        for (int i = 0, j = 0; i < n; i++) {
            l[lOff + j] = 1.0;
            j += n - i;
        }
    }
}
