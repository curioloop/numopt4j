/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * L-BFGS-B Subspace Minimization - Pure Java implementation.
 *
 * This module implements the subspace minimization step of the L-BFGS-B algorithm.
 * The subspace minimization computes an approximate solution of the subspace problem:
 *
 *   m̃ₖ(d̃) ≡ d̃ᵀr̃ᶜ + ½d̃ᵀB̃ₖr̃ᶜ
 *
 * along the subspace unconstrained Newton direction:
 *
 *   d̃ᵘ = -B̃ₖ⁻¹r̃ᶜ
 *
 * then backtrack towards the feasible region to obtain optimal direction (optional):
 *
 *   d̃* = α* × d̃ᵘ
 *
 * Reference: C implementation in lbfgsb/subspace.c
 * Reference: Go implementation in lbfgsb/subsapce.go
 *
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import static com.curioloop.numopt4j.linalg.blas.BLAS.*;
import com.curioloop.numopt4j.optim.Bound;

/**
 * Subspace minimization for the L-BFGS-B algorithm.
 *
 * <p>This class computes the subspace minimizer by:</p>
 * <ol>
 *   <li>Computing the reduced gradient r̃ᶜ = -Zᵀ(g + B(xᶜ - xₖ))</li>
 *   <li>Computing the unconstrained Newton direction d̃ᵘ = -B̃ₖ⁻¹r̃ᶜ</li>
 *   <li>Projecting onto the feasible region and backtracking if needed</li>
 * </ol>
 *
 * @see LBFGSBWorkspace
 * @see CauchyPoint
 */
final class SubspaceMinimizer implements LBFGSBConstants {

    private SubspaceMinimizer() {}

    /**
     * Computes the reduced gradient r̃ᶜ = -Zᵀ(g + B(xᶜ - xₖ)).
     *
     * <p>Given:</p>
     * <ul>
     *   <li>xₖ current location (x)</li>
     *   <li>gₖ the gradient value of f(x) (g)</li>
     *   <li>xᶜ the Cauchy point (z)</li>
     *   <li>Sₖ, Yₖ the correction matrices of Bₖ (ws, wy)</li>
     *   <li>c = Wᵀ(xᶜ - x), computed during Cauchy point search</li>
     * </ul>
     *
     * <p>The reduced gradient is computed as:</p>
     * <pre>
     *   r = -r̃ᶜ = -Zᵀ(g + θ(xᶜ-x) - WMc) = Zᵀ(-g - θ(xᶜ-x) + WMc)
     * </pre>
     *
     * <p>Where:</p>
     * <pre>
     *   W = [Y, θS]  (correction matrices)
     *   M = [-D    Lᵀ ]⁻¹
     *       [ L   θSᵀS]
     * </pre>
     *
     * <p>Reference: C function reduce_gradient() in subspace.c</p>
     * <p>Reference: Go function reduceGradient() in subsapce.go</p>
     *
     * @param n Problem dimension
     * @param m Maximum number of L-BFGS corrections
     * @param x Current point xₖ
     * @param g Gradient gₖ
     * @param z Cauchy point xᶜ
     * @param r Output: reduced gradient (size = free)
     * @param ws Workspace containing iteration state
     * @return 0 on success, negative on error
     *
     */
    static int reduceGradient(int n, int m,
                              double[] x, double[] g,
                              LBFGSBWorkspace ws) {
        double[] buffer = ws.getBuffer();
        int[] iBuffer = ws.getIntBuffer();

        int col = ws.getCol();
        int head = ws.getHead();
        int free = ws.getFree();
        double theta = ws.getTheta();
        boolean constrained = ws.isConstrained();

        // Index array: index[0:free] contains indices of free variables
        int indexOffset = ws.getIndexOffset();

        // BFGS correction matrices
        int wsOffset = ws.getWsOffset();  // S matrix (n x m)
        int wyOffset = ws.getWyOffset();  // Y matrix (n x m)
        int syOffset = ws.getSyOffset();  // S'Y matrix
        int wtOffset = ws.getWtOffset();  // Cholesky factor
        int waOffset = ws.getWaOffset();  // Work array

        // Workspace arrays offsets
        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();

        // Workspace arrays:
        // c[2m:4m] = W'(x^c - x), computed during Cauchy point search
        // v[0:2m]  = M*c, temporary for bmv result
        int cOffset = waOffset + 2 * m;  // c = W'(x^c - x)
        int vOffset = waOffset;          // v = M*c (temporary)
        int diagInvOffset = ws.getDiagInvOffset();
        int sqrtDiagInvOffset = ws.getSqrtDiagInvOffset();

        // Handle unconstrained case specially - matches Go exactly
        if (!constrained && col > 0) {
            // If the problem is unconstrained and col > 0, set r = -g
            for (int i = 0; i < n; i++) {
                buffer[rOffset + i] = -g[i];
            }
            return ERR_NONE;
        }

        // Compute r = -θ(x^c - x) - g for free variables
        for (int i = 0; i < free; i++) {
            int k = iBuffer[indexOffset + i];  // Index of free variable
            buffer[rOffset + i] = -theta * (buffer[zOffset + k] - x[k]) - g[k];
        }

        // If no BFGS corrections, we're done
        if (col == 0) {
            return ERR_NONE;
        }

        // Compute v = M * c using bmv
        int info = CauchyPoint.bmv(m, col, buffer, syOffset, buffer, wtOffset,
                                   buffer, cOffset, buffer, vOffset,
                                   buffer, diagInvOffset, buffer, sqrtDiagInvOffset);
        if (info != ERR_NONE) {
            return info;
        }

        // Compute r += W * M * c for free variables
        //
        // W = [Y, θS], so:
        //   W * M * c = Y * (Mc)_1 + θS * (Mc)_2
        //
        // For each free variable i with index k:
        //   r[i] += sum_j (Y[k,j] * v[j] + θ * S[k,j] * v[col+j])
        int ptr = head;
        for (int j = 0; j < col; j++) {
            double mc1 = buffer[vOffset + j];              // (Mc)_1[j]
            double mc2 = theta * buffer[vOffset + col + j]; // θ * (Mc)_2[j]

            for (int i = 0; i < free; i++) {
                int k = iBuffer[indexOffset + i];  // Index of free variable
                // r[i] += Y[k,j] * mc1 + S[k,j] * mc2
                buffer[rOffset + i] += buffer[wyOffset + k * m + ptr] * mc1 + buffer[wsOffset + k * m + ptr] * mc2;
            }

            ptr = (ptr + 1) % m;
        }

        return ERR_NONE;
    }


    /**
     * Computes the optimal direction by subspace minimization.
     *
     * <p>This subroutine computes an approximate solution of the subspace problem:</p>
     * <pre>
     *   m̃ₖ(d̃) ≡ d̃ᵀr̃ᶜ + ½d̃ᵀB̃ₖr̃ᶜ
     * </pre>
     *
     * <p>along the subspace unconstrained Newton direction:</p>
     * <pre>
     *   d̃ᵘ = -B̃ₖ⁻¹r̃ᶜ
     * </pre>
     *
     * <p>then backtrack towards the feasible region to obtain optimal direction (optional):</p>
     * <pre>
     *   d̃* = α* × d̃ᵘ
     * </pre>
     *
     * <p>Given the L-BFGS matrix and the Sherman-Morrison formula:</p>
     * <pre>
     *   B̃ₖ = (1/θ)I - (1/θ)ZᵀW[ (I-(1/θ)MWᵀZZᵀW)⁻¹M ]WᵀZ(1/θ)
     * </pre>
     *
     * <p>With N ≡ I - (1/θ)MWᵀZZᵀW, the formula for the unconstrained Newton direction is:</p>
     * <pre>
     *   d̃ᵘ = (1/θ)r̃ᶜ + (1/θ²)ZᵀWN⁻¹MZᵀW
     * </pre>
     *
     * <p>Then form middle K = M⁻¹N = (N⁻¹M)⁻¹ to avoid inverting N (see formk):</p>
     * <pre>
     *   d̃ᵘ = (1/θ)r̃ᶜ + (1/θ²)ZᵀWK⁻¹WᵀZr̃ᶜ
     * </pre>
     *
     * <p>Finally the computation of K⁻¹v could be replaced with solving v = Kx by factorization K = LELᵀ</p>
     *
     * <p>Reference: C function optimal_direction() in subspace.c</p>
     * <p>Reference: Go function optimalDirection() in subsapce.go</p>
     *
     * @param n Problem dimension
     * @param m Maximum number of L-BFGS corrections
     * @param x Current point xₖ
     * @param g Gradient gₖ
     * @param bounds Array of Bounds for each variable. Null means unbounded.
     * @param z Cauchy point xᶜ (modified: becomes subspace minimizer x̂)
     * @param r Reduced gradient r̃ᶜ (modified: becomes Newton direction d̃ᵘ)
     * @param ws Workspace containing iteration state
     * @return 0 on success, negative on error (ERR_SINGULAR_TRIANGULAR)
     *
     */
    static int optimalDirection(int n, int m,
                                double[] x, double[] g,
                                Bound[] bounds,
                                LBFGSBWorkspace ws) {
        double[] buffer = ws.getBuffer();
        int[] iBuffer = ws.getIntBuffer();

        int col = ws.getCol();
        int col2 = 2 * col;
        int head = ws.getHead();
        int free = ws.getFree();
        int m2 = 2 * m;
        double theta = ws.getTheta();

        // If no free variables, nothing to do
        if (free <= 0) {
            return ERR_NONE;
        }

        // Index array: index[0:free] contains indices of free variables (Z)
        int indexOffset = ws.getIndexOffset();

        // BFGS correction matrices
        int wsOffset = ws.getWsOffset();  // S matrix (n x m)
        int wyOffset = ws.getWyOffset();  // Y matrix (n x m)

        // K matrix (LEL^T factorization stored in wn)
        int wnOffset = ws.getWnOffset();

        // Workspace arrays:
        // wv[0:2m] = K^{-1}W^T Zr̃^c (temporary workspace)
        // xp[0:n]  = safeguard for projected Newton direction
        int waOffset = ws.getWaOffset();
        int wvOffset = waOffset;          // v = K^{-1}W^T Zr̃^c
        int xpOffset = ws.getXpOffset();  // Safeguard copy of z

        int zOffset = ws.getZOffset();
        int rOffset = ws.getROffset();

        // ========================================================================
        // Compute v = Wᵀ Z r̃ᶜ
        //
        // W = [Y, θS], so:
        //   v_y[j] = Σᵢ (Y[k,j] * r[i]) for free variable i with index k
        //   v_s[j] = θ × Σᵢ (S[k,j] * r[i]) for free variable i with index k
        // ========================================================================

        int ptr = head;
        for (int j = 0; j < col; j++) {
            double yr = 0.0;
            double sr = 0.0;
            for (int i = 0; i < free; i++) {
                int k = iBuffer[indexOffset + i];  // Index of free variable
                yr += buffer[wyOffset + k * m + ptr] * buffer[rOffset + i];
                sr += buffer[wsOffset + k * m + ptr] * buffer[rOffset + i];
            }
            buffer[wvOffset + j] = yr;
            buffer[wvOffset + col + j] = theta * sr;
            ptr = (ptr + 1) % m;
        }

        // ========================================================================
        // Compute K⁻¹v = (LELᵀ)⁻¹v = (L⁻ᵀE⁻¹L⁻¹)v
        //
        // Lᵀ stored in the upper triangle of WN
        // E⁻¹ = [-I  0]⁻¹ = [-I  0]
        //       [ 0  I]     [ 0  I]
        // ========================================================================

        // Compute L⁻¹v by solving Lx = (Lᵀ)ᵀx = v
        // Lᵀ is upper triangular, so we solve Lᵀᵀ x = v (uplo='U', trans='T' for transpose)
        if (dtrsl(buffer, wnOffset, m2, col2, buffer, wvOffset, Uplo.Upper, Trans.Trans) != 0) {
            return ERR_SINGULAR_TRIANGULAR;  // Singular triangular matrix
        }

        // Compute E⁻¹(L⁻¹v): negate first col elements
        dscal(col, -1.0, buffer, wvOffset, 1);

        // Compute L⁻ᵀ(E⁻¹L⁻¹v) by solving Lᵀx = E⁻¹L⁻¹v
        // Lᵀ is upper triangular (uplo='U', trans='N' for no transpose)
        if (dtrsl(buffer, wnOffset, m2, col2, buffer, wvOffset, Uplo.Upper, Trans.NoTrans) != 0) {
            return ERR_SINGULAR_TRIANGULAR;  // Singular triangular matrix
        }

        // ========================================================================
        // Compute r̃ᶜ + (1/θ)ZᵀW(K⁻¹WᵀZr̃ᶜ)
        //
        // buffer[rOffset + i] += Σⱼ (Y[k,j] × wv[j] / θ + S[k,j] × wv[col+j])
        // ========================================================================

        ptr = head;
        for (int j = 0; j < col; j++) {
            int js = col + j;
            for (int i = 0; i < free; i++) {
                int k = iBuffer[indexOffset + i];  // Index of free variable
                buffer[rOffset + i] += (buffer[wyOffset + k * m + ptr] * buffer[wvOffset + j] / theta)
                                     + (buffer[wsOffset + k * m + ptr] * buffer[wvOffset + js]);
            }
            ptr = (ptr + 1) % m;
        }

        // Scale r̃ᶜ + (1/θ)ZᵀWK⁻¹WᵀZr̃ᶜ by 1/θ
        // Note: Go uses d[i] *= one / theta which is equivalent to d[i] /= theta
        for (int i = 0; i < free; i++) {
            buffer[rOffset + i] *= 1.0 / theta;
        }

        // ========================================================================
        // Perform projection along unconstrained Newton direction d̃ᵘ
        // Compute subspace minimizer x̂ = 𝚙𝚛𝚘𝚓(xᶜ + d̃ᵘ)
        // ========================================================================

        // Save z to xp for safeguard
        dcopy(n, buffer, zOffset, 1, buffer, xpOffset, 1);

        // Project x^c + d̃^u onto feasible region
        boolean projected = false;
        for (int i = 0; i < free; i++) {
            int k = iBuffer[indexOffset + i];  // Index of free variable
            double dk = buffer[rOffset + i];
            double xk = buffer[zOffset + k];
            double zk = xk + dk;
            Bound b = bounds != null ? bounds[k] : null;
            if (b != null) {
                if (b.hasLower() && zk < b.getLower()) {
                    zk = b.getLower();
                    projected = true;
                }
                if (b.hasUpper() && zk > b.getUpper()) {
                    zk = b.getUpper();
                    projected = true;
                }
            }
            buffer[zOffset + k] = zk;
        }

        // Store solution status in workspace
        if (projected) {
            ws.setWord(SOLUTION_BEYOND_BOX);
        } else {
            ws.setWord(SOLUTION_WITHIN_BOX);
        }

        // ========================================================================
        // Check sign of the directional derivative
        // sgn = (x̂ - xₖ)ᵀgₖ
        //
        // If sgn > 0, the direction is not a descent direction, need to backtrack
        // ========================================================================

        double sgn = 0.0;
        if (projected) {
            for (int i = 0; i < n; i++) {
                sgn += (buffer[zOffset + i] - x[i]) * g[i];  // (x̂ - xₖ) × gₖ
            }
        }

        // ========================================================================
        // When the direction x̂ - xₖ is not a direction of strong descent,
        // truncate the path from xₖ to x̂ to satisfy the constraints
        //
        // sgn ≤ 0  ⇒  d̃* = d̃ᵘ (keep current z)
        // sgn > 0  ⇒  d̃* = α* × d̃ᵘ (backtrack)
        // ========================================================================

        if (sgn > 0.0) {
            // Restore z from xp - matches Go: copy(x[:n], xp[:n])
            dcopy(n, buffer, xpOffset, 1, buffer, zOffset, 1);

            // Search positive optimal step
            // α* = 𝚖𝚊𝚡 { α : α ≤ 1, lᵢ - xᶜᵢ ≤ α × d̃ᵘᵢ ≤ uᵢ - xᶜᵢ (i ∈ 𝓕) }
            double alpha = 1.0;
            int ibd = 0;

            for (int i = 0; i < free; i++) {
                int k = iBuffer[indexOffset + i];  // Index of free variable
                double dk = buffer[rOffset + i];
                Bound b = bounds[k];
                if (b != null) {
                    double stp = alpha;

                    // Match Go logic exactly:
                    // if dk < zero && bk.hint <= bndBoth (i.e., BOUND_LOWER or BOUND_BOTH)
                    // if dk > zero && bk.hint >= bndBoth (i.e., BOUND_UPPER or BOUND_BOTH)
                    if (dk < 0.0 && b.hasLower()) {
                        // Moving towards lower bound
                        double span = b.getLower() - buffer[zOffset + k];
                        if (span >= 0.0) {
                            stp = 0.0;
                        } else if (dk * alpha < span) {
                            stp = span / dk;
                        }
                    } else if (dk > 0.0 && b.hasUpper()) {
                        // Moving towards upper bound
                        double span = b.getUpper() - buffer[zOffset + k];
                        if (span <= 0.0) {
                            stp = 0.0;
                        } else if (dk * alpha > span) {
                            stp = span / dk;
                        }
                    }

                    if (stp < alpha) {
                        alpha = stp;
                        ibd = i;
                    }
                }
            }

            // If alpha < 1, fix the blocking variable at its bound
            if (alpha < 1.0) {
                double dk = buffer[rOffset + ibd];
                int k = iBuffer[indexOffset + ibd];
                Bound b = Bound.of(bounds, k, Bound.unbounded());
                if (dk > 0.0) {
                    buffer[zOffset + k] = b.getUpper();
                    buffer[rOffset + ibd] = 0.0;
                } else if (dk < 0.0) {
                    buffer[zOffset + k] = b.getLower();
                    buffer[rOffset + ibd] = 0.0;
                }
            }

            // x̂ = xᶜ + d̃* = xᶜ + (α* × d̃ᵘ)
            //   x̂ᵢ = xᶜᵢ           if i ∉ 𝓕
            //   x̂ᵢ = xᶜᵢ + Zd̃*ᵢ   otherwise
            for (int i = 0; i < free; i++) {
                int k = iBuffer[indexOffset + i];  // Index of free variable
                buffer[zOffset + k] += alpha * buffer[rOffset + i];
            }
        }

        return ERR_NONE;
    }

}
