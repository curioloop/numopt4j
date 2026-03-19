/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * L-BFGS-B Cauchy Point Computation - Pure Java implementation.
 *
 * This module computes the Generalized Cauchy Point (GCP) for the L-BFGS-B algorithm.
 *
 * Given:
 *   - xₖ current location
 *   - fₖ the function value of f(x)
 *   - gₖ the gradient value of f(x)
 *   - Sₖ, Yₖ the correction matrices of Bₖ
 *
 * The quadratic model without bounds of f(x) at xₖ is:
 *
 *   mₖ(x) = fₖ + gₖᵀ(x-xₖ) + ½(x-xₖ)ᵀBₖ(x-xₖ)
 *
 * The GCP is defined as the first local minimizer of mₖ(x) along the piecewise
 * linear path 𝚙𝚛𝚘𝚓(xₖ - tgₖ) obtained by projecting points along the steepest
 * descent direction xₖ - tgₖ onto the feasible region.
 *
 * Final return:
 *   - GCP : xᶜ
 *   - Cauchy direction : dᶜ = 𝚙𝚛𝚘𝚓(xₖ - tgₖ) - xₖ
 *
 * Reference: C implementation in lbfgsb/cauchy.c
 * Reference: Go implementation in lbfgsb/cauchy.go
 *
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import static com.curioloop.numopt4j.linalg.blas.BLAS.*;
import com.curioloop.numopt4j.optim.Bound;

/**
 * Cauchy point computation for the L-BFGS-B algorithm.
 *
 * <p>This class computes the Generalized Cauchy Point (GCP), which is the first
 * local minimizer of the quadratic model along the projected steepest descent path.</p>
 *
 * <h2>Algorithm Overview</h2>
 * <p>The GCP computation involves:</p>
 * <ol>
 *   <li>Computing breakpoints where variables hit their bounds</li>
 *   <li>Processing breakpoints in order using heap sort</li>
 *   <li>Updating quadratic model derivatives at each breakpoint</li>
 *   <li>Finding the minimizer within the current segment</li>
 * </ol>
 *
 * @see LBFGSBWorkspace
 * @see LBFGSBCore
 */
final class CauchyPoint implements LBFGSBConstants {

    private CauchyPoint() {}

    // ========================================================================
    // Heap Sort Implementation (hpsolb)
    // ========================================================================

    /**
     * Heap sort output minimum breakpoint (hpsolb).
     *
     * <p>Given t[0:n] and order[0:n]:</p>
     * <ul>
     *   <li>Build min-heap on t[0:n] if sorted == false</li>
     *   <li>Swap the top element to the tail: t[0] ⇄ t[n-1]</li>
     *   <li>Recover heap t[0:n-1] by shifting down t[0]</li>
     * </ul>
     *
     * <p>After calling this function:</p>
     * <ul>
     *   <li>t[n-1] contains the minimum value that was at t[0]</li>
     *   <li>t[0:n-1] is a valid min-heap</li>
     *   <li>order array is updated correspondingly</li>
     * </ul>
     *
     * <p>Reference: C function heap_sort_out() in cauchy.c</p>
     * <p>Reference: Go function heapSortOut() in cauchy.go</p>
     *
     * @param n Number of elements
     * @param t Breakpoint times array
     * @param tOffset Offset into t array
     * @param order Order array (indices)
     * @param orderOffset Offset into order array
     * @param sorted True if heap is already built, false to build heap first
     */
    static void heapSortOut(int n, double[] t, int tOffset,
                            int[] order, int orderOffset, boolean sorted) {
        if (n <= 0) {
            return;
        }

        // Build min-heap on t[0:n] if not already sorted
        if (!sorted) {
            for (int k = 1; k < n; k++) {
                // Add t[k] to the heap t[0:k-1]
                int i = k;
                double val = t[tOffset + i];
                int idx = order[orderOffset + i];

                // Shift up: compare with parent and swap if smaller
                while (i > 0 && i < n) {
                    int j = (i - 1) / 2;  // Parent index
                    if (val < t[tOffset + j]) {
                        // Shift down the parent
                        t[tOffset + i] = t[tOffset + j];
                        order[orderOffset + i] = order[orderOffset + j];
                        i = j;
                    } else {
                        // Already a heap
                        break;
                    }
                }
                t[tOffset + i] = val;
                order[orderOffset + i] = idx;
            }
        }

        if (n > 1) {
            // Pop the least (top) element of heap
            double topVal = t[tOffset];
            int topIdx = order[orderOffset];

            // Move the bottom element to top: t[0] = t[n-1] and trim the heap to t[0:n-1]
            double val = t[tOffset + n - 1];
            int idx = order[orderOffset + n - 1];

            // Shift down t[0] until heap property is recovered
            int i = 0;  // t[i] is parent
            for (;;) {
                int j = 2 * i + 1;  // Left child index
                if (j < n) {
                    // Select the smaller child when right child is available
                    if (j + 1 < n && t[tOffset + j + 1] < t[tOffset + j]) {
                        j++;
                    }
                    if (t[tOffset + j] < val) {
                        // Shift up the smaller child
                        t[tOffset + i] = t[tOffset + j];
                        order[orderOffset + i] = order[orderOffset + j];
                        i = j;
                    } else {
                        // Stop when parent is smaller than children
                        break;
                    }
                } else {
                    break;
                }
            }

            // Now t[0:n-1] is a heap
            t[tOffset + i] = val;
            order[orderOffset + i] = idx;

            // Store the least element at t[n-1]
            t[tOffset + n - 1] = topVal;
            order[orderOffset + n - 1] = topIdx;
        }
    }

    // ========================================================================
    // BMV Matrix-Vector Multiplication
    // ========================================================================

    /**
     * BMV Matrix-Vector Multiplication: p = Mv.
     *
     * <p>Given 2m vector v = [v₁, v₂]ᵀ, calculate matrix product p = Mv with
     * 2m × 2m middle matrix:</p>
     * <pre>
     *     M = [ -D    Lᵀ  ]⁻¹
     *         [  L   θSᵀS ]
     * </pre>
     *
     * <h3>Algorithm:</h3>
     * <ol>
     *   <li>Calculate upper triangular matrix Jᵀ by applying Cholesky factorization to
     *       symmetric positive definite matrix: (θSᵀS + LD⁻¹Lᵀ) = JJᵀ</li>
     *   <li>Reorder the blocks to get M⁻¹ = (AB)⁻¹ = B⁻¹A⁻¹</li>
     *   <li>Calculate p = Bv by solving B⁻¹p = v</li>
     *   <li>Calculate p = ABv = Mv by solving A⁻¹p = Bv</li>
     * </ol>
     *
     * <p>Matrices D and L are calculated from SᵀY:</p>
     * <ul>
     *   <li>D = diag{sᵢᵀyᵢ} for i = 1,...,col</li>
     *   <li>Lᵢⱼ = sᵢᵀyⱼ for i &gt; j (strictly lower triangular)</li>
     * </ul>
     *
     * <p>Reference: C function bmv() in cauchy.c</p>
     * <p>Reference: Go function bmv() in cauchy.go</p>
     *
     * @param m Maximum number of corrections
     * @param col Current number of corrections stored
     * @param sy SᵀY matrix (m × m)
     * @param syOffset Offset into sy array
     * @param wt JJᵀ Cholesky factor (m × m)
     * @param wtOffset Offset into wt array
     * @param v Input vector (2*col)
     * @param vOffset Offset into v array
     * @param p Output vector p = Mv (2*col)
     * @param pOffset Offset into p array
     * @param diagInv Pre-allocated array for 1/Dᵢᵢ (size m)
     * @param diagInvOffset Offset into diagInv array
     * @param sqrtDiagInv Pre-allocated array for 1/√Dᵢᵢ (size m)
     * @param sqrtDiagInvOffset Offset into sqrtDiagInv array
     * @return 0 on success, negative on error (ERR_SINGULAR_TRIANGULAR)
     *
     */
    static int bmv(int m, int col,
                   double[] sy, int syOffset,
                   double[] wt, int wtOffset,
                   double[] v, int vOffset,
                   double[] p, int pOffset,
                   double[] diagInv, int diagInvOffset,
                   double[] sqrtDiagInv, int sqrtDiagInvOffset) {
        if (col == 0) {
            return ERR_NONE;
        }

        // Matrices D and L can be calculated from SᵀY:
        //   D = diag{sᵢᵀyᵢ} for i = 1,...,col
        //   Lᵢⱼ = sᵢᵀyⱼ for i > j (strictly lower triangular)

        // Pointers to v₁, v₂ and p₁, p₂
        int v1Off = vOffset;
        int v2Off = vOffset + col;
        int p1Off = pOffset;
        int p2Off = pOffset + col;

        // ====================================================================
        // Precompute diagonal elements and their reciprocals for efficiency
        // Using pre-allocated workspace arrays instead of allocating new arrays
        // ====================================================================
        for (int i = 0; i < col; i++) {
            double dii = sy[syOffset + i * m + i];
            if (dii <= 0.0) {
                return ERR_SINGULAR_TRIANGULAR;
            }
            diagInv[diagInvOffset + i] = 1.0 / dii;
            sqrtDiagInv[sqrtDiagInvOffset + i] = 1.0 / Math.sqrt(dii);
        }

        // ====================================================================
        // PART I: Solve [ D¹ᐟ²      O  ] [ p₁ ] = [ v₁ ]
        //               [ -LD⁻¹ᐟ²   J  ] [ p₂ ]   [ v₂ ]
        //
        // From first row:  D¹ᐟ²p₁ = v₁  ⇒  p₁ = D⁻¹ᐟ²v₁
        // From second row: -LD⁻¹ᐟ²p₁ + Jp₂ = v₂  ⇒  p₂ = J⁻¹(v₂ + LD⁻¹v₁)
        // ====================================================================

        // Calculate v₂ + LD⁻¹v₁ and store in p₂
        p[p2Off] = v[v2Off];
        for (int i = 1; i < col; i++) {
            // Calculate (LD⁻¹v₁)ᵢ = ∑(Lᵢⱼ * v₁ⱼ / Dⱼⱼ) for j < i
            double sum = 0.0;
            int iRow = syOffset + i * m;
            for (int j = 0; j < i; j++) {
                // Lᵢⱼ = sy[i*m + j] (lower triangular part)
                sum += sy[iRow + j] * v[v1Off + j] * diagInv[diagInvOffset + j];
            }
            // p₂ᵢ = v₂ᵢ + (LD⁻¹v₁)ᵢ
            p[p2Off + i] = v[v2Off + i] + sum;
        }

        // Calculate p₂ by solving triangular system Jp₂ = v₂ + LD⁻¹v₁
        // J is upper triangular stored in wt
        // Use uplo='U', trans='T' for solving Jᵀx = b (matches Go solveUpperT)
        if (dtrsl(wt, wtOffset, m, col, p, p2Off, Uplo.Upper, Trans.Trans) != 0) {
            return ERR_SINGULAR_TRIANGULAR;  // Singular triangular matrix
        }

        // Solve p₁ = D⁻¹ᐟ²v₁
        for (int i = 0; i < col; i++) {
            p[p1Off + i] = v[v1Off + i] * sqrtDiagInv[sqrtDiagInvOffset + i];
        }

        // ====================================================================
        // PART II: Solve [ -D¹ᐟ²  D⁻¹ᐟ²Lᵀ ] [ p₁ ] = [ ṗ₁ ]
        //                [  O     Jᵀ      ] [ p₂ ]   [ ṗ₂ ]
        //
        // From second row: Jᵀp₂ = ṗ₂  ⇒  p₂ = J⁻ᵀṗ₂
        // From first row:  -D¹ᐟ²p₁ + D⁻¹ᐟ²Lᵀp₂ = ṗ₁
        //                  ⇒  p₁ = -D⁻¹ᐟ²(ṗ₁ - D⁻¹ᐟ²Lᵀp₂)
        //                      = -D⁻¹ᐟ²ṗ₁ + D⁻¹Lᵀp₂
        // ====================================================================

        // Calculate p₂ by solving Jᵀp₂ = ṗ₂
        // J is upper triangular stored in wt
        // Use uplo='U', trans='N' for solving Jx = b (matches Go solveUpperN)
        if (dtrsl(wt, wtOffset, m, col, p, p2Off, Uplo.Upper, Trans.NoTrans) != 0) {
            return ERR_SINGULAR_TRIANGULAR;  // Singular triangular matrix
        }

        // Calculate p₁ = -D⁻¹ᐟ²ṗ₁ + D⁻¹Lᵀp₂
        for (int i = 0; i < col; i++) {
            // First term: -D⁻¹ᐟ²ṗ₁
            p[p1Off + i] *= -sqrtDiagInv[sqrtDiagInvOffset + i];
        }

        for (int i = 0; i < col; i++) {
            // Calculate (D⁻¹Lᵀp₂)ᵢ = ∑(Lⱼᵢ * p₂ⱼ / Dᵢᵢ) for j > i
            // Note: Lᵀ has Lⱼᵢ in position (i, j) where j > i
            double sum = 0.0;
            double invDii = diagInv[diagInvOffset + i];
            for (int j = i + 1; j < col; j++) {
                // Lⱼᵢ = sy[j*m + i] (L is stored in lower triangle of sy)
                sum += sy[syOffset + j * m + i] * p[p2Off + j] * invDii;
            }
            // Add to p₁ᵢ
            p[p1Off + i] += sum;
        }

        return ERR_NONE;
    }

    // ========================================================================
    // Free Variable Identification (freev)
    // ========================================================================

    /**
     * Count entering and leaving variables and build the index set of free variables.
     *
     * <p>This subroutine counts the entering and leaving variables when iter &gt; 0,
     * and finds the index set of free and active variables at the GCP.</p>
     *
     * <h3>Index arrays:</h3>
     * <ul>
     *   <li>index[0:free] are indices of free variables</li>
     *   <li>index[free:n] are indices of bound variables</li>
     * </ul>
     *
     * <h3>State arrays (for tracking changes):</h3>
     * <ul>
     *   <li>state[0:enter] have changed from bound to free</li>
     *   <li>state[leave:n] have changed from free to bound</li>
     * </ul>
     *
     * <p>Reference: C function free_var() in cauchy.c</p>
     * <p>Reference: Go function freeVar() in cauchy.go</p>
     *
     * @param n Problem dimension
     * @param ws Workspace containing iteration state
     * @return 1 if K matrix needs recomputation, 0 otherwise
     */
    static int freeVar(int n, LBFGSBWorkspace ws) {
        int[] iBuffer = ws.getIntBuffer();

        int indexOffset = ws.getIndexOffset();
        int whereOffset = ws.getWhereOffset();

        // index[0:n] for free/bound variables
        // index[n:2n] for entering/leaving variables (state)
        int stateOffset = indexOffset + n;

        int iter = ws.getIter();
        boolean constrained = ws.isConstrained();
        int oldFree = ws.getFree();

        int enter = 0;
        int leave = n;

        // Count entering and leaving variables for iter > 0
        if (iter > 0 && constrained) {
            // Check variables that were free in previous iteration
            for (int i = 0; i < oldFree; i++) {
                int k = iBuffer[indexOffset + i];
                if (iBuffer[whereOffset + k] > VAR_FREE) {
                    // Variable is now at a bound - leaving free set
                    leave--;
                    iBuffer[stateOffset + leave] = k;
                }
            }

            // Check variables that were at bounds in previous iteration
            for (int i = oldFree; i < n; i++) {
                int k = iBuffer[indexOffset + i];
                if (iBuffer[whereOffset + k] <= VAR_FREE) {
                    // Variable is now free - entering free set
                    iBuffer[stateOffset + enter] = k;
                    enter++;
                }
            }
        }

        ws.setEnter(enter);
        ws.setLeave(leave);

        // Build the index set of free and active variables at the GCP
        int freeCount = 0;
        int activeCount = n;

        for (int i = 0; i < n; i++) {
            if (iBuffer[whereOffset + i] <= VAR_FREE) {
                // Free variable (VAR_FREE or VAR_UNBOUND)
                iBuffer[indexOffset + freeCount] = i;
                freeCount++;
            } else {
                // Bound variable (VAR_AT_LOWER, VAR_AT_UPPER, or VAR_FIXED)
                activeCount--;
                iBuffer[indexOffset + activeCount] = i;
            }
        }

        ws.setFree(freeCount);
        ws.setActive(n - freeCount);

        // Return whether K matrix needs to be recomputed
        return ((leave < n) || (enter > 0) || ws.isUpdated()) ? 1 : 0;
    }

    // ========================================================================
    // Cauchy Point Computation
    // ========================================================================

    /**
     * Compute the Generalized Cauchy Point (GCP) by piecewise linear path search.
     *
     * <p>Given:</p>
     * <ul>
     *   <li>xₖ current location</li>
     *   <li>fₖ the function value of f(x)</li>
     *   <li>gₖ the gradient value of f(x)</li>
     *   <li>Sₖ, Yₖ the correction matrices of Bₖ</li>
     * </ul>
     *
     * <p>The quadratic model without bounds of f(x) at xₖ is:</p>
     * <pre>
     *   mₖ(x) = fₖ + gₖᵀ(x-xₖ) + ½(x-xₖ)ᵀBₖ(x-xₖ)
     * </pre>
     *
     * <p>This subroutine computes the GCP, defined as the first local minimizer of mₖ(x),
     * along the piecewise linear path 𝚙𝚛𝚘𝚓(xₖ - tgₖ) obtained by projecting points
     * along the steepest descent direction xₖ - tgₖ onto the feasible region.</p>
     *
     * <h3>Breakpoint computation:</h3>
     * <pre>
     *   tᵢ = (xᵢ - uᵢ)/gᵢ  if gᵢ &lt; 0
     *   tᵢ = (xᵢ - lᵢ)/gᵢ  if gᵢ &gt; 0
     *   tᵢ = ∞             otherwise
     * </pre>
     *
     * <h3>Search direction:</h3>
     * <pre>
     *   dᵢ = 0    if tᵢ = 0
     *   dᵢ = -gᵢ  otherwise
     * </pre>
     *
     * <h3>Corrections of B:</h3>
     * <pre>
     *   W = [Y  θS]   M = [ -D    Lᵀ  ]⁻¹
     *                     [  L   θSᵀS ]
     * </pre>
     *
     * <h3>Derivative updates at each breakpoint:</h3>
     * <pre>
     *   f′ = f′ + f″Δtᵢ + gᵢ² + θgᵢzᵢ - gᵢwᵀᵢMc
     *   f″ = f″ - θgᵢ² - 2gᵢwᵀᵢMp - gᵢ²wᵀᵢMwᵢ
     * </pre>
     *
     * <p>Reference: C function cauchy_point() in cauchy.c</p>
     * <p>Reference: Go function cauchy() in cauchy.go</p>
     *
     * @param n Problem dimension
     * @param m Maximum number of L-BFGS corrections
     * @param x Current point xₖ
     * @param g Gradient gₖ
     * @param bounds Array of Bounds for each variable. Null means unbounded.
     * @param ws Workspace containing iteration state (result written to workspace z)
     * @return 0 on success, negative on error
     *
     */
    static int compute(int n, int m,
                       double[] x, double[] g,
                       Bound[] bounds,
                       LBFGSBWorkspace ws) {

        double[] buffer = ws.getBuffer();
        int[] iBuffer = ws.getIntBuffer();

        int col = ws.getCol();
        int col2 = 2 * col;
        double theta = ws.getTheta();

        // Array offsets
        int wsOffset = ws.getWsOffset();   // S matrix
        int wyOffset = ws.getWyOffset();   // Y matrix
        int syOffset = ws.getSyOffset();   // S'Y matrix
        int wtOffset = ws.getWtOffset();   // Cholesky factor
        int dOffset = ws.getDOffset();     // Search direction d
        int tOffset = ws.getTOffset();     // Breakpoint times
        int waOffset = ws.getWaOffset();   // Work array (8m)
        int whereOffset = ws.getWhereOffset();
        int indexOffset = ws.getIndexOffset();
        int diagInvOffset = ws.getDiagInvOffset();
        int sqrtDiagInvOffset = ws.getSqrtDiagInvOffset();

        // Use second half of index array for order
        int orderOffset = indexOffset + n;

        // Workspace arrays from wa (8*m total):
        // p[0:2m]   = Wᵀd = [Yᵀd, θSᵀd]ᵀ
        // c[2m:4m]  = Wᵀ(xᶜ - x)
        // w[4m:6m]  = Wᵢ (row of W at breakpoint)
        // v[6m:8m]  = M? (temporary for bmv)
        int pOffset = waOffset;
        int cOffset = waOffset + 2 * m;
        int wOffset = waOffset + 4 * m;
        int vOffset = waOffset + 6 * m;

        // Precompute z offset (workspace stores GCP z)
        int zOffset = ws.getZOffset();

        // Check if projected gradient norm is zero: ‖proj g‖∞ = 0 → ∀ gᵢ = 0
        if (ws.getSbgNorm() <= 0.0) {
            // xᶜ = x (store into workspace z)
            dcopy(n, x, 0, 1, buffer, zOffset, 1);
            return ERR_NONE;
        }

        // Initialize p to zero
        for (int i = 0; i < col2; i++) {
            buffer[pOffset + i] = 0.0;
        }

        // Initialize f′ = gᵀd = -dᵀd = ∑(-dᵢ²)
        // Initialize f″ = -θf′ - pᵀMp
        double f1 = 0.0;
        double f2 = 0.0;

        int nFree = n;      // Number of free variables
        int nBreak = 0;     // Number of breakpoints
        double bkMin = 0.0;
        int idxMin = 0;
        boolean bounded = true;

        // Loop over all variables to determine:
        // 1. Variable status (iwhere)
        // 2. Search direction d
        // 3. Breakpoints t
        // 4. Initialize p = Wᵀd
        int head = ws.getHead();

        for (int i = 0; i < n; i++) {
            double negG = -g[i];
            Bound b = Bound.of(bounds, i, Bound.unbounded());

            double tl = 0.0;
            double tu = 0.0;

            if (iBuffer[whereOffset + i] != VAR_FIXED && iBuffer[whereOffset + i] != VAR_UNBOUND) {
                // If xᵢ is not a constant and has bounds, compute xᵢ - lᵢ and uᵢ - xᵢ
                if (b.hasLower()) {
                    tl = x[i] - b.getLower();
                }
                if (b.hasUpper()) {
                    tu = b.getUpper() - x[i];
                }

                iBuffer[whereOffset + i] = VAR_FREE;

                // If a variable is close enough to a bound we treat it as at bound
                if (b.hasLower() && tl <= 0.0) {
                    if (negG <= 0.0) {
                        // xᵢ ≤ lᵢ and -gᵢ ≤ 0 means xₖ₊₁ᵢ = xₖᵢ - gₖᵢ < lᵢ
                        iBuffer[whereOffset + i] = VAR_AT_LOWER;
                    }
                } else if (b.hasUpper() && tu <= 0.0) {
                    if (negG >= 0.0) {
                        // xᵢ ≥ uᵢ and -gᵢ ≥ 0 means xₖ₊₁ᵢ = xₖᵢ - gₖᵢ > uᵢ
                        iBuffer[whereOffset + i] = VAR_AT_UPPER;
                    }
                } else {
                    if (Math.abs(negG) <= 0.0) {
                        // gᵢ = 0, variable won't move
                        iBuffer[whereOffset + i] = VAR_NOT_MOVE;
                    }
                }
            }

            // Set search direction and update p
            if (iBuffer[whereOffset + i] != VAR_FREE && iBuffer[whereOffset + i] != VAR_UNBOUND) {
                // Fixed variable: dᵢ = 0
                buffer[dOffset + i] = 0.0;
            } else {
                // Free variable: dᵢ = -gᵢ
                buffer[dOffset + i] = negG;
                f1 -= negG * negG;  // f′ += -dᵢ²

                // Update p = Wᵀd:
                // pᵧ[j] += wy[i,j] * (-gᵢ)
                // pₛ[j] += ws[i,j] * (-gᵢ)
                int pyOffset = pOffset;
                int psOffset = pOffset + col;
                int ptr = head;
                for (int j = 0; j < col; j++) {
                    buffer[pyOffset + j] += buffer[wyOffset + i * m + ptr] * negG;
                    buffer[psOffset + j] += buffer[wsOffset + i * m + ptr] * negG;
                    ptr = (ptr + 1) % m;
                }

                // Compute breakpoint for this variable
                if (b.hasLower() && negG < 0.0) {
                    // xᵢ + dᵢ is bounded below, compute tᵢ = (xᵢ - lᵢ) / (-dᵢ)
                    iBuffer[orderOffset + nBreak] = i;
                    buffer[tOffset + nBreak] = tl / (-negG);
                    if (nBreak == 0 || buffer[tOffset + nBreak] < bkMin) {
                        bkMin = buffer[tOffset + nBreak];
                        idxMin = nBreak;
                    }
                    nBreak++;
                } else if (b.hasUpper() && negG > 0.0) {
                    // xᵢ + dᵢ is bounded above, compute tᵢ = (uᵢ - xᵢ) / dᵢ
                    iBuffer[orderOffset + nBreak] = i;
                    buffer[tOffset + nBreak] = tu / negG;
                    if (nBreak == 0 || buffer[tOffset + nBreak] < bkMin) {
                        bkMin = buffer[tOffset + nBreak];
                        idxMin = nBreak;
                    }
                    nBreak++;
                } else {
                    // xᵢ + dᵢ is not bounded
                    nFree--;
                    iBuffer[orderOffset + nFree] = i;
                    if (Math.abs(negG) > 0.0) {
                        bounded = false;
                    }
                }
            }
        }

        // Complete initialization of p for θ ≠ 1
        if (theta != 1.0) {
            int psOffset = pOffset + col;
            dscal(col, theta, buffer, psOffset, 1);
        }

        // Initialize GCP in workspace buffer: xᶜ = x
        dcopy(n, x, 0, 1, buffer, zOffset, 1);

        // If d is zero vector, return with initial xᶜ as GCP
        if (nBreak == 0 && nFree == n) {
            return ERR_NONE;
        }

        // Initialize c = Wᵀ(xᶜ - x) = 0
        for (int i = 0; i < col2; i++) {
            buffer[cOffset + i] = 0.0;
        }

        // Initialize f″ = -θf′
        f2 = -theta * f1;
        double orgF2 = f2;

        // Compute f″ -= pᵀMp using bmv
        if (col > 0) {
            int info = bmv(m, col, buffer, syOffset, buffer, wtOffset,
                          buffer, pOffset, buffer, vOffset,
                          buffer, diagInvOffset, buffer, sqrtDiagInvOffset);
            if (info != ERR_NONE) {
                return info;
            }
            f2 -= ddot(col2, buffer, vOffset, 1, buffer, pOffset, 1);
        }

        // Δtₘᵢₙ = -f′/f″
        double deltaMin = -f1 / f2;
        double deltaSum = 0.0;
        ws.setSeg(1);

        // Search along piecewise linear path
        boolean found = (nBreak == 0);
        int nLeft = nBreak;

        for (int iter = 1; nLeft > 0; iter++) {
            int tIdx;
            double tVal, tOld;

            if (iter == 1) {
                // Use the smallest breakpoint found during initialization
                tVal = bkMin;
                tIdx = iBuffer[orderOffset + idxMin];
                tOld = 0.0;
            } else {
                if (iter == 2) {
                    // Swap the used smallest breakpoint with the last one before heapsort
                    int nLast = nBreak - 1;
                    if (idxMin != nLast) {
                        double tmpT = buffer[tOffset + idxMin];
                        int tmpO = iBuffer[orderOffset + idxMin];
                        buffer[tOffset + idxMin] = buffer[tOffset + nLast];
                        iBuffer[orderOffset + idxMin] = iBuffer[orderOffset + nLast];
                        buffer[tOffset + nLast] = tmpT;
                        iBuffer[orderOffset + nLast] = tmpO;
                    }
                }
                // Update heap structure (if iter=2, build heap)
                heapSortOut(nLeft, buffer, tOffset, iBuffer, orderOffset, iter > 2);
                tOld = buffer[tOffset + nLeft];
                tVal = buffer[tOffset + nLeft - 1];
                tIdx = iBuffer[orderOffset + nLeft - 1];
            }

            // Compute dt = t[nLeft] - t[nLeft + 1]
            double tDelta = tVal - tOld;

            // If minimizer is within this interval (Δtₘᵢₙ < Δtᵢ), locate GCP and return
            if (deltaMin < tDelta) {
                found = true;
                break;
            }

            // Fix one variable and reset its d component to zero
            deltaSum += tDelta;
            nLeft--;

            double dBreak = buffer[dOffset + tIdx];           // -gᵢ
            double d2Break = dBreak * dBreak;                 // gᵢ²
            buffer[dOffset + tIdx] = 0.0;                    // dᵢ = 0

            // Update xᶜ and variable status
            if (dBreak > 0.0) {
                Bound b = Bound.of(bounds, tIdx, Bound.unbounded());
                buffer[zOffset + tIdx] = b.getUpper(); // xᶜᵢ = uᵢ (dᵢ > 0)
                iBuffer[whereOffset + tIdx] = VAR_AT_UPPER;
            } else {
                Bound b = Bound.of(bounds, tIdx, Bound.unbounded());
                buffer[zOffset + tIdx] = b.getLower(); // xᶜᵢ = lᵢ (dᵢ < 0)
                iBuffer[whereOffset + tIdx] = VAR_AT_LOWER;
            }
            double zBreak = buffer[zOffset + tIdx] - x[tIdx];                // zᵢ = xᶜᵢ - xᵢ

            // All n variables are fixed, return with xᶜ as GCP
            if (nLeft == 0 && nBreak == n) {
                deltaMin = tDelta;
                break;
            }

            // Update derivative information:
            // f′ = f′ + f″Δtᵢ + gᵢ² + θgᵢzᵢ - gᵢwᵀᵢMc
            // f″ = f″ - θgᵢ² - 2gᵢwᵀᵢMp - gᵢ²wᵀᵢMwᵢ
            f1 += f2 * tDelta + d2Break - theta * dBreak * zBreak;
            f2 -= theta * d2Break;

            // Process matrix product with middle matrix M
            if (col > 0) {
                // c = c + pΔtᵢ
                daxpy(col2, tDelta, buffer, pOffset, 1, buffer, cOffset, 1);

                // w = Wᵢ (row of W at breakpoint, 2m elements)
                int w1Offset = wOffset;
                int w2Offset = wOffset + col;
                int ptr = head;
                for (int j = 0; j < col; j++) {
                    buffer[w1Offset + j] = buffer[wyOffset + tIdx * m + ptr];        // Yᵢ
                    buffer[w2Offset + j] = theta * buffer[wsOffset + tIdx * m + ptr]; // θSᵢ
                    ptr = (ptr + 1) % m;
                }

                // v = Mw (2m)
                int info = bmv(m, col, buffer, syOffset, buffer, wtOffset,
                              buffer, wOffset, buffer, vOffset,
                              buffer, diagInvOffset, buffer, sqrtDiagInvOffset);
                if (info != ERR_NONE) {
                    return info;
                }

                double wmc = ddot(col2, buffer, cOffset, 1, buffer, vOffset, 1);  // wᵀMc
                double wmp = ddot(col2, buffer, pOffset, 1, buffer, vOffset, 1);  // wᵀMp
                double wmw = ddot(col2, buffer, wOffset, 1, buffer, vOffset, 1);  // wᵀMw

                // p = p + (-gᵢ)w
                daxpy(col2, -dBreak, buffer, wOffset, 1, buffer, pOffset, 1);

                f1 += dBreak * wmc;                              // += -gᵢwᵀᵢMc
                f2 += 2.0 * dBreak * wmp - d2Break * wmw;        // += -2gᵢwᵀᵢMp - gᵢ²wᵀᵢMwᵢ
            }

            // Ensure f″ doesn't become too small
            f2 = Math.max(EPS * orgF2, f2);
            deltaMin = -f1 / f2;  // Δtₘᵢₙ = -f′/f″

            if (nLeft == 0 && bounded) {
                f1 = 0.0;
                f2 = 0.0;
                deltaMin = 0.0;
            }
        }

        // Handle remaining variables
        if (nLeft == 0 || found) {
            deltaMin = Math.max(deltaMin, 0.0);  // Δtₘᵢₙ = max(Δtₘᵢₙ, 0)
            deltaSum += deltaMin;                  // tₒₗₐ = tₒₗₐ + Δtₘᵢₙ

            // Move free variables and variables whose breakpoints haven't been reached:
            // xᶜᵢ = xᵢ + tₒₗₐ * dᵢ (for dᵢ ≠ 0), write into workspace z
            daxpy(n, deltaSum, buffer, dOffset, 1, buffer, zOffset, 1);
        }

        // Update c = c + Δtₘᵢₙ * p = Wᵀ(xᶜ - x)
        // which will be used in computing r = Zᵀ(B(xᶜ - x) + g)
        if (col > 0) {
            daxpy(col2, deltaMin, buffer, pOffset, 1, buffer, cOffset, 1);
        }

        return ERR_NONE;
    }

}
