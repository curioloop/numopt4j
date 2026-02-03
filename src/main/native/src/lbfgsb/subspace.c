/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * L-BFGS-B subspace minimization functions.
 * Based on the Go implementation in subsapce.go.
 *
 * This file implements the subspace minimization step of the L-BFGS-B algorithm.
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
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/* Variable Status Constants (matches Go varWhere type) */
#define VAR_NOT_MOVE  -3
#define VAR_UNBOUND   -1
#define VAR_FREE       0
#define VAR_AT_LOWER   1
#define VAR_AT_UPPER   2
#define VAR_FIXED      3

/* Solution status constants (matching Go solutionXxx constants) */
#define SOLUTION_UNKNOWN    -1
#define SOLUTION_WITHIN_BOX  0
#define SOLUTION_BEYOND_BOX  1

/* Forward declaration for bmv from cauchy.c */
extern int bmv(int m, int col, const double* sy, const double* wt,
               const double* v, double* p);

/* External BLAS functions */
extern void dcopy(int n, const double* x, int incx, double* y, int incy);
extern void daxpy(int n, double a, const double* x, int incx, double* y, int incy);
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern void dscal(int n, double a, double* x, int incx);

/* External LINPACK functions */
extern int dtrsl(double* t, int ldt, int n, double* b, int job);

/* Constants */

/* Job codes for dtrsl (matching Go constants in linpack.go) */
#define SOLVE_LOWER_N 0  /* Solve L*x = b (lower triangular, no transpose) */
#define SOLVE_UPPER_N 1  /* Solve U*x = b (upper triangular, no transpose) */
#define SOLVE_LOWER_T 2  /* Solve L'*x = b (lower triangular, transpose) */
#define SOLVE_UPPER_T 3  /* Solve U'*x = b (upper triangular, transpose) */

/**
 * Subroutine reduceGradient (cmprlb)
 *
 * This subroutine computes r̃ᶜ = -Zᵀ(g + B(xᶜ - xₖ))
 *
 * Given:
 *   - xₖ current location (x)
 *   - gₖ the gradient value of f(x) (g)
 *   - xᶜ the Cauchy point (z)
 *   - Sₖ, Yₖ the correction matrices of Bₖ (ws, wy)
 *   - c = Wᵀ(xᶜ - x), computed during Cauchy point search
 *
 * The reduced gradient is computed as:
 *   r = -r̃ᶜ = -Zᵀ(g + θ(xᶜ-x) - WMc) = Zᵀ(-g - θ(xᶜ-x) + WMc)
 *
 * Where:
 *   W = [Y, θS]  (correction matrices)
 *   M = [-D    Lᵀ ]⁻¹
 *       [ L   θSᵀS]
 *
 * Matches Go implementation in subsapce.go reduceGradient function.
 */
int reduce_gradient(int n, int m, const double* x, const double* g,
                    const double* z, double* r, LbfgsbWorkspace* ws) {
    int i, j, k;
    int ptr;
    double mc1, mc2;
    
    int col = ws->col;
    int head = ws->head;
    int free = ws->free;
    double theta = ws->theta;
    int constrained = ws->constrained;
    
    /* Index array: index[0:free] contains indices of free variables */
    int* inx = ws->index;
    
    /* BFGS correction matrices */
    double* ws_arr = ws->ws;  /* S matrix (n x m) */
    double* wy = ws->wy;      /* Y matrix (n x m) */
    
    /* Workspace arrays:
     * c[2m:4m] = W'(x^c - x), computed during Cauchy point search
     * v[0:2m]  = M*c, temporary for bmv result
     */
    double* c = ws->wa + 2 * m;  /* c = W'(x^c - x) */
    double* v = ws->wa;          /* v = M*c (temporary) */
    
    /* Handle unconstrained case specially - matches Go exactly */
    if (!constrained && col > 0) {
        /* If the problem is unconstrained and col > 0, set r = -g */
        for (i = 0; i < n; i++) {
            r[i] = -g[i];
        }
        return 0;
    }
    
    /* Compute r = -θ(x^c - x) - g for free variables */
    for (i = 0; i < free; i++) {
        k = inx[i];  /* Index of free variable */
        r[i] = -theta * (z[k] - x[k]) - g[k];
    }
    
    /* If no BFGS corrections, we're done */
    if (col == 0) {
        return 0;
    }
    
    /* Compute v = M * c using bmv */
    int info = bmv(m, col, ws->sy, ws->wt, c, v);
    if (info != 0) {
        return info;
    }
    
    /* Compute r += W * M * c for free variables
     * 
     * W = [Y, θS], so:
     *   W * M * c = Y * (Mc)_1 + θS * (Mc)_2
     * 
     * For each free variable i with index k:
     *   r[i] += sum_j (Y[k,j] * v[j] + θ * S[k,j] * v[col+j])
     */
    ptr = head;
    for (j = 0; j < col; j++) {
        mc1 = v[j];              /* (Mc)_1[j] */
        mc2 = theta * v[col + j]; /* θ * (Mc)_2[j] */
        
        for (i = 0; i < free; i++) {
            k = inx[i];  /* Index of free variable */
            /* r[i] += Y[k,j] * mc1 + S[k,j] * mc2 */
            r[i] += wy[k * m + ptr] * mc1 + ws_arr[k * m + ptr] * mc2;
        }
        
        ptr = (ptr + 1) % m;
    }
    
    return 0;
}

/**
 * Subroutine optimalDirection (subsm)
 *
 * This subroutine computes an approximate solution of the subspace problem
 *
 *   m̃ₖ(d̃) ≡ d̃ᵀr̃ᶜ + ½d̃ᵀB̃ₖr̃ᶜ
 *
 * along the subspace unconstrained Newton direction
 *
 *   d̃ᵘ = -B̃ₖ⁻¹r̃ᶜ
 *
 * then backtrack towards the feasible region to obtain optimal direction (optional)
 *
 *   d̃* = α* × d̃ᵘ
 *
 * Given the L-BFGS matrix and the Sherman-Morrison formula
 *
 *   B̃ₖ = (1/θ)I - (1/θ)ZᵀW[ (I-(1/θ)MWᵀZZᵀW)⁻¹M ]WᵀZ(1/θ)
 *
 * With N ≡ I - (1/θ)MWᵀZZᵀW, the formula for the unconstrained Newton direction is
 *
 *   d̃ᵘ = (1/θ)r̃ᶜ + (1/θ²)ZᵀWN⁻¹MZᵀW
 *
 * Then form middle K = M⁻¹N = (N⁻¹M)⁻¹ to avoid inverting N (see formk)
 *
 *   d̃ᵘ = (1/θ)r̃ᶜ + (1/θ²)ZᵀWK⁻¹WᵀZr̃ᶜ
 *
 * Finally the computation of K⁻¹v could be replaced with solving v = Kx by factorization K = LELᵀ
 *
 * The K matrix factorization is:
 *   M⁻¹N = K = LELᵀ = [ LLᵀ           L⁻¹(-Laᵀ+Rzᵀ) ]
 *                     [ (-La+Rz)L⁻ᵀ   S'AA'Sθ       ]
 *
 * Matches Go implementation in subsapce.go optimalDirection function.
 */
int optimal_direction(int n, int m, const double* x, const double* g,
                      const double* lower, const double* upper,
                      const int* bound_type, double* z, double* r,
                      LbfgsbWorkspace* ws) {
    int i, j, k;
    int ptr;
    double dk, xk, span;
    double alpha, stp;
    int ibd;
    int projected;
    double sgn;
    
    int col = ws->col;
    int col2 = 2 * col;
    int head = ws->head;
    int free = ws->free;
    int m2 = 2 * m;
    double theta = ws->theta;
    
    /* If no free variables, nothing to do */
    if (free <= 0) {
        return 0;
    }
    
    /* Index array: index[0:free] contains indices of free variables (Z) */
    int* inx = ws->index;
    
    /* BFGS correction matrices */
    double* ws_arr = ws->ws;  /* S matrix (n x m) */
    double* wy = ws->wy;      /* Y matrix (n x m) */
    
    /* K matrix (LEL^T factorization stored in wn) */
    double* wn = ws->wn;
    
    /* Workspace arrays:
     * wv[0:2m] = K^{-1}W^T Zr̃^c (temporary workspace)
     * xp[0:n]  = safeguard for projected Newton direction
     */
    double* wv = ws->wa;      /* v = K^{-1}W^T Zr̃^c */
    double* xp = ws->xp;      /* Safeguard copy of z */
    
    /* d = r is the Newton direction (will be modified in place) */
    double* d = r;
    
    /* ========================================================================
     * Compute v = Wᵀ Z r̃ᶜ
     * 
     * W = [Y, θS], so:
     *   v_y[j] = Σᵢ (Y[k,j] * r[i]) for free variable i with index k
     *   v_s[j] = θ × Σᵢ (S[k,j] * r[i]) for free variable i with index k
     * ======================================================================== */
    
    ptr = head;
    for (j = 0; j < col; j++) {
        double yr = ZERO;
        double sr = ZERO;
        for (i = 0; i < free; i++) {
            k = inx[i];  /* Index of free variable */
            yr += wy[k * m + ptr] * d[i];
            sr += ws_arr[k * m + ptr] * d[i];
        }
        wv[j] = yr;
        wv[col + j] = theta * sr;
        ptr = (ptr + 1) % m;
    }
    
    /* ========================================================================
     * Compute K⁻¹v = (LELᵀ)⁻¹v = (L⁻ᵀE⁻¹L⁻¹)v
     * 
     * Lᵀ stored in the upper triangle of WN
     * E⁻¹ = [-I  0]⁻¹ = [-I  0]
     *       [ 0  I]     [ 0  I]
     * ======================================================================== */
    
    /* Compute L⁻¹v by solving Lx = (Lᵀ)ᵀx = v
     * Lᵀ is upper triangular, so we solve Lᵀᵀ x = v (job = SOLVE_UPPER_T for transpose) */
    if (dtrsl(wn, m2, col2, wv, SOLVE_UPPER_T) != 0) {
        return -1;  /* Singular triangular matrix */
    }
    
    /* Compute E⁻¹(L⁻¹v): negate first col elements */
    dscal(col, -ONE, wv, 1);
    
    /* Compute L⁻ᵀ(E⁻¹L⁻¹v) by solving Lᵀx = E⁻¹L⁻¹v
     * Lᵀ is upper triangular (job = SOLVE_UPPER_N for no transpose) */
    if (dtrsl(wn, m2, col2, wv, SOLVE_UPPER_N) != 0) {
        return -1;  /* Singular triangular matrix */
    }
    
    /* ========================================================================
     * Compute r̃ᶜ + (1/θ)ZᵀW(K⁻¹WᵀZr̃ᶜ)
     * 
     * d[i] += Σⱼ (Y[k,j] × wv[j] / θ + S[k,j] × wv[col+j])
     * ======================================================================== */
    
    ptr = head;
    for (j = 0; j < col; j++) {
        int js = col + j;
        for (i = 0; i < free; i++) {
            k = inx[i];  /* Index of free variable */
            d[i] += (wy[k * m + ptr] * wv[j] / theta) + (ws_arr[k * m + ptr] * wv[js]);
        }
        ptr = (ptr + 1) % m;
    }
    
    /* Scale r̃ᶜ + (1/θ)ZᵀWK⁻¹WᵀZr̃ᶜ by 1/θ 
     * Note: Go uses d[i] *= one / theta which is equivalent to d[i] /= theta */
    for (i = 0; i < free; i++) {
        d[i] *= ONE / theta;
    }
    
    /* ========================================================================
     * Perform projection along unconstrained Newton direction d̃ᵘ
     * Compute subspace minimizer x̂ = 𝚙𝚛𝚘𝚓(xᶜ + d̃ᵘ)
     * ======================================================================== */
    
    /* Save z to xp for safeguard */
    dcopy(n, z, 1, xp, 1);
    
    /* Project x^c + d̃^u onto feasible region */
    projected = 0;
    for (i = 0; i < free; i++) {
        k = inx[i];  /* Index of free variable */
        dk = d[i];
        xk = z[k];
        int bt = bound_type ? bound_type[k] : BOUND_NONE;
        
        switch (bt) {
            case BOUND_NONE:
                /* Unbound variable */
                z[k] = xk + dk;
                break;
            case BOUND_LOWER:
                z[k] = fmax(lower[k], xk + dk);
                if (z[k] == lower[k]) projected = 1;
                break;
            case BOUND_UPPER:
                z[k] = fmin(upper[k], xk + dk);
                if (z[k] == upper[k]) projected = 1;
                break;
            case BOUND_BOTH:
                z[k] = fmin(upper[k], fmax(lower[k], xk + dk));
                if (z[k] == lower[k] || z[k] == upper[k]) projected = 1;
                break;
        }
    }
    
    /* Store solution status in workspace */
    if (projected) {
        ws->word = SOLUTION_BEYOND_BOX;
    } else {
        ws->word = SOLUTION_WITHIN_BOX;
    }
    
    /* ========================================================================
     * Check sign of the directional derivative
     * sgn = (x̂ - xₖ)ᵀgₖ
     * 
     * If sgn > 0, the direction is not a descent direction, need to backtrack
     * ======================================================================== */
    
    sgn = ZERO;
    if (projected) {
        for (i = 0; i < n; i++) {
            sgn += (z[i] - x[i]) * g[i];  /* (x̂ - xₖ) × gₖ */
        }
    }
    
    /* ========================================================================
     * When the direction x̂ - xₖ is not a direction of strong descent,
     * truncate the path from xₖ to x̂ to satisfy the constraints
     * 
     * sgn ≤ 0  ⇒  d̃* = d̃ᵘ (keep current z)
     * sgn > 0  ⇒  d̃* = α* × d̃ᵘ (backtrack)
     * ======================================================================== */
    
    if (sgn > ZERO) {
        /* Restore z from xp - matches Go: copy(x[:n], xp[:n]) */
        dcopy(n, xp, 1, z, 1);
        
        /* Search positive optimal step
         * α* = 𝚖𝚊𝚡 { α : α ≤ 1, lᵢ - xᶜᵢ ≤ α × d̃ᵘᵢ ≤ uᵢ - xᶜᵢ (i ∈ 𝓕) }
         */
        alpha = ONE;
        ibd = 0;
        
        for (i = 0; i < free; i++) {
            k = inx[i];  /* Index of free variable */
            dk = d[i];
            int bt = bound_type ? bound_type[k] : BOUND_NONE;
            
            if (bt != BOUND_NONE) {
                stp = alpha;
                
                /* Match Go logic exactly:
                 * if dk < zero && bk.hint <= bndBoth (i.e., BOUND_LOWER or BOUND_BOTH)
                 * if dk > zero && bk.hint >= bndBoth (i.e., BOUND_UPPER or BOUND_BOTH)
                 */
                if (dk < ZERO && (bt == BOUND_LOWER || bt == BOUND_BOTH)) {
                    /* Moving towards lower bound */
                    span = lower[k] - z[k];
                    if (span >= ZERO) {
                        stp = ZERO;
                    } else if (dk * alpha < span) {
                        stp = span / dk;
                    }
                } else if (dk > ZERO && (bt == BOUND_UPPER || bt == BOUND_BOTH)) {
                    /* Moving towards upper bound */
                    span = upper[k] - z[k];
                    if (span <= ZERO) {
                        stp = ZERO;
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
        
        /* If alpha < 1, fix the blocking variable at its bound */
        if (alpha < ONE) {
            dk = d[ibd];
            k = inx[ibd];
            if (dk > ZERO) {
                z[k] = upper[k];
                d[ibd] = ZERO;
            } else if (dk < ZERO) {
                z[k] = lower[k];
                d[ibd] = ZERO;
            }
        }
        
        /* x̂ = xᶜ + d̃* = xᶜ + (α* × d̃ᵘ)
         *   x̂ᵢ = xᶜᵢ           if i ∉ 𝓕
         *   x̂ᵢ = xᶜᵢ + Zd̃*ᵢ   otherwise
         */
        for (i = 0; i < free; i++) {
            k = inx[i];  /* Index of free variable */
            z[k] += alpha * d[i];
        }
    }
    
    return 0;
}
