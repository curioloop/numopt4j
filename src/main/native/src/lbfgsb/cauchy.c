/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * L-BFGS-B Cauchy point computation module.
 * Based on the Go implementation in lbfgsb/cauchy.go.
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
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * Variable Status Constants (matches Go varWhere type)
 * ============================================================================
 *
 * where[i] records the status of the current x variables:
 *   - where[i] = -3 : xᵢ is free and has bounds, but is not moved
 *   - where[i] = -1 : xᵢ is always free, i.e., it has no bounds
 *   - where[i] =  0 : xᵢ is free and has bounds, and is moved
 *   - where[i] =  1 : xᵢ is fixed at lᵢ, and uᵢ ≠ lᵢ
 *   - where[i] =  2 : xᵢ is fixed at uᵢ, and uᵢ ≠ lᵢ
 *   - where[i] =  3 : xᵢ is always fixed, i.e., uᵢ=xᵢ=lᵢ
 */

#define VAR_NOT_MOVE  -3  /* xᵢ is free but won't move (gᵢ = 0) */
#define VAR_UNBOUND   -1  /* xᵢ has no bounds */
#define VAR_FREE       0  /* xᵢ is free with bounds */
#define VAR_AT_LOWER   1  /* xᵢ is at lower bound lᵢ */
#define VAR_AT_UPPER   2  /* xᵢ is at upper bound uᵢ */
#define VAR_FIXED      3  /* xᵢ is fixed (lᵢ = uᵢ) */

/* External BLAS functions */
extern void dcopy(int n, const double* x, int incx, double* y, int incy);
extern void daxpy(int n, double a, const double* x, int incx, double* y, int incy);
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern void dscal(int n, double a, double* x, int incx);

/* External LINPACK functions */
extern int dtrsl(double* t, int ldt, int n, double* b, int job);

/* Constants */

/* ============================================================================
 * Heap Sort Implementation (hpsolb)
 * ============================================================================ */

/**
 * Heap sort output minimum breakpoint (hpsolb)
 *
 * Given t[0:n] and order[0:n]:
 *   - Build min-heap on t[0:n] if sorted == 0
 *   - Swap the top element to the tail: t[0] ⇄ t[n-1]
 *   - Recover heap t[0:n-1] by shifting down t[0]
 *
 * After calling this function:
 *   - t[n-1] contains the minimum value that was at t[0]
 *   - t[0:n-1] is a valid min-heap
 *   - order array is updated correspondingly
 *
 * This matches the Go implementation heapSortOut in cauchy.go.
 */
void heap_sort_out(int n, double* t, int* order, int sorted) {
    int i, j, k;
    double val;
    int idx;
    double top_val;
    int top_idx;
    
    if (n <= 0) {
        return;
    }
    
    /* Build min-heap on t[0:n] if not already sorted */
    if (!sorted) {
        for (k = 1; k < n; k++) {
            /* Add t[k] to the heap t[0:k-1] */
            i = k;
            val = t[i];
            idx = order[i];
            
            /* Shift up: compare with parent and swap if smaller */
            while (i > 0 && i < n) {
                j = (i - 1) / 2;  /* Parent index */
                if (val < t[j]) {
                    /* Shift down the parent */
                    t[i] = t[j];
                    order[i] = order[j];
                    i = j;
                } else {
                    /* Already a heap */
                    break;
                }
            }
            t[i] = val;
            order[i] = idx;
        }
    }
    
    if (n > 1) {
        /* Pop the least (top) element of heap */
        top_val = t[0];
        top_idx = order[0];
        
        /* Move the bottom element to top: t[0] = t[n-1] and trim the heap to t[0:n-1] */
        val = t[n - 1];
        idx = order[n - 1];
        
        /* Shift down t[0] until heap property is recovered */
        i = 0;  /* t[i] is parent */
        for (;;) {
            j = 2 * i + 1;  /* Left child index */
            if (j < n) {
                /* Select the smaller child when right child is available */
                if (j + 1 < n && t[j + 1] < t[j]) {
                    j++;
                }
                if (t[j] < val) {
                    /* Shift up the smaller child */
                    t[i] = t[j];
                    order[i] = order[j];
                    i = j;
                } else {
                    /* Stop when parent is smaller than children */
                    break;
                }
            } else {
                break;
            }
        }
        
        /* Now t[0:n-1] is a heap */
        t[i] = val;
        order[i] = idx;
        
        /* Store the least element at t[n-1] */
        t[n - 1] = top_val;
        order[n - 1] = top_idx;
    }
}

/* ============================================================================
 * BMV Matrix-Vector Multiplication
 * ============================================================================ */

/**
 * BMV Matrix-Vector Multiplication: p = Mv
 *
 * Given 2m vector v = [v₁, v₂]ᵀ, calculate matrix product p = Mv with
 * 2m × 2m middle matrix:
 *
 *     M = [ -D    Lᵀ  ]⁻¹
 *         [  L   θSᵀS ]
 *
 * Algorithm:
 *
 * 1. Calculate upper triangular matrix Jᵀ by applying Cholesky factorization to
 *    symmetric positive definite matrix:
 *
 *      (θSᵀS + LD⁻¹Lᵀ) = JJᵀ
 *
 * 2. Reorder the blocks to get M⁻¹ = (AB)⁻¹ = B⁻¹A⁻¹
 *
 *     [ -D    Lᵀ  ] = [ D¹ᐟ²      O  ] [ -D¹ᐟ²  D⁻¹ᐟ²Lᵀ ]
 *     [  L   θSᵀS ]   [ -LD⁻¹ᐟ²   J  ] [  O     Jᵀ      ]
 *
 * 3. Calculate p = Bv by solving B⁻¹p = v
 *
 *     [ D¹ᐟ²      O  ] [ p₁ ] = [ v₁ ]
 *     [ -LD⁻¹ᐟ²   J  ] [ p₂ ]   [ v₂ ]
 *
 * 4. Calculate p = ABv = Mv by solving A⁻¹p = Bv
 *
 *     [ -D¹ᐟ²  D⁻¹ᐟ²Lᵀ ] [ p₁ ] = [ ṗ₁ ]
 *     [  O     Jᵀ      ] [ p₂ ]   [ ṗ₂ ]
 *
 * Matrices D and L are calculated from SᵀY:
 *   D = diag{sᵢᵀyᵢ} for i = 1,...,col
 *   Lᵢⱼ = sᵢᵀyⱼ for i > j (strictly lower triangular)
 *
 * This matches the Go implementation bmv in cauchy.go.
 *
 * @param m      Maximum number of corrections
 * @param col    Current number of corrections stored
 * @param sy     SᵀY matrix (m × m)
 * @param wt     JJᵀ Cholesky factor (m × m)
 * @param v      Input vector (2*col)
 * @param p      Output vector p = Mv (2*col)
 * @return       0 on success, negative on error
 */
int bmv(int m, int col, const double* sy, const double* wt,
        const double* v, double* p) {
    int i, j;
    double sum;
    
    if (col == 0) {
        return 0;
    }
    
    /* Matrices D and L can be calculated from SᵀY:
     *   D = diag{sᵢᵀyᵢ} for i = 1,...,col
     *   Lᵢⱼ = sᵢᵀyⱼ for i > j (strictly lower triangular)
     */
    
    /* Pointers to v₁, v₂ and p₁, p₂ */
    const double* v1 = v;
    const double* v2 = v + col;
    double* p1 = p;
    double* p2 = p + col;
    
    /* ========================================================================
     * PART I: Solve [ D¹ᐟ²      O  ] [ p₁ ] = [ v₁ ]
     *               [ -LD⁻¹ᐟ²   J  ] [ p₂ ]   [ v₂ ]
     *
     * From first row:  D¹ᐟ²p₁ = v₁  ⇒  p₁ = D⁻¹ᐟ²v₁
     * From second row: -LD⁻¹ᐟ²p₁ + Jp₂ = v₂  ⇒  p₂ = J⁻¹(v₂ + LD⁻¹v₁)
     * ======================================================================== */
    
    /* Calculate v₂ + LD⁻¹v₁ and store in p₂ */
    p2[0] = v2[0];
    for (i = 1; i < col; i++) {
        /* Calculate (LD⁻¹v₁)ᵢ = ∑(Lᵢⱼ * v₁ⱼ / Dⱼⱼ) for j < i */
        sum = ZERO;
        for (j = 0; j < i; j++) {
            /* Lᵢⱼ = sy[i*m + j] (lower triangular part)
             * Dⱼⱼ = sy[j*m + j] (diagonal) */
            sum += sy[i * m + j] * v1[j] / sy[j * m + j];
        }
        /* p₂ᵢ = v₂ᵢ + (LD⁻¹v₁)ᵢ */
        p2[i] = v2[i] + sum;
    }
    
    /* Calculate p₂ by solving triangular system Jp₂ = v₂ + LD⁻¹v₁
     * J is upper triangular stored in wt
     * Use job=11 for solving Jᵀx = b (matches Go solveUpperT)
     */
    if (dtrsl((double*)wt, m, col, p2, 11) != 0) {
        return -1;  /* Singular triangular matrix */
    }
    
    /* Solve p₁ = D⁻¹ᐟ²v₁ */
    for (i = 0; i < col; i++) {
        double d_ii = sy[i * m + i];  /* Dᵢᵢ = sy[i*m + i] */
        if (d_ii <= ZERO) {
            return -2;  /* Non-positive diagonal element */
        }
        p1[i] = v1[i] / sqrt(d_ii);
    }
    
    /* ========================================================================
     * PART II: Solve [ -D¹ᐟ²  D⁻¹ᐟ²Lᵀ ] [ p₁ ] = [ ṗ₁ ]
     *                [  O     Jᵀ      ] [ p₂ ]   [ ṗ₂ ]
     *
     * From second row: Jᵀp₂ = ṗ₂  ⇒  p₂ = J⁻ᵀṗ₂
     * From first row:  -D¹ᐟ²p₁ + D⁻¹ᐟ²Lᵀp₂ = ṗ₁
     *                  ⇒  p₁ = -D⁻¹ᐟ²(ṗ₁ - D⁻¹ᐟ²Lᵀp₂)
     *                      = -D⁻¹ᐟ²ṗ₁ + D⁻¹Lᵀp₂
     * ======================================================================== */
    
    /* Calculate p₂ by solving Jᵀp₂ = ṗ₂
     * J is upper triangular stored in wt
     * Use job=10 for solving Jx = b (matches Go solveUpperN)
     */
    if (dtrsl((double*)wt, m, col, p2, 10) != 0) {
        return -1;  /* Singular triangular matrix */
    }
    
    /* Calculate p₁ = -D⁻¹ᐟ²ṗ₁ + D⁻¹Lᵀp₂ */
    for (i = 0; i < col; i++) {
        double d_ii = sy[i * m + i];
        /* First term: -D⁻¹ᐟ²ṗ₁ */
        p1[i] /= -sqrt(d_ii);
    }
    
    for (i = 0; i < col; i++) {
        /* Calculate (D⁻¹Lᵀp₂)ᵢ = ∑(Lⱼᵢ * p₂ⱼ / Dᵢᵢ) for j > i
         * Note: Lᵀ has Lⱼᵢ in position (i, j) where j > i */
        sum = ZERO;
        for (j = i + 1; j < col; j++) {
            /* Lⱼᵢ = sy[j*m + i] (L is stored in lower triangle of sy) */
            sum += sy[j * m + i] * p2[j] / sy[i * m + i];
        }
        /* Add to p₁ᵢ */
        p1[i] += sum;
    }
    
    return 0;
}

/* ============================================================================
 * Free Variable Identification (freev)
 * ============================================================================ */

/**
 * Count entering and leaving variables and build the index set of free variables (freev)
 *
 * This subroutine counts the entering and leaving variables when iter > 0,
 * and finds the index set of free and active variables at the GCP.
 *
 * Index arrays:
 *   - index[0:free] are indices of free variables
 *   - index[free:n] are indices of bound variables
 *
 * State arrays (for tracking changes):
 *   - state[0:enter] have changed from bound to free
 *   - state[leave:n] have changed from free to bound
 *
 * This matches the Go implementation freeVar in cauchy.go.
 *
 * @param n   Problem dimension
 * @param ws  Workspace containing iteration state
 * @return    1 if K matrix needs recomputation, 0 otherwise
 */
int free_var(int n, LbfgsbWorkspace* ws) {
    int i, k;
    int enter, leave;
    int free_count, active_count;
    
    int* index = ws->index;       /* index[0:n] for free/bound variables */
    int* state = ws->index + n;   /* index[n:2n] for entering/leaving variables */
    int* iwhere = ws->iwhere;
    
    int iter = ws->iter;
    int constrained = ws->constrained;
    int old_free = ws->free;
    
    enter = 0;
    leave = n;
    
    /* Count entering and leaving variables for iter > 0 */
    if (iter > 0 && constrained) {
        /* Check variables that were free in previous iteration */
        for (i = 0; i < old_free; i++) {
            k = index[i];
            if (iwhere[k] > VAR_FREE) {
                /* Variable is now at a bound - leaving free set */
                leave--;
                state[leave] = k;
            }
        }
        
        /* Check variables that were at bounds in previous iteration */
        for (i = old_free; i < n; i++) {
            k = index[i];
            if (iwhere[k] <= VAR_FREE) {
                /* Variable is now free - entering free set */
                state[enter] = k;
                enter++;
            }
        }
    }
    
    ws->enter = enter;
    ws->leave = leave;
    
    /* Build the index set of free and active variables at the GCP */
    free_count = 0;
    active_count = n;
    
    for (i = 0; i < n; i++) {
        if (iwhere[i] <= VAR_FREE) {
            /* Free variable (VAR_FREE or VAR_UNBOUND) */
            index[free_count] = i;
            free_count++;
        } else {
            /* Bound variable (VAR_AT_LOWER, VAR_AT_UPPER, or VAR_FIXED) */
            active_count--;
            index[active_count] = i;
        }
    }
    
    ws->free = free_count;
    ws->active = n - free_count;
    
    /* Return whether K matrix needs to be recomputed */
    return (leave < n) || (enter > 0) || ws->updated;
}

/* ============================================================================
 * Cauchy Point Computation
 * ============================================================================ */

/**
 * Compute the Generalized Cauchy Point (GCP) by piecewise linear path search.
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
 * This subroutine computes the GCP, defined as the first local minimizer of mₖ(x),
 * along the piecewise linear path 𝚙𝚛𝚘𝚓(xₖ - tgₖ) obtained by projecting points
 * along the steepest descent direction xₖ - tgₖ onto the feasible region.
 *
 * Breakpoint computation:
 *   tᵢ = (xᵢ - uᵢ)/gᵢ  if gᵢ < 0
 *   tᵢ = (xᵢ - lᵢ)/gᵢ  if gᵢ > 0
 *   tᵢ = ∞             otherwise
 *
 * Search direction:
 *   dᵢ = 0    if tᵢ = 0
 *   dᵢ = -gᵢ  otherwise
 *
 * Corrections of B:
 *   W = [Y  θS]   M = [ -D    Lᵀ  ]⁻¹
 *                     [  L   θSᵀS ]
 *
 * Derivative updates at each breakpoint:
 *   f′ = f′ + f″Δtᵢ + gᵢ² + θgᵢzᵢ - gᵢwᵀᵢMc
 *   f″ = f″ - θgᵢ² - 2gᵢwᵀᵢMp - gᵢ²wᵀᵢMwᵢ
 *
 * Final return:
 *   - GCP : xᶜ
 *   - Cauchy direction : dᶜ = 𝚙𝚛𝚘𝚓(xₖ - tgₖ) - xₖ
 *
 * This matches the Go implementation cauchy in cauchy.go.
 *
 * @param n          Problem dimension
 * @param m          Maximum number of L-BFGS corrections
 * @param x          Current point xₖ
 * @param g          Gradient gₖ
 * @param lower      Lower bounds l
 * @param upper      Upper bounds u
 * @param bound_type Bound type for each variable
 * @param z          Output: GCP xᶜ
 * @param ws         Workspace containing iteration state
 * @return           0 on success, negative on error
 */
int cauchy_point(int n, int m, const double* x, const double* g,
                 const double* lower, const double* upper,
                 const int* bound_type, double* z,
                 LbfgsbWorkspace* ws) {
    int i, j;
    double neg_g, tl, tu;
    double f1, f2, org_f2;
    double bk_min, delta_min, delta_sum, t_delta;
    int idx_min, n_free, n_break, n_left;
    int bounded;
    
    int col = ws->col;
    int col2 = 2 * col;
    double theta = ws->theta;
    
    /* Search direction d */
    double* d = ws->d;
    
    /* Breakpoint time array and order array
     * order stores the breakpoints in the piecewise linear path and free variables:
     *   - order[0:left] are indices of breakpoints which have not been encountered
     *   - order[left:break] are indices of encountered breakpoints
     *   - order[free:n] are indices of variables with no bounds along search direction
     */
    double* t = ws->t;
    int* order = ws->index + n;  /* Use second half of index array for order */
    int* iwhere = ws->iwhere;
    
    /* Workspace arrays from wa (8*m total):
     * p[0:2m]   = Wᵀd = [Yᵀd, θSᵀd]ᵀ
     * c[2m:4m]  = Wᵀ(xᶜ - x)
     * w[4m:6m]  = Wᵢ (row of W at breakpoint)
     * v[6m:8m]  = M? (temporary for bmv)
     */
    double* p = ws->wa;
    double* c = ws->wa + 2 * m;
    double* w = ws->wa + 4 * m;
    double* v = ws->wa + 6 * m;
    
    /* Check if projected gradient norm is zero: ‖𝚙𝚛𝚘𝚓 g‖∞ = 0 → ∀ gᵢ = 0 */
    if (ws->sbg_norm <= ZERO) {
        /* xᶜ = x */
        dcopy(n, x, 1, z, 1);
        return 0;
    }
    
    /* Initialize p to zero */
    for (i = 0; i < col2; i++) {
        p[i] = ZERO;
    }
    
    /* Initialize f′ = gᵀd = -dᵀd = ∑(-dᵢ²)
     * Initialize f″ = -θf′ - pᵀMp */
    f1 = ZERO;
    f2 = ZERO;
    
    n_free = n;      /* Number of free variables */
    n_break = 0;     /* Number of breakpoints */
    bk_min = ZERO;
    idx_min = 0;
    bounded = 1;     /* Assume all variables are bounded */
    
    /* Loop over all variables to determine:
     * 1. Variable status (iwhere)
     * 2. Search direction d
     * 3. Breakpoints t
     * 4. Initialize p = Wᵀd
     */
    for (i = 0; i < n; i++) {
        neg_g = -g[i];
        int bt = bound_type ? bound_type[i] : BOUND_NONE;
        
        tl = ZERO;
        tu = ZERO;
        
        if (iwhere[i] != VAR_FIXED && iwhere[i] != VAR_UNBOUND) {
            /* If xᵢ is not a constant and has bounds, compute xᵢ - lᵢ and uᵢ - xᵢ */
            if (bt == BOUND_LOWER || bt == BOUND_BOTH) {
                tl = x[i] - lower[i];
            }
            if (bt == BOUND_UPPER || bt == BOUND_BOTH) {
                tu = upper[i] - x[i];
            }
            
            iwhere[i] = VAR_FREE;
            
            /* If a variable is close enough to a bound we treat it as at bound */
            if ((bt == BOUND_LOWER || bt == BOUND_BOTH) && tl <= ZERO) {
                if (neg_g <= ZERO) {
                    /* xᵢ ≤ lᵢ and -gᵢ ≤ 0 means xₖ₊₁ᵢ = xₖᵢ - gₖᵢ < lᵢ */
                    iwhere[i] = VAR_AT_LOWER;
                }
            } else if ((bt == BOUND_UPPER || bt == BOUND_BOTH) && tu <= ZERO) {
                if (neg_g >= ZERO) {
                    /* xᵢ ≥ uᵢ and -gᵢ ≥ 0 means xₖ₊₁ᵢ = xₖᵢ - gₖᵢ > uᵢ */
                    iwhere[i] = VAR_AT_UPPER;
                }
            } else {
                if (fabs(neg_g) <= ZERO) {
                    /* gᵢ = 0, variable won't move */
                    iwhere[i] = VAR_NOT_MOVE;
                }
            }
        }
        
        /* Set search direction and update p */
        if (iwhere[i] != VAR_FREE && iwhere[i] != VAR_UNBOUND) {
            /* Fixed variable: dᵢ = 0 */
            d[i] = ZERO;
        } else {
            /* Free variable: dᵢ = -gᵢ */
            d[i] = neg_g;
            f1 -= neg_g * neg_g;  /* f′ += -dᵢ² */
            
            /* Update p = Wᵀd:
             * pᵧ[j] += wy[i,j] * (-gᵢ)
             * pₛ[j] += ws[i,j] * (-gᵢ)
             */
            double* py = p;
            double* ps = p + col;
            int ptr = ws->head;
            for (j = 0; j < col; j++) {
                py[j] += ws->wy[i * m + ptr] * neg_g;
                ps[j] += ws->ws[i * m + ptr] * neg_g;
                ptr = (ptr + 1) % m;
            }
            
            /* Compute breakpoint for this variable */
            if ((bt == BOUND_LOWER || bt == BOUND_BOTH) && bt != BOUND_NONE && neg_g < ZERO) {
                /* xᵢ + dᵢ is bounded below, compute tᵢ = (xᵢ - lᵢ) / (-dᵢ) */
                order[n_break] = i;
                t[n_break] = tl / (-neg_g);
                if (n_break == 0 || t[n_break] < bk_min) {
                    bk_min = t[n_break];
                    idx_min = n_break;
                }
                n_break++;
            } else if ((bt == BOUND_UPPER || bt == BOUND_BOTH) && neg_g > ZERO) {
                /* xᵢ + dᵢ is bounded above, compute tᵢ = (uᵢ - xᵢ) / dᵢ */
                order[n_break] = i;
                t[n_break] = tu / neg_g;
                if (n_break == 0 || t[n_break] < bk_min) {
                    bk_min = t[n_break];
                    idx_min = n_break;
                }
                n_break++;
            } else {
                /* xᵢ + dᵢ is not bounded */
                n_free--;
                order[n_free] = i;
                if (fabs(neg_g) > ZERO) {
                    bounded = 0;
                }
            }
        }
    }
    
    /* Complete initialization of p for θ ≠ 1 */
    if (theta != ONE) {
        double* ps = p + col;
        dscal(col, theta, ps, 1);
    }
    
    /* Initialize GCP: xᶜ = x */
    dcopy(n, x, 1, z, 1);
    
    /* If d is zero vector, return with initial xᶜ as GCP */
    if (n_break == 0 && n_free == n) {
        return 0;
    }
    
    /* Initialize c = Wᵀ(xᶜ - x) = 0 */
    for (i = 0; i < col2; i++) {
        c[i] = ZERO;
    }
    
    /* Initialize f″ = -θf′ */
    f2 = -theta * f1;
    org_f2 = f2;
    
    /* Compute f″ -= pᵀMp using bmv */
    if (col > 0) {
        int info = bmv(m, col, ws->sy, ws->wt, p, v);
        if (info != 0) {
            return info;
        }
        f2 -= ddot(col2, v, 1, p, 1);
    }
    
    /* Δtₘᵢₙ = -f′/f″ */
    delta_min = -f1 / f2;
    delta_sum = ZERO;
    
    /* Search along piecewise linear path */
    int found = (n_break == 0);
    n_left = n_break;
    
    for (int iter = 1; n_left > 0; iter++) {
        int t_idx;
        double t_val, t_old;
        
        if (iter == 1) {
            /* Use the smallest breakpoint found during initialization */
            t_val = bk_min;
            t_idx = order[idx_min];
            t_old = ZERO;
        } else {
            if (iter == 2) {
                /* Swap the used smallest breakpoint with the last one before heapsort */
                int n_last = n_break - 1;
                if (idx_min != n_last) {
                    double tmp_t = t[idx_min];
                    int tmp_o = order[idx_min];
                    t[idx_min] = t[n_last];
                    order[idx_min] = order[n_last];
                    t[n_last] = tmp_t;
                    order[n_last] = tmp_o;
                }
            }
            /* Update heap structure (if iter=2, build heap) */
            heap_sort_out(n_left, t, order, iter > 2);
            t_old = t[n_left];
            t_val = t[n_left - 1];
            t_idx = order[n_left - 1];
        }
        
        /* Compute dt = t[n_left] - t[n_left + 1] */
        t_delta = t_val - t_old;
        
        /* If minimizer is within this interval (Δtₘᵢₙ < Δtᵢ), locate GCP and return */
        if (delta_min < t_delta) {
            found = 1;
            break;
        }
        
        /* Fix one variable and reset its d component to zero */
        delta_sum += t_delta;
        n_left--;
        
        double d_break = d[t_idx];           /* -gᵢ */
        double d2_break = d_break * d_break; /* gᵢ² */
        d[t_idx] = ZERO;                     /* dᵢ = 0 */
        
        /* Update xᶜ and variable status */
        if (d_break > ZERO) {
            z[t_idx] = upper[t_idx];         /* xᶜᵢ = uᵢ (dᵢ > 0) */
            iwhere[t_idx] = VAR_AT_UPPER;
        } else {
            z[t_idx] = lower[t_idx];         /* xᶜᵢ = lᵢ (dᵢ < 0) */
            iwhere[t_idx] = VAR_AT_LOWER;
        }
        double z_break = z[t_idx] - x[t_idx];  /* zᵢ = xᶜᵢ - xᵢ */
        
        /* All n variables are fixed, return with xᶜ as GCP */
        if (n_left == 0 && n_break == n) {
            delta_min = t_delta;
            break;
        }
        
        /* Update derivative information:
         * f′ = f′ + f″Δtᵢ + gᵢ² + θgᵢzᵢ - gᵢwᵀᵢMc
         * f″ = f″ - θgᵢ² - 2gᵢwᵀᵢMp - gᵢ²wᵀᵢMwᵢ
         */
        f1 += f2 * t_delta + d2_break - theta * d_break * z_break;
        f2 -= theta * d2_break;
        
        /* Process matrix product with middle matrix M */
        if (col > 0) {
            /* c = c + pΔtᵢ */
            daxpy(col2, t_delta, p, 1, c, 1);
            
            /* w = Wᵢ (row of W at breakpoint, 2m elements) */
            double* w1 = w;
            double* w2 = w + col;
            int ptr = ws->head;
            for (j = 0; j < col; j++) {
                w1[j] = ws->wy[t_idx * m + ptr];        /* Yᵢ */
                w2[j] = theta * ws->ws[t_idx * m + ptr]; /* θSᵢ */
                ptr = (ptr + 1) % m;
            }
            
            /* v = Mw (2m) */
            int info = bmv(m, col, ws->sy, ws->wt, w, v);
            if (info != 0) {
                return info;
            }
            
            double wmc = ddot(col2, c, 1, v, 1);  /* wᵀMc */
            double wmp = ddot(col2, p, 1, v, 1);  /* wᵀMp */
            double wmw = ddot(col2, w, 1, v, 1);  /* wᵀMw */
            
            /* p = p + (-gᵢ)w */
            daxpy(col2, -d_break, w, 1, p, 1);
            
            f1 += d_break * wmc;                      /* += -gᵢwᵀᵢMc */
            f2 += 2.0 * d_break * wmp - d2_break * wmw;  /* += -2gᵢwᵀᵢMp - gᵢ²wᵀᵢMwᵢ */
        }
        
        /* Ensure f″ doesn't become too small */
        f2 = fmax(EPS * org_f2, f2);
        delta_min = -f1 / f2;  /* Δtₘᵢₙ = -f′/f″ */
        
        if (n_left == 0 && bounded) {
            f1 = ZERO;
            f2 = ZERO;
            delta_min = ZERO;
        }
    }
    
    /* Handle remaining variables */
    if (n_left == 0 || found) {
        delta_min = fmax(delta_min, ZERO);  /* Δtₘᵢₙ = max(Δtₘᵢₙ, 0) */
        delta_sum += delta_min;              /* tₒₗₐ = tₒₗₐ + Δtₘᵢₙ */
        
        /* Move free variables and variables whose breakpoints haven't been reached:
         * xᶜᵢ = xᵢ + tₒₗₐ * dᵢ (for dᵢ ≠ 0)
         */
        daxpy(n, delta_sum, d, 1, z, 1);
    }
    
    /* Update c = c + Δtₘᵢₙ * p = Wᵀ(xᶜ - x)
     * which will be used in computing r = Zᵀ(B(xᶜ - x) + g)
     */
    if (col > 0) {
        daxpy(col2, delta_min, p, 1, c, 1);
    }
    
    return 0;
}
