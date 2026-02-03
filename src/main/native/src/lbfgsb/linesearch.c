/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * L-BFGS-B line search functions.
 * Based on the Go implementation in lbfgsb/linesearch.go.
 * 
 * This implementation matches the Go version exactly.
 *
 * Line Search Algorithm
 * =====================
 * 
 * Perform a line search along dₖ subject to the bounds on the problem.
 * The λₖ starts with the unit steplength and ensures fₖ₊₁ = f(xₖ + λₖdₖ),
 * gₖ₊₁ = f′ₖ₊₁ satisfies the strong Wolfe conditions:
 *
 *   Sufficient decrease condition (Armijo):
 *     fₖ₊₁ ≤ fₖ + α·λₖ·gₖᵀdₖ     (α = 10⁻³)
 *
 *   Curvature condition:
 *     |gₖ₊₁ᵀdₖ| ≤ β·|gₖᵀdₖ|      (β = 0.9)
 *
 * where:
 *   fₖ     - function value at current iterate xₖ
 *   gₖ     - gradient at current iterate xₖ
 *   dₖ     - search direction
 *   λₖ     - step length (to be determined)
 *   α      - sufficient decrease parameter (typically 10⁻³)
 *   β      - curvature parameter (typically 0.9)
 *
 * The step update formula is:
 *   xₖ₊₁ = xₖ + λₖdₖ
 *
 * References:
 *   J.J. Moré and D.J. Thuente, "Line Search Algorithms with Guaranteed
 *   Sufficient Decrease", ACM Transactions on Mathematical Software,
 *   Vol. 20, No. 3, September 1994, pp. 286-307.
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * Line Search Constants (matching Go linesearch.go)
 * 
 * These parameters control the Wolfe conditions:
 *   α (alpha) - sufficient decrease parameter for Armijo condition
 *   β (beta)  - curvature parameter for curvature condition
 *   ε (eps)   - relative tolerance for step convergence
 * ============================================================================ */

#define SEARCH_NO_BND    1.0e+10   /* searchNoBnd - maximum step when unconstrained */
#define SEARCH_ALPHA     1.0e-3    /* searchAlpha - α: sufficient decrease parameter */
#define SEARCH_BETA      0.9       /* searchBeta  - β: curvature parameter */
#define SEARCH_EPS       0.1       /* searchEps   - ε: relative tolerance */

/* Error codes are defined in optimizer.h:
 * ERR_NONE = 0, ERR_DERIVATIVE = -4, ERR_LINE_SEARCH_TOL = -7 */

/* ============================================================================
 * Minpack Constants and Types (matching Go minpack.go)
 * ============================================================================ */

#define MINPACK_P5         0.5
#define MINPACK_P66        0.66
#define MINPACK_XTRAP_LOWER 1.1
#define MINPACK_XTRAP_UPPER 4.0

#define MINPACK_STAGE_ARMIJO 1
#define MINPACK_STAGE_WOLFE  2

#define MINPACK_SEARCH_START  0
#define MINPACK_SEARCH_CONV   (1 << 5)
#define MINPACK_SEARCH_FG     (1 << 6)
#define MINPACK_SEARCH_ERROR  (1 << 7)
#define MINPACK_SEARCH_WARN   (1 << 8)

#define MINPACK_ERR_OVER_LOWER   (MINPACK_SEARCH_ERROR | 1)
#define MINPACK_ERR_OVER_UPPER   (MINPACK_SEARCH_ERROR | 2)
#define MINPACK_ERR_NEG_INIT_G   (MINPACK_SEARCH_ERROR | 3)
#define MINPACK_ERR_NEG_ALPHA    (MINPACK_SEARCH_ERROR | 4)
#define MINPACK_ERR_NEG_BETA     (MINPACK_SEARCH_ERROR | 5)
#define MINPACK_ERR_NEG_EPS      (MINPACK_SEARCH_ERROR | 6)
#define MINPACK_ERR_LOWER        (MINPACK_SEARCH_ERROR | 7)
#define MINPACK_ERR_UPPER        (MINPACK_SEARCH_ERROR | 8)

#define MINPACK_WARN_ROUND_ERR   (MINPACK_SEARCH_WARN | 9)
#define MINPACK_WARN_REACH_EPS   (MINPACK_SEARCH_WARN | 10)
#define MINPACK_WARN_REACH_MAX   (MINPACK_SEARCH_WARN | 11)
#define MINPACK_WARN_REACH_MIN   (MINPACK_SEARCH_WARN | 12)

typedef struct {
    double alpha, beta, eps, lower, upper;
} MinpackSearchTol;

typedef struct {
    int bracket, stage;
    double g0, gx, gy, f0, fx, fy, stx, sty;
    double width[2], bound[2];
} MinpackSearchCtx;

/* Forward declaration */
double minpack_scalar_search(double f, double g, double stp, int* task,
                             const MinpackSearchTol* tol, MinpackSearchCtx* ctx);

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

/* ============================================================================
 * BLAS-like Helper Functions (inline for performance)
 * ============================================================================ */

/**
 * Compute dot product of two vectors: result = x·y
 */
static double ddot(int n, const double* x, int incx, const double* y, int incy) {
    double result = 0.0;
    int i;
    
    if (n <= 0) return 0.0;
    
    if (incx == 1 && incy == 1) {
        for (i = 0; i < n; i++) {
            result += x[i] * y[i];
        }
    } else {
        int ix = 0, iy = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        for (i = 0; i < n; i++) {
            result += x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return result;
}

/**
 * Copy vector: y = x
 */
static void dcopy(int n, const double* x, int incx, double* y, int incy) {
    int i;
    
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        memcpy(y, x, (size_t)n * sizeof(double));
    } else {
        int ix = 0, iy = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        for (i = 0; i < n; i++) {
            y[iy] = x[ix];
            ix += incx;
            iy += incy;
        }
    }
}

/* ============================================================================
 * Public Functions
 * ============================================================================ */

/**
 * init_line_search - Initialize line search parameters
 * 
 * This matches the Go implementation in linesearch.go initLineSearch function.
 *
 * Computes:
 *   ‖d‖₂ = √(dᵀd)     - Euclidean norm of search direction
 *   λₘₐₓ              - Maximum step length respecting bounds
 *   λ₀                - Initial step length
 *
 * Initial step selection:
 *   - First iteration (not boxed): λ₀ = min(1/‖d‖₂, λₘₐₓ)
 *   - Otherwise: λ₀ = 1
 *
 * Maximum step computation for bounded variables:
 *   For each variable xᵢ with bound constraint:
 *     - If dᵢ < 0 and has lower bound: λₘₐₓ = min(λₘₐₓ, (lᵢ - xᵢ)/dᵢ)
 *     - If dᵢ > 0 and has upper bound: λₘₐₓ = min(λₘₐₓ, (uᵢ - xᵢ)/dᵢ)
 */
double init_line_search(int n, int m, const double* x,
                        const double* lower, const double* upper,
                        const int* bound_type, LbfgsbWorkspace* ws,
                        int* out_task) {
    int i;
    double d_i, span;
    double step_max;
    double stp;
    
    const double* d = ws->d;
    
    /* Compute d·d and ||d||_2 */
    /* ‖d‖₂ = √(dᵀd) - Euclidean norm of search direction */
    ws->d_sqrt = ddot(n, d, 1, d, 1);  /* dᵀd */
    ws->d_norm = sqrt(ws->d_sqrt);      /* ‖d‖₂ */
    
    /* Determine the maximum step length λₘₐₓ */
    step_max = SEARCH_NO_BND;
    
    if (ws->constrained) {
        if (ws->iter == 0) {
            step_max = ONE;
        } else {
            /* Compute maximum step that keeps x within bounds:
             * For each bounded variable, compute the step to the bound
             * and take the minimum across all variables. */
            for (i = 0; i < n; i++) {
                int bt = bound_type ? bound_type[i] : BOUND_NONE;
                
                if (bt != BOUND_NONE) {
                    d_i = d[i];
                    
                    /* Check lower bound: dᵢ < 0 and has lower bound
                     * λₘₐₓ = min(λₘₐₓ, (lᵢ - xᵢ)/dᵢ) */
                    if (d_i < ZERO && (bt == BOUND_LOWER || bt == BOUND_BOTH)) {
                        span = lower[i] - x[i];  /* lᵢ - xᵢ */
                        if (span >= ZERO) {
                            step_max = ZERO;  /* Variable fixed at lower bound */
                        } else if (d_i * step_max < span) {
                            step_max = span / d_i;  /* Constrain step to bound */
                        }
                    }
                    /* Check upper bound: dᵢ > 0 and has upper bound
                     * λₘₐₓ = min(λₘₐₓ, (uᵢ - xᵢ)/dᵢ) */
                    else if (d_i > ZERO && (bt == BOUND_UPPER || bt == BOUND_BOTH)) {
                        span = upper[i] - x[i];  /* uᵢ - xᵢ */
                        if (span <= ZERO) {
                            step_max = ZERO;  /* Variable fixed at upper bound */
                        } else if (d_i * step_max > span) {
                            step_max = span / d_i;  /* Constrain step to bound */
                        }
                    }
                }
            }
        }
    }
    
    /* Initialize search tolerances for Wolfe conditions:
     *   α (alpha) - sufficient decrease: f(λ) ≤ f(0) + α·λ·f′(0)
     *   β (beta)  - curvature: |f′(λ)| ≤ β·|f′(0)|
     *   ε (eps)   - relative tolerance for convergence */
    ws->search_tol.alpha = SEARCH_ALPHA;
    ws->search_tol.beta = SEARCH_BETA;
    ws->search_tol.eps = SEARCH_EPS;
    ws->search_tol.lower = ZERO;
    ws->search_tol.upper = step_max;
    
    /* Set initial step length λ₀:
     *   - First iteration (not boxed): λ₀ = min(1/‖d‖₂, λₘₐₓ)
     *   - Otherwise: λ₀ = 1 */
    if (ws->iter == 0 && !ws->boxed) {
        /* First iteration and not all variables boxed: use scaled step */
        stp = ONE / ws->d_norm;  /* λ₀ = 1/‖d‖₂ */
        if (stp > step_max) {
            stp = step_max;
        }
    } else {
        stp = ONE;  /* λ₀ = 1 */
    }
    
    /* Initialize line search state */
    ws->num_eval = 0;
    ws->num_back = 0;
    
    /* Set task to SearchStart */
    *out_task = MINPACK_SEARCH_START;
    
    return stp;
}

/**
 * perform_line_search - Perform line search along search direction
 * 
 * This matches the Go implementation in linesearch.go performLineSearch function.
 *
 * Performs a line search along dₖ to find a step λₖ satisfying the strong
 * Wolfe conditions:
 *
 *   Sufficient decrease (Armijo condition):
 *     f(xₖ + λₖdₖ) ≤ f(xₖ) + α·λₖ·gₖᵀdₖ
 *
 *   Curvature condition:
 *     |g(xₖ + λₖdₖ)ᵀdₖ| ≤ β·|gₖᵀdₖ|
 *
 * where:
 *   gₖᵀdₖ  - directional derivative at xₖ (must be negative for descent)
 *   α      - sufficient decrease parameter (10⁻³)
 *   β      - curvature parameter (0.9)
 *
 * The new iterate is computed as:
 *   xₖ₊₁ = xₖ + λₖdₖ
 *
 * Or equivalently using stored values:
 *   xₖ₊₁ = λₖdₖ + t    (where t = xₖ before line search)
 *
 * Special case: if λₖ = 1, use the Cauchy point z directly:
 *   xₖ₊₁ = xᶜ
 */
int perform_line_search(int n, double* x, double f, const double* g,
                        double* stp, int* task,
                        LbfgsbWorkspace* ws, int* out_done) {
    int i;
    double gd;
    int done;
    int info = ERR_NONE;
    
    const double* d = ws->d;
    const double* t = ws->t;  /* t = xₖ (saved before line search) */
    const double* z = ws->z;  /* z = xᶜ (Cauchy point) */
    
    /* Compute directional derivative gd = gᵀd = gₖᵀdₖ */
    gd = ddot(n, g, 1, d, 1);
    ws->gd = gd;
    
    /* First evaluation: check descent direction
     * Line search requires gₖᵀdₖ < 0 (descent direction) */
    if (ws->num_eval == 0) {
        ws->gd_old = gd;
        if (gd >= ZERO) {
            /* Line search is impossible when directional derivative ≥ 0
             * This means d is not a descent direction */
            *out_done = 0;
            return ERR_DERIVATIVE;
        }
    }
    
    /* Convert SearchTol to MinpackSearchTol */
    MinpackSearchTol tol;
    tol.alpha = ws->search_tol.alpha;
    tol.beta = ws->search_tol.beta;
    tol.eps = ws->search_tol.eps;
    tol.lower = ws->search_tol.lower;
    tol.upper = ws->search_tol.upper;
    
    /* Convert SearchCtx to MinpackSearchCtx */
    MinpackSearchCtx ctx;
    ctx.bracket = ws->search_ctx.bracket;
    ctx.stage = ws->search_ctx.stage;
    ctx.g0 = ws->search_ctx.g0;
    ctx.gx = ws->search_ctx.gx;
    ctx.gy = ws->search_ctx.gy;
    ctx.f0 = ws->search_ctx.f0;
    ctx.fx = ws->search_ctx.fx;
    ctx.fy = ws->search_ctx.fy;
    ctx.stx = ws->search_ctx.stx;
    ctx.sty = ws->search_ctx.sty;
    ctx.width[0] = ws->search_ctx.width[0];
    ctx.width[1] = ws->search_ctx.width[1];
    ctx.bound[0] = ws->search_ctx.bound[0];
    ctx.bound[1] = ws->search_ctx.bound[1];
    
    /* Call scalar search */
    *stp = minpack_scalar_search(f, gd, *stp, task, &tol, &ctx);
    
    /* Copy context back */
    ws->search_ctx.bracket = ctx.bracket;
    ws->search_ctx.stage = ctx.stage;
    ws->search_ctx.g0 = ctx.g0;
    ws->search_ctx.gx = ctx.gx;
    ws->search_ctx.gy = ctx.gy;
    ws->search_ctx.f0 = ctx.f0;
    ws->search_ctx.fx = ctx.fx;
    ws->search_ctx.fy = ctx.fy;
    ws->search_ctx.stx = ctx.stx;
    ws->search_ctx.sty = ctx.sty;
    ws->search_ctx.width[0] = ctx.width[0];
    ws->search_ctx.width[1] = ctx.width[1];
    ws->search_ctx.bound[0] = ctx.bound[0];
    ws->search_ctx.bound[1] = ctx.bound[1];
    
    /* Check if done (converged, warning, or error) */
    done = (*task & (MINPACK_SEARCH_CONV | MINPACK_SEARCH_WARN | MINPACK_SEARCH_ERROR)) != 0;
    
    if (!done) {
        /* Try another x: compute xₖ₊₁ = λₖdₖ + xₖ */
        if (*stp == ONE) {
            dcopy(n, z, 1, x, 1);  /* x = xᶜ (Cauchy point when λ = 1) */
        } else {
            for (i = 0; i < n; i++) {
                x[i] = (*stp) * d[i] + t[i];  /* xₖ₊₁ = λₖdₖ + xₖ */
            }
        }
        /* Note: num_eval is incremented by the caller (driver), not here */
    } else if (*task & MINPACK_SEARCH_ERROR) {
        info = ERR_LINE_SEARCH_TOL;
    }
    
    *out_done = done;
    return info;
}
