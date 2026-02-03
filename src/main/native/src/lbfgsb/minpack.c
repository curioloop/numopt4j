/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * L-BFGS-B line search utility functions (minpack).
 * Based on the Moré-Thuente line search algorithm.
 * 
 * This implementation matches the Go version in minpack.go exactly.
 *
 * References:
 *   J.J. Moré and D.J. Thuente, "Line Search Algorithms with Guaranteed
 *   Sufficient Decrease", ACM Transactions on Mathematical Software,
 *   Vol. 20, No. 3, September 1994, pp. 286-307.
 */

#include "optimizer.h"
#include <math.h>

/* ============================================================================
 * Minpack Constants (matching Go minpack.go)
 * 
 * Line search parameters:
 *   p5  = 0.5   - bisection factor
 *   p66 = 0.66  - safeguard factor for step bounds
 *   xTrapLower = 1.1 - extrapolation lower factor
 *   xTrapUpper = 4.0 - extrapolation upper factor
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

/* ============================================================================
 * Moré-Thuente Line Search Algorithm
 * 
 * The algorithm finds a step λ satisfying the strong Wolfe conditions:
 * 
 *   Sufficient decrease (Armijo condition):
 *     f(λ) ≤ f(0) + α·λ·f′(0)
 * 
 *   Curvature condition:
 *     |f′(λ)| ≤ β·|f′(0)|
 * 
 * where:
 *   α (alpha) - sufficient decrease parameter (typically 1e-4)
 *   β (beta)  - curvature parameter (typically 0.9)
 *   f(0)      - function value at current point
 *   f′(0)     - directional derivative at current point (must be negative)
 *   λ         - step length
 * 
 * The algorithm uses a modified function to ensure progress:
 *   ψ(λ) = f(λ) - f(0) - α·λ·f′(0)
 * 
 * If ψ(λ) ≤ 0 and f′(λ) ≥ 0 for some step, then the interval is chosen
 * so that it contains a minimizer of f.
 * ============================================================================ */

/**
 * minpack_scalar_step (dcstep)
 * 
 * This subroutine computes a safeguarded step for a search procedure and 
 * updates an interval that contains a step that satisfies a sufficient 
 * decrease and a curvature condition.
 * 
 * The parameter stx contains the step with the least function value. If 
 * bracket is set to true then a minimizer has been bracketed in an interval 
 * with endpoints stx and sty. The parameter stp contains the current step.
 * The subroutine assumes that if bracket is set to true then
 *     min(stx,sty) < stp < max(stx,sty),
 * and that the derivative at stx is negative in the direction of the step.
 * 
 * Step Selection Strategy:
 * 
 *   Case 1: f(stp) > f(stx) - Higher function value
 *     The minimum is bracketed. Use cubic or average of cubic/quadratic.
 * 
 *   Case 2: f(stp) ≤ f(stx) and f′(stp)·f′(stx) < 0 - Opposite sign derivatives
 *     The minimum is bracketed. Use cubic or secant step.
 * 
 *   Case 3: f(stp) ≤ f(stx) and |f′(stp)| < |f′(stx)| - Same sign, decreasing
 *     Use cubic step if it tends to infinity or minimum is beyond stp.
 * 
 *   Case 4: f(stp) ≤ f(stx) and |f′(stp)| ≥ |f′(stx)| - Same sign, not decreasing
 *     Use cubic step if bracketed, otherwise use bounds.
 * 
 * Cubic interpolation formula:
 *   θ = 3·(f(stx) - f(stp))/(stp - stx) + f′(stx) + f′(stp)
 *   γ = s·√((θ/s)² - (f′(stx)/s)·(f′(stp)/s))
 *   where s = max(|θ|, |f′(stx)|, |f′(stp)|)
 * 
 * This matches the Go implementation in minpack.go scalarStep function exactly.
 */
void minpack_scalar_step(
    double* stx, double* fx, double* dx,
    double* sty, double* fy, double* dy,
    double* stp, double fp, double dp,
    int* bracket, const double* bound) {
    
    double gamma, p, q, r, s, sgnd;
    double stpc, stpf, stpq, theta;
    double stpmin, stpmax;
    
    stpmin = bound[0];
    stpmax = bound[1];
    
    /* Sign of dp * (dx / |dx|) - determines if derivatives have same sign */
    sgnd = dp * (*dx / fabs(*dx));
    
    /* ========================================================================
     * First case: A higher function value. The minimum is bracketed.
     * If the cubic step is closer to stx than the quadratic step, the cubic 
     * step is taken, otherwise the average of the cubic and quadratic steps 
     * is taken.
     * ======================================================================== */
    if (fp > *fx) {
        theta = THREE * (*fx - fp) / (*stp - *stx) + *dx + dp;
        s = fmax(fmax(fabs(theta), fabs(*dx)), fabs(dp));
        gamma = s * sqrt((theta / s) * (theta / s) - (*dx / s) * (dp / s));
        if (*stp < *stx) {
            gamma = -gamma;
        }
        p = (gamma - *dx) + theta;
        q = ((gamma - *dx) + gamma) + dp;
        r = p / q;
        stpc = *stx + r * (*stp - *stx);
        stpq = *stx + ((*dx / ((*fx - fp) / (*stp - *stx) + *dx)) / TWO) * (*stp - *stx);
        if (fabs(stpc - *stx) < fabs(stpq - *stx)) {
            stpf = stpc;
        } else {
            stpf = stpc + (stpq - stpc) / TWO;
        }
        *bracket = 1;
    }
    /* ========================================================================
     * Second case: A lower function value and derivatives of opposite sign.
     * The minimum is bracketed. If the cubic step is farther from stp than 
     * the secant step, the cubic step is taken, otherwise the secant step 
     * is taken.
     * ======================================================================== */
    else if (sgnd < ZERO) {
        theta = THREE * (*fx - fp) / (*stp - *stx) + *dx + dp;
        s = fmax(fmax(fabs(theta), fabs(*dx)), fabs(dp));
        gamma = s * sqrt((theta / s) * (theta / s) - (*dx / s) * (dp / s));
        if (*stp > *stx) {
            gamma = -gamma;
        }
        p = (gamma - dp) + theta;
        q = ((gamma - dp) + gamma) + *dx;
        r = p / q;
        stpc = *stp + r * (*stx - *stp);
        stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        if (fabs(stpc - *stp) > fabs(stpq - *stp)) {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        *bracket = 1;
    }
    /* ========================================================================
     * Third case: A lower function value, derivatives of the same sign,
     * and the magnitude of the derivative decreases.
     * The cubic step is computed only if either:
     *   - the cubic tends to infinity in the direction of the step
     *   - the minimum of the cubic is beyond stp.
     * Otherwise the cubic step is defined to be the secant step.
     * ======================================================================== */
    else if (fabs(dp) < fabs(*dx)) {
        theta = THREE * (*fx - fp) / (*stp - *stx) + *dx + dp;
        s = fmax(fmax(fabs(theta), fabs(*dx)), fabs(dp));
        /* The case gamma = 0 only arises if the cubic does not tend
         * to infinity in the direction of the step. */
        gamma = s * sqrt((theta / s) * (theta / s) - (*dx / s) * (dp / s));
        if (*stp > *stx) {
            gamma = -gamma;
        }
        p = (gamma - dp) + theta;
        q = (gamma + (*dx - dp)) + gamma;
        r = p / q;
        if (r < ZERO && gamma != ZERO) {
            stpc = *stp + r * (*stx - *stp);
        } else if (*stp > *stx) {
            stpc = stpmax;
        } else {
            stpc = stpmin;
        }
        stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        if (*bracket) {
            /* A minimizer has been bracketed.
             * If the cubic step is closer to stp than the secant step,
             * the cubic step is taken, otherwise the secant step is taken. */
            if (fabs(stpc - *stp) < fabs(stpq - *stp)) {
                stpf = stpc;
            } else {
                stpf = stpq;
            }
            if (*stp > *stx) {
                stpf = fmin(*stp + MINPACK_P66 * (*sty - *stp), stpf);
            } else {
                stpf = fmax(*stp + MINPACK_P66 * (*sty - *stp), stpf);
            }
        } else {
            /* A minimizer has not been bracketed.
             * If the cubic step is farther from stp than the secant step,
             * the cubic step is taken, otherwise the secant step is taken. */
            if (fabs(stpc - *stp) > fabs(stpq - *stp)) {
                stpf = stpc;
            } else {
                stpf = stpq;
            }
            stpf = fmin(stpmax, stpf);
            stpf = fmax(stpmin, stpf);
        }
    }
    /* ========================================================================
     * Fourth case: A lower function value, derivatives of the same sign,
     * and the magnitude of the derivative does not decrease.
     * If the minimum is not bracketed, the step is either stpmin or stpmax,
     * otherwise the cubic step is taken.
     * ======================================================================== */
    else {
        if (*bracket) {
            theta = THREE * (fp - *fy) / (*sty - *stp) + *dy + dp;
            s = fmax(fmax(fabs(theta), fabs(*dy)), fabs(dp));
            gamma = s * sqrt((theta / s) * (theta / s) - (*dy / s) * (dp / s));
            if (*stp > *sty) {
                gamma = -gamma;
            }
            p = (gamma - dp) + theta;
            q = ((gamma - dp) + gamma) + *dy;
            r = p / q;
            stpc = *stp + r * (*sty - *stp);
            stpf = stpc;
        } else if (*stp > *stx) {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }
    
    /* ========================================================================
     * Update the interval which contains a minimizer.
     * ======================================================================== */
    if (fp > *fx) {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if (sgnd < ZERO) {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    
    /* Compute the new step */
    *stp = stpf;
}

/**
 * minpack_scalar_search (dcsrch)
 * 
 * This subroutine finds a step λ that satisfies the strong Wolfe conditions:
 * 
 *   Sufficient decrease condition (Armijo):
 *     f(λ) ≤ f(0) + α·λ·f′(0)
 * 
 *   Curvature condition:
 *     |f′(λ)| ≤ β·|f′(0)|
 * 
 * Each call of the subroutine updates an interval with endpoints stx and sty.
 * 
 * The interval is initially chosen so that it contains a minimizer of the
 * modified function:
 *   ψ(λ) = f(λ) - f(0) - α·λ·f′(0)
 * 
 * If ψ(λ) ≤ 0 and f′(λ) ≥ 0 for some step, then the interval is chosen so
 * that it contains a minimizer of f.
 * 
 * If α < β and if, for example, the function is bounded below, then there
 * is always a step which satisfies both conditions.
 * 
 * If no step can be found that satisfies both conditions, then the algorithm
 * stops with a warning. In this case stp only satisfies the sufficient
 * decrease condition.
 * 
 * Parameters:
 *   f     - On initial entry: function value at 0
 *           On subsequent entries: function value at stp
 *   g     - On initial entry: derivative at 0 (must be negative)
 *           On subsequent entries: derivative at stp
 *   stp   - Current estimate of satisfactory step
 *   task  - Search state (MINPACK_SEARCH_START, MINPACK_SEARCH_FG, etc.)
 *   tol   - Tolerance parameters (α, β, ε, lower, upper)
 *   ctx   - Search context (bracket state, interval endpoints, etc.)
 * 
 * Returns:
 *   Updated step length. If task = MINPACK_SEARCH_CONV, the step satisfies
 *   both Wolfe conditions.
 * 
 * This matches the Go implementation in minpack.go ScalarSearch function exactly.
 */
double minpack_scalar_search(
    double f, double g, double stp,
    int* task,
    const MinpackSearchTol* tol,
    MinpackSearchCtx* ctx) {
    
    double stpmin, stpmax;
    double gtest, ftest;
    
    /* ========================================================================
     * Initialization block
     * ======================================================================== */
    if (*task == MINPACK_SEARCH_START) {
        /* Check the input arguments for errors */
        if (stp < tol->lower) {
            *task = MINPACK_ERR_OVER_LOWER;
        } else if (stp > tol->upper) {
            *task = MINPACK_ERR_OVER_UPPER;
        } else if (g >= ZERO) {
            *task = MINPACK_ERR_NEG_INIT_G;
        } else if (tol->alpha < ZERO) {
            *task = MINPACK_ERR_NEG_ALPHA;
        } else if (tol->beta < ZERO) {
            *task = MINPACK_ERR_NEG_BETA;
        } else if (tol->eps < ZERO) {
            *task = MINPACK_ERR_NEG_EPS;
        } else if (tol->lower < ZERO) {
            *task = MINPACK_ERR_LOWER;
        } else if (tol->upper < tol->lower) {
            *task = MINPACK_ERR_UPPER;
        }
        
        /* Exit if there are errors on input */
        if (*task & MINPACK_SEARCH_ERROR) {
            return stp;
        }
        
        /* Initialize local variables */
        ctx->bracket = 0;
        ctx->stage = MINPACK_STAGE_ARMIJO;
        ctx->f0 = f;
        ctx->g0 = g;
        ctx->width[0] = tol->upper - tol->lower;
        ctx->width[1] = ctx->width[0] / MINPACK_P5;
        
        /* Initialize the points and their corresponding function and derivative values */
        ctx->stx = ZERO;
        ctx->fx = ctx->f0;
        ctx->gx = ctx->g0;
        ctx->sty = ZERO;
        ctx->fy = ctx->f0;
        ctx->gy = ctx->g0;
        ctx->bound[0] = ZERO;
        ctx->bound[1] = stp + MINPACK_XTRAP_UPPER * stp;
        
        *task = MINPACK_SEARCH_FG;
        return stp;
    }
    
    /* ========================================================================
     * Test for convergence or warnings
     * 
     * Convergence is achieved when both Wolfe conditions are satisfied:
     *   1. Sufficient decrease: f ≤ f₀ + α·stp·g₀  (f ≤ ftest)
     *   2. Curvature condition: |g| ≤ β·|g₀|
     * ======================================================================== */
    gtest = tol->alpha * ctx->g0;
    ftest = ctx->f0 + stp * gtest;
    
    stpmin = ctx->bound[0];
    stpmax = ctx->bound[1];
    
    if (ctx->bracket && (stp <= stpmin || stp >= stpmax)) {
        *task = MINPACK_WARN_ROUND_ERR;
    } else if (ctx->bracket && (stpmax - stpmin) <= tol->eps * stpmax) {
        *task = MINPACK_WARN_REACH_EPS;
    } else if (stp == tol->upper && f <= ftest && g <= gtest) {
        *task = MINPACK_WARN_REACH_MAX;
    } else if (stp == tol->lower && (f > ftest || g >= gtest)) {
        *task = MINPACK_WARN_REACH_MIN;
    } else if (f <= ftest && fabs(g) <= tol->beta * (-ctx->g0)) {
        *task = MINPACK_SEARCH_CONV;
    }
    
    if (*task & (MINPACK_SEARCH_WARN | MINPACK_SEARCH_CONV)) {
        return stp;
    }
    
    /* ========================================================================
     * Update search stage
     * 
     * Stage 1 (Armijo): Looking for sufficient decrease
     * Stage 2 (Wolfe):  Sufficient decrease achieved, looking for curvature
     * 
     * Transition from Armijo to Wolfe when:
     *   f ≤ ftest (sufficient decrease) AND g ≥ 0 (positive derivative)
     * ======================================================================== */
    if (ctx->stage == MINPACK_STAGE_ARMIJO && f <= ftest && g >= ZERO) {
        ctx->stage = MINPACK_STAGE_WOLFE;
    }
    
    /* ========================================================================
     * A modified function is used to predict the step during the first stage
     * if a lower function value has been obtained but the decrease is not
     * sufficient.
     * 
     * Modified function and derivatives:
     *   ψ(λ) = f(λ) - f(0) - α·λ·f′(0)
     *   ψ′(λ) = f′(λ) - α·f′(0)
     * 
     * The modified values are:
     *   fm  = f - stp·gtest     (modified function at stp)
     *   fxm = fx - stx·gtest    (modified function at stx)
     *   fym = fy - sty·gtest    (modified function at sty)
     *   gm  = g - gtest         (modified derivative at stp)
     *   gxm = gx - gtest        (modified derivative at stx)
     *   gym = gy - gtest        (modified derivative at sty)
     * ======================================================================== */
    if (ctx->stage == MINPACK_STAGE_ARMIJO && f <= ctx->fx && f > ftest) {
        /* Define the modified function and derivative values */
        double fm = f - stp * gtest;
        double fxm = ctx->fx - ctx->stx * gtest;
        double fym = ctx->fy - ctx->sty * gtest;
        double gm = g - gtest;
        double gxm = ctx->gx - gtest;
        double gym = ctx->gy - gtest;
        
        /* Call scalar_step to update interval of uncertainty and compute new step */
        minpack_scalar_step(&ctx->stx, &fxm, &gxm, 
                            &ctx->sty, &fym, &gym, 
                            &stp, fm, gm, 
                            &ctx->bracket, ctx->bound);
        
        /* Reset the function and derivative values for f */
        ctx->fx = fxm + ctx->stx * gtest;
        ctx->fy = fym + ctx->sty * gtest;
        ctx->gx = gxm + gtest;
        ctx->gy = gym + gtest;
    } else {
        /* Call scalar_step to update interval of uncertainty and compute new step */
        minpack_scalar_step(&ctx->stx, &ctx->fx, &ctx->gx, 
                            &ctx->sty, &ctx->fy, &ctx->gy, 
                            &stp, f, g, 
                            &ctx->bracket, ctx->bound);
    }
    
    /* ========================================================================
     * Decide if a bisection step is needed.
     * ======================================================================== */
    if (ctx->bracket) {
        if (fabs(ctx->sty - ctx->stx) >= MINPACK_P66 * ctx->width[1]) {
            stp = ctx->stx + MINPACK_P5 * (ctx->sty - ctx->stx);
        }
        ctx->width[1] = ctx->width[0];
        ctx->width[0] = fabs(ctx->sty - ctx->stx);
    }
    
    /* ========================================================================
     * Set the minimum and maximum steps allowed for stp.
     * ======================================================================== */
    if (ctx->bracket) {
        stpmin = fmin(ctx->stx, ctx->sty);
        stpmax = fmax(ctx->stx, ctx->sty);
    } else {
        stpmin = stp + MINPACK_XTRAP_LOWER * (stp - ctx->stx);
        stpmax = stp + MINPACK_XTRAP_UPPER * (stp - ctx->stx);
    }
    ctx->bound[0] = stpmin;
    ctx->bound[1] = stpmax;
    
    /* Force the step to be within the bounds */
    stp = fmin(fmax(stp, tol->lower), tol->upper);
    
    /* If further progress is not possible, let stp be the best point obtained
     * so far. */
    if ((ctx->bracket && (stp <= stpmin || stp >= stpmax)) ||
        (ctx->bracket && stpmax - stpmin <= tol->eps * stpmax)) {
        stp = ctx->stx;
    }
    
    *task = MINPACK_SEARCH_FG;
    return stp;
}
