/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * SLSQP (Sequential Least Squares Programming) algorithm implementation.
 * Complete implementation based on Dieter Kraft's algorithm.
 *
 * This file corresponds to solver.go in the Go implementation.
 *
 * ============================================================================
 * SLSQP Algorithm Overview
 * ============================================================================
 *
 * SLSQP solves NLP (general constrained NonLinear optimization Problem) with
 * SQP (Sequential Quadratic Programming):
 *
 *   minimize 𝒇(𝐱) subject to
 *     - equality constraints: 𝒄ⱼ(𝐱) = 0  (j = 1 ··· mₑ)
 *     - inequality constraints: 𝒄ⱼ(𝐱) ≥ 0  (j = mₑ+1 ··· m)
 *     - boundaries: 𝒍ᵢ ≤ 𝐱ᵢ ≤ 𝒖ᵢ (i = 1 ··· n)
 *
 * SQP decomposes NLP into a series of QP sub-problems, each of which solves
 * a descent direction 𝐝 and step length 𝛂, ensuring that 𝒇(𝐱 + 𝛂𝐝) < 𝒇(𝐱)
 * and the updated 𝐱 satisfies the constraints.
 *
 * ============================================================================
 * Lagrangian Function
 * ============================================================================
 *
 * The Lagrangian function of NLP is:
 *   ℒ(𝐱,𝛌) = 𝒇(𝐱) - ∑𝛌ⱼ𝒄ⱼ(𝐱)
 *
 * which is a linear approximation of constraints 𝒄ⱼ(𝐱).
 *
 * ============================================================================
 * QP Subproblem
 * ============================================================================
 *
 * A quadratic approximation of ℒ(𝐱,𝛌) at location 𝐱ᵏ is a standard QP problem:
 *
 *   minimize ½ 𝐝ᵀ𝐁ᵏ𝐝 + 𝜵𝒇(𝐱ᵏ)𝐝 subject to
 *     - 𝜵𝒄ⱼ(𝐱ᵏ)𝐝 + 𝒄ⱼ(𝐱ᵏ) = 0  (j = 1 ··· mₑ)
 *     - 𝜵𝒄ⱼ(𝐱ᵏ)𝐝 + 𝒄ⱼ(𝐱ᵏ) ≥ 0  (j = mₑ+1 ··· m)
 *
 * With a symmetric Hessian approximation 𝐁ᵏ ≈ 𝜵²ℒ(𝐱ᵏ,𝛌ᵏ), the descent
 * search direction 𝐝 is determined by the above problem.
 *
 * ============================================================================
 * Merit Function (L1 Penalty)
 * ============================================================================
 *
 * The merit function with L1 penalty is:
 *   𝟇(𝐱;𝛒) = 𝒇(𝐱) + ∑𝛒ⱼ‖𝒄ⱼ(𝐱)‖₁
 *
 * where:
 *   - ‖𝒄ⱼ(𝐱)‖₁ = |𝒄ⱼ(𝐱)| = max[𝒄ⱼ(𝐱),-𝒄ⱼ(𝐱)]    (j = 1 ··· mₑ)
 *   - ‖𝒄ⱼ(𝐱)‖₁ = |min[0,𝒄ⱼ(𝐱)]| = max[0,-𝒄ⱼ(𝐱)] (j = mₑ+1 ··· m)
 *
 * The penalty parameters 𝛒 are updated iteratively:
 *   𝛒ⱼᵏ⁺¹ = max[ ½(𝛒ⱼᵏ+|𝛌ⱼ|), |𝛌ⱼ| ] (j = 1 ··· m)
 *
 * ============================================================================
 * Directional Derivative of Merit Function
 * ============================================================================
 *
 * The directional derivative of the merit function along 𝐝 is:
 *   𝜵𝞥(𝐝;𝐱ᵏ,𝛒ᵏ) = 𝜵𝒇(𝐱ᵏ)ᵀ𝐝 - ∑𝛒ᵏⱼ‖𝒄ⱼ(𝐱ᵏ)‖₁
 *
 * For augmented QP with slack variable 𝛅:
 *   𝜵𝞥 = 𝜵𝒇(𝐱ᵏ)ᵀ𝐝 - (1 - 𝛅)∑𝛒ᵏⱼ‖𝒄ⱼ(𝐱ᵏ)‖₁
 *
 * ============================================================================
 * Line Search (Armijo Condition)
 * ============================================================================
 *
 * The step length 𝛂 is obtained by line-search with Armijo condition:
 *   𝞥(𝐱ᵏ+𝛂𝐝;𝛌,𝛒) - 𝞥(𝐱ᵏ;𝛌,𝛒) < η · 𝛂 · 𝜵𝞥(𝐝;𝐱ᵏ,𝛒ᵏ) (0<η<0.5)
 *
 * In this implementation, η = 0.1 is used.
 *
 * ============================================================================
 * BFGS Update Formula
 * ============================================================================
 *
 * The modified BFGS formula for constrained optimization:
 *   - 𝐁ᵏ⁺¹ = 𝐁ᵏ + 𝐪𝐪ᵀ/𝐪ᵀ𝐬 - 𝐁ᵏ𝐬𝐬ᵀ𝐁ᵏ/𝐬ᵀ𝐁ᵏ𝐬
 *   - 𝐬 = 𝐱ᵏ⁺¹ - 𝐱ᵏ
 *   - 𝐪 = 𝛉𝛈 + (1-𝛉)𝐁ᵏ𝐬
 *   - 𝛈 = 𝜵ℒ(𝐱ᵏ⁺¹,𝛌ᵏ) - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ)
 *   - if 𝐬ᵀ𝛈 ≥ ⅕ 𝐬ᵀ𝐁ᵏ𝐬 : 𝛉 = 1
 *   - otherwise : 𝛉 = ⅘ 𝐬ᵀ𝐁ᵏ𝐬 / (𝐬ᵀ𝐁ᵏ𝐬 - 𝐬ᵀ𝛈)
 *
 * The matrix is stored as 𝐁 = 𝐋𝐃𝐋ᵀ where:
 *   - 𝐋 is a strict lower triangular matrix
 *   - 𝐃 is a diagonal matrix
 *
 * ============================================================================
 * Convergence Criteria
 * ============================================================================
 *
 * After obtaining the solution 𝐝 to the QP problem:
 *   - C_vio = ∑‖𝒄ⱼ(𝐱ᵏ)‖₁                    (constraint violation)
 *   - C_opt = |𝜵𝒇(𝐱ᵏ)ᵀ𝐝| + |𝛌ᵏ|ᵀ×‖𝒄(𝐱ᵏ)‖₁  (optimality measure)
 *   - C_stp = ‖𝐝‖₂                           (step length)
 *
 * After line-search finds the step 𝛂:
 *   - Ĉ_vio = ∑‖𝒄ⱼ(𝐱ᵏ + 𝛂𝐝)‖₁
 *   - Ĉ_opt = |𝒇(𝐱ᵏ + 𝛂𝐝) - 𝒇(𝐱ᵏ)|
 *   - Ĉ_stp = ‖𝐝‖₂
 *
 * Reference: Dieter Kraft, "A software package for sequential quadratic
 * programming". DFVLR-FB 88-28, 1988
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

/* External BLAS functions */
extern void dcopy(int n, const double* x, int incx, double* y, int incy);
extern void daxpy(int n, double a, const double* x, int incx, double* y, int incy);
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern double dnrm2(int n, const double* x, int incx);
extern void dscal(int n, double a, double* x, int incx);
extern void dzero(int n, double* x, int incx);

/* External solver functions */
extern int lsq(int m, int meq, int n, int nl,
               double* l, double* g, double* a, double* b,
               double* xl, double* xu,
               double* x, double* y, double* w, int* jw,
               int maxIter, double infBnd, double* norm);
extern void compositeT(int n, double* l, double* z, double sigma, double* w);

/* Constants */
#define INF_BND 1e20

/* Golden section ratio: 1 / phi^2 where phi = (1 + sqrt(5)) / 2 */
#if defined(__GNUC__) || defined(__clang__)
#define INV_PHI2 (1.0 / ((1.0 + __builtin_sqrt(5.0)) / 2.0 * (1.0 + __builtin_sqrt(5.0)) / 2.0))
#else
static inline double get_inv_phi2(void) {
    static double val = 0.0;
    if (val == 0.0) {
        double phi = (1.0 + sqrt(5.0)) / 2.0;
        val = 1.0 / (phi * phi);
    }
    return val;
}
#define INV_PHI2 (get_inv_phi2())
#endif

/* Mode constants for LSQ */
#define MODE_OK              0
#define MODE_HAS_SOLUTION    0
#define MODE_CONS_INCOMPAT  -2
#define MODE_LSEI_SINGULAR  -4

/* ============================================================================
 * Convergence Check Functions
 * ============================================================================
 *
 * These functions implement the convergence criteria for SLSQP:
 *
 * check_stop: Checks multiple termination criteria in order:
 *   1. If constraint violation ≥ tol or bad_qp or f is NaN → continue
 *   2. If |f - f₀| < tol → converged (function value change)
 *   3. If ‖s‖₂ < tol → converged (step length)
 *   4. If |f| < f_eval_tol (if enabled) → converged
 *   5. If |f - f₀| < f_diff_tol (if enabled) → converged
 *   6. If ‖x - x₀‖₂ < x_diff_tol (if enabled) → converged
 *
 * check_conv: Computes constraint violation and calls check_stop:
 *   h₃ = ∑‖𝒄ⱼ(𝐱)‖₁ (L1 norm of constraint violations)
 *
 * These correspond to checkStop and checkConv in solver.go.
 */

/**
 * Check if optimization should stop based on various criteria.
 * 
 * This function matches Go's checkStop function in solver.go.
 * It checks multiple termination criteria in order:
 * 1. Constraint violation >= tol or bad_qp or f is NaN -> continue
 * 2. |f - f0| < tol -> converged
 * 3. ||s||_2 < tol -> converged
 * 4. |f| < f_eval_tol (if f_eval_tol >= 0) -> converged
 * 5. |f - f0| < f_diff_tol (if f_diff_tol >= 0) -> converged
 * 6. ||x - x0||_2 < x_diff_tol (if x_diff_tol >= 0) -> converged
 * 
 * @param vio Constraint violation sum
 * @param tol Tolerance for convergence
 * @param bad_qp Whether QP was inconsistent
 * @param f Current function value
 * @param f0 Previous function value
 * @param s Search direction
 * @param n Problem dimension
 * @param config Configuration with extended tolerances
 * @param x Current position
 * @param x0 Previous position
 * @param u Workspace for x difference computation
 * @return 1 if converged, 0 otherwise
 * 
 * Requirements: 16.2, 16.3
 */
static int check_stop(
    double vio, double tol, int bad_qp,
    double f, double f0,
    const double* s, int n,
    const SlsqpConfig* config,
    const double* x, const double* x0,
    double* u)
{
    /* Matches Go: if vio >= tol || ctx.bad || math.IsNaN(loc.f) { return false } */
    if (vio >= tol || bad_qp || isnan(f)) {
        return 0;
    }
    
    /* Matches Go: case math.Abs(loc.f-ctx.f0) < tol */
    if (fabs(f - f0) < tol) {
        return 1;
    }
    
    /* Matches Go: case dnrm2(spec.n, ctx.s, 1) < tol */
    if (dnrm2(n, s, 1) < tol) {
        return 1;
    }
    
    /* Matches Go: case stop.FEvalTolerance >= zero && math.Abs(loc.f) < stop.FEvalTolerance */
    if (config->f_eval_tol >= ZERO && fabs(f) < config->f_eval_tol) {
        return 1;
    }
    
    /* Matches Go: case stop.FDiffTolerance >= zero && math.Abs(loc.f-ctx.f0) < stop.FDiffTolerance */
    if (config->f_diff_tol >= ZERO && fabs(f - f0) < config->f_diff_tol) {
        return 1;
    }
    
    /* Matches Go: case stop.XDiffTolerance >= zero */
    if (config->x_diff_tol >= ZERO) {
        /* Compute ||x - x0||_2 */
        dcopy(n, x, 1, u, 1);
        daxpy(n, -ONE, x0, 1, u, 1);
        if (dnrm2(n, u, 1) < config->x_diff_tol) {
            return 1;
        }
    }
    
    return 0;
}

/**
 * Compute constraint violation and check convergence.
 * 
 * This function matches Go's checkConv function in solver.go.
 * It computes the L1 norm of constraint violations and calls checkStop.
 * 
 * @param c Constraint values
 * @param m Total number of constraints
 * @param meq Number of equality constraints
 * @param tol Tolerance for convergence
 * @param bad_qp Whether QP was inconsistent
 * @param f Current function value
 * @param f0 Previous function value
 * @param s Search direction
 * @param n Problem dimension
 * @param config Configuration with extended tolerances
 * @param x Current position
 * @param x0 Previous position
 * @param u Workspace
 * @param vio_out Output: constraint violation (can be NULL)
 * @return 1 if converged, 0 otherwise
 * 
 * Requirements: 16.2
 */
static int check_conv(
    const double* c, int m, int meq,
    double tol, int bad_qp,
    double f, double f0,
    const double* s, int n,
    const SlsqpConfig* config,
    const double* x, const double* x0,
    double* u,
    double* vio_out)
{
    /* Compute constraint violation: h3 = sum of ||c_j||_1 */
    /* Matches Go: for j, c := range ss.location.c { ... h3 += math.Max(-c, h1) } */
    double h3 = ZERO;
    for (int j = 0; j < m; j++) {
        double h1 = (j < meq) ? c[j] : ZERO;
        h3 += fmax(-c[j], h1);
    }
    
    if (vio_out) {
        *vio_out = h3;
    }
    
    return check_stop(h3, tol, bad_qp, f, f0, s, n, config, x, x0, u);
}

/* ============================================================================
 * Exact Line Search (Golden Section + Quadratic Interpolation)
 * ============================================================================
 *
 * This implements Brent's method for finding the minimum of a unimodal function
 * without derivatives. The algorithm combines:
 *   1. Golden section search to narrow the interval
 *   2. Quadratic (parabolic) interpolation to accelerate convergence
 *
 * Golden Section Ratio:
 *   c = 1/φ² where φ = (1 + √5)/2 ≈ 1.618 (golden ratio)
 *   c ≈ 0.381966...
 *
 * The algorithm maintains a bracketing interval [a, b] and three points:
 *   - x: current best point (lowest function value)
 *   - w: second best point
 *   - v: previous value of w
 *
 * Convergence test:
 *   |x - m| ≤ tol2 - 0.5*(b - a)
 *   where m = (a + b)/2 is the midpoint
 *
 * Reference: R.P. Brent, "Algorithms for Minimization without Derivatives",
 * Prentice-Hall, 1973.
 */

/**
 * Exact line search using combination of golden section and quadratic interpolation.
 * 
 * This function finds the argument x where the function f(x) takes its minimum
 * in the interval [alpha_lower, alpha_upper]. It uses a combination of:
 * 1. Golden section search to narrow the interval
 * 2. Quadratic (parabolic) interpolation to accelerate convergence
 * 
 * The function operates in a reverse communication style matching Go's findMin:
 * - FIND_NOOP: Initialize and return first evaluation point
 * - FIND_INIT: Process initial function value, return next point
 * - FIND_NEXT: Process function value, return next point or converge
 * - FIND_CONV: Search has converged
 * 
 * @param mode Search mode (input/output)
 * @param work Working storage for search state
 * @param f Current function value at the point returned by previous call
 * @param tol Desired length of interval of uncertainty
 * @param alpha_lower Lower bound of search interval
 * @param alpha_upper Upper bound of search interval
 * @return Current best step length (evaluation point)
 * 
 * Requirements: 15.2, 15.3
 */
static double find_min(
    FindMode* mode,
    FindWork* work,
    double f,
    double tol,
    double alpha_lower,
    double alpha_upper)
{
    double c = INV_PHI2;  /* Golden section ratio */
    double ax = alpha_lower;
    double bx = alpha_upper;
    
    switch (*mode) {
    case FIND_INIT:
        /* Main loop starts - process initial function value */
        work->fx = f;
        work->fv = work->fx;
        work->fw = work->fv;
        break;
        
    case FIND_NEXT:
        /* Process function value at u, update interval */
        work->fu = f;
        
        /* Update a, b, v, w, and x based on new function value */
        /* Matches Go: if u, x := w.u, w.x; w.fu > w.fx { ... } */
        if (work->fu > work->fx) {
            /* New point is worse than current best */
            if (work->u < work->x) {
                work->a = work->u;
            }
            if (work->u >= work->x) {
                work->b = work->u;
            }
            
            /* Update v and w if appropriate */
            if (work->fu <= work->fw || fabs(work->w - work->x) <= ZERO) {
                work->v = work->w;
                work->fv = work->fw;
                work->w = work->u;
                work->fw = work->fu;
            } else if (work->fu <= work->fv || fabs(work->v - work->x) <= ZERO || 
                       fabs(work->v - work->w) <= ZERO) {
                work->v = work->u;
                work->fv = work->fu;
            }
        } else {
            /* New point is better than or equal to current best */
            if (work->u >= work->x) {
                work->a = work->x;
            }
            if (work->u < work->x) {
                work->b = work->x;
            }
            
            /* Shift v <- w <- x <- u */
            work->v = work->w;
            work->fv = work->fw;
            work->w = work->x;
            work->fw = work->fx;
            work->x = work->u;
            work->fx = work->fu;
        }
        break;
        
    default:
        /* FIND_NOOP: Initialization - matches Go's default case */
        work->a = ax;
        work->b = bx;
        work->e = ZERO;
        
        /* Initial point using golden section from left: w.v = w.a + c*(w.b-w.a) */
        work->v = work->a + c * (work->b - work->a);
        work->w = work->v;
        work->x = work->v;
        
        *mode = FIND_INIT;
        return work->x;
    }
    
    /* Compute midpoint and tolerances */
    work->m = 0.5 * (work->a + work->b);
    work->tol1 = SQRT_EPS * fabs(work->x) + tol;
    work->tol2 = 2.0 * work->tol1;
    
    /* Test for convergence: |x - m| <= tol2 - 0.5*(b - a) */
    /* Matches Go: if math.Abs(w.x-w.m) <= w.tol2-0.5*(w.b-w.a) */
    if (fabs(work->x - work->m) <= work->tol2 - 0.5 * (work->b - work->a)) {
        /* End of main loop - converged */
        *mode = FIND_CONV;
        return work->x;
    }
    
    /* Parabolic interpolation or golden-section step */
    /* Matches Go: r, q, p, d, e := zero, zero, zero, w.d, w.e */
    double r_val = ZERO;
    double q_val = ZERO;
    double p_val = ZERO;
    double d_val = work->d;
    double e_val = work->e;
    
    if (fabs(e_val) > work->tol1) {
        /* Fit parabola - matches Go's parabola fitting */
        double fx = work->fx;
        double fw = work->fw;
        double fv = work->fv;
        double x_pt = work->x;
        double w_pt = work->w;
        double v_pt = work->v;
        
        r_val = (x_pt - w_pt) * (fx - fv);
        q_val = (x_pt - v_pt) * (fx - fw);
        p_val = (x_pt - v_pt) * q_val - (x_pt - w_pt) * r_val;
        q_val = 2.0 * (q_val - r_val);
        
        if (q_val > ZERO) {
            p_val = -p_val;
        }
        if (q_val < ZERO) {
            q_val = -q_val;
        }
        
        r_val = e_val;
        e_val = d_val;
    }
    
    /* Store interpolation parameters */
    work->r = r_val;
    work->q = q_val;
    work->p = p_val;
    
    /* Decide whether to use parabolic step or golden section */
    /* Matches Go: if a, b, x := w.a, w.b, w.x; math.Abs(p) >= 0.5*math.Abs(q*r) || p <= q*(a-x) || p >= q*(b-x) */
    double a_pt = work->a;
    double b_pt = work->b;
    double x_pt = work->x;
    
    if (fabs(p_val) >= 0.5 * fabs(q_val * r_val) || 
        p_val <= q_val * (a_pt - x_pt) || 
        p_val >= q_val * (b_pt - x_pt)) {
        /* Golden-section step */
        if (x_pt >= work->m) {
            e_val = a_pt - x_pt;
        } else {
            e_val = b_pt - x_pt;
        }
        d_val = c * e_val;
    } else {
        /* Parabolic interpolation step */
        /* Matches Go: if w.u-a < w.tol2 || b-w.u < w.tol2 { ... } else { d = p / q } */
        double u_temp = x_pt + p_val / q_val;
        if (u_temp - a_pt < work->tol2 || b_pt - u_temp < work->tol2) {
            /* Ensure not too close to bounds - use copysign like Go */
            d_val = (work->m > x_pt) ? work->tol1 : -work->tol1;
        } else {
            d_val = p_val / q_val;
        }
    }
    
    /* Ensure step is not too small - matches Go: if math.Abs(d) < w.tol1 { d = math.Copysign(w.tol1, d) } */
    if (fabs(d_val) < work->tol1) {
        d_val = (d_val > ZERO) ? work->tol1 : -work->tol1;
    }
    
    /* Store step information */
    work->d = d_val;
    work->e = e_val;
    
    /* Compute next evaluation point: w.u = w.x + w.d */
    work->u = work->x + work->d;
    
    *mode = FIND_NEXT;
    return work->u;
}

/* ============================================================================
 * Exact Line Search Integration
 * ============================================================================ */

/**
 * Perform exact line search step using golden section and quadratic interpolation.
 * 
 * This function integrates find_min into the SLSQP optimization loop.
 * It handles the reverse communication protocol and updates the position
 * x = x0 + alpha * s for merit function evaluation.
 * 
 * Matches Go's exactSearch function:
 * - If mode != findConv: call findMin, set x = x0 + alpha * s
 * - If mode == findConv: scale s = alpha * s (final step)
 * 
 * @param ws Workspace containing line search state (fw, line_mode)
 * @param t Current merit function value at the point returned by previous call
 * @param tol Desired length of interval of uncertainty
 * @param alpha_lower Lower bound of search interval (typically 0.1)
 * @param alpha_upper Upper bound of search interval (typically 1.0)
 * @param n Problem dimension
 * @param x Current position (output: updated to x0 + alpha * s)
 * @param x0 Initial position
 * @param s Search direction (output: scaled to alpha * s when converged)
 * @return FindMode: FIND_CONV if converged, otherwise FIND_INIT or FIND_NEXT
 * 
 * Requirements: 15.1, 15.2
 */
static FindMode exact_line_search(
    SlsqpWorkspace* ws,
    double t,
    double tol,
    double alpha_lower,
    double alpha_upper,
    int n,
    double* x,
    const double* x0,
    double* s)
{
    FindMode mode = (FindMode)ws->line_mode;
    
    /* Matches Go: if mode != findConv { ... } else { dscal(s.n, c.alpha, c.s, 1) } */
    if (mode != FIND_CONV) {
        /* Call find_min to get the next step length */
        /* Matches Go: c.alpha, mode = findMin(mode, &c.fw, t, c.tol, *s.Line.Alpha) */
        ws->alpha = find_min(&mode, &ws->fw, t, tol, alpha_lower, alpha_upper);
        ws->line_mode = (int)mode;
        
        /* Update position: x = x0 + alpha * s */
        /* Matches Go: dcopy(s.n, c.x0, 1, x, 1); daxpy(s.n, c.alpha, c.s, 1, x, 1) */
        dcopy(n, x0, 1, x, 1);
        daxpy(n, ws->alpha, s, 1, x, 1);
    } else {
        /* Search converged: scale s to final step s = alpha * s */
        /* Matches Go: dscal(s.n, c.alpha, c.s, 1) */
        dscal(n, ws->alpha, s, 1);
    }
    
    return mode;
}

/* ============================================================================
 * Main Optimization Function
 * ============================================================================
 *
 * This implements the main SQP iteration loop, corresponding to mainLoop()
 * in solver.go.
 *
 * Algorithm Flow:
 * ─────────────────────────────────────────────────────────────────────────────
 * 1. Initialize: evaluate f, g, c, a and reset BFGS (L = I, D = I)
 *
 * 2. Main loop (while mode == OK):
 *    a. Increment iteration counter, check max iterations
 *
 *    b. Transform bounds: 𝒍 - 𝐱ᵏ ≤ 𝐝 ≤ 𝒖 - 𝐱ᵏ
 *
 *    c. Solve QP subproblem via LSQ to get:
 *       - Search direction 𝐝 (stored in s)
 *       - Lagrange multipliers 𝛌 (stored in r)
 *
 *    d. Handle inconsistent constraints with augmented QP:
 *       - Add slack variable 𝛅 with penalty 𝛒
 *       - Solve augmented problem with increasing 𝛒 (10² → 10⁷)
 *
 *    e. Update multipliers for L1-test:
 *       v[i] = g[i] - 𝛌ᵀ∇c[i]  (gradient of Lagrangian)
 *
 *    f. Compute optimality and feasibility measures:
 *       - h₁ = |∇f(x)ᵀd| + |𝛌|ᵀ×‖c(x)‖₁  (optimality)
 *       - h₂ = ∑‖cⱼ(x)‖₁                   (feasibility)
 *
 *    g. Check convergence: h₁ < acc && h₂ < acc
 *
 *    h. Compute directional derivative of merit function:
 *       h₃ = ∇f(x)ᵀd - (1-𝛅)∑𝛒ⱼ‖cⱼ(x)‖₁
 *       If h₃ ≥ 0 (ascent direction), reset BFGS
 *
 *    i. Line search (inexact or exact):
 *       - Inexact: Armijo backtracking with η = 0.1
 *       - Exact: Golden section + quadratic interpolation
 *
 *    j. BFGS update:
 *       - Compute 𝛈 = ∇ℒ(x^{k+1},𝛌) - ∇ℒ(x^k,𝛌)
 *       - Compute B^k*s via L*D*L'*s
 *       - Apply modified BFGS with damping factor 𝛉
 *       - Update LDL' factorization via compositeT
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

/**
 * SLSQP main optimization loop.
 * 
 * This function implements the main iteration loop of the SLSQP algorithm,
 * matching the Go implementation in solver.go mainLoop().
 * 
 * The algorithm flow is:
 * 1. Initialize context (initCtx): evaluate f, g, c, a and reset BFGS
 * 2. Main loop:
 *    a. Increment iteration counter and check max iterations
 *    b. Solve QP subproblem to get search direction s and multipliers r
 *    c. Handle inconsistent constraints with augmented QP
 *    d. Check convergence criteria
 *    e. Perform line search (inexact or exact)
 *    f. Update BFGS approximation
 * 
 * Requirements: 16.1, 16.2, 16.3
 */
EXPORT OptStatus slsqp_optimize(const SlsqpConfig* config,
                                 SlsqpWorkspace* ws,
                                 SlsqpResult* result) {
    /* Validate inputs */
    if (!config || !ws || !result) {
        return STATUS_INVALID_ARG;
    }
    
    int n = config->n;
    int meq = config->meq;
    int mineq = config->mineq;
    int m = meq + mineq;
    
    if (n <= 0 || !config->x || !config->obj_eval) {
        return STATUS_INVALID_ARG;
    }
    
    /* Initialize result */
    result->f = 0.0;
    result->iterations = 0;
    result->status = STATUS_CONVERGED;
    
    /* Reset workspace */
    slsqp_workspace_reset(ws);
    
    double* x = config->x;
    double* l = ws->l;
    double* g = ws->g;
    double* c = ws->c;
    double* a = ws->a;
    double* s = ws->s;
    double* u = ws->u;
    double* v = ws->v;
    double* r = ws->r;
    double* mu = ws->mu;
    double* x0 = ws->x0;
    double* w = ws->w;
    int* jw = ws->jw;
    
    const double* lower = config->lower;
    const double* upper = config->upper;
    double acc = config->accuracy;
    int max_iter = config->max_iter;
    long max_time = config->max_time;
    long start_time_us = (max_time > 0) ? get_time_us() : 0;
    int nnls_iter = config->nnls_iter > 0 ? config->nnls_iter : 3 * n;
    
    int n1 = n + 1;
    int n2 = n * n1 / 2;
    int la = (m > 1) ? m : 1;
    
    double tol = TEN * acc;
    int reset_count = 0;
    int bad_qp = 0;
    int iter = 0;  /* Iteration counter - matches Go's ctx.iter */
    
    /* Initialize LDL^T factorization: L = I, D = I */
    /* Matches Go's resetBFGS() */
    dzero(n2 + 1, l, 1);
    for (int i = 0, j = 0; i < n; i++) {
        l[j] = ONE;
        j += n - i;
    }
    reset_count = 1;  /* Matches Go: ctx.reset++ in resetBFGS */
    
    /* Initialize multipliers and search direction */
    /* Matches Go: dzero(c.s); dzero(c.mu) in initCtx */
    dzero(n + 1, s, 1);
    dzero(m > 0 ? m : 1, mu, 1);
    
    /* Evaluate initial function and gradient - matches Go's evalLoc(evalFunc) then evalLoc(evalGrad) */
    double f = config->obj_eval(config->eval_ctx, x, g, n);
    
    if (isnan(f) || isinf(f)) {
        result->status = STATUS_CALLBACK_ERROR;
        return STATUS_CALLBACK_ERROR;
    }
    
    /* Evaluate constraints and their gradients using batch callbacks */
    /* Constraint Jacobian a is stored in column-major order: a[constraint_idx + la * variable_idx] */
    /* This matches Go's storage: dcopy(o.n, tmp, 1, loc.a[i:], mda) */
    if (config->eq_eval && meq > 0) {
        /* Batch evaluate equality constraints: c[0..meq-1] and Jacobian rows 0..meq-1 */
        config->eq_eval(config->eval_ctx, x, c, a, meq, n);
    }
    if (config->ineq_eval && mineq > 0) {
        /* Batch evaluate inequality constraints: c[meq..m-1] and Jacobian rows meq..m-1 */
        config->ineq_eval(config->eval_ctx, x, &c[meq], &a[meq], mineq, n);
    }
    
    /* Variables for extended termination criteria */
    double f_old = f;  /* Previous function value for f_diff_tol check */
    double* x_old = (double*)alloca(n * sizeof(double));  /* Previous position for x_diff_tol check */
    dcopy(n, x, 1, x_old, 1);
    
    /* Main iteration loop - matches Go's mainLoop() */
    /* Go pattern: for mode == OK { if ctx.iter++; ctx.iter > spec.Stop.MaxIterations { return SQPExceedMaxIter } ... } */
    int mode = MODE_OK;  /* Matches Go's sqpMode */
    while (mode == MODE_OK) {
        /* Increment iteration counter at start of loop - matches Go: if ctx.iter++; ctx.iter > spec.Stop.MaxIterations */
        iter++;
        if (iter > max_iter) {
            iter--;  /* Matches Go: ctx.iter-- */
            result->f = f;
            result->iterations = iter;
            result->status = STATUS_MAX_ITER;
            return STATUS_MAX_ITER;
        }
        
        /* Check time limit */
        if (max_time > 0) {
            long elapsed_us = get_time_us() - start_time_us;
            if (elapsed_us >= max_time) {
                result->f = f;
                result->iterations = iter;
                result->status = STATUS_MAX_TIME;
                return STATUS_MAX_TIME;
            }
        }
        
        ws->iter = iter;
        
        /* Transfer bounds from l <= x <= u to l - x^k <= d <= u - x^k */
        for (int i = 0; i < n; i++) {
            u[i] = (lower && !isnan(lower[i])) ? lower[i] - x[i] : -INF_BND;
            v[i] = (upper && !isnan(upper[i])) ? upper[i] - x[i] : INF_BND;
        }
        
        /* Solve QP subproblem to get search direction s and multipliers r */
        double norm;
        int lsq_mode = lsq(m, meq, n, n2 + 1, l, g, a, c, u, v,
                       s, r, w, jw, nnls_iter, INF_BND, &norm);
        
        /* Handle singular C matrix case */
        /* Matches Go: if mode == LSEISingularC && n == meq { mode = ConsIncompatible } */
        if (lsq_mode == MODE_LSEI_SINGULAR && n == meq) {
            lsq_mode = MODE_CONS_INCOMPAT;
        }
        
        double h4 = ONE;
        /* Matches Go: if ctx.bad = mode == ConsIncompatible; ctx.bad { ... } */
        bad_qp = (lsq_mode == MODE_CONS_INCOMPAT);
        
        if (bad_qp) {
            /* Form augmented QP relaxation */
            /* Matches Go's augmented QP setup */
            double* aug_a = &a[n * la];
            for (int j = 0; j < m; j++) {
                if (j < meq) {
                    aug_a[j] = -c[j];  /* -c_j(x^k) */
                } else {
                    /* Matches Go: a[j] = math.Max(-c, zero) */
                    aug_a[j] = (c[j] <= ZERO) ? -c[j] : ZERO;  /* -zeta_j * c_j(x^k) */
                }
            }
            g[n] = ZERO;
            l[n2] = HUN;  /* rho = 100 */
            dzero(n, s, 1);
            s[n] = ONE;   /* delta = 1 */
            u[n] = ZERO;
            v[n] = ONE;   /* 0 <= delta <= 1 */
            
            /* Try to solve augmented problem with increasing penalty */
            /* Matches Go: for relax := 0; relax <= 5; relax++ { ... } */
            for (int relax = 0; relax <= 5; relax++) {
                lsq_mode = lsq(m, meq, n + 1, n2 + 1, l, g, a, c, u, v,
                           s, r, w, jw, nnls_iter, INF_BND, &norm);
                h4 = ONE - s[n];  /* 1 - delta */
                if (lsq_mode != MODE_CONS_INCOMPAT) {
                    break;
                }
                l[n2] *= TEN;  /* rho = rho * 10 */
            }
        }
        
        /* Unable to solve LSQ even the augmented one */
        /* Matches Go: if mode != HasSolution { return } */
        if (lsq_mode != MODE_HAS_SOLUTION) {
            result->f = f;
            result->iterations = iter;
            result->status = STATUS_CONSTRAINT_INCOMPATIBLE;
            return STATUS_CONSTRAINT_INCOMPATIBLE;
        }
        
        /* Update multipliers for L1-test */
        for (int i = 0; i < n; i++) {
            /* a[i*la:(i+1)*la] in Go is column i of the constraint Jacobian */
            v[i] = g[i] - ddot(m, &a[i * la], 1, r, 1);
        }
        
        /* Save current state */
        double f0 = f;
        dcopy(n, x, 1, x0, 1);
        
        /* Compute optimality and feasibility measures */
        double gs = ddot(n, g, 1, s, 1);  /* g'*d */
        double h1 = fabs(gs);              /* |g'*d| */
        double h2 = ZERO;                  /* constraint violation */
        
        for (int j = 0; j < m; j++) {
            double h3 = (j < meq) ? c[j] : ZERO;
            h2 += fmax(-c[j], h3);         /* ||c_j(x^k)||_1 */
            h3 = fabs(r[j]);               /* |lambda_j| */
            h1 += h3 * fabs(c[j]);         /* |lambda_j| * ||c_j(x^k)||_1 */
            mu[j] = fmax(h3, (mu[j] + h3) / 2);  /* rho_j^{k+1} */
        }
        
        /* Check convergence */
        if (h1 < acc && h2 < acc && !bad_qp && !isnan(f)) {
            result->f = f;
            result->iterations = iter;
            result->status = STATUS_CONVERGED;
            return STATUS_CONVERGED;
        }
        
        /* Compute directional derivative of merit function */
        h1 = ZERO;
        for (int j = 0; j < m; j++) {
            double h3_tmp = (j < meq) ? c[j] : ZERO;
            h1 += mu[j] * fmax(-c[j], h3_tmp);
        }
        
        double t0 = f + h1;  /* Merit function at x^k */
        double h3 = gs - h1 * h4;  /* Directional derivative */
        
        if (h3 >= ZERO) {
            /* Reset BFGS when ascent direction is generated */
            /* Matches Go: mode = ss.resetBFGS(); if ctx.reset > 5 { return } */
            reset_count++;
            if (reset_count > 5) {
                /* Check relaxed convergence */
                /* Matches Go's checkConv(ctx.tol, SearchNotDescent) */
                double vio = ZERO;
                if (check_conv(c, m, meq, tol, bad_qp, f, f0, s, n, config, x, x0, u, &vio)) {
                    result->f = f;
                    result->iterations = iter;
                    result->status = STATUS_CONVERGED;
                    return STATUS_CONVERGED;
                }
                /* Not converged even with relaxed tolerance */
                result->f = f;
                result->iterations = iter;
                result->status = STATUS_LINE_SEARCH_FAILED;
                return STATUS_LINE_SEARCH_FAILED;
            }
            /* Reset L = I, D = I */
            dzero(n2 + 1, l, 1);
            for (int i = 0, j = 0; i < n; i++) {
                l[j] = ONE;
                j += n - i;
            }
            continue;
        }
        
        /* Line search with merit function */
        /* Matches Go: ctx.line = 0; ctx.alpha = spec.Line.Alpha.Upper; ss.inexactSearch(); h3 *= ctx.alpha */
        int line_count = 0;
        double alpha = 1.0;  /* Will be set properly below */
        double alpha_min = 0.1;
        double alpha_max = 1.0;
        
        /* Store t0 in workspace for line search */
        ws->t0 = t0;
        
        if (config->exact_search) {
            /* Initialize exact line search */
            /* Matches Go: ctx.line = int(findNoop); ss.exactSearch(math.NaN()) */
            ws->line_mode = FIND_NOOP;
            
            /* First call to exactSearch with NaN to initialize */
            FindMode fm = exact_line_search(ws, NAN, tol, alpha_min, alpha_max, n, x, x0, s);
            (void)fm;  /* First call just initializes */
        } else {
            /* Initialize inexact line search */
            /* Matches Go: ctx.line = 0; ctx.alpha = spec.Line.Alpha.Upper; ss.inexactSearch(); h3 *= ctx.alpha */
            line_count = 1;
            alpha = alpha_max;
            
            /* inexactSearch: scale s by alpha, then x = x0 + s */
            dscal(n, alpha, s, 1);  /* s = alpha * d */
            dcopy(n, x0, 1, x, 1);
            daxpy(n, ONE, s, 1, x, 1);  /* x = x0 + s */
            
            /* Project onto bounds - matches Go's inexactSearch */
            for (int i = 0; i < n; i++) {
                double lb = (lower && !isnan(lower[i])) ? lower[i] : -INF_BND;
                double ub = (upper && !isnan(upper[i])) ? upper[i] : INF_BND;
                if (lb > -INF_BND && x[i] < lb) {
                    x[i] = lb;
                } else if (ub < INF_BND && x[i] > ub) {
                    x[i] = ub;
                }
            }
            
            h3 *= alpha;  /* Update directional derivative */
        }
        
        /* Line search loop - matches Go: for mode = evalFunc; mode == evalFunc; { mode = ss.lineSearch(&h3) } */
        int ls_mode = 1;  /* 1 = evalFunc, 0 = done */
        for (int ls_iter = 0; ls_iter < 20 && ls_mode == 1; ls_iter++) {
            /* Evaluate function at current point - matches Go's evalLoc(evalFunc) */
            f = config->obj_eval(config->eval_ctx, x, g, n);
            
            if (isnan(f) || isinf(f)) {
                result->f = f0;
                result->iterations = iter;
                result->status = STATUS_CALLBACK_ERROR;
                return STATUS_CALLBACK_ERROR;
            }
            
            /* Evaluate constraints using batch callbacks */
            if (config->eq_eval && meq > 0) {
                config->eq_eval(config->eval_ctx, x, c, a, meq, n);
            }
            if (config->ineq_eval && mineq > 0) {
                config->ineq_eval(config->eval_ctx, x, &c[meq], &a[meq], mineq, n);
            }
            
            /* Compute merit function at new point: t = f + sum(mu_j * ||c_j||_1) */
            double t = f;
            for (int j = 0; j < m; j++) {
                double tmp = (j < meq) ? c[j] : ZERO;
                t += mu[j] * fmax(-c[j], tmp);
            }
            
            /* Line search decision - matches Go's lineSearch function */
            if (!config->exact_search) {
                /* Inexact line search (Armijo backtracking) */
                /* Matches Go: if h1 := t - ctx.t0; !li.Exact { ... } */
                double h1_ls = t - t0;
                
                if (h1_ls <= h3 / 10.0 || line_count > 10) {
                    /* Armijo condition satisfied or max iterations reached */
                    /* Matches Go: *h3, mode = ss.checkConv(ctx.acc, evalGrad) */
                    double vio = ZERO;
                    
                    /* Use check_conv which matches Go's checkConv -> checkStop */
                    if (check_conv(c, m, meq, acc, bad_qp, f, f0, s, n, config, x, x0, u, &vio)) {
                        result->f = f;
                        result->iterations = iter;
                        result->status = STATUS_CONVERGED;
                        return STATUS_CONVERGED;
                    }
                    h3 = vio;  /* Update h3 to violation for later use */
                    ls_mode = 0;  /* Exit line search, continue to BFGS update */
                } else {
                    /* Armijo condition not satisfied, reduce step */
                    /* Matches Go: ctx.alpha = math.Min(math.Max(*h3/(2*(*h3-h1)), al), au) */
                    alpha = h3 / (2.0 * (h3 - h1_ls));
                    alpha = fmax(fmin(alpha, alpha_max), alpha_min);
                    
                    /* Call inexactSearch again - scales s by alpha */
                    /* Matches Go: ss.inexactSearch(); *h3 *= ctx.alpha */
                    line_count++;
                    dscal(n, alpha, s, 1);  /* s = alpha * s (accumulates) */
                    dcopy(n, x0, 1, x, 1);
                    daxpy(n, ONE, s, 1, x, 1);  /* x = x0 + s */
                    
                    /* Project onto bounds */
                    for (int i = 0; i < n; i++) {
                        double lb = (lower && !isnan(lower[i])) ? lower[i] : -INF_BND;
                        double ub = (upper && !isnan(upper[i])) ? upper[i] : INF_BND;
                        if (lb > -INF_BND && x[i] < lb) {
                            x[i] = lb;
                        } else if (ub < INF_BND && x[i] > ub) {
                            x[i] = ub;
                        }
                    }
                    
                    h3 *= alpha;  /* Update directional derivative */
                    ls_mode = 1;  /* Continue line search */
                }
            } else {
                /* Exact line search using golden section + quadratic interpolation */
                /* Matches Go: if ss.exactSearch(t) == findConv { ... } */
                FindMode fm = exact_line_search(ws, t, tol, alpha_min, alpha_max, n, x, x0, s);
                
                if (fm == FIND_CONV) {
                    /* Search converged - s has been scaled to alpha * s */
                    /* Matches Go: *h3, mode = ss.checkConv(ctx.acc, evalGrad) */
                    double vio = ZERO;
                    
                    /* Use check_conv which matches Go's checkConv -> checkStop */
                    if (check_conv(c, m, meq, acc, bad_qp, f, f0, s, n, config, x, x0, u, &vio)) {
                        result->f = f;
                        result->iterations = iter;
                        result->status = STATUS_CONVERGED;
                        return STATUS_CONVERGED;
                    }
                    h3 = vio;
                    ls_mode = 0;  /* Exit line search, continue to BFGS update */
                } else {
                    /* Need more function evaluations */
                    ls_mode = 1;  /* Continue line search */
                }
            }
        }
        
        /* After line search, check if mode indicates convergence */
        /* In Go, if mode == OK after lineSearch, the function returns */
        /* This is handled by the check_conv calls above */
        
        /* BFGS update */
        /* Matches Go: if mode == evalGrad { mode = ss.updateBFGS() } */
        /* First, evaluate gradients - matches Go's evalLoc(evalGrad) in updateBFGS */
        /* Note: gradients were already evaluated in the line search loop above */
        
        /* Compute eta = grad_L(x^{k+1}, lambda^k) - grad_L(x^k, lambda^k) */
        for (int i = 0; i < n; i++) {
            /* a[i*la:(i+1)*la] in Go is column i of the constraint Jacobian */
            u[i] = g[i] - ddot(m, &a[i * la], 1, r, 1) - v[i];
        }
        
        /* Compute L'*s */
        for (int i = 0, k = 0; i < n; i++) {
            k++;
            double sm = ZERO;
            for (int j = i + 1; j < n; j++) {
                sm += l[k] * s[j];
                k++;
            }
            v[i] = s[i] + sm;
        }
        
        /* Compute D*L'*s */
        for (int i = 0, k = 0; i < n; i++) {
            v[i] = l[k] * v[i];
            k += n - i;
        }
        
        /* Compute L*D*L'*s = B^k*s */
        for (int i = n - 1; i >= 0; i--) {
            int k = i;
            double sm = ZERO;
            for (int j = 0; j < i; j++) {
                sm += l[k] * v[j];
                k += n - 1 - j;
            }
            v[i] += sm;
        }
        
        h1 = ddot(n, s, 1, u, 1);  /* s'*eta */
        h2 = ddot(n, s, 1, v, 1);  /* s'*B^k*s */
        h3 = 0.2 * h2;
        
        if (h1 < h3) {
            /* theta = 4/5 * s'*B^k*s / (s'*B^k*s - s'*eta) */
            h4 = (h2 - h3) / (h2 - h1);
            h1 = h3;
            dscal(n, h4, u, 1);
            daxpy(n, ONE - h4, v, 1, u, 1);
        }
        
        if (h1 == ZERO || h2 == ZERO) {
            /* Reset BFGS - matches Go resetBFGS() */
            reset_count++;
            if (reset_count > 5) {
                /* Check relaxed convergence in case of positive directional derivative */
                /* This matches Go's checkConv(ctx.tol, SearchNotDescent) */
                /* Note: Go uses ctx.tol (= 10 * acc) for relaxed convergence */
                double vio = ZERO;
                if (check_conv(c, m, meq, tol, bad_qp, f, f0, s, n, config, x, x0, u, &vio)) {
                    result->f = f;
                    result->iterations = iter;
                    result->status = STATUS_CONVERGED;
                    return STATUS_CONVERGED;
                }
                /* Not converged even with relaxed tolerance */
                result->f = f;
                result->iterations = iter;
                result->status = STATUS_LINE_SEARCH_FAILED;
                return STATUS_LINE_SEARCH_FAILED;
            }
            /* Reset L = I, D = I */
            dzero(n2 + 1, l, 1);
            for (int i = 0, j = 0; i < n; i++) {
                l[j] = ONE;
                j += n - i;
            }
            /* Note: Do NOT reset reset_count here - Go doesn't reset ctx.reset after BFGS reset */
        } else {
            /* Update LDL^T factorization */
            compositeT(n, l, u, ONE / h1, NULL);
            compositeT(n, l, v, -ONE / h2, u);
            /* Note: Do NOT reset reset_count here - Go doesn't reset ctx.reset after successful BFGS update */
        }
        
        /* Store current function value for extended termination check in next iteration */
        result->f = f;
        
        /* Note: The extended termination criteria check has been integrated into check_conv */
        /* which is called during line search convergence checks above */
        
        /* Update x_old for next iteration's extended termination check */
        dcopy(n, x, 1, x_old, 1);
        f_old = f;
    }
    
    /* Maximum iterations reached */
    result->f = f;
    result->iterations = max_iter;
    result->status = STATUS_MAX_ITER;
    return STATUS_MAX_ITER;
}
