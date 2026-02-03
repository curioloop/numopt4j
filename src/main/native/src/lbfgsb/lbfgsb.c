/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * L-BFGS-B algorithm implementation - Main entry and iteration driver.
 * Based on the Go implementation in lbfgsb/optimize.go and lbfgsb/driver.go.
 *
 * ============================================================================
 * L-BFGS-B Algorithm Overview
 * ============================================================================
 *
 * L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bound constraints)
 * is a quasi-Newton method for solving large-scale bound-constrained optimization:
 *
 *   minimize f(x)
 *   subject to l ≤ x ≤ u
 *
 * where f: ℝⁿ → ℝ is a smooth function, and l, u ∈ ℝⁿ are bound vectors.
 *
 * The algorithm maintains a limited-memory approximation Bₖ of the Hessian matrix
 * using the most recent m correction pairs {sᵢ, yᵢ} where:
 *   sᵢ = xᵢ₊₁ - xᵢ  (step)
 *   yᵢ = gᵢ₊₁ - gᵢ  (gradient difference)
 *
 * The L-BFGS approximation has the form:
 *   Bₖ = θI - WₖMₖWₖᵀ
 *
 * where:
 *   θ = yₖᵀyₖ / yₖᵀsₖ  (scaling factor)
 *   Wₖ = [Y, θS]       (2m × n matrix of corrections)
 *   Mₖ = middle matrix computed from correction pairs
 *
 * ============================================================================
 * Main Iteration Flow (matches Go driver.go mainLoop)
 * ============================================================================
 *
 * The optimization proceeds as follows:
 *
 * 1. INITIALIZATION
 *    - Project initial point x₀ onto feasible region [l, u]
 *    - Compute f₀ = f(x₀) and g₀ = ∇f(x₀)
 *    - Check initial convergence: ‖proj(g₀)‖∞ ≤ pgtol
 *
 * 2. MAIN LOOP (for k = 0, 1, 2, ...)
 *
 *    2a. GENERALIZED CAUCHY POINT (GCP) - cauchy.c
 *        Find xᶜ by minimizing the quadratic model mₖ(x) along the
 *        piecewise linear path from xₖ to the projected steepest descent:
 *
 *        mₖ(x) = fₖ + gₖᵀ(x-xₖ) + ½(x-xₖ)ᵀBₖ(x-xₖ)
 *
 *        The GCP identifies the set of free variables 𝓕 (not at bounds).
 *
 *    2b. SUBSPACE MINIMIZATION - subspace.c
 *        If |𝓕| > 0 and col > 0, minimize mₖ(x) over the subspace of
 *        free variables starting from xᶜ:
 *
 *        minimize   m̃ₖ(d̃) = d̃ᵀr̃ᶜ + ½d̃ᵀB̃ₖd̃
 *        subject to lᵢ - xᶜᵢ ≤ d̃ᵢ ≤ uᵢ - xᶜᵢ  (i ∈ 𝓕)
 *
 *        where:
 *          B̃ₖ = ZₖᵀBₖZₖ  (reduced Hessian)
 *          r̃ᶜ = Zₖᵀ(gₖ + Bₖ(xᶜ-xₖ))  (reduced gradient)
 *          Zₖ is the n × |𝓕| selection matrix for free variables
 *
 *        This yields the search direction dₖ = x̂ - xₖ.
 *
 *    2c. LINE SEARCH - linesearch.c, minpack.c
 *        Find step length λₖ satisfying Wolfe conditions:
 *
 *        Sufficient decrease: f(xₖ + λₖdₖ) ≤ f(xₖ) + αλₖgₖᵀdₖ
 *        Curvature condition: |g(xₖ + λₖdₖ)ᵀdₖ| ≤ β|gₖᵀdₖ|
 *
 *        Set xₖ₊₁ = xₖ + λₖdₖ.
 *
 *    2d. CONVERGENCE CHECK
 *        Check stopping criteria:
 *        - Projected gradient: ‖proj(gₖ₊₁)‖∞ ≤ pgtol
 *        - Function reduction: (fₖ - fₖ₊₁)/max(|fₖ|,|fₖ₊₁|,1) ≤ factr×ε
 *        - Iteration limit: k+1 > max_iter
 *        - Evaluation limit: total_eval ≥ max_eval
 *
 *    2e. BFGS UPDATE - update.c
 *        If not converged, update the L-BFGS approximation:
 *        - Compute sₖ = xₖ₊₁ - xₖ and yₖ = gₖ₊₁ - gₖ
 *        - Add (sₖ, yₖ) to correction history (circular buffer)
 *        - Update θ = yₖᵀyₖ / yₖᵀsₖ
 *        - Form T matrix (Cholesky factor of θSᵀS + LD⁻¹Lᵀ)
 *
 * 3. TERMINATION
 *    Return final x, f(x), and convergence status.
 *
 * ============================================================================
 * Error Recovery
 * ============================================================================
 *
 * When numerical issues occur (singular matrices, failed line search), the
 * algorithm attempts recovery by resetting the BFGS approximation to the
 * identity matrix (Bₖ = I). This is equivalent to restarting with steepest
 * descent. A maximum of 5 resets is allowed per optimization run.
 *
 * ============================================================================
 * References
 * ============================================================================
 *
 * [1] R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, "A Limited Memory Algorithm
 *     for Bound Constrained Optimization", SIAM J. Scientific Computing,
 *     16(5):1190-1208, 1995.
 *
 * [2] C. Zhu, R. H. Byrd, P. Lu, and J. Nocedal, "Algorithm 778: L-BFGS-B:
 *     Fortran Subroutines for Large-Scale Bound-Constrained Optimization",
 *     ACM Trans. Math. Software, 23(4):550-560, 1997.
 *
 * ============================================================================
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>

/* ============================================================================
 * Variable Status Constants (matches Go varWhere type in base.go)
 * ============================================================================
 *
 * These constants track the status of each variable during optimization:
 *
 * VAR_NOT_MOVE (-3): Variable is free with bounds but did not move
 *                    (Go: varNotMove)
 * VAR_UNBOUND (-1):  Variable has no bounds, always free
 *                    (Go: varUnbound)
 * VAR_FREE (0):      Variable is free and has bounds
 *                    (Go: varFree)
 * VAR_AT_LOWER (1):  Variable is at lower bound lᵢ, with uᵢ ≠ lᵢ
 *                    (Go: varAtLB)
 * VAR_AT_UPPER (2):  Variable is at upper bound uᵢ, with uᵢ ≠ lᵢ
 *                    (Go: varAtUB)
 * VAR_FIXED (3):     Variable is always fixed (uᵢ = xᵢ = lᵢ)
 *                    (Go: varFixed)
 */

#define VAR_NOT_MOVE  -3
#define VAR_UNBOUND   -1
#define VAR_FREE       0
#define VAR_AT_LOWER   1
#define VAR_AT_UPPER   2
#define VAR_FIXED      3

/* ============================================================================
 * Solution Status Constants (matches Go solutionWithinBox/solutionBeyondBox)
 * ============================================================================
 *
 * These constants indicate the result of subspace minimization:
 *
 * SOLUTION_UNKNOWN (-1):    Subspace minimization not performed
 *                           (Go: solutionUnknown = -1)
 * SOLUTION_WITHIN_BOX (0):  Unconstrained minimizer is within bounds
 *                           (Go: solutionWithinBox = 0)
 * SOLUTION_BEYOND_BOX (1):  Minimizer hit a bound constraint
 *                           (Go: solutionBeyondBox = 1)
 */
#define SOLUTION_UNKNOWN    -1
#define SOLUTION_WITHIN_BOX  0
#define SOLUTION_BEYOND_BOX  1

/* ============================================================================
 * Line Search Constants (matches Go linesearch.go constants)
 * ============================================================================
 *
 * SEARCH_NO_BND:    Large value indicating no bound on step length
 * SEARCH_ALPHA:     Sufficient decrease parameter (Armijo condition)
 *                   f(x + λd) ≤ f(x) + α·λ·g'·d
 * SEARCH_BETA:      Curvature condition parameter
 *                   |g(x + λd)'·d| ≤ β·|g'·d|
 * SEARCH_EPS:       Relative tolerance for step length
 * SEARCH_BACK_EXIT: Maximum backtracking iterations before failure
 * SEARCH_BACK_SLOW: Threshold for "too many searches" warning
 */
#define SEARCH_NO_BND    1.0e+10
#define SEARCH_ALPHA     1.0e-3
#define SEARCH_BETA      0.9
#define SEARCH_EPS       0.1
#define SEARCH_BACK_EXIT 20
#define SEARCH_BACK_SLOW 10

/* ============================================================================
 * Error Codes (defined in optimizer.h, documented here for reference)
 * ============================================================================
 *
 * ERR_NONE (0):                No error
 * ERR_NOT_POS_DEF (-1):        Matrix not positive definite (1st Cholesky)
 * ERR_NOT_POS_DEF_2ND_K (-2):  2nd K matrix not positive definite
 * ERR_NOT_POS_DEF_T (-3):      T matrix not positive definite
 * ERR_DERIVATIVE (-4):         Invalid derivative (not descent direction)
 * ERR_SINGULAR_TRIANGULAR (-5): Triangular matrix is singular
 * ERR_LINE_SEARCH_FAILED (-6): Line search failed to find valid step
 * ERR_LINE_SEARCH_TOL (-7):    Line search tolerance error
 * ERR_TOO_MANY_RESETS (-8):    Too many BFGS matrix resets
 *
 * These match Go's errInfo type in base.go:
 *   ok = 0, errNotPosDef1stK = -1, errNotPosDef2ndK = -2, etc.
 */

/* Minpack line search task constants (matches Go SearchTask in base.go) */
#define MINPACK_SEARCH_START  0
#define MINPACK_SEARCH_CONV   (1 << 5)
#define MINPACK_SEARCH_FG     (1 << 6)
#define MINPACK_SEARCH_ERROR  (1 << 7)
#define MINPACK_SEARCH_WARN   (1 << 8)

/* ============================================================================
 * Forward Declarations for Module Functions
 * ============================================================================
 *
 * These functions implement the core L-BFGS-B algorithm components:
 *
 * Cauchy Point Computation (cauchy.c):
 *   cauchy_point - Compute Generalized Cauchy Point (GCP)
 *   bmv          - Matrix-vector product with L-BFGS matrix B
 *   heap_sort_out - Sort breakpoints for piecewise linear path
 *   free_var     - Identify free variables at GCP
 *
 * Subspace Minimization (subspace.c):
 *   reduce_gradient    - Compute reduced gradient r̃ᶜ = Zᵀ(g + B(xᶜ-x))
 *   optimal_direction  - Find optimal direction in subspace
 *
 * BFGS Update (update.c):
 *   update_correction - Add new (s,y) pair to correction history
 *   form_t           - Form T matrix (Cholesky of θSᵀS + LD⁻¹Lᵀ)
 *   form_k           - Form K matrix for subspace minimization
 *
 * Projection Operations (project.c):
 *   proj_grad_norm   - Compute ‖proj(g)‖∞
 *   proj_init_active - Project initial point and identify active set
 *   project_x        - Project x onto feasible region [l, u]
 *
 * Line Search (linesearch.c):
 *   init_line_search    - Initialize line search parameters
 *   perform_line_search - Execute one line search iteration
 */
extern int cauchy_point(int n, int m, const double* x, const double* g,
                        const double* lower, const double* upper,
                        const int* bound_type, double* z, LbfgsbWorkspace* ws);
extern int bmv(int m, int col, const double* sy, const double* wt,
               const double* v, double* p);
extern void heap_sort_out(int n, double* t, int* order, int sorted);
extern int free_var(int n, LbfgsbWorkspace* ws);

extern int reduce_gradient(int n, int m, const double* x, const double* g,
                           const double* z, double* r, LbfgsbWorkspace* ws);
extern int optimal_direction(int n, int m, const double* x, const double* g,
                             const double* lower, const double* upper,
                             const int* bound_type, double* z, double* r,
                             LbfgsbWorkspace* ws);

extern void update_correction(int n, int m, const double* s, const double* y,
                              LbfgsbWorkspace* ws);
extern int form_t(int m, LbfgsbWorkspace* ws);
extern int form_k(int n, int m, LbfgsbWorkspace* ws);

extern double proj_grad_norm(int n, const double* x, const double* g,
                             const double* lower, const double* upper,
                             const int* bound_type);
extern void proj_init_active(int n, double* x, const double* lower,
                             const double* upper, const int* bound_type,
                             int* iwhere, int* out_projected,
                             int* out_constrained, int* out_boxed);
extern void project_x(int n, double* x, const double* lower, const double* upper,
                      const int* bound_type);

extern double init_line_search(int n, int m, const double* x,
                               const double* lower, const double* upper,
                               const int* bound_type, LbfgsbWorkspace* ws,
                               int* out_task);
extern int perform_line_search(int n, double* x, double f, const double* g,
                               double* stp, int* task,
                               LbfgsbWorkspace* ws, int* out_done);

/* ============================================================================
 * External BLAS Functions (from blas.c)
 * ============================================================================
 *
 * These are Level 1 BLAS operations used throughout the algorithm:
 *
 * dcopy(n, x, incx, y, incy) - Copy vector: y ← x
 * daxpy(n, a, x, incx, y, incy) - Vector update: y ← y + a·x
 * ddot(n, x, incx, y, incy) - Dot product: returns xᵀy
 * dnrm2(n, x, incx) - Euclidean norm: returns ‖x‖₂
 * dscal(n, a, x, incx) - Scale vector: x ← a·x
 * damax(n, x, incx) - Maximum absolute value: returns max|xᵢ|
 */
extern void dcopy(int n, const double* x, int incx, double* y, int incy);
extern void daxpy(int n, double a, const double* x, int incx, double* y, int incy);
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern double dnrm2(int n, const double* x, int incx);
extern void dscal(int n, double a, double* x, int incx);
extern double damax(int n, const double* x, int incx);

/* Constants */

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/* ============================================================================
 * BFGS Reset Function
 * ============================================================================
 *
 * This function resets the L-BFGS approximation to the identity matrix (B = I).
 * It is called when numerical issues occur during optimization:
 *
 * - Singular triangular system in Cauchy point computation
 * - Non-positive definite matrix in Cholesky factorization
 * - Line search failure with existing BFGS corrections
 *
 * The reset clears the correction history and restarts with steepest descent.
 * This matches Go's iterBFGS.reset() method in base.go.
 *
 * Recovery is limited to 5 resets per optimization run to prevent infinite
 * loops in pathological cases.
 *
 * @param ws Workspace containing BFGS state
 * @return 0 on success, ERR_TOO_MANY_RESETS if reset limit exceeded
 */
static int reset_bfgs_to_identity(LbfgsbWorkspace* ws) {
    if (!ws) {
        return ERR_TOO_MANY_RESETS;
    }
    
    /* Increment reset counter */
    ws->reset_count++;
    
    /* Check if we've exceeded the maximum number of resets */
    if (ws->reset_count > 5) {
        return ERR_TOO_MANY_RESETS;
    }
    
    /* Reset BFGS approximation to identity matrix
     * This is done by clearing the correction history:
     * - col = 0: No corrections stored
     * - head = 0: Reset circular buffer head
     * - tail = 0: Reset circular buffer tail
     * - theta = 1.0: Reset scaling factor to 1 (identity)
     * - updates = 0: Reset update counter
     * - updated = 0: Mark as not updated
     * 
     * With col = 0, the algorithm will use B = θI = I as the
     * initial Hessian approximation, effectively restarting
     * the quasi-Newton method from scratch.
     *
     * This matches Go's iterBFGS.reset() in base.go:
     *   c.col = 0
     *   c.head = 0
     *   c.tail = 0
     *   c.theta = one
     *   c.updates = 0
     *   c.updated = false
     */
    ws->col = 0;
    ws->head = 0;
    ws->tail = 0;
    ws->theta = 1.0;
    ws->updates = 0;
    ws->updated = 0;
    
    return 0;
}

/* ============================================================================
 * Main Optimization Function: lbfgsb_optimize
 * ============================================================================
 *
 * This function implements the main L-BFGS-B iteration loop, matching the
 * Go implementation in driver.go (mainLoop function).
 *
 * The iteration flow is:
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │ INITIALIZATION                                                      │
 *   │   1. Project x₀ onto feasible region [l, u]                        │
 *   │   2. Compute f₀ = f(x₀) and g₀ = ∇f(x₀)                           │
 *   │   3. Check initial convergence: ‖proj(g₀)‖∞ ≤ pgtol               │
 *   └─────────────────────────────────────────────────────────────────────┘
 *                                    │
 *                                    ▼
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │ MAIN LOOP (for k = 0, 1, 2, ...)                                   │
 *   │                                                                     │
 *   │   ┌─────────────────────────────────────────────────────────────┐  │
 *   │   │ Step 1: Search GCP (Generalized Cauchy Point)               │  │
 *   │   │   - If unconstrained and col > 0: z ← x (skip GCP)         │  │
 *   │   │   - Else: compute xᶜ by cauchy_point()                     │  │
 *   │   │   - Identify free variables by free_var()                  │  │
 *   │   └─────────────────────────────────────────────────────────────┘  │
 *   │                              │                                      │
 *   │                              ▼                                      │
 *   │   ┌─────────────────────────────────────────────────────────────┐  │
 *   │   │ Step 2: Minimize Subspace (if free > 0 and col > 0)        │  │
 *   │   │   - Form K matrix by form_k() if needed                    │  │
 *   │   │   - Compute reduced gradient by reduce_gradient()          │  │
 *   │   │   - Find optimal direction by optimal_direction()          │  │
 *   │   └─────────────────────────────────────────────────────────────┘  │
 *   │                              │                                      │
 *   │                              ▼                                      │
 *   │   ┌─────────────────────────────────────────────────────────────┐  │
 *   │   │ Step 3: Line Search                                         │  │
 *   │   │   - Compute search direction d = z - x                     │  │
 *   │   │   - Initialize line search by init_line_search()           │  │
 *   │   │   - Save current state (x, f, g)                           │  │
 *   │   │   - Iterate perform_line_search() until done               │  │
 *   │   │   - Restore state if line search fails                     │  │
 *   │   └─────────────────────────────────────────────────────────────┘  │
 *   │                              │                                      │
 *   │                              ▼                                      │
 *   │   ┌─────────────────────────────────────────────────────────────┐  │
 *   │   │ Step 4: New Iteration Checks                                │  │
 *   │   │   - Check iteration limit: iter > max_iter                 │  │
 *   │   │   - Check evaluation limit: total_eval ≥ max_eval          │  │
 *   │   │   - Check gradient threshold: ‖d‖ ≤ pgtol·(1+|f|)          │  │
 *   │   └─────────────────────────────────────────────────────────────┘  │
 *   │                              │                                      │
 *   │                              ▼                                      │
 *   │   ┌─────────────────────────────────────────────────────────────┐  │
 *   │   │ Step 5: Convergence Check                                   │  │
 *   │   │   - Compute ‖proj(g)‖∞ by proj_grad_norm()                 │  │
 *   │   │   - Check: ‖proj(g)‖∞ ≤ pgtol                              │  │
 *   │   │   - Check: (f_old - f)/max(|f_old|,|f|,1) ≤ factr·ε        │  │
 *   │   └─────────────────────────────────────────────────────────────┘  │
 *   │                              │                                      │
 *   │                              ▼                                      │
 *   │   ┌─────────────────────────────────────────────────────────────┐  │
 *   │   │ Step 6: BFGS Update                                         │  │
 *   │   │   - Compute s = x_new - x_old                              │  │
 *   │   │   - Compute y = g_new - g_old                              │  │
 *   │   │   - Update corrections by update_correction()              │  │
 *   │   │   - Form T matrix by form_t()                              │  │
 *   │   └─────────────────────────────────────────────────────────────┘  │
 *   │                              │                                      │
 *   │                              ▼                                      │
 *   │                    Continue to next iteration                       │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * Error Recovery:
 *   When numerical issues occur (info != 0), the algorithm attempts to
 *   recover by resetting the BFGS approximation to identity (B = I).
 *   This matches Go's handling: if info != ok { info = ok; ctx.reset() }
 *
 * @param config  Configuration parameters (read-only)
 * @param ws      Workspace (read-write)
 * @param result  Result output
 * @return        Optimization status
 */
EXPORT OptStatus lbfgsb_optimize(const LbfgsbConfig* config,
                                  LbfgsbWorkspace* ws,
                                  LbfgsbResult* result) {
    /* Validate inputs */
    if (!config || !ws || !result) {
        return STATUS_INVALID_ARG;
    }
    
    int n = config->n;
    int m = config->m;
    
    if (n <= 0 || m <= 0 || !config->x || !config->eval) {
        return STATUS_INVALID_ARG;
    }
    
    /* Initialize result */
    result->f = 0.0;
    result->iterations = 0;
    result->evaluations = 0;
    result->status = STATUS_CONVERGED;
    
    /* Reset workspace */
    lbfgsb_workspace_reset(ws);
    
    double* x = config->x;
    double* g = ws->g;
    double* z = ws->z;
    double* d = ws->d;
    double* xp = ws->xp;
    double* t = ws->t;
    double* r = ws->r;
    int* iwhere = ws->iwhere;
    
    const double* lower = config->lower;
    const double* upper = config->upper;
    const int* bound_type = config->bound_type;
    
    double factr = config->factr;
    double pgtol = config->pgtol;
    int max_iter = config->max_iter;
    int max_eval = config->max_eval;
    long max_time = config->max_time;
    long start_time_us = (max_time > 0) ? get_time_us() : 0;
    
    /* ========================================================================
     * Initialization (matching Go driver.go mainLoop initialization)
     * ========================================================================
     *
     * This section corresponds to Go's mainLoop initialization:
     *   ctx.clear()
     *   ctx.global.reset()
     *   d.printInit()
     *   projInitActive(loc, spec, ctx)
     *   task = d.nextLocation(iterLoop)  // Compute f₀ and g₀
     *   task = d.checkConvergence(task)  // Check initial convergence
     */
    
    /* Project initial point and initialize variable status
     * This matches Go's projInitActive(loc, spec, ctx)
     *
     * For each variable xᵢ:
     *   - If lᵢ ≤ xᵢ ≤ uᵢ: keep xᵢ, mark as free or at bound
     *   - If xᵢ < lᵢ: set xᵢ = lᵢ, mark as at lower bound
     *   - If xᵢ > uᵢ: set xᵢ = uᵢ, mark as at upper bound
     */
    int projected, constrained, boxed;
    proj_init_active(n, x, lower, upper, bound_type, iwhere,
                     &projected, &constrained, &boxed);
    
    ws->constrained = constrained;
    ws->boxed = boxed;
    
    /* Initialize free variable count (all variables start as potentially free)
     * This matches Go's iterCtx.clear() initialization */
    ws->free = n;
    ws->active = 0;
    ws->enter = 0;
    ws->leave = n;
    ws->updated = 0;
    ws->updates = 0;
    
    /* Initialize index array: all variables are initially free
     * index[i] = i means variable i is in the free set
     * This will be updated by free_var() after GCP computation */
    for (int i = 0; i < n; i++) {
        ws->index[i] = i;
    }
    
    /* Calculate f₀ and g₀ (matching Go's d.nextLocation(iterLoop))
     * This is the first function evaluation at the projected initial point */
    double f = config->eval(config->eval_ctx, x, g, n);
    ws->total_eval++;
    
    if (isnan(f) || isinf(f)) {
        result->status = STATUS_CALLBACK_ERROR;
        return STATUS_CALLBACK_ERROR;
    }
    
    ws->f = f;
    
    /* Compute projected gradient norm (matching Go's d.checkConvergence)
     *
     * The projected gradient is defined as:
     *   proj(g)ᵢ = gᵢ           if lᵢ < xᵢ < uᵢ  (free variable)
     *            = min(gᵢ, 0)   if xᵢ = lᵢ       (at lower bound)
     *            = max(gᵢ, 0)   if xᵢ = uᵢ       (at upper bound)
     *
     * The infinity norm ‖proj(g)‖∞ = max|proj(g)ᵢ| measures how far
     * the current point is from satisfying the KKT conditions.
     */
    double sbg_norm = proj_grad_norm(n, x, g, lower, upper, bound_type);
    ws->sbg_norm = sbg_norm;
    
    /* Check initial convergence
     * If ‖proj(g₀)‖∞ ≤ pgtol, the initial point already satisfies
     * the optimality conditions (within tolerance) */
    if (sbg_norm <= pgtol) {
        result->f = f;
        result->iterations = 0;
        result->evaluations = ws->total_eval;
        result->status = STATUS_GRAD_TOL;
        return STATUS_GRAD_TOL;
    }
    
    /* ========================================================================
     * Main iteration loop (matching Go driver.go mainLoop)
     * ========================================================================
     *
     * The main loop structure matches Go's:
     *   for task == iterLoop {
     *       if info != ok { info = ok; ctx.reset() }
     *       if info, wrk = d.searchGCP(); info != ok { continue }
     *       if info = d.minimizeSubspace(wrk); info != ok { continue }
     *       if info = d.searchOptimalStep(&task); info != ok { continue }
     *       task = d.newIteration(task)
     *       task = d.checkConvergence(task)
     *       d.printIter()
     *       if task == iterLoop { info = d.updateBFGS() }
     *   }
     */
    
    int info = 0;  /* Error info from subroutines (0 = ok) */
    int wrk = 0;   /* Whether K matrix needs recomputation */
    
    for (int iter = 0; iter < max_iter; iter++) {
        ws->iter = iter;
        ws->f_old = f;
        
        /* Handle BFGS reset if needed (matching Go's info != ok check)
         *
         * When a numerical error occurred in the previous iteration
         * (singular matrix, non-positive definite Cholesky, etc.),
         * we reset the BFGS approximation to identity and restart.
         *
         * This matches Go's:
         *   if info != ok {
         *       info = ok
         *       ctx.reset()
         *       if log.enable(LogLast) {
         *           log.log("Refreshing LBFGS memory and restarting iteration.\n")
         *       }
         *   }
         */
        if (info != 0) {
            info = 0;
            int reset_status = reset_bfgs_to_identity(ws);
            if (reset_status != 0) {
                /* Too many resets - give up */
                result->f = f;
                result->iterations = ws->iter;
                result->evaluations = ws->total_eval;
                result->status = STATUS_LINE_SEARCH_FAILED;
                return STATUS_LINE_SEARCH_FAILED;
            }
            /* Reset successful - continue with identity BFGS matrix */
        }
        
        /* ====================================================================
         * Step 1: Search GCP (matching Go's d.searchGCP())
         * ====================================================================
         *
         * The Generalized Cauchy Point (GCP) xᶜ is found by minimizing
         * the quadratic model mₖ(x) along the piecewise linear path
         * from xₖ toward the projected steepest descent direction.
         *
         * The path is defined by breakpoints where variables hit bounds.
         * At each segment, the model is quadratic, so we can find the
         * minimum analytically.
         *
         * Special case: If the problem is unconstrained and we have
         * BFGS corrections (col > 0), we skip GCP and use z = x directly.
         * This matches Go's:
         *   if !ctx.constrained && ctx.col > 0 {
         *       dcopy(spec.n, loc.x, 1, ctx.z, 1)
         *       wrk = ctx.updated
         *       ctx.seg = 0
         *   }
         */
        
        /* Skip the search for GCP if unconstrained and col > 0 */
        if (!ws->constrained && ws->col > 0) {
            /* Copy x to z directly */
            dcopy(n, x, 1, z, 1);
            wrk = ws->updated;
            ws->seg = 0;
        } else {
            /* Compute the Generalized Cauchy Point (GCP)
             * This finds xᶜ by exploring the piecewise linear path */
            int cauchy_info = cauchy_point(n, m, x, g, lower, upper, bound_type, z, ws);
            if (cauchy_info != 0) {
                /* Singular triangular system detected - try BFGS reset */
                info = cauchy_info;
                continue;
            }
            
            /* Count entering/leaving variables and build free variable index set
             * This identifies which variables are free at the GCP */
            wrk = free_var(n, ws);
        }
        
        /* ====================================================================
         * Step 2: Minimize subspace (matching Go's d.minimizeSubspace(wrk))
         * ====================================================================
         *
         * Subspace minimization finds the optimal direction within the
         * subspace of free variables, starting from the GCP xᶜ.
         *
         * The subspace problem is:
         *   minimize   m̃ₖ(d̃) = d̃ᵀr̃ᶜ + ½d̃ᵀB̃ₖd̃
         *   subject to lᵢ - xᶜᵢ ≤ d̃ᵢ ≤ uᵢ - xᶜᵢ  (i ∈ 𝓕)
         *
         * where:
         *   B̃ₖ = ZₖᵀBₖZₖ = θI - ZᵀWMWᵀZ  (reduced Hessian)
         *   r̃ᶜ = Zₖᵀ(gₖ + Bₖ(xᶜ-xₖ))     (reduced gradient)
         *
         * The unconstrained solution is:
         *   d̃ᵘ = -B̃ₖ⁻¹r̃ᶜ = r̃ᶜ/θ + ZᵀW(I-MWᵀZZᵀW/θ)⁻¹MWᵀZ/θ²
         *
         * This requires solving a system involving the K matrix:
         *   K = [-D - YᵀZZᵀY/θ    Laᵀ - Rzᵀ]
         *       [La - Rz          θSᵀAAᵀS  ]
         *
         * Skip if no free variables (free = 0) or no BFGS corrections (col = 0).
         */
        
        ws->word = SOLUTION_UNKNOWN;
        
        /* Only perform subspace minimization if:
         * - There are free variables (free > 0)
         * - There are BFGS corrections (col > 0)
         */
        if (ws->free > 0 && ws->col > 0) {
            /* Build K matrix if needed (wrk = true means K needs update)
             * K = LELᵀ factorization for solving the subspace problem */
            if (wrk) {
                int formk_info = form_k(n, m, ws);
                if (formk_info != 0) {
                    /* Non-positive definite matrix - try BFGS reset */
                    info = formk_info;
                    continue;
                }
            }
            
            /* Compute reduced gradient r̃ᶜ = -Zᵀ(g + B(xᶜ - xₖ)) */
            int rg_info = reduce_gradient(n, m, x, g, z, r, ws);
            if (rg_info != 0) {
                /* Singular triangular matrix - try BFGS reset */
                info = rg_info;
                continue;
            }
            
            /* Compute optimal direction x̂ = xᶜ + d̃⁎
             * This solves the bounded subspace problem */
            int od_info = optimal_direction(n, m, x, g, lower, upper, bound_type, z, r, ws);
            if (od_info != 0) {
                /* Singular triangular matrix - try BFGS reset */
                info = od_info;
                continue;
            }
        }
        
        /* ====================================================================
         * Step 3: Search optimal step (matching Go's d.searchOptimalStep(&task))
         * ====================================================================
         *
         * Line search finds the step length λₖ satisfying Wolfe conditions:
         *
         * Sufficient decrease (Armijo):
         *   f(xₖ + λₖdₖ) ≤ f(xₖ) + α·λₖ·gₖᵀdₖ
         *
         * Curvature condition:
         *   |g(xₖ + λₖdₖ)ᵀdₖ| ≤ β·|gₖᵀdₖ|
         *
         * where α = SEARCH_ALPHA = 1e-3 and β = SEARCH_BETA = 0.9.
         *
         * The search direction is dₖ = z - x where z is the result of
         * GCP + subspace minimization.
         *
         * This matches Go's:
         *   for i, x := range x { d[i] = z[i] - x }
         *   initLineSearch(loc, spec, ctx)
         *   loc.save(ctx.t, &ctx.fOld, ctx.r)
         *   for !done { info, done = performLineSearch(...) }
         */
        
        /* Compute search direction d = z - x */
        dcopy(n, z, 1, d, 1);
        daxpy(n, -ONE, x, 1, d, 1);
        
        /* Initialize line search */
        int task;
        double stp = init_line_search(n, m, x, lower, upper, bound_type, ws, &task);
        
        /* Save original x, f, g (matching Go's loc.save(ctx.t, &ctx.fOld, ctx.r))
         * This allows restoration if line search fails */
        dcopy(n, x, 1, t, 1);   /* Save x to t */
        ws->f_old = f;          /* Save f */
        dcopy(n, g, 1, r, 1);   /* Save g to r */
        
        /* Line search loop
         * Iterate until Wolfe conditions are satisfied or failure */
        int done = 0;
        while (!done) {
            int ls_info = perform_line_search(n, x, f, g, &stp, &task, ws, &done);
            
            if (ls_info == 0 && ws->num_back < SEARCH_BACK_EXIT) {
                if (!done) {
                    /* Need function evaluation at new point
                     * x has been updated to x + stp*d by perform_line_search */
                    f = config->eval(config->eval_ctx, x, g, n);
                    ws->total_eval++;
                    ws->num_eval++;
                    ws->num_back = ws->num_eval - 1;
                    
                    /* Check for callback error */
                    if (isnan(f) || isinf(f)) {
                        /* Restore previous iterate */
                        dcopy(n, t, 1, x, 1);
                        f = ws->f_old;
                        dcopy(n, r, 1, g, 1);
                        result->f = f;
                        result->iterations = ws->iter;
                        result->evaluations = ws->total_eval;
                        result->status = STATUS_CALLBACK_ERROR;
                        return STATUS_CALLBACK_ERROR;
                    }
                }
                continue;
            }
            
            /* Line search failed or too many backtracking steps */
            if (ws->col == 0) {
                /* No BFGS corrections - abnormal termination
                 * Cannot recover without correction history */
                /* Restore previous iterate */
                dcopy(n, t, 1, x, 1);
                f = ws->f_old;
                dcopy(n, r, 1, g, 1);
                result->f = f;
                result->iterations = ws->iter + 1;
                result->evaluations = ws->total_eval;
                result->status = STATUS_LINE_SEARCH_FAILED;
                return STATUS_LINE_SEARCH_FAILED;
            } else {
                /* Try BFGS reset - matches Go's warnRestartLoop */
                info = -1;
            }
            break;
        }
        
        if (!done) {
            /* Restore the previous iterate (matching Go's loc.load) */
            dcopy(n, t, 1, x, 1);
            f = ws->f_old;
            dcopy(n, r, 1, g, 1);
            
            if (info != 0) {
                continue;  /* Will reset BFGS at start of next iteration */
            }
        }
        
        ws->f = f;
        
        /* ====================================================================
         * Step 4: New iteration (matching Go's d.newIteration(task))
         * ====================================================================
         *
         * Check stopping criteria related to iteration/evaluation limits
         * and gradient descent threshold.
         *
         * This matches Go's:
         *   w.iter++
         *   if w.iter > o.stop.MaxIterations { iter = OverIterLimit }
         *   else if w.totalEval >= o.stop.MaxEvaluations { iter = OverEvalLimit }
         *   else if w.dNorm <= o.stop.GradDescentThreshold*(1.0+math.Abs(loc.f)) {
         *       iter = OverGradThresh
         *   }
         */
        
        ws->iter++;
        
        /* Check iteration limit */
        if (ws->iter > max_iter) {
            result->f = f;
            result->iterations = ws->iter;
            result->evaluations = ws->total_eval;
            result->status = STATUS_MAX_ITER;
            return STATUS_MAX_ITER;
        }
        
        /* Check evaluation limit */
        if (ws->total_eval >= max_eval) {
            result->f = f;
            result->iterations = ws->iter;
            result->evaluations = ws->total_eval;
            result->status = STATUS_MAX_EVAL;
            return STATUS_MAX_EVAL;
        }
        
        /* Check time limit */
        if (max_time > 0) {
            long elapsed_us = get_time_us() - start_time_us;
            if (elapsed_us >= max_time) {
                result->f = f;
                result->iterations = ws->iter;
                result->evaluations = ws->total_eval;
                result->status = STATUS_MAX_TIME;
                return STATUS_MAX_TIME;
            }
        }
        
        /* Compute d_norm = ‖d‖₂ for gradient descent threshold check */
        ws->d_norm = dnrm2(n, d, 1);
        
        /* Check gradient descent threshold
         * This is an alternative stopping criterion based on step size */
        if (ws->d_norm <= pgtol * (ONE + fabs(f))) {
            result->f = f;
            result->iterations = ws->iter;
            result->evaluations = ws->total_eval;
            result->status = STATUS_GRAD_TOL;
            return STATUS_GRAD_TOL;
        }
        
        /* ====================================================================
         * Step 5: Check convergence (matching Go's d.checkConvergence(task))
         * ====================================================================
         *
         * Check convergence based on:
         * 1. Projected gradient norm: ‖proj(g)‖∞ ≤ pgtol
         * 2. Function value reduction: (f_old - f)/max(|f_old|,|f|,1) ≤ factr·ε
         *
         * This matches Go's:
         *   w.sbgNrm = projGradNorm(loc, &o.iterSpec)
         *   if w.sbgNrm <= o.stop.ProjGradTolerance { iter = ConvGradProgNorm }
         *   else if w.iter > 0 {
         *       tolEps := o.epsilon * o.stop.EpsAccuracyFactor
         *       change := math.Max(math.Abs(w.fOld), math.Max(math.Abs(loc.f), one))
         *       if w.fOld-loc.f <= tolEps*change { iter = ConvEnoughAccuracy }
         *   }
         */
        
        /* Compute projected gradient norm */
        sbg_norm = proj_grad_norm(n, x, g, lower, upper, bound_type);
        ws->sbg_norm = sbg_norm;
        
        /* Check projected gradient tolerance
         * Convergence if gradient is small enough at the current point */
        if (sbg_norm <= pgtol) {
            result->f = f;
            result->iterations = ws->iter;
            result->evaluations = ws->total_eval;
            result->status = STATUS_GRAD_TOL;
            return STATUS_GRAD_TOL;
        }
        
        /* Check function value convergence
         * Convergence if function value is not decreasing significantly */
        if (ws->iter > 0) {
            double tol_eps = EPS * factr;
            double change = fmax(fabs(ws->f_old), fmax(fabs(f), ONE));
            if (ws->f_old - f <= tol_eps * change) {
                result->f = f;
                result->iterations = ws->iter;
                result->evaluations = ws->total_eval;
                result->status = STATUS_FUNC_TOL;
                return STATUS_FUNC_TOL;
            }
        }
        
        /* ====================================================================
         * Step 6: Update BFGS (matching Go's d.updateBFGS())
         * ====================================================================
         *
         * Update the L-BFGS approximation with the new correction pair:
         *   sₖ = xₖ₊₁ - xₖ  (step)
         *   yₖ = gₖ₊₁ - gₖ  (gradient difference)
         *
         * The update is accepted if yₖᵀsₖ > ε·‖yₖ‖² (curvature condition).
         * This ensures the BFGS approximation remains positive definite.
         *
         * After adding the correction pair, we form the T matrix:
         *   T = θSᵀS + LD⁻¹Lᵀ
         * where L is the strict lower triangle of SᵀY and D = diag(sᵢᵀyᵢ).
         *
         * This matches Go's:
         *   updateCorrection(loc, spec, ctx)
         *   info = formT(spec, ctx)
         */
        
        /* Compute s = x_new - x_old (stored in d, which is no longer needed)
         * Note: t contains the saved x_old from before line search */
        double* s = d;
        dcopy(n, x, 1, s, 1);
        daxpy(n, -ONE, t, 1, s, 1);  /* s = x - t (x_new - x_old) */
        
        /* Compute y = g_new - g_old (stored in r, which contains g_old)
         * Note: r contains the saved g_old from before line search */
        double* y = r;
        dscal(n, -ONE, y, 1);        /* y = -g_old */
        daxpy(n, ONE, g, 1, y, 1);   /* y = g - g_old */
        
        /* Update BFGS matrices with new (s, y) pair
         * This adds the correction to the circular buffer and updates θ */
        update_correction(n, m, s, y, ws);
        
        /* Form T matrix if update was performed
         * T = θSᵀS + LD⁻¹Lᵀ is needed for subspace minimization */
        if (ws->updated) {
            int formt_info = form_t(m, ws);
            if (formt_info != 0) {
                /* form_t failed - T matrix is not positive definite
                 * Try to recover by resetting BFGS approximation */
                info = formt_info;
                /* Continue to next iteration which will reset BFGS */
            }
        }
    }
    
    /* Maximum iterations reached */
    result->f = f;
    result->iterations = ws->iter;
    result->evaluations = ws->total_eval;
    result->status = STATUS_MAX_ITER;
    return STATUS_MAX_ITER;
}
