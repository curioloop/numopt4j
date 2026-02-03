/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * Utility functions for workspace management.
 *
 * This file provides workspace allocation and initialization utilities for
 * L-BFGS-B and SLSQP optimization algorithms. These functions are C-specific
 * implementations that handle memory layout and pointer assignment.
 *
 * Go Correspondence:
 *   - L-BFGS-B workspace: lbfgsb/base.go (iterWork.init, iterCtx.clear)
 *   - SLSQP workspace: slsqp/optimize.go (Optimizer.Init)
 *
 * Key Differences from Go:
 *   - Go uses garbage-collected slices; C requires explicit memory management
 *   - C implementation uses a single contiguous memory block with aligned offsets
 *   - C provides explicit workspace_size() functions for pre-allocation
 *   - Go workspace initialization is split across struct methods; C consolidates
 *
 * Memory Layout Strategy:
 *   The C implementation uses a single pre-allocated memory block to avoid
 *   multiple allocations and improve cache locality. Arrays are aligned to
 *   8-byte boundaries for optimal double precision performance.
 */

#include "optimizer.h"
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Alignment for memory allocation (8 bytes for double)
 * Ensures proper alignment for double precision floating point operations */
#define ALIGN_SIZE 8
#define ALIGN_UP(x) (((x) + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1))

/* ============================================================================
 * L-BFGS-B Workspace Functions
 *
 * These functions manage workspace for the L-BFGS-B algorithm.
 *
 * Go Correspondence:
 *   - iterWork.init() in lbfgsb/base.go allocates the same arrays
 *   - iterCtx.clear() and iterBFGS.reset() handle state initialization
 *
 * Array Dimensions (matching Go implementation in base.go):
 *   - ws, wy: n × m (correction matrices S and Y)
 *   - sy, ss, wt: m × m (inner products and Cholesky factor)
 *   - wn, snd: 2m × 2m = 4m² (LELᵀ factorization matrices)
 *   - z, r, d, t, xp, g: n (working vectors)
 *   - wa: 8m (temporary workspace)
 *   - index: 2 × n (free/active variable indices)
 *   - iwhere: n (variable status flags)
 * ============================================================================ */

/**
 * Calculate the required workspace size for L-BFGS-B.
 *
 * This function is C-specific - Go uses garbage-collected slices that are
 * allocated individually in iterWork.init() (base.go).
 *
 * Memory layout (all arrays 8-byte aligned):
 *   - ws:     n × m doubles     (S matrix: stores sₖ = xₖ₊₁ - xₖ)
 *   - wy:     n × m doubles     (Y matrix: stores yₖ = gₖ₊₁ - gₖ)
 *   - sy:     m × m doubles     (SᵀY = (sᵀy)₁, (sᵀy)₂, ···, (sᵀy)ₘ)
 *   - ss:     m × m doubles     (SᵀS = (sᵀs)₁, (sᵀs)₂, ···, (sᵀs)ₘ)
 *   - wt:     m × m doubles     (Cholesky factor of θSᵀS + LD⁻¹Lᵀ)
 *   - wn:     4 × m × m doubles (LELᵀ factorization workspace)
 *   - snd:    4 × m × m doubles (Lower triangular storage)
 *   - z:      n doubles         (Cauchy point / Newton point)
 *   - r:      n doubles         (Reduced gradient)
 *   - d:      n doubles         (Search direction)
 *   - t:      n doubles         (Temporary storage for x backup)
 *   - xp:     n doubles         (Safeguard projected Newton direction)
 *   - g:      n doubles         (Gradient backup)
 *   - wa:     8 × m doubles     (Shared temporary workspace)
 *   - index:  2 × n ints        (Free/active variable indices)
 *   - iwhere: n ints            (Variable status: varFree, varAtLB, etc.)
 *
 * @param n Problem dimension (number of variables)
 * @param m Number of L-BFGS corrections (limited memory parameter)
 * @return Required size in bytes, or 0 if parameters are invalid
 *
 * Requirements: 3.2
 */
EXPORT size_t lbfgsb_workspace_size(int n, int m) {
    if (n <= 0 || m <= 0) return 0;
    
    /* Double arrays */
    size_t ws_size   = ALIGN_UP((size_t)n * m * sizeof(double));
    size_t wy_size   = ALIGN_UP((size_t)n * m * sizeof(double));
    size_t sy_size   = ALIGN_UP((size_t)m * m * sizeof(double));
    size_t ss_size   = ALIGN_UP((size_t)m * m * sizeof(double));
    size_t wt_size   = ALIGN_UP((size_t)m * m * sizeof(double));
    size_t wn_size   = ALIGN_UP((size_t)4 * m * m * sizeof(double));
    size_t snd_size  = ALIGN_UP((size_t)4 * m * m * sizeof(double));
    size_t z_size    = ALIGN_UP((size_t)n * sizeof(double));
    size_t r_size    = ALIGN_UP((size_t)n * sizeof(double));
    size_t d_size    = ALIGN_UP((size_t)n * sizeof(double));
    size_t t_size    = ALIGN_UP((size_t)n * sizeof(double));
    size_t xp_size   = ALIGN_UP((size_t)n * sizeof(double));
    size_t g_size    = ALIGN_UP((size_t)n * sizeof(double));
    size_t wa_size   = ALIGN_UP((size_t)8 * m * sizeof(double));
    
    /* Integer arrays */
    size_t index_size  = ALIGN_UP((size_t)2 * n * sizeof(int));
    size_t iwhere_size = ALIGN_UP((size_t)n * sizeof(int));
    
    return ws_size + wy_size + sy_size + ss_size + wt_size + 
           wn_size + snd_size + z_size + r_size + d_size + 
           t_size + xp_size + g_size + wa_size + index_size + iwhere_size;
}

/**
 * Initialize L-BFGS-B workspace pointers.
 *
 * Go Correspondence:
 *   This function combines the functionality of:
 *   - iterWork.init() in base.go (array allocation)
 *   - iterCtx.clear() in base.go (state initialization via reset)
 *
 * The Go implementation allocates each slice separately:
 *   w.ws = make([]float64, m*n)
 *   w.wy = make([]float64, m*n)
 *   ... etc.
 *
 * The C implementation assigns pointers into a pre-allocated contiguous block,
 * which provides better cache locality and simpler memory management.
 *
 * @param ws Workspace structure to initialize
 * @param memory Pre-allocated memory block (from lbfgsb_workspace_size)
 * @param n Problem dimension
 * @param m Number of L-BFGS corrections
 *
 * Requirements: 3.2
 */
EXPORT void lbfgsb_workspace_init(LbfgsbWorkspace* ws, void* memory, int n, int m) {
    if (!ws || !memory || n <= 0 || m <= 0) return;
    
    ws->n = n;
    ws->m = m;
    
    char* ptr = (char*)memory;
    
    /* Assign double arrays */
    ws->ws = (double*)ptr;
    ptr += ALIGN_UP((size_t)n * m * sizeof(double));
    
    ws->wy = (double*)ptr;
    ptr += ALIGN_UP((size_t)n * m * sizeof(double));
    
    ws->sy = (double*)ptr;
    ptr += ALIGN_UP((size_t)m * m * sizeof(double));
    
    ws->ss = (double*)ptr;
    ptr += ALIGN_UP((size_t)m * m * sizeof(double));
    
    ws->wt = (double*)ptr;
    ptr += ALIGN_UP((size_t)m * m * sizeof(double));
    
    ws->wn = (double*)ptr;
    ptr += ALIGN_UP((size_t)4 * m * m * sizeof(double));
    
    ws->snd = (double*)ptr;
    ptr += ALIGN_UP((size_t)4 * m * m * sizeof(double));
    
    ws->z = (double*)ptr;
    ptr += ALIGN_UP((size_t)n * sizeof(double));
    
    ws->r = (double*)ptr;
    ptr += ALIGN_UP((size_t)n * sizeof(double));
    
    ws->d = (double*)ptr;
    ptr += ALIGN_UP((size_t)n * sizeof(double));
    
    ws->t = (double*)ptr;
    ptr += ALIGN_UP((size_t)n * sizeof(double));
    
    ws->xp = (double*)ptr;
    ptr += ALIGN_UP((size_t)n * sizeof(double));
    
    ws->g = (double*)ptr;
    ptr += ALIGN_UP((size_t)n * sizeof(double));
    
    ws->wa = (double*)ptr;
    ptr += ALIGN_UP((size_t)8 * m * sizeof(double));
    
    /* Assign integer arrays */
    ws->index = (int*)ptr;
    ptr += ALIGN_UP((size_t)2 * n * sizeof(int));
    
    ws->iwhere = (int*)ptr;
    
    /* Initialize state */
    lbfgsb_workspace_reset(ws);
}

/**
 * Reset L-BFGS-B workspace for a new optimization.
 *
 * Go Correspondence:
 *   This function combines the functionality of:
 *   - iterBFGS.reset() in base.go:
 *       c.col = 0; c.head = 0; c.tail = 0; c.theta = one
 *       c.updates = 0; c.updated = false
 *   - iterCtx.clear() in base.go:
 *       c.word = solutionWithinBox; c.free = 0; c.active = 0
 *       c.leave = 0; c.enter = 0; c.fOld = zero; c.sbgNrm = zero
 *       c.dSqrt = zero; c.dNorm = zero; c.iter = 0; c.seg = 0
 *       c.totalEval = 0; c.totalSegGCP = 0
 *       c.gd = zero; c.stp = zero; c.gdOld = zero
 *       c.numEval = 0; c.numBack = 0
 *
 * State Variables:
 *   - iter: iteration counter (Go: ctx.iter)
 *   - col: number of corrections stored, 0 ≤ col ≤ m (Go: ctx.col)
 *   - head, tail: circular buffer pointers for S/Y matrices
 *   - theta: scaling factor θ for B₀ = θI (Go: ctx.theta, initialized to 1.0)
 *   - free, active, enter, leave: variable set tracking
 *   - updated, updates: BFGS update tracking
 *   - constrained, boxed: problem type flags
 *
 * Note: The C implementation also zeros out work arrays, which Go handles
 * implicitly through make() zero-initialization.
 *
 * @param ws Workspace to reset
 *
 * Requirements: 3.2
 */
EXPORT void lbfgsb_workspace_reset(LbfgsbWorkspace* ws) {
    if (!ws) return;
    
    ws->iter = 0;
    ws->col = 0;
    ws->head = 0;
    ws->tail = 0;
    ws->total_eval = 0;
    ws->f = 0.0;
    ws->f_old = 0.0;
    ws->theta = 1.0;
    ws->sbg_norm = 0.0;
    
    /* Free variable tracking */
    ws->free = 0;
    ws->active = 0;
    ws->enter = 0;
    ws->leave = 0;
    ws->updated = 0;
    ws->updates = 0;
    ws->constrained = 0;
    ws->boxed = 0;
    
    /* BFGS reset recovery state */
    ws->reset_count = 0;
    
    /* Zero out work arrays */
    int n = ws->n;
    int m = ws->m;
    
    memset(ws->ws, 0, (size_t)n * m * sizeof(double));
    memset(ws->wy, 0, (size_t)n * m * sizeof(double));
    memset(ws->sy, 0, (size_t)m * m * sizeof(double));
    memset(ws->ss, 0, (size_t)m * m * sizeof(double));
    memset(ws->wt, 0, (size_t)m * m * sizeof(double));
    memset(ws->wn, 0, (size_t)4 * m * m * sizeof(double));
    memset(ws->snd, 0, (size_t)4 * m * m * sizeof(double));
    memset(ws->z, 0, (size_t)n * sizeof(double));
    memset(ws->r, 0, (size_t)n * sizeof(double));
    memset(ws->d, 0, (size_t)n * sizeof(double));
    memset(ws->t, 0, (size_t)n * sizeof(double));
    memset(ws->xp, 0, (size_t)n * sizeof(double));
    memset(ws->g, 0, (size_t)n * sizeof(double));
    memset(ws->wa, 0, (size_t)8 * m * sizeof(double));
    memset(ws->index, 0, (size_t)2 * n * sizeof(int));
    memset(ws->iwhere, 0, (size_t)n * sizeof(int));
}

/* ============================================================================
 * SLSQP Workspace Functions
 *
 * These functions manage workspace for the SLSQP algorithm.
 *
 * Go Correspondence:
 *   - Optimizer.Init() in slsqp/optimize.go allocates workspace
 *   - sqpCtx struct in slsqp/base.go defines the workspace layout
 *
 * Array Dimensions (matching Go implementation in optimize.go):
 *   - l: (n+1)*(n+2)/2 (LDLᵀ factor of approximate Hessian)
 *   - x0: n (initial position backup)
 *   - g: n+1 (gradient, extra element for augmented objective)
 *   - c: max(1,m) (constraint values)
 *   - a: max(1,m) × (n+1) (constraint Jacobian, column-major)
 *   - mu: max(1,m) (penalty multipliers)
 *   - s, u, v: n+1 (search direction and bound differences)
 *   - r: 2n+m+2 (Lagrange multipliers for all constraints)
 *   - w: large workspace for LSQ, LSI, LSEI solvers
 *   - jw: max(mineq_total, n1-mineq_total) integers
 * ============================================================================ */

/**
 * Calculate the required workspace size for SLSQP.
 *
 * This function is C-specific - Go uses garbage-collected slices allocated
 * in Optimizer.Init() (optimize.go).
 *
 * Go Implementation Reference (optimize.go):
 *   la := max(1, m)
 *   ll := (n + 1) * (n + 2) / 2  // LDLᵀ factor size
 *   lr := n + n + m + 2          // multiplier array size
 *
 *   totwk := n1*(n1+1) + meq*(n1+1) + mineq*(n1+1) +  // LSQ
 *            (n1-meq+1)*(mineq+2) + 2*mineq +          // LSI
 *            (n1+mineq)*(n1-meq) + 2*meq + n1 +        // LSEI
 *            n1*n/2 + 2*m + 3*n + 3*n1 + 1             // SLSQP
 *
 * Array dimensions match Go implementation in optimize.go:
 * - l: (n+1)*(n+2)/2 = n*(n+1)/2 + n + 1 (LDLᵀ factor, needs extra n+1 for augmented QP)
 * - r: 2n + m + 2 (multipliers for constraints and bounds)
 *
 * @param n Problem dimension (number of variables)
 * @param meq Number of equality constraints
 * @param mineq Number of inequality constraints
 * @return Required size in bytes, or 0 if parameters are invalid
 *
 * Requirements: 3.2, 8.1, 8.2, 8.3
 */
EXPORT size_t slsqp_workspace_size(int n, int meq, int mineq) {
    if (n <= 0) return 0;
    
    int m = meq + mineq;
    int n1 = n + 1;
    int la = (m > 0) ? m : 1;
    
    /* Calculate workspace sizes */
    int mineq_total = mineq + 2 * n1;  /* Including bound constraints */
    
    /* Double arrays */
    /* l array: Go uses ll := (n + 1) * (n + 2) / 2 in optimize.go
     * This equals n*(n+1)/2 + n + 1, which is n2 + n + 1 where n2 = n*(n+1)/2
     * The extra n+1 elements are needed for the augmented QP (l[n2] = rho) */
    size_t l_size   = ALIGN_UP((size_t)((n + 1) * (n + 2) / 2) * sizeof(double));
    size_t x0_size  = ALIGN_UP((size_t)n * sizeof(double));
    size_t g_size   = ALIGN_UP((size_t)n1 * sizeof(double));
    size_t c_size   = ALIGN_UP((size_t)la * sizeof(double));
    size_t a_size   = ALIGN_UP((size_t)la * n1 * sizeof(double));
    size_t mu_size  = ALIGN_UP((size_t)la * sizeof(double));
    size_t s_size   = ALIGN_UP((size_t)n1 * sizeof(double));
    size_t u_size   = ALIGN_UP((size_t)n1 * sizeof(double));
    size_t v_size   = ALIGN_UP((size_t)n1 * sizeof(double));
    size_t r_size   = ALIGN_UP((size_t)(m + 2 * n + 2) * sizeof(double));
    
    /* General workspace for LSQ, LSI, LSEI, SLSQP */
    size_t w_size = ALIGN_UP(
        (size_t)(n1 * (n1 + 1) + meq * (n1 + 1) + mineq_total * (n1 + 1) +
        (n1 - meq + 1) * (mineq_total + 2) + 2 * mineq_total +
        (n1 + mineq_total) * (n1 - meq) + 2 * meq + n1 +
        n1 * n / 2 + 2 * m + 3 * n + 3 * n1 + 1) * sizeof(double)
    );
    
    /* Integer workspace */
    int jw_size_val = (mineq_total > n1 - mineq_total) ? mineq_total : (n1 - mineq_total);
    size_t jw_size = ALIGN_UP((size_t)jw_size_val * sizeof(int));
    
    return l_size + x0_size + g_size + c_size + a_size + mu_size +
           s_size + u_size + v_size + r_size + w_size + jw_size;
}

/**
 * Initialize SLSQP workspace pointers.
 *
 * Go Correspondence:
 *   This function corresponds to Optimizer.Init() in optimize.go.
 *
 * The Go implementation creates overlapping slices from a single workspace:
 *   wrk := make([]float64, totwk)
 *   w.sqpCtx = sqpCtx{
 *       r:  wrk[ir : ir+lr],  // r overlaps s: (m + 2) - max(1, m)
 *       l:  wrk[il : il+ll],  // l overlaps x0: n
 *       x0: wrk[ix : ix+n],
 *       mu: wrk[im : im+la],
 *       s:  wrk[is : is+n1*1],
 *       u:  wrk[is+n1*1 : is+n1*2],
 *       v:  wrk[is+n1*2 : is+n1*3],
 *       w:  wrk[is+n1*3:],
 *       jw: make([]int, max(mineq, n1-mineq)),
 *   }
 *
 * Note: The Go implementation uses overlapping slices for memory efficiency.
 * The C implementation uses separate non-overlapping regions for simplicity
 * and to avoid potential aliasing issues.
 *
 * Array dimensions match Go implementation in optimize.go:
 * - l: (n+1)*(n+2)/2 = n*(n+1)/2 + n + 1 (LDLᵀ factor)
 * - r: 2n + m + 2 (multipliers)
 *
 * @param ws Workspace structure to initialize
 * @param memory Pre-allocated memory block (from slsqp_workspace_size)
 * @param n Problem dimension
 * @param meq Number of equality constraints
 * @param mineq Number of inequality constraints
 *
 * Requirements: 3.2, 8.1, 8.2, 8.3
 */
EXPORT void slsqp_workspace_init(SlsqpWorkspace* ws, void* memory, 
                                  int n, int meq, int mineq) {
    if (!ws || !memory || n <= 0) return;
    
    int m = meq + mineq;
    int n1 = n + 1;
    int la = (m > 0) ? m : 1;
    int mineq_total = mineq + 2 * n1;
    
    ws->n = n;
    ws->m = m;
    ws->meq = meq;
    
    char* ptr = (char*)memory;
    
    /* Assign double arrays */
    /* l array: Go uses ll := (n + 1) * (n + 2) / 2 in optimize.go */
    ws->l = (double*)ptr;
    ptr += ALIGN_UP((size_t)((n + 1) * (n + 2) / 2) * sizeof(double));
    
    ws->x0 = (double*)ptr;
    ptr += ALIGN_UP((size_t)n * sizeof(double));
    
    ws->g = (double*)ptr;
    ptr += ALIGN_UP((size_t)n1 * sizeof(double));
    
    ws->c = (double*)ptr;
    ptr += ALIGN_UP((size_t)la * sizeof(double));
    
    ws->a = (double*)ptr;
    ptr += ALIGN_UP((size_t)la * n1 * sizeof(double));
    
    ws->mu = (double*)ptr;
    ptr += ALIGN_UP((size_t)la * sizeof(double));
    
    ws->s = (double*)ptr;
    ptr += ALIGN_UP((size_t)n1 * sizeof(double));
    
    ws->u = (double*)ptr;
    ptr += ALIGN_UP((size_t)n1 * sizeof(double));
    
    ws->v = (double*)ptr;
    ptr += ALIGN_UP((size_t)n1 * sizeof(double));
    
    ws->r = (double*)ptr;
    ptr += ALIGN_UP((size_t)(m + 2 * n + 2) * sizeof(double));
    
    ws->w = (double*)ptr;
    size_t w_size = (size_t)(n1 * (n1 + 1) + meq * (n1 + 1) + mineq_total * (n1 + 1) +
        (n1 - meq + 1) * (mineq_total + 2) + 2 * mineq_total +
        (n1 + mineq_total) * (n1 - meq) + 2 * meq + n1 +
        n1 * n / 2 + 2 * m + 3 * n + 3 * n1 + 1) * sizeof(double);
    ptr += ALIGN_UP(w_size);
    
    ws->jw = (int*)ptr;
    
    /* Initialize state */
    slsqp_workspace_reset(ws);
}

/**
 * Reset SLSQP workspace for a new optimization.
 *
 * Go Correspondence:
 *   The Go implementation doesn't have an explicit reset function.
 *   Instead, workspace state is implicitly reset when Optimizer.Fit()
 *   is called, as the sqpCtx fields are initialized in the solver.
 *
 * State Variables:
 *   - iter: iteration counter (Go: w.iter in optimize.go)
 *   - mode: current SQP mode (Go: sqpMode in base.go)
 *   - acc: solution accuracy (Go: ctx.acc in base.go)
 *   - f0: initial function value for line search (Go: ctx.f0)
 *   - alpha: line search step length (Go: ctx.alpha, initialized to 1.0)
 *
 * Note: Unlike L-BFGS-B, SLSQP workspace arrays are not zeroed here
 * because they are overwritten during each iteration.
 *
 * @param ws Workspace to reset
 *
 * Requirements: 3.2
 */
EXPORT void slsqp_workspace_reset(SlsqpWorkspace* ws) {
    if (!ws) return;
    
    ws->iter = 0;
    ws->mode = 0;
    ws->acc = 0.0;
    ws->f0 = 0.0;
    ws->alpha = 1.0;
}
