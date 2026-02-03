/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * LSQ - Least Squares Quadratic Programming
 * Solves QP subproblem for SLSQP using LSEI.
 * 
 * This implementation follows the Go version in solver.go LSQ function.
 * 
 * LSQ (Least Squares Quadratic programming) solves the problem:
 * 
 *   minimize ‖ 𝐃¹ᐟ²𝐋ᵀ𝐱 + 𝐃⁻¹ᐟ²𝐋⁻¹𝐠 ‖₂ subject to
 *     - 𝐀ⱼ𝐱 - 𝐛ⱼ = 0  (j = 1 ··· mₑ)
 *     - 𝐀ⱼ𝐱 - 𝐛ⱼ ≥ 0  (j = mₑ+1 ··· m)
 *     - 𝒍ᵢ ≤ 𝐱ᵢ ≤ 𝒖ᵢ (i = 1 ··· n)
 * 
 * where:
 *   - 𝐋 is an n × n lower triangular matrix with unit diagonal elements
 *   - 𝐃 is an n × n diagonal matrix
 *   - 𝐠 is an n-vector
 *   - 𝐀 is an m × n matrix
 *   - 𝐛 is an m-vector
 * 
 * The QP subproblem arises from the BFGS approximation 𝐁 = 𝐋𝐃𝐋ᵀ (LDLᵀ factorization)
 * of the Hessian of the Lagrangian in the SQP method.
 * 
 * LSQ is transformed to LSEI problem 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐂𝐱 = 𝐝 and 𝐆𝐱 ≥ 𝐡 with:
 *   - 𝐄 = 𝐃¹ᐟ²𝐋ᵀ (n × n upper triangular)
 *   - 𝐟 = -𝐃⁻¹ᐟ²𝐋⁻¹𝐠 (n-vector)
 *   - 𝐂 = { 𝐀ⱼ: j = 1 ··· mₑ } (mₑ × n matrix)
 *   - 𝐝 = { -𝐛ⱼ: j = 1 ··· mₑ } (mₑ-vector)
 *   - 𝐆ⱼ = { 𝐀ⱼ: j = mₑ+1 ··· m } ((m-mₑ+2n) × n matrix)
 *   - 𝐡ⱼ = { -𝐛ⱼ: j = mₑ+1 ··· m } ((m-mₑ+2n)-vector)
 * 
 * Bound Constraint Transformation:
 * The bounds 𝒍 ≤ 𝐱 ≤ 𝒖 are equivalent to inequality constraints 𝐈𝐱 ≥ 𝒍 and -𝐈𝐱 ≥ -𝒖:
 *   - 𝐆ⱼ = { 𝐈ⱼ: j = m+1 ··· m+n }     𝐡ⱼ = { 𝒍ⱼ: j = m+1 ··· m+n }
 *   - 𝐆ⱼ = { -𝐈ⱼ: j = m+n ··· m+2n }   𝐡ⱼ = { -𝒖ⱼ: j = m+n ··· m+2n }
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/* External functions */
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern void dscal(int n, double a, double* x, int incx);
extern void dcopy(int n, const double* x, int incx, double* y, int incy);
extern int lsei(double* c, double* d, double* e, double* f,
                double* g, double* h,
                int lc, int mc, int le, int me, int lg, int mg, int n,
                double* x, double* w, int* jw, int maxIter,
                double* norm);

/**
 * lsq - Least Squares Quadratic Programming
 * 
 * Solves the QP subproblem:
 * 
 *   minimize ‖ 𝐃¹ᐟ²𝐋ᵀ𝐱 + 𝐃⁻¹ᐟ²𝐋⁻¹𝐠 ‖₂
 * 
 * subject to:
 *   - 𝐀ⱼ𝐱 - 𝐛ⱼ = 0  (j = 1 ··· mₑ)   [equality constraints]
 *   - 𝐀ⱼ𝐱 - 𝐛ⱼ ≥ 0  (j = mₑ+1 ··· m) [inequality constraints]
 *   - 𝒍 ≤ 𝐱 ≤ 𝒖                       [bound constraints]
 * 
 * The LDLᵀ factorization 𝐁 = 𝐋𝐃𝐋ᵀ is used where:
 *   - 𝐋 is lower triangular with unit diagonal
 *   - 𝐃 is diagonal
 * 
 * This is transformed to LSEI problem 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ s.t. 𝐂𝐱 = 𝐝, 𝐆𝐱 ≥ 𝐡:
 *   - 𝐄 = 𝐃¹ᐟ²𝐋ᵀ
 *   - 𝐟 = -𝐃⁻¹ᐟ²𝐋⁻¹𝐠
 *   - 𝐂 = { 𝐀ⱼ: j = 1 ··· mₑ }
 *   - 𝐝 = { -𝐛ⱼ: j = 1 ··· mₑ }
 *   - 𝐆 = { 𝐀ⱼ: j = mₑ+1 ··· m } ∪ { ±𝐈 for bounds }
 *   - 𝐡 = { -𝐛ⱼ: j = mₑ+1 ··· m } ∪ { 𝒍, -𝒖 for bounds }
 * 
 * @param m       Total number of constraints (m = mₑ + mᵢₙₑ)
 * @param meq     Number of equality constraints (mₑ)
 * @param n       Number of variables
 * @param nl      Length of l array: n(n+1)/2 + 1 for normal, +1 for augmented
 * @param l       𝐋 + 𝐃 in packed form (LDLᵀ factorization of Hessian)
 * @param g       Gradient vector 𝐠 = 𝜵𝒇(𝐱ᵏ)
 * @param a       Constraint Jacobian 𝐀 (column-major, leading dimension = max(m,1))
 * @param b       Constraint values 𝐛 = 𝒄(𝐱ᵏ)
 * @param xl      Lower bounds 𝒍
 * @param xu      Upper bounds 𝒖
 * @param x       Output: solution vector 𝐱
 * @param y       Output: Lagrange multipliers 𝛌 (m + 2n elements)
 * @param w       Working array
 * @param jw      Working array (integer)
 * @param maxIter Maximum iterations for NNLS solver
 * @param infBnd  Infinity bound value (bounds beyond this are ignored)
 * @param norm    Output: residual norm ‖𝐄𝐱 - 𝐟‖₂
 * @return        Status code:
 *                 0 = HasSolution (success)
 *                -2 = ConsIncompatible (constraints incompatible)
 *                -3 = LSISingularE (singular E matrix in LSI)
 *                -4 = LSEISingularC (singular C matrix in LSEI)
 *                -5 = HFTIRankDefect (rank defect in HFTI)
 */
int lsq(int m, int meq, int n, int nl,
        double* l, double* g, double* a, double* b,
        double* xl, double* xu,
        double* x, double* y, double* w, int* jw,
        int maxIter, double infBnd, double* norm) {
    
    int i, j, i2, i3, i4, la, mineq, m1, n1, n2, n3, bnd, status;
    int e0, f0, c0, d0, g0, h0, w0;
    double diag;
    
    mineq = m - meq;
    m1 = mineq + n + n;  /* Total inequality constraints including bounds */
    la = (m > 1) ? m : 1;  /* Leading dimension of A, matches Go: max(m, 1) */
    
    /* Determine problem type */
    n1 = n + 1;
    if ((n + 1) * n / 2 + 1 == nl) {
        /* Solve the original problem m × n */
        n2 = 0;
        n3 = n;
    } else {
        /* Solve the augmented problem m × (n+1) */
        n2 = 1;
        n3 = n - 1;
    }
    
    /* Working space indices - matches Go implementation
     * Layout: [E(n×n) | f(n) | C(meq×n) | d(meq) | G(m1×n) | h(m1) | workspace]
     */
    e0 = 0;                    /* Start index of 𝐄: n×n upper triangular */
    f0 = n * n;                /* Start index of 𝐟: n-vector */
    c0 = f0 + n;               /* Start index of 𝐂: meq×n matrix */
    d0 = c0 + meq * n;         /* Start index of 𝐝: meq-vector */
    g0 = d0 + meq;             /* Start index of 𝐆: m1×n matrix */
    h0 = g0 + m1 * n;          /* Start index of 𝐡: m1-vector */
    w0 = h0 + m1;              /* Start index of workspace */
    
    /* =========================================================================
     * Recover matrix 𝐄 and vector 𝐟 from 𝐋, 𝐃, and 𝐠
     * 
     * LDLᵀ Factorization Recovery:
     *   𝐄 = 𝐃¹ᐟ²𝐋ᵀ  (upper triangular)
     *   𝐟 = -𝐃⁻¹ᐟ²𝐋⁻¹𝐠
     * 
     * For each column j:
     *   𝐄ⱼ = 𝐃¹ᐟ²ⱼⱼ × 𝐋ⱼᵀ
     *   𝐟ⱼ = 𝐃⁻¹ᐟ²ⱼⱼ × (𝐋⁻¹𝐠)ⱼ
     * 
     * where (𝐋⁻¹𝐠)ⱼ = 𝐠ⱼ - ∑ᵢ𝐋ⱼᵢ(𝐋⁻¹𝐠)ᵢ (forward substitution, 𝐋ⱼⱼ = 1)
     * ========================================================================= */
    i2 = 0;
    i3 = 0;
    i4 = 0;
    
    for (j = 0; j < n3; j++) {
        i = n - j;
        diag = sqrt(l[i2]);  /* 𝐃¹ᐟ²ⱼⱼ = √𝐃ⱼⱼ */
        
        /* Zero out column: dzero(w[i3 : i3+i]) */
        for (int k = 0; k < i; k++) {
            w[i3 + k] = 0.0;
        }
        
        /* 𝐄ⱼ = 𝐋ⱼᵀ : dcopy(i-n2, l[i2:], 1, w[i3:], n) */
        dcopy(i - n2, &l[i2], 1, &w[i3], n);
        
        /* 𝐄ⱼ = 𝐃¹ᐟ² × 𝐋ⱼᵀ : dscal(i-n2, diag, w[i3:], n) */
        dscal(i - n2, diag, &w[i3], n);
        
        /* 𝐄ⱼⱼ = 𝐃¹ᐟ²ⱼⱼ : w[i3] = diag */
        w[i3] = diag;
        
        /* 𝐟ⱼ = 𝐃⁻¹ᐟ²ⱼⱼ × (𝐋⁻¹𝐠)ⱼ
         * 
         * Forward substitution for 𝐲 = 𝐋⁻¹𝐠:
         *   𝐲ⱼ = (𝐠ⱼ - ∑ᵢ𝐋ⱼᵢ𝐲ᵢ) / 𝐋ⱼⱼ
         * Since 𝐋ⱼⱼ = 1:
         *   (𝐋⁻¹𝐠)ⱼ = 𝐠ⱼ - ∑ᵢ𝐋ⱼᵢ(𝐋⁻¹𝐠)ᵢ
         * 
         * Matches Go: w[f0+j] = (g[j] - ddot(j, w[i4:], 1, w[f0:], 1)) / diag */
        w[f0 + j] = (g[j] - ddot(j, &w[i4], 1, &w[f0], 1)) / diag;
        
        i2 += i - n2;
        i3 += n1;
        i4 += n;
    }
    
    /* Handle augmented problem case (for inconsistent constraints relaxation)
     * In augmented problem, an extra variable 𝛅 is added with 𝐄ⱼⱼ = 𝛒 (penalty) */
    if (n2 == 1) {
        w[i3] = l[nl - 1];  /* 𝐄ⱼⱼ = 𝛒 (penalty parameter) */
        /* dzero(w[i4 : i4+n3]) */
        for (int k = 0; k < n3; k++) {
            w[i4 + k] = 0.0;
        }
        w[f0 + n3] = 0.0;   /* 𝐟ⱼ = 0 */
    }
    
    /* 𝐟 = -𝐃⁻¹ᐟ²𝐋⁻¹𝐠 : negate the computed values */
    dscal(n, -1.0, &w[f0], 1);
    
    /* =========================================================================
     * Recover matrix 𝐂 and vector 𝐝 from equality constraints
     * 
     * 𝐂 = { 𝐀ⱼ: j = 1 ··· mₑ }  (equality constraint Jacobian)
     * 𝐝 = { -𝐛ⱼ: j = 1 ··· mₑ } (negated equality constraint values)
     * ========================================================================= */
    if (meq > 0) {
        /* Recover matrix 𝐂 from upper part of 𝐀
         * Matches Go: for i := 0; i < meq; i++ { dcopy(n, a[i:], la, w[c0+i:], meq) } */
        for (i = 0; i < meq; i++) {
            dcopy(n, &a[i], la, &w[c0 + i], meq);
        }
        /* Recover vector 𝐝 from upper part of 𝐛
         * 𝐝ⱼ = -𝐛ⱼ = -𝒄ⱼ(𝐱ᵏ)
         * Matches Go: dcopy(meq, b, 1, w[d0:], 1); dscal(meq, -one, w[d0:], 1) */
        dcopy(meq, b, 1, &w[d0], 1);
        dscal(meq, -1.0, &w[d0], 1);
    }
    
    /* =========================================================================
     * Recover matrix 𝐆 and vector 𝐡 from inequality constraints
     * 
     * 𝐆 = { 𝐀ⱼ: j = mₑ+1 ··· m }  (inequality constraint Jacobian)
     * 𝐡 = { -𝐛ⱼ: j = mₑ+1 ··· m } (negated inequality constraint values)
     * ========================================================================= */
    if (mineq > 0) {
        /* Recover matrix 𝐆 from lower part of 𝐀
         * 𝐆ⱼ = 𝐀ⱼ = -𝒄ⱼ(𝐱ᵏ)
         * Matches Go: for i := 0; i < mineq; i++ { dcopy(n, a[meq+i:], la, w[g0+i:], m1) } */
        for (i = 0; i < mineq; i++) {
            dcopy(n, &a[meq + i], la, &w[g0 + i], m1);
        }
        /* Recover vector 𝐡 from lower part of 𝐛
         * 𝐡ⱼ = -𝐛ⱼ = -𝒄ⱼ(𝐱ᵏ)
         * Matches Go: dcopy(mineq, b[meq:], 1, w[h0:], 1); dscal(mineq, -one, w[h0:], 1) */
        dcopy(mineq, &b[meq], 1, &w[h0], 1);
        dscal(mineq, -1.0, &w[h0], 1);
    }
    
    /* =========================================================================
     * Bound Constraint Transformation
     * 
     * Transform bounds 𝒍 ≤ 𝐱 ≤ 𝒖 to inequality constraints:
     * 
     * Lower bounds (𝐱 ≥ 𝒍):
     *   𝐆ⱼ = 𝐈ⱼ (j-th row of identity matrix)
     *   𝐡ⱼ = 𝒍ⱼ
     * 
     * Upper bounds (𝐱 ≤ 𝒖 ⟺ -𝐱 ≥ -𝒖):
     *   𝐆ⱼ = -𝐈ⱼ (negated j-th row of identity)
     *   𝐡ⱼ = -𝒖ⱼ
     * 
     * Matches Go: bnd := mineq; for i, l := range xl { ... }
     * ========================================================================= */
    bnd = mineq;
    
    /* Lower bounds: 𝐆ⱼ = 𝐈ⱼ, 𝐡ⱼ = 𝒍ⱼ (constraint: 𝐱ᵢ ≥ 𝒍ᵢ) */
    for (i = 0; i < n; i++) {
        if (!isnan(xl[i]) && xl[i] > -infBnd) {
            int ip = g0 + bnd;
            int il = h0 + bnd;
            w[il] = xl[i];  /* 𝐡ⱼ = 𝒍ⱼ */
            w[ip] = 0.0;    /* 𝐆ⱼ = 𝐈ⱼ (start with zeros) */
            /* Zero out row, then set diagonal element to 1 */
            for (int k = 0; k < n; k++) {
                w[ip + m1 * k] = 0.0;
            }
            w[ip + m1 * i] = 1.0;  /* 𝐆ⱼᵢ = 1 */
            bnd++;
        }
    }
    
    /* Upper bounds: 𝐆ⱼ = -𝐈ⱼ, 𝐡ⱼ = -𝒖ⱼ (constraint: -𝐱ᵢ ≥ -𝒖ᵢ ⟺ 𝐱ᵢ ≤ 𝒖ᵢ) */
    for (i = 0; i < n; i++) {
        if (!isnan(xu[i]) && xu[i] < infBnd) {
            int ip = g0 + bnd;
            int il = h0 + bnd;
            w[il] = -xu[i];  /* 𝐡ⱼ = -𝒖ⱼ */
            w[ip] = 0.0;     /* 𝐆ⱼ = -𝐈ⱼ (start with zeros) */
            /* Zero out row, then set diagonal element to -1 */
            for (int k = 0; k < n; k++) {
                w[ip + m1 * k] = 0.0;
            }
            w[ip + m1 * i] = -1.0;  /* 𝐆ⱼᵢ = -1 */
            bnd++;
        }
    }
    
    /* Calculate number of NaN bounds (unused bound constraints)
     * nan = total possible bounds - actual bounds used
     * Matches Go: nan := (n + n) - (bnd - mineq) */
    int nan_count = (n + n) - (bnd - mineq);
    
    /* =========================================================================
     * Call LSEI solver
     * 
     * Solve: 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐂𝐱 = 𝐝 and 𝐆𝐱 ≥ 𝐡
     * 
     * Parameters:
     *   - C(meq×n), d(meq): equality constraints
     *   - E(n×n), f(n): least squares objective
     *   - G(m1×n), h(m1-nan): inequality constraints (excluding unused bounds)
     * 
     * Matches Go: norm, mode := LSEI(w[c0:d0], w[d0:g0], w[e0:f0], w[f0:c0], 
     *                               w[g0:h0], w[h0:w0], max(1, meq), meq, n, n, 
     *                               m1, m1-nan, n, x, w[w0:], jw, maxIter)
     * ========================================================================= */
    int meq_max = (meq > 1) ? meq : 1;
    
    status = lsei(&w[c0], &w[d0], &w[e0], &w[f0], &w[g0], &w[h0],
                  meq_max, meq, n, n, m1, m1 - nan_count, n,
                  x, &w[w0], jw, maxIter, norm);
    
    /* =========================================================================
     * Process results
     * 
     * If solution found:
     *   1. Restore Lagrange multipliers 𝛌 from workspace
     *   2. Set unused multipliers to NaN
     *   3. Enforce bounds on solution (project onto feasible region)
     * 
     * Matches Go: if mode == HasSolution { ... }
     * ========================================================================= */
    if (status == 0) {
        /* Restore Lagrange multipliers 𝛌
         * Matches Go: dcopy(m, w[w0:], 1, y, 1) */
        dcopy(m, &w[w0], 1, y, 1);
        
        /* Set unused multipliers to NaN (for bounds that weren't active)
         * Matches Go: if n3 > 0 { y[m] = math.NaN(); dcopy(n3+n3, y[m:], 0, y[m:], 1) }
         * Note: dcopy with incx=0 copies the same value to all elements */
        if (n3 > 0) {
            y[m] = NAN;
            for (i = 1; i < n3 + n3; i++) {
                y[m + i] = NAN;
            }
        }
        
        /* Enforce lower bounds on solution: 𝐱ᵢ = max(𝐱ᵢ, 𝒍ᵢ)
         * Matches Go: for i, l := range xl { if !math.IsNaN(l) && l > -infBnd && x[i] < l { x[i] = l } } */
        for (i = 0; i < n; i++) {
            if (!isnan(xl[i]) && xl[i] > -infBnd && x[i] < xl[i]) {
                x[i] = xl[i];
            }
        }
        /* Enforce upper bounds on solution: 𝐱ᵢ = min(𝐱ᵢ, 𝒖ᵢ)
         * Matches Go: for i, u := range xu { if !math.IsNaN(u) && u < infBnd && x[i] > u { x[i] = u } } */
        for (i = 0; i < n; i++) {
            if (!isnan(xu[i]) && xu[i] < infBnd && x[i] > xu[i]) {
                x[i] = xu[i];
            }
        }
    }
    
    return status;
}
