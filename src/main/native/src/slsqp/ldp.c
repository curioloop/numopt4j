/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * LDP (Least Distance Programming)
 *
 * Solves the problem 𝚖𝚒𝚗 ‖ 𝐱 ‖₂ subject to 𝐆𝐱 ≥ 𝐡.
 *   - 𝐆 is m × n matrix (no assumption need to be made for its rank)
 *   - 𝐱 ∈ ℝⁿ
 *   - 𝐡 ∈ ℝᵐ
 *
 * NNLS could solve LDP by given:
 *   - an (n+1) × m matrix 𝐀 = [𝐆 : 𝐡]ᵀ
 *   - an (n+1)-vector 𝐛 = [Oₙ : 1]
 *
 * Assume m-vector 𝐮 is optimal solution to NNLS solution:
 *   - the residual is an (n+1)-vector 𝐫 = 𝐀𝐮 - 𝐛  = [𝐆ᵀ𝐮 : 𝐡ᵀ𝐮 - 1]ᵀ = [𝐫₁ ··· 𝐫ₙ : 𝐫ₙ₊₁]ᵀ
 *   - The dual vector is an m-vector 𝐰 = 𝐀ᵀ(𝐛 - 𝐀𝐮) = 𝐀ᵀ𝐫
 *
 * The 𝐰ᵀ𝐮 = 0 which is given by:
 *   - 𝐰ᵢ ≥ 0 → 𝐮ᵢ = 0
 *   - 𝐰ᵢ = 0 → 𝐮ᵢ > 0
 *
 * Thus the norm-2 of NNLS residual satisfied: ‖ 𝐫 ‖₂ = 𝐫ᵀ𝐫 = 𝐫ᵀ(𝐀𝐮 - 𝐛) = (𝐀ᵀ𝐫)𝐮 - 𝐫ᵀ𝐛 = 𝐰ᵀ𝐮 - 𝐫ₙ₊₁ = - 𝐫ₙ₊₁
 *   - ‖ 𝐫 ‖₂ > 0 → 𝐫ₙ₊₁ < 0
 *   - ‖ 𝐫 ‖₂ = 0 → 𝐫ₙ₊₁ = 0
 *
 * Constraints 𝐆𝐱 ≥ 𝐡 is satisfied when ‖ 𝐫 ‖₂ > 0 since:
 *
 *   (𝐆𝐱 - 𝐡)‖ 𝐫 ‖₂ = [𝐆:𝐡][𝐱:-1]ᵀ(-𝐫ₙ₊₁) = 𝐀ᵀ𝐫 = 𝐰 ≥ 0
 *
 * Substitute LDP to the KKT conditions:
 *   - 𝒇(𝐱) = ½‖ 𝐱 ‖₂                   →  𝜵𝒇(𝐱) = 𝐱
 *   - 𝒈ⱼ(𝐱) = 0  (j = 1 ··· mₑ)        →  𝜵𝒈ⱼ(𝐱) = 0
 *   - 𝒈ⱼ(𝐱) = 𝐡ⱼ -𝐆ⱼ𝐱 (j = mₑ+1 ··· m) →  𝜵𝒈ⱼ(𝐱) = -𝐆
 *
 * the optimality conditions for LDP are given:
 *   - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ) = 𝐱ᵏ - 𝐆ᵀ𝛌ᵏ = 0
 *   - 𝛌ᵏⱼ ≥ 0 ∀j
 *   - 𝛌ᵏⱼ(𝐡ⱼ -𝐆ⱼ𝐱) = 0 ∀j
 *
 * Solution of LDP is given by 𝐱 = [𝐫₁ ··· 𝐫ₙ]ᵀ/(-𝐫ₙ₊₁) = 𝐆ᵀ𝐮 / ‖ 𝐫 ‖₂.
 * The Lagrange multiplier of LDP inequality constraint 𝛌 = 𝐆⁻¹𝐱 = 𝐮 / ‖ 𝐫 ‖₂.
 *
 * References
 * ----------
 * C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
 * Chapters 23, Algorithm 23.27.
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/* External functions */
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern double dnrm2(int n, const double* x, int incx);
extern void dcopy(int n, const double* x, int incx, double* y, int incy);
extern int nnls(int m, int n, double* a, int mda,
                double* b, double* x, double* w,
                double* z, int* index, int maxIter,
                double* rnorm);

/**
 * ldp - Least Distance Programming
 *
 * Solves 𝚖𝚒𝚗 ‖ 𝐱 ‖₂ subject to 𝐆𝐱 ≥ 𝐡
 *
 * @param m       Number of constraints (rows in 𝐆)
 * @param n       Number of variables (columns in 𝐆)
 * @param g       Constraint matrix 𝐆 (column-major, m × n)
 * @param mdg     Leading dimension of 𝐆
 * @param h       Constraint vector 𝐡 (m-vector)
 * @param x       Output: solution vector 𝐱 (n-vector)
 * @param w       Working array of length (n+1)×(m+2)+2m
 *                On return, w[0:m] contains Lagrange multipliers 𝛌
 * @param jw      Working array of length m
 * @param maxIter Maximum iterations for NNLS
 * @param xnorm   Output: ‖ 𝐱 ‖₂
 * @return        Status code (0 = success, -1 = bad argument, -2 = constraints incompatible)
 */
int ldp(int m, int n, double* g, int mdg,
        double* h, double* x, double* w,
        int* jw, int maxIter, double* xnorm) {
    
    int i, j, iw, status;
    double fac, rnorm;
    double *a, *b, *z, *u, *dv;
    
    if (n <= 0) {
        return -1;  /* Bad argument */
    }
    
    if (m <= 0) {
        *xnorm = 0.0;
        return 0;  /* OK */
    }
    
    /* Working space layout:
     * 𝐰[:(n+1)×m]                     =  (n+1)×m matrix 𝐀
     * 𝐰[(n+1)×m:(n+1)×(m+1)]          =  (n+1)-vector 𝐛
     * 𝐰[(n+1)×(m+1):(n+1)×(m+2)]      =  (n+1)-vector 𝐳 (working space)
     * 𝐰[(n+1)×(m+2):(n+1)×(m+2)+m]    =  m-vector 𝐮
     * 𝐰[(n+1)×(m+2)+m:(n+1)×(m+2)+2m] =  m-vector 𝐰 (dual)
     */
    
    iw = 0;
    a = &w[iw];
    iw += m * (n + 1);
    b = &w[iw];
    iw += (n + 1);
    z = &w[iw];
    iw += (n + 1);
    u = &w[iw];
    iw += m;
    dv = &w[iw];
    
    for (j = 0; j < m; j++) {
        /* Copy 𝐆ᵀ into first n rows and m columns of 𝐀 */
        dcopy(n, &g[j], mdg, &a[j * (n + 1)], 1);
        /* Copy 𝐡ᵀ into row n+1 of 𝐀 */
        a[j * (n + 1) + n] = h[j];
    }
    
    /* Initialize 𝐛 = [Oₙ : 1] */
    for (i = 0; i < n; i++) {
        b[i] = 0.0;
    }
    b[n] = 1.0;
    
    /* Solve NNLS problem: 𝚖𝚒𝚗 ‖ 𝐀𝐮 - 𝐛 ‖₂ subject to 𝐮 ≥ 0 */
    status = nnls(n + 1, m, a, n + 1, b, u, dv, z, jw, maxIter, &rnorm);
    
    if (status == 0) {
        if (rnorm <= 0.0) {
            /* ‖ 𝐫 ‖₂ = 0 → constraints incompatible */
            return -2;
        }
        
        /* fac = -𝐫ₙ₊₁ = 1 - 𝐡ᵀ𝐮 */
        fac = 1.0 - ddot(m, h, 1, u, 1);
        
        if (isnan(fac) || fac < EPS) {
            /* Constraints incompatible */
            return -2;
        }
        
        fac = 1.0 / fac;
        
        /* 𝐱 = 𝐆ᵀ𝐮 / ‖ 𝐫 ‖₂ */
        for (j = 0; j < n; j++) {
            x[j] = ddot(m, &g[mdg * j], 1, u, 1) * fac;
        }
        
        /* Store Lagrange multipliers: 𝛌 = 𝐮 / ‖ 𝐫 ‖₂ */
        for (j = 0; j < m; j++) {
            w[j] = u[j] * fac;
        }
        
        *xnorm = dnrm2(n, x, 1);  /* ‖ 𝐱 ‖₂ */
        return 0;  /* Success */
    }
    
    return status;  /* NNLS error */
}
