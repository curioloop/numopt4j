/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * NNLS (Non-Negative Least-Squares)
 *
 * Solves a least-squares problem 𝚖𝚒𝚗 ‖ 𝐀𝐱 - 𝐛 ‖₂ subject to 𝐱 ≥ 0 with active-set method.
 *   - 𝐀 is m × n column-major matrix with 𝚛𝚊𝚗𝚔(𝐀) = n (the columns of 𝐀 are linearly independent)
 *   - 𝐱 ∈ ℝⁿ
 *   - 𝐛 ∈ ℝᵐ
 *
 * There are two index sets ℤ(zero) and ℙ(pivot):
 *   - 𝐱ⱼ = 0, j ∈ ℤ : variable indexed in active set ℤ will be held at the value zero
 *   - 𝐱ⱼ > 0, j ∈ ℙ : variable indexed in passive set ℙ will be free to take any positive value
 *
 * When 𝐱ⱼ < 0 occurred, NNLS will change its value to a non-negative value and move its index j from ℙ to ℤ.
 *
 * The m × k matrix 𝐀ₖ is a subset columns of 𝐀 defined by indices of ℙ.
 * NNLS applies QR decomposition 𝐐𝐀ₖ = [𝐑ₖᵀ:O]ᵀ to solve least-squares [𝐀ₖ:O]𝐱 ≅ 𝐛
 * where 𝐐 is m × m orthogonal matrix and 𝐑ₖ is k × k upper triangular matrix.
 *
 * Once 𝐐 and 𝐑ₖ is computed, the solution is given by 𝐱߮ = [𝐑ₖ⁻¹:O]𝐐𝐛.
 *
 * Let 𝐛 = [𝐛₁:𝐛₂] (𝐛₁ ∈ ℝⁿ, 𝐛₂ ∈ ℝᵐ⁻ⁿ) and rewrite 𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂ to 𝚖𝚒𝚗‖ 𝐐ᵀ𝐐[𝐑ₙ:O]𝐱 - 𝐐ᵀ[𝐛₁:𝐛₂] ‖₂
 *   - the solution 𝐱 satisfied 𝐑ₙ𝐱 = 𝐐ᵀ𝐛₁ (𝐐ᵀ𝐐 = 𝐈ₘ)
 *   - the residual is given by 𝐫 = 𝐐𝐐ᵀ[𝐛₁:𝐛₂]ᵀ - 𝐐[𝐑ₙᵀ𝐱:O]ᵀ = 𝐐[O:𝐐ᵀ𝐛₂]
 *   - the norm of residual is given by ‖ 𝐫 ‖₂ = ‖ 𝐐ᵀ𝐛₂ ‖₂
 *
 * The input will be treated as a whole m × (n+1) working space 𝐐[𝐀:𝐛] where
 *   - space of matrix 𝐀 will be used to store the 𝐐𝐀 result
 *   - space of vector 𝐛 will be used to store the 𝐐𝐛 result
 *
 * Optimality Conditions
 * ---------------------
 * Given a problem 𝚖𝚒𝚗 𝒇(𝐱) subject to 𝒉ⱼ(𝐱) = 0 (j = 1 ··· mₑ) and 𝒈ⱼ(𝐱) ≤ 0 (j = mₑ+1 ··· m),
 * its optimality at location 𝐱ᵏ are given by below KKT conditions:
 *   - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ) = 𝜵𝒇(𝐱ᵏ) + ∑𝛌ᵏⱼ𝜵𝒈ⱼ(𝐱ᵏ) = 0
 *   - 𝒈ⱼ(𝐱ᵏ) = 0   (j = 1 ··· mₑ)
 *   - 𝒈ⱼ(𝐱ᵏ) ≤ 0   (j = mₑ+1 ··· m)
 *   - 𝛌ᵏⱼ ≥ 0      (j = mₑ+1 ··· m)
 *   - 𝛌ᵏⱼ𝒈ⱼ(𝐱) = 0  (j = mₑ+1 ··· m)
 *
 * and substitute NNLS to the KKT conditions:
 *   - 𝒇(𝐱) = ½𝐱ᵀ𝐀𝐱 - 2𝐛ᵀ𝐀𝐱 + ½𝐛ᵀ𝐛  →  𝜵𝒇(𝐱) = 𝐀ᵀ(𝐀𝐱 + 𝐛)
 *   - 𝒈ⱼ(𝐱) = 0  (j = 1 ··· mₑ)    →  𝜵𝒈ⱼ(𝐱) = 0
 *   - 𝒈ⱼ(𝐱) = -𝐱ⱼ (j = mₑ+1 ··· m) →  𝜵𝒈ⱼ(𝐱) = -1
 *
 * the optimality conditions for NNLS are given:
 *   - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ) = 𝐀ᵀ(𝐀𝐱ᵏ + 𝐛) - ∑𝛌ᵏⱼ = 0
 *   - 𝛌ᵏⱼ ≥ 0 ∀j
 *   - 𝛌ᵏⱼ𝒈ⱼ(𝐱) = 0 ∀j
 *
 * NNLS introduces a dual m-vector 𝐰 = -𝝺 = -𝜵𝒇(𝐱) = 𝐀ᵀ(𝐛 - 𝐀𝐱) and optimality is given by:
 *   - 𝐰ⱼ = 0, ∀j ∈ ℙ
 *   - 𝐰ⱼ ≤ 0, ∀j ∈ ℤ
 *
 * Active Set Method
 * -----------------
 * The optimality of the activity set method is described by the KKT condition.
 *
 * Let 𝐱ᵏ be a feasible vector, the inequality constraints 𝒈ⱼ(𝐱ᵏ) (j = mₑ+1 ··· m) has two status:
 *   - active inequality constraints : 𝒈ⱼ(𝐱ᵏ) = 0
 *   - passive inequality constraints : 𝒈ⱼ(𝐱ᵏ) < 0
 *
 * Recall the 𝝺 describes how 𝒇(𝐱) change when relaxing constraints 𝒈ⱼ(𝐱) ≤ 0 → 𝛆 with a interruption 𝛆 > 0:
 *   - for 𝛌ⱼ < 0, relax the 𝒈ⱼ(𝐱) will decrease 𝒇(𝐱)
 *   - for 𝛌ⱼ > 0, relax the 𝒈ⱼ(𝐱) will increase 𝒇(𝐱)
 *
 * When we found some active constraints with 𝛌ⱼ < 0:
 *   - relax 𝒈ⱼ(𝐱) and move it from ℤ to ℙ
 *   - form a new pure equality constrain sub-problem EQP base on new ℤ
 *   - solve EQP with variable elimination method
 *
 * Assume 𝐬 is the EQP solution, then there is 𝒇(𝐬) < 𝒇(𝐱ᵏ) and:
 *   - if 𝐬 is feasible, update ℤ and ℙ and solve new EQP until feasible solution is not change
 *   - if 𝐬 is infeasible, we just obtain a descending direction 𝐝 = 𝐬 - 𝐱ᵏ and need to find
 *     a step length α > 0 such that 𝐱ᵏ + α𝐝 is feasible.
 *
 * The α can be obtained by projecting the infeasible 𝐬 to the boundaries defined by ℙ.
 *
 * Once new location 𝐱ᵏ⁺¹ = 𝐱ᵏ + α𝐝 is determined, update ℤ and ℙ and solve new EQP again.
 *
 * In case of NNLS, the EQP is a unconstrained least-squares problem 𝚖𝚒𝚗 ½‖ 𝐀ᴾ𝐱 - 𝐛 ‖₂.
 * The matrix 𝐀ᴾ is a matrix containing only the variables currently in ℙ.
 * Thus the solution is given by 𝐬 = [(𝐀ᴾ)ᵀ𝐀ᴾ]⁻¹(𝐀ᴾ)ᵀ𝐛 which is actually computed by QR decomposition.
 *
 * Non-negative Solution
 * ---------------------
 * Consider an m × (n+1) augmented matrix [𝐀:𝐛] defined by least-squares problem 𝐀𝐱 ≅ 𝐛.
 *
 * Let 𝐐 be an m × m orthogonal matrix that zeros the sub-diagonal elements in first n-1 cols of 𝐀.
 *
 *      n     1       n-1  1   1
 *     ┌┴┐   ┌┴┐      ┌┴┐ ┌┴┐ ┌┴┐
 *  𝐐[  𝐀 ﹕  𝐛 ] = ⎡  𝐑   𝒔   𝒖 ⎤ ]╴ n-1
 *                  ⎣  ０   𝒕   𝒗 ⎦ ]╴ m-n+1
 *
 * where 𝐑 is an m × m upper triangular full-rank matrix.
 *
 * Since orthogonal transformation preserves the relationship between the columns of augmented matrix:
 *
 *    (𝐐𝐀)ᵀ𝐐𝐛 ＝ 𝐀ᵀ𝐛 ＝ ⎡ 𝑹ᵀ ０ ⎤⎡ 𝒖 ⎤ ＝ ⎡    𝑹ᵀ𝒖   ⎤
 *                      ⎣ 𝒔ᵀ  𝒕ᵀ ⎦⎣ 𝒗 ⎦   ⎣ 𝒔ᵀ𝒖 + 𝒕ᵀ𝒗 ⎦
 *
 *                              n-1    1
 *                            ┌──┴──┐ ┌┴┐
 *    Assume there is 𝐀ᵀ𝐛 = [ 0 ··· 0  ω ]ᵀ = [𝑹ᵀ𝒖 : 𝒔ᵀ𝒖 + 𝒕ᵀ𝒗]ᵀ.
 *    Since 𝐑 is non-singular, 𝑹ᵀ𝒖 has only the trivial solution 𝒖 = 0 which means 𝒕ᵀ𝒗 = ω.
 *
 * The n-th component of solution to 𝐀𝐱 ≅ 𝐛 is the least squares solution of 𝒕𝐱ₙ ≅ 𝒗 which is 𝐱ₙ = 𝒕ᵀ𝒗/𝒕ᵀ𝒕 = ω/𝒕ᵀ𝒕.
 *
 * Thus when the n-th component of 𝐀ᵀ𝐛 is positive (ω > 0), then the n-th component of solution satisfied 𝐱ₙ > 0.
 *
 * References
 * ----------
 * C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
 * Chapters 23, Algorithm 23.10.
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/* Factor for checking linear independence */
#define FACTOR 0.01

/* External functions */
extern double h1(int pivot, int start, int m, double* u, int inc);
extern void h2(int pivot, int start, int m, double* u, int incu,
               double up, double* c, int incc, int mdc, int nc);
extern void g1(double a, double b, double* c, double* s, double* sig);
extern void g2(double c, double s, double* a, double* b);
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern void daxpy(int n, double a, const double* x, int incx, double* y, int incy);
extern double dnrm2(int n, const double* x, int incx);

/**
 * nnls - Non-Negative Least Squares
 * 
 * Solves 𝚖𝚒𝚗 ‖𝐀𝐱 - 𝐛‖₂ subject to 𝐱 ≥ 0
 * 
 * @param m       Number of rows in 𝐀
 * @param n       Number of columns in 𝐀
 * @param a       Matrix 𝐀 (column-major), modified on return to 𝐐𝐀
 * @param mda     Leading dimension of 𝐀
 * @param b       Vector 𝐛, modified on return to 𝐐𝐛
 * @param x       Output: solution vector 𝐱 of primal problem
 * @param w       Output: dual vector 𝐰 describing the weight of constraint
 * @param z       Working array of length m
 * @param index   Working array of length n, stores ℙ ∪ ℤ = {0,...,n-1}
 *                ℙ = index[:np] defines the subset columns of 𝐀
 *                ℤ = index[z1:]
 * @param maxIter Maximum iterations (0 = 3*n)
 * @param rnorm   Output: residual norm ‖𝐐ᵀ𝐛₂‖₂
 * @return        Status code (0 = success, 1 = exceeded max iterations, negative = error)
 */
int nnls(int m, int n, double* a, int mda,
         double* b, double* x, double* w,
         double* z, int* index, int maxIter,
         double* rnorm) {
    
    int i, ii, ip, iter, iz, izmax, j, jj, jz, l, np, z1;
    double alpha, asave, cc, sm, ss, t, unorm, up, wmax, ztest;
    double* aj;
    
    if (m <= 0 || n <= 0 || mda < m) {
        return -1;  /* Bad argument */
    }
    
    if (maxIter <= 0) {
        maxIter = 3 * n;
    }
    
    np = 0;   /* Number of elements in set ℙ */
    z1 = 0;   /* Start index of set ℤ */
    
    /* Initialize index = ℙ ∪ ℤ = {0,...,n-1} */
    for (i = 0; i < n; i++) {
        index[i] = i;
    }
    
    /* Start from 𝐱 = O and all indices are initially in ℤ */
    for (i = 0; i < n; i++) {
        x[i] = 0.0;
    }
    
    iter = 0;
    
    /* Main loop: continued until no more active constraints can be set free */
    for (;;) {
        /* Quit if all coefficients are positive: ℤ = ∅ (𝐱 ≥ 0),
           or if m columns of 𝐀 have been triangularized */
        if (z1 >= n || np >= m) {
            goto compute_rnorm;
        }
        
        /* Compute components of the dual vector 𝐰 = 𝐀ᵀ(𝐛 - 𝐀𝐱) (negative gradient).
         * Since 𝐰ⱼ = 0 for j ∈ ℙ, we only compute 𝐰ⱼ for j ∈ ℤ.
         * Given 𝐱ⱼ = 0 for j ∈ ℤ, the update simplifies to 𝐰 = 𝐀ᵀ𝐛. */
        for (iz = z1; iz < n; iz++) {
            j = index[iz];
            w[j] = ddot(m - np, &a[np + mda * j], 1, &b[np], 1);
        }
        
        for (;;) {
            /* Find index t ∈ ℤ such that 𝐰ₜ = 𝚊𝚛𝚐 𝚖𝚊𝚡 { 𝐰ⱼ: j ∈ ℤ } */
            wmax = 0.0;
            izmax = 0;
            for (iz = z1; iz < n; iz++) {
                j = index[iz];
                if (w[j] > wmax) {
                    wmax = w[j];
                    izmax = iz;
                }
            }
            
            /* Quit when 𝐰ⱼ ≤ 0, ∀j ∈ ℤ (no more constraint could be relaxed)
             * This indicates satisfaction of the Kuhn-Tucker conditions */
            if (wmax <= 0.0) {
                goto compute_rnorm;
            }
            
            /* Move index t from ℤ to ℙ */
            iz = izmax;
            j = index[iz];
            aj = &a[mda * j];
            
            /* Given j-th column of 𝐀, compute corresponding Householder vector 𝐮.
             * Save the pivot-th component of j-th column 𝐀ₚⱼ. */
            asave = aj[np];
            up = h1(np, np + 1, m, aj, 1);
            /* Now the pivot-th component of j-th column is (𝐐𝐀)ₚⱼ.
             * The pivot-th component of 𝐮 is returned as 𝐮ₚ. */
            
            /* Check new diagonal element to avoid near linear dependence */
            unorm = dnrm2(np, aj, 1);  /* ‖𝐮‖₂ */
            if (fabs(aj[np]) * FACTOR >= unorm * EPS) {
                /* Column j is sufficiently independent.
                 * Compute Householder transformation z = 𝐐𝐛 = [-σ‖𝐛‖₂ 0 ··· 0]ᵀ */
                memcpy(z, b, m * sizeof(double));
                h2(np, np + 1, m, aj, 1, up, z, 1, 1, 1);
                
                /* Solve 𝐐(𝐀𝐱)ⱼ ≅ 𝐐𝐛ⱼ for proposed new value for 𝐱ⱼ
                 * 𝐱 = (𝐐𝐀)⁺𝐐𝐛 */
                ztest = z[np] / aj[np];
                
                if (ztest > 0.0) {
                    /* Accept j: 𝐱ⱼ > 0 */
                    
                    /* Update b = 𝐐𝐛 */
                    memcpy(b, z, m * sizeof(double));
                    
                    /* Move j from ℤ to ℙ */
                    index[iz] = index[z1];
                    index[z1] = j;
                    z1++;
                    np++;
                    
                    /* Apply Householder transformations to cols in new ℤ */
                    if (z1 < n) {
                        for (jz = z1; jz < n; jz++) {
                            jj = index[jz];
                            h2(np - 1, np, m, aj, 1, up, &a[mda * jj], 1, mda, 1);
                        }
                    }
                    
                    /* Zero sub-diagonal elements in col j */
                    for (i = np; i < m; i++) {
                        aj[i] = 0.0;
                    }
                    
                    /* Set 𝐰ⱼ = 0 for j ∈ ℙ */
                    w[j] = 0.0;
                    break;
                }
            }
            
            /* Reject j as a candidate to be moved from ℤ to ℙ,
             * restore 𝐀ₚⱼ and test dual coefficients again */
            aj[np] = asave;
            w[j] = 0.0;
        }
        
        /* Inner loop: When new j joins ℙ, the coefficients of the free variables
         * in the unconstrained solution 𝐬 may turn negative.
         * The inner loop continues until all violating variables have been moved to ℤ. */
        for (;;) {
            /* Compute EQP solution 𝐬 by solving triangular system 𝐱߮ = [𝐑ₖ⁻¹:O]𝐐𝐛 */
            for (ip = np - 1; ip >= 0; ip--) {
                if (ip < np - 1) {
                    jj = index[ip + 1];
                    daxpy(ip + 1, -z[ip + 1], &a[mda * jj], 1, z, 1);
                }
                jj = index[ip];
                z[ip] /= a[ip + mda * jj];
            }
            
            /* Check iteration count */
            if (++iter > maxIter) {
                *rnorm = (np < m) ? dnrm2(m - np, &b[np], 1) : 0.0;
                return 1;  /* Exceeded max iterations */
            }
            
            /* See if all new constrained coefficients are feasible.
             * Find index t ∈ ℙ such that 𝐱ₜ/(𝐱ₜ-𝐳ₜ) = 𝚊𝚛𝚐 𝚖𝚒𝚗 { 𝐱ⱼ/(𝐱ⱼ-𝐳ⱼ) : 𝐳ⱼ ≤ 0, j ∈ ℙ } */
            alpha = 2.0;
            jj = -1;
            for (ip = 0; ip < np; ip++) {
                l = index[ip];
                if (z[ip] <= 0.0) {
                    /* Found unfeasible coefficient, compute alpha.
                     * ɑ = 𝐱ₜ/(𝐱ₜ-𝐳ₜ) */
                    t = -x[l] / (z[ip] - x[l]);
                    if (alpha > t) {
                        alpha = t;
                        jj = ip;
                    }
                }
            }
            
            /* If all coefficients are feasible, exit inner loop to main loop */
            if (jj < 0) {
                for (ip = 0; ip < np; ip++) {
                    l = index[ip];
                    x[l] = z[ip];
                }
                break;
            }
            
            /* Interpolate between x and z: 𝐱 = 𝐱 + ɑ(𝐬 - 𝐱) */
            for (ip = 0; ip < np; ip++) {
                l = index[ip];
                x[l] += alpha * (z[ip] - x[l]);
            }
            
            /* Move coefficient i from ℙ to ℤ */
            i = index[jj];
            for (;;) {
                x[i] = 0.0;
                if (++jj < np) {
                    for (j = jj; j < np; j++) {
                        ii = index[j];
                        double* ci = &a[mda * ii];
                        index[j - 1] = ii;
                        g1(ci[j - 1], ci[j], &cc, &ss, &ci[j - 1]);
                        ci[j] = 0.0;
                        for (l = 0; l < n; l++) {
                            if (l != ii) {
                                double* cl = &a[mda * l];
                                g2(cc, ss, &cl[j - 1], &cl[j]);
                            }
                        }
                        g2(cc, ss, &b[j - 1], &b[j]);
                    }
                }
                
                np--;
                z1--;
                index[z1] = i;
                
                /* See if the remaining coefficients in ℙ are feasible.
                 * They should be because of the way ɑ was determined.
                 * If any are infeasible, it is due to round-off error.
                 * Any that are non-positive will be set to zero and moved from ℙ to ℤ. */
                break;
            }
            
            /* Copy b into z, then solve again and loop back */
            memcpy(z, b, m * sizeof(double));
        }
    }
    
compute_rnorm:
    /* Calculate norm-2 of the residual vector: ‖𝐐ᵀ𝐛₂‖₂ */
    if (np < m) {
        *rnorm = dnrm2(m - np, &b[np], 1);
    } else {
        *rnorm = 0.0;
        for (i = 0; i < n; i++) {
            w[i] = 0.0;
        }
    }
    return 0;  /* Success */
}
