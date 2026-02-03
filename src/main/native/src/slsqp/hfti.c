/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * HFTI - Householder Forward Triangulation with column Interchanges
 * Solves least-squares problem 𝐀𝐗 ≅ 𝐁 using Householder transformations.
 * Based on Lawson & Hanson, "Solving Least Squares Problems", Chapter 14.
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/* Factor for recomputing column norms */
#define FACTOR 0.001

/* External functions */
extern double h1(int pivot, int start, int m, double* u, int inc);
extern void h2(int pivot, int start, int m, double* u, int incu,
               double up, double* c, int incc, int mdc, int nc);

/**
 * hfti - Householder Forward Triangulation with column Interchanges
 *
 * Solve a least-squares problem 𝐀𝐗 ≅ 𝐁 where:
 *   - 𝐀 is m × n matrix with 𝚙𝚜𝚎𝚞𝚍𝚘-𝚛𝚊𝚗𝚔(𝐀) = k
 *   - 𝐗 is n × nb matrix having column vectors 𝐱ⱼ
 *   - 𝐁 is m × nb matrix
 *
 * # Basics
 *
 * Recall the least-squares problem 𝐀𝐱 ≅ 𝐛 where 𝚛𝚊𝚗𝚔(𝐀) = k with below orthogonal transformation.
 *
 *   𝐀ₘₓₙ = 𝐇ₘₓₘ[𝐑ₖₓₖ ೦]𝐊ᵀₙₓₙ   𝐊ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ   𝐇ᵀ𝐛 = [𝐠₁ 𝐠₂]ᵀ
 *
 * where 𝐇 and 𝐊 are orthogonal, 𝐑 is full-rank, 𝐲₁, 𝐠₁ is k-vector and 𝐲₂, 𝐠₂ is (n-k)-vector, such that:
 *   - ‖ 𝐀𝐱 - 𝐛 ‖₂ = ‖ 𝐑𝐲₁ - 𝐠₁ ‖₂ + ‖𝐠₂‖₂ (since orthogonal transformation preserve the norm)
 *   - 𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂ = 𝚖𝚒𝚗‖ 𝐑𝐲₁ - 𝐠₁ ‖₂    (since ‖𝐠₂‖₂ is constant)
 *   - 𝐲₁ = 𝐑⁻¹𝐠₁                          (since 𝐑 is invertible)
 *   - 𝐲₂ is arbitrary                     (usually set 𝐲₂ = O)
 *
 * The unique solution of minimum length is given by 𝐱 = 𝐊[𝐲₁ 𝐲₂]ᵀ = 𝐊[𝐑⁻¹𝐠₁ ೦]ᵀ and the norm of residual satisfies ‖𝐫‖ = ‖𝐠₂‖.
 *
 * When 𝚛𝚊𝚗𝚔(𝐀) = k < 𝚖𝚒𝚗(m,n), there exist orthogonal matrix 𝐐 and permutation matrix 𝐏 such that 𝐐𝐀𝐏 = 𝐑
 *
 *   ⎡𝐑₁₁ 𝐑₁₂⎤  where 𝐑₁₁ is k × k matrix, 𝐑₁₂ is k × (n-k) matrix
 *   ⎣ ೦  𝐑₂₂⎦    and 𝐑₂₂ is (n-k) × (n-k) matrix
 *
 *   - permutation matrix 𝐏 interchange column of 𝐀 resulting first k columns of 𝐀𝐏 is linearly independent
 *   - orthogonal matrix 𝐐 interchange column of 𝐀 resulting 𝐐𝐀𝐏 is zero below the main diagonal
 *
 * HFTI assume 𝐀 is rank-deficient that make problem very ill-conditioned.
 *
 * To stabilizing such problem, HFTI first figure out a 𝚙𝚜𝚎𝚞𝚍𝚘-𝚛𝚊𝚗𝚔(𝐀) = k < 𝛍 where 𝛍 = 𝚖𝚒𝚗(m,n) by computing 𝐑.
 * By setting 𝐑₂₂ = ೦ and replace the 𝐀 with 𝐀߬ = 𝐐ᵀ[𝐑₁₁ 𝐑₁₂]ₙₓₙ𝐏ᵀ and 𝐛 with 𝐜 = 𝐐ᵀ𝐛 = [𝐜₁ 𝐜₂]ᵀ the problem become 𝐀߬ 𝐱 ≅ 𝐜.
 *
 * Since [𝐑₁₁:𝐑₁₂]ₖₓₙ is full-row-rank, its triangulation can be obtained by orthogonal transformation 𝐊
 * such that [𝐑₁₁:𝐑₁₂]𝐊ₙₓₙ = [𝐖ₖₓₖ:೦] and 𝐊ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ.
 *   - For forward triangulation, 𝐖 is a non-singular upper triangular matrix
 *   - For backward triangulation, 𝐖 is a non-singular lower triangular matrix
 *
 * The minimum length solution of 𝐀߬ 𝐱 ≅ 𝐜 is given by 𝐱 = 𝐏𝐊[𝐲₁ 𝐲₂]ᵀ = 𝐏𝐊[𝐖⁻¹𝐜₁ ೦]ᵀ.
 * Note that 𝐖 is triangular, computation of 𝐖⁻¹𝐜₁ is simple.
 *
 * # Pseudo Rank
 *
 * The pseudo-rank is not a nature of 𝐀 but determined by a user-specified tolerance 𝛕 > 0.
 * All sub-diagonal elements in 𝐑 = 𝐐𝐀𝐏 are zero and its diagonal elements satisfy |rᵢ₊₁| < |rᵢ| where i = 1, ..., 𝛍-1.
 * The pseudo-rank k equal to the number of diagonal elements of 𝐑 exceeding 𝛕 in magnitude.
 *
 * # Column Pivoting
 *
 *   𝐏 is constructed as product of 𝛍 transposition matrix 𝐏₁ × ··· × 𝐏ᵤ
 *   where 𝐏ⱼ = (j, pⱼ) denotes the interchange between column j and pⱼ.
 *
 *   𝐐 is constructed as product of 𝛍 Householder matrix 𝐐ᵤ × ··· × 𝐐₁
 *   where 𝐐ⱼ corresponding to the j column after interchange interchange.
 *
 * This column is the best candidate for numerical stability.
 * For the construction of j-th Householder transformation, we consider remaining columns j,...,n
 * and select the 𝝺-th column whose sum of squares of components in rows j,...,m is greatest.
 *
 * # Algorithm Outline
 *
 * HFTI first transforms the augmented matrix [𝐀:𝐁] ≡ [𝐑:𝐂] = [𝐐𝐀𝐏:𝐐𝐁] using
 * pre-multiplying Householder transformation 𝐐 with column interchange 𝐏
 * where 𝐀𝐏 is linearly independent and 𝐐 resulting all sub-diagonal elements in 𝐀𝐏 are zero.
 *
 * After determining the pseudo-rank k by diagonal element of 𝐑, apply forward triangulation
 * to 𝐑𝐊 = [𝐖:೦] using Householder transformation 𝐊.
 *
 * Then solve triangular system 𝐖𝐲₁ = 𝐜₁ and apply 𝐊 to 𝐲₁.
 * Finally the solution 𝐱 is obtained by rearranging the 𝐊𝐲₁ = 𝐊𝐖⁻¹𝐜₁ by 𝐏.
 *
 * # Memory Layout
 *
 * The space of input data 𝐀 is will be modified to store the intermediate results:
 *
 *          k        n-k
 *      ┌───┴───┐  ┌──┴──┐
 *   ⎡ w₁₁ w₁₂ w₁₃ k₁₄ k₁₅ ⎤┐          the data that define 𝐐 occupy the lower triangular part of 𝐀
 *   ⎥ u₁₂ w₂₂ w₂₃ k₂₄ k₂₅ ⎥├ k        the data that define 𝐊 occupy the rectangular portion of 𝐀
 *   ⎥ u₁₃ u₂₃ w₃₃ k₃₄ k₃₅ ⎥┘          the data that define 𝐖 occupy the rectangular portion of 𝐀
 *   ⎥ u₁₄ u₂₄ u₃₄  †   †  ⎥┐
 *   ⎥ u₁₅ u₂₅ u₃₅ u₄₅  †  ⎥├ n-k
 *   ⎣ u₁₆ u₂₆ u₃₆ u₄₆ u₅₆ ⎦┘
 *
 * And 3 × 𝚖𝚒𝚗(m,n) additional working space required:
 *
 *   g: [ u₁₁ u₂₂ u₃₃ u₄₄ u₅₅ ]    the pivot scalars for 𝐐
 *   h: [ k₁₁ k₂₂ k₃₃  †   †  ]    the pivot scalars for 𝐊
 *   p: [ p₁  p₂  p₃  p₄  p₅  ]    interchange record define 𝐏
 *
 * # References
 *
 *   C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
 *   Chapters 14, Algorithm 14.9.
 *
 * @param m      Number of rows in 𝐀 (either m ≥ n or m < n is permitted)
 * @param n      Number of columns in 𝐀 (no restriction on 𝚛𝚊𝚗𝚔(𝐀))
 * @param a      Matrix 𝐀 (column-major), modified on return to store intermediate results
 * @param mda    Leading dimension of 𝐀
 * @param b      Matrix 𝐁 (column-major), contains solution 𝐗 in first n rows on return
 * @param mdb    Leading dimension of 𝐁
 * @param nb     Number of columns in 𝐁 (right-hand sides), if nb = 0 no reference to b
 * @param tau    Absolute tolerance 𝛕 for pseudo-rank determination
 * @param rnorm  Output: residual norms ‖𝐠₂‖ for each column of 𝐁
 * @param h      Working array of length n (column norms and 𝐊 pivots)
 * @param g      Working array of length min(m,n) (𝐐 pivots)
 * @param ip     Working array of length min(m,n) (permutation 𝐏)
 * @return       Pseudo-rank k
 */
int hfti(int m, int n, double* a, int mda,
         double* b, int mdb, int nb,
         double tau, double* rnorm,
         double* h, double* g, int* ip) {
    
    int diag, i, j, jb, k, l, lmax;
    double hmax, sm, t, up, unorm, cl, v;
    
    diag = (m < n) ? m : n;
    if (diag <= 0) {
        return 0;
    }
    
    hmax = 0.0;
    
    for (j = 0; j < diag; j++) {
        /* Update the squared column lengths and find lmax. */
        lmax = j;
        
        if (j > 0) {
            v = -1e308;  /* Use large negative value for NaN-safe comparison */
            for (l = j; l < n; l++) {
                t = a[(j - 1) + mda * l];
                h[l] -= t * t;
                if (!(h[l] <= v)) {  /* handles NaN correctly */
                    lmax = l;
                    v = h[l];
                }
            }
        }
        
        /* Compute squared column lengths and find lmax. */
        if (j == 0 || FACTOR * h[lmax] < hmax * EPS) {
            v = -1e308;
            for (l = j; l < n; l++) {
                sm = 0.0;
                for (i = j; i < m; i++) {
                    t = a[i + mda * l];
                    sm += t * t;
                }
                h[l] = sm;
                if (!(h[l] <= v)) {
                    lmax = l;
                    v = h[l];
                }
            }
            hmax = h[lmax];
        }
        
        /* Perform column interchange 𝐏 if needed. */
        ip[j] = lmax;
        if (ip[j] != j) {
            /* Swap columns j and lmax */
            for (i = 0; i < m; i++) {
                t = a[i + mda * j];
                a[i + mda * j] = a[i + mda * lmax];
                a[i + mda * lmax] = t;
            }
            h[lmax] = h[j];
        }
        
        /* Compute the j-th transformation and apply it to 𝐀 and 𝐁. */
        i = (j + 1 < n - 1) ? j + 1 : n - 1;
        h[j] = h1(j, j + 1, m, &a[mda * j], 1);                              /* 𝐐 */
        h2(j, j + 1, m, &a[mda * j], 1, h[j], &a[mda * i], 1, mda, n - j - 1); /* 𝐑 = 𝐐𝐀𝐏 */
        h2(j, j + 1, m, &a[mda * j], 1, h[j], b, 1, mdb, nb);                  /* 𝐂 = 𝐐𝐁 */
    }
    
    /* Determine the pseudo-rank
     * k = 𝚖𝚊𝚡ⱼ |𝐑ⱼⱼ| > 𝛕 */
    k = diag;
    for (j = 0; j < diag; j++) {
        if (fabs(a[j + mda * j]) <= tau) {
            k = j;
            break;
        }
    }
    
    /* Compute the norms of the residual vectors ‖𝐠₂‖ ≡ ‖𝐜₂‖ */
    for (jb = 0; jb < nb; jb++) {
        sm = 0.0;
        if (k < m) {
            for (i = k; i < m; i++) {
                t = b[i + mdb * jb];
                sm += t * t;
            }
        }
        rnorm[jb] = sqrt(sm);
    }
    
    if (k > 0) {
        /* If the pseudo-rank is less than n,
         * compute Householder decomposition of first k rows. */
        if (k < n) {
            for (i = k - 1; i >= 0; i--) {
                g[i] = h1(i, k, n, &a[i], mda);                    /* 𝐊 */
                h2(i, k, n, &a[i], mda, g[i], a, mda, 1, i);       /* 𝐑₁₁𝐊 = 𝐖 */
            }
        }
        
        /* If 𝐁 is provided, compute 𝐗 */
        for (jb = 0; jb < nb; jb++) {
            double* cb = &b[mdb * jb];
            
            /* Solve k × k triangular system 𝐖𝐲₁ = 𝐜₁ */
            for (i = k - 1; i >= 0; i--) {
                sm = 0.0;
                for (j = i + 1; j < k; j++) {
                    sm += a[i + mda * j] * cb[j];
                }
                cb[i] = (cb[i] - sm) / a[i + mda * i];
            }
            
            /* Complete computation of solution vector. */
            if (k < n) {
                /* 𝐊𝐲₂ = O */
                for (i = k; i < n; i++) {
                    cb[i] = 0.0;
                }
                /* 𝐊𝐲₁ = 𝐊𝐖⁻¹𝐜₁ */
                for (i = 0; i < k; i++) {
                    h2(i, k, n, &a[i], mda, g[i], cb, 1, mdb, 1);
                }
            }
            
            /* Re-order solution vector 𝐊𝐲 by 𝐏 to obtain 𝐱. */
            for (j = diag - 1; j >= 0; j--) {
                l = ip[j];
                if (l != j) {
                    t = cb[l];
                    cb[l] = cb[j];
                    cb[j] = t;
                }
            }
        }
    } else if (nb > 0) {
        for (jb = 0; jb < nb; jb++) {
            for (i = 0; i < n; i++) {
                b[i + mdb * jb] = 0.0;
            }
        }
    }
    
    /* The solution vectors 𝐗 are now in the first n rows of 𝐁. */
    return k;
}
