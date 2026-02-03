/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * LSEI - Least Squares with Equality and Inequality constraints
 * LSI  - Least Squares with Inequality constraints
 * Based on Lawson & Hanson, "Solving Least Squares Problems", Chapter 20, 23.
 *
 * LSEI (Least-Squares with linear Equality & Inequality) solves the problem:
 *   𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂  subject to  𝐂𝐱 = 𝐝  and  𝐆𝐱 ≥ 𝐡
 *
 * where:
 *   - 𝐄 is m × n matrix (no assumption need to be made for its rank)
 *   - 𝐱 ∈ ℝⁿ
 *   - 𝐟 ∈ ℝᵐ
 *   - 𝐂 is m1 × n matrix with 𝚛𝚊𝚗𝚔(𝐂) = k = m1 < n
 *   - 𝐝 ∈ ℝᵐ¹
 *   - 𝐆 is m2 × n matrix
 *   - 𝐡 ∈ ℝᵐ²
 *
 * LSE Problem:
 * -----------
 * Consider a LSE (Least-Squares with linear Equality) problem:
 *   𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂  subject to  𝐂𝐱 = 𝐝
 *
 * Given an orthogonal transformation of matrix 𝐂 where 𝐇 and 𝐊 are orthogonal, 𝐑 is full-rank:
 *   𝐂ₘ₁ₓₙ = 𝐇ₘ₁ₓₘ₁[𝐑ₖₓₖ ೦]𝐊ᵀₘ₁ₓₙ
 *
 * Its pseudo-inverse is defined by 𝐂⁺ = 𝐊𝐑⁺𝐇ᵀ where 𝐑⁺ = [𝐑⁻¹ ೦].
 *
 * Define partition 𝐊 = [𝐊₁ 𝐊₂] and [𝐊₁ 𝐊₂]ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ where
 * 𝐊₁ is an n × k matrix, 𝐊₂ is an n × (n-k) matrix.
 *
 * Assume k = m1 such that 𝐇 = 𝐈 and let 𝐊 satisfied that 𝐂𝐊 is lower triangular:
 *
 *   ⎡ 𝐂 ⎤ 𝐊 = ⎡ 𝐂߬₁  ೦  ⎤
 *   ⎣ 𝐀 ⎦     ⎣ 𝐀߬₁  𝐀߬₂ ⎦
 *
 * The solution of LSE problem is given by 𝐱߮ = 𝐊[𝐲߮₁ 𝐲߮₂]ᵀ where:
 *   - 𝐲߮₁ is obtained by solving triangular system 𝐂߬₁𝐲₁ = 𝐝
 *   - 𝐲߮₂ is obtained by solving least-squares 𝐀߬₂𝐲₂ ≅ 𝐛 - 𝐀߬₁𝐲߮₁
 *
 * Reduce to LSI:
 * -------------
 * Using the conclusion of LSE, the equality constraints can be eliminated by introducing
 * orthogonal basis 𝐊 = [𝐊₁:𝐊₂] of null space 𝐂𝐊₂ = 0 and let 𝐊ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ such that:
 *
 *              mᶜ  n-mᶜ
 *             ┌┴┐  ┌┴┐
 *   ⎡ 𝐂 ⎤ 𝐊 = ⎡ 𝐂߬₁   ೦  ⎤ ]╴mᶜ       𝐱 = 𝐊⎡ 𝐲₁ ⎤ ]╴ mᶜ
 *   ⎥ 𝐄 ⎥     ⎥ 𝐄߬₁   𝐄߬₂ ⎥ ]╴mᵉ            ⎣ 𝐲₂ ⎦ ]╴ n-mᶜ
 *   ⎣ 𝐆 ⎦     ⎣ 𝐆߬₁   𝐆߬₂ ⎦ ]╴mᵍ
 *
 * The 𝐲߮₁ is determined as solution of triangular system 𝐂߬₁𝐲₁ = 𝐝,
 * and 𝐲߮₂ is the solution of LSI problem:
 *   𝚖𝚒𝚗‖ 𝐄߬₂𝐲₂ - (𝐟 - 𝐄߬₂𝐲߮₁) ‖₂  subject to  𝐆߬₂𝐲₂ ≥ 𝐡 - 𝐆߬₁𝐲߮₁
 *
 * Finally the solution of LSEI problem is given by 𝐱߮ = 𝐊[𝐲߮₁ 𝐲߮₂]ᵀ.
 *
 * Lagrange Multiplier:
 * -------------------
 * The optimality conditions (KKT) for LSEI are given:
 *   - 𝜵ℒ(𝐱ᵏ,𝛍ᵏ,𝛌ᵏ) = 𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐂ᵀ𝛍ᵏ - 𝐆ᵀ𝛌ᵏ = 0
 *   - 𝛌ᵏⱼ ≥ 0 (j = mₑ+1 ··· m)
 *   - 𝛌ᵏⱼ(𝐡ⱼ - 𝐆ⱼ𝐱) = 0 (j = mₑ+1 ··· m)
 *
 * Multiplier of equality constraints is given by:
 *   𝛍ᵏ = (𝐂ᵀ)⁻¹[𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐆ᵀ𝛌ᵏ]
 *
 * References:
 *   C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
 *   Prentice Hall, 1974. (revised 1995 edition)
 *   Chapters 20, Algorithm 20.24.
 *   Chapters 23, Section 6.
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/* External functions */
extern double h1(int pivot, int start, int m, double* u, int inc);
extern void h2(int pivot, int start, int m, double* u, int incu,
               double up, double* c, int incc, int mdc, int nc);
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern void daxpy(int n, double a, const double* x, int incx, double* y, int incy);
extern double dnrm2(int n, const double* x, int incx);
extern void dcopy(int n, const double* x, int incx, double* y, int incy);
extern int ldp(int m, int n, double* g, int mdg,
               double* h, double* x, double* w,
               int* jw, int maxIter, double* xnorm);
extern int hfti(int m, int n, double* a, int mda,
                double* b, int mdb, int nb,
                double tau, double* rnorm,
                double* h, double* g, int* ip);

/**
 * lsi - Least Squares with Inequality constraints
 * 
 * LSI (Least-Squares with linear Inequality) solves the problem:
 *   𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂  subject to  𝐆𝐱 ≥ 𝐡
 *
 * where:
 *   - 𝐄 is m × n matrix with 𝚛𝚊𝚗𝚔(𝐄) = n
 *   - 𝐟 ∈ ℝⁿ
 *   - 𝐆 is mg × n matrix
 *   - 𝐡 ∈ ℝᵐᵍ
 *
 * Consider below orthogonal decomposition of 𝐄:
 *
 *                   n    m-n
 *                  ┌┴┐   ┌┴┐
 *   𝐄 = 𝐐⎡𝐑 ೦⎤𝐊ᵀ ≡ [ 𝐐₁ : 𝐐₂ ]⎡𝐑⎤ 𝐊ᵀ
 *        ⎣೦ ೦⎦                 ⎣೦⎦
 *
 * where:
 *   - 𝐐 is m × m orthogonal
 *   - 𝐊 is n × n orthogonal
 *   - 𝐑 is n × n non-singular
 *
 * By introducing orthogonal change of variable 𝐱 = 𝐊ᵀ𝐲 one can obtain:
 *
 *   ⎡𝐐₁ᵀ⎤(𝐄𝐱 - 𝐟) = ⎡𝐑𝐲 - 𝐐₁ᵀ𝐟⎤
 *   ⎣𝐐₂ᵀ⎦          ⎣   𝐐₂ᵀ𝐟  ⎦
 *
 * Since orthogonal transformation does not change matrix norm and ‖ 𝐐₂ᵀ𝐟 ‖₂ is constant,
 * the LSI objective could be rewritten as 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ = 𝚖𝚒𝚗‖ 𝐑𝐲 - 𝐐₁ᵀ𝐟 ‖₂.
 *
 * By following definitions:
 *   - 𝐟߫₁ = 𝐐₁ᵀ𝐟
 *   - 𝐟߫₂ = 𝐐₂ᵀ𝐟
 *   - 𝐳 = 𝐑𝐲 - 𝐟߫₁
 *   - 𝐱 = 𝐊𝐑⁻¹(𝐳 + 𝐟߫₁)
 *
 * The LSI problem is equivalent to LDP problem:
 *   𝚖𝚒𝚗 ‖ 𝐳 ‖₂  subject to  𝐆𝐊𝐑⁻¹𝐳 ≥ 𝐡 - 𝐆𝐊𝐑⁻¹𝐟߫₁
 *
 * The residual vector norm of LSI problem can be computed from (‖ 𝐳 ‖₂ + ‖ 𝐟߫₂ ‖₂)¹ᐟ².
 *
 * References:
 *   C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
 *   Prentice Hall, 1974. (revised 1995 edition)
 *   Chapters 23, Section 5.
 * 
 * @param e       Matrix E (column-major), modified on return
 * @param f       Vector f, modified on return
 * @param g       Matrix G (column-major), modified on return
 * @param h       Vector h, modified on return
 * @param le      Leading dimension of E
 * @param me      Number of rows in E
 * @param lg      Leading dimension of G
 * @param mg      Number of rows in G (constraints)
 * @param n       Number of variables
 * @param x       Output: solution vector
 * @param w       Working array of length (n+1)*(mg+2)+2*mg
 * @param jw      Working array of length mg
 * @param maxIter Maximum iterations for LDP
 * @param xnorm   Output: residual norm
 * @return        Status code (0 = success, negative = error)
 */
int lsi(double* e, double* f, double* g, double* h,
        int le, int me, int lg, int mg, int n,
        double* x, double* w, int* jw, int maxIter,
        double* xnorm) {
    
    int i, j, status;
    double t, diag;
    
    if (n < 1) {
        return -1;  /* Bad argument */
    }
    
    /* QR-factors of 𝐄 and application to 𝐟 */
    for (i = 0; i < n; i++) {
        j = (i + 1 < n - 1) ? i + 1 : n - 1;
        t = h1(i, i + 1, me, &e[i * le], 1);
        h2(i, i + 1, me, &e[i * le], 1, t, &e[j * le], 1, le, n - i - 1);  /* 𝐐𝐄 = 𝐑 (triangular) */
        h2(i, i + 1, me, &e[i * le], 1, t, f, 1, 1, 1);                     /* 𝐐𝐟 = [ 𝐟߫₁ : 𝐟߫₂ ] */
    }
    
    /* Transform 𝐆 and 𝐡 to get LDP */
    for (i = 0; i < mg; i++) {
        for (j = 0; j < n; j++) {
            diag = e[j + le * j];
            if (fabs(diag) < EPS || isnan(diag)) {
                return -3;  /* 𝚛𝚊𝚗𝚔(𝐄) < n (E is singular) */
            }
            /* 𝐆𝐊𝐑⁻¹ (𝐊 = 𝐈ₙ) */
            g[i + lg * j] = (g[i + lg * j] - ddot(j, &g[i], lg, &e[j * le], 1)) / diag;
        }
        h[i] -= ddot(n, &g[i], lg, f, 1);  /* 𝐡 - 𝐆𝐊𝐑⁻¹𝐟߫₁ */
    }
    
    /* Solve LDP: 𝚖𝚒𝚗 ‖ 𝐳 ‖₂  subject to  𝐆𝐊𝐑⁻¹𝐳 ≥ 𝐡 - 𝐆𝐊𝐑⁻¹𝐟߫₁ */
    status = ldp(mg, n, g, lg, h, x, w, jw, maxIter, xnorm);
    
    if (status == 0) {
        /* 𝐳 + 𝐟߫₁ */
        daxpy(n, 1.0, f, 1, x, 1);
        
        /* 𝐊𝐑⁻¹(𝐳 + 𝐟߫₁) */
        for (i = n - 1; i >= 0; i--) {
            j = (i + 1 < n - 1) ? i + 1 : n - 1;
            x[i] = (x[i] - ddot(n - i - 1, &e[i + le * j], le, &x[j], 1)) / e[i + le * i];
        }
        
        /* Compute residual norm: (‖ 𝐳 ‖₂ + ‖ 𝐟߫₂ ‖₂)¹ᐟ² */
        j = (n < me - 1) ? n : me - 1;
        t = dnrm2(me - n, &f[j], 1);  /* ‖ 𝐟߫₂ ‖₂ */
        *xnorm = sqrt((*xnorm) * (*xnorm) + t * t);
    }
    
    return status;
}

/**
 * lsei - Least Squares with Equality and Inequality constraints
 * 
 * Solves the LSEI problem:
 *   𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂  subject to  𝐂𝐱 = 𝐝  and  𝐆𝐱 ≥ 𝐡
 * 
 * @param c       Matrix C (column-major), modified on return
 * @param d       Vector d, modified on return
 * @param e       Matrix E (column-major), modified on return
 * @param f       Vector f, modified on return
 * @param g       Matrix G (column-major), modified on return
 * @param h       Vector h, modified on return
 * @param lc      Leading dimension of C
 * @param mc      Number of equality constraints (rows in C)
 * @param le      Leading dimension of E
 * @param me      Number of rows in E
 * @param lg      Leading dimension of G
 * @param mg      Number of inequality constraints (rows in G)
 * @param n       Number of variables
 * @param x       Output: solution vector
 * @param w       Working array: 2×mc+me+(me+mg)×(n-mc) + (n-mc+1)×(mg+2)+2×mg
 * @param jw      Working array: max(mg, min(me, n-mc))
 * @param maxIter Maximum iterations for LDP
 * @param norm    Output: residual norm
 * @return        Status code (0 = success, negative = error)
 *                Multipliers returned as 𝛍 = w[0:mc] and 𝛌 = w[mc:mc+mg]
 */
int lsei(double* c, double* d, double* e, double* f,
         double* g, double* h,
         int lc, int mc, int le, int me, int lg, int mg, int n,
         double* x, double* w, int* jw, int maxIter,
         double* norm) {
    
    int i, j, l, iw, status, rank;
    double t, diag, up;
    double *ws, *wp, *we, *wf, *wg;
    int k;
    
    if (n < 1 || mc > n) {
        return -1;  /* Bad argument */
    }
    
    l = n - mc;
    
    /* Working space layout (matching Go implementation):
     * w[0:mc]                          = Lagrange multipliers for equality constraints (𝛍)
     * w[mc:mc+(l+1)*(mg+2)+2*mg]       = workspace for LSI (ws)
     * w[...+mc]                        = Householder pivots for 𝐊 (wp)
     * w[...+me*l]                      = 𝐄߬₂ (we)
     * w[...+me]                        = 𝐟 - 𝐄߬₁𝐲߮₁ (wf)
     * w[...+mg*l]                      = 𝐆߬₂ (wg)
     */
    
    iw = mc;
    ws = &w[iw];
    iw += (l + 1) * (mg + 2) + 2 * mg;
    wp = &w[iw];
    iw += mc;
    we = &w[iw];
    iw += me * l;
    wf = &w[iw];
    iw += me;
    wg = &w[iw];
    
    /* Triangularize 𝐂 and apply factors to 𝐄 and 𝐆 */
    for (i = 0; i < mc; i++) {
        j = (i + 1 < lc - 1) ? i + 1 : lc - 1;
        wp[i] = h1(i, i + 1, n, &c[i], lc);
        h2(i, i + 1, n, &c[i], lc, wp[i], &c[j], lc, 1, mc - i - 1);  /* 𝐂𝐊 = [𝐂߬₁ ೦] */
        h2(i, i + 1, n, &c[i], lc, wp[i], e, le, 1, me);               /* 𝐄𝐊 = [𝐄߬₁ 𝐄߬₂] */
        h2(i, i + 1, n, &c[i], lc, wp[i], g, lg, 1, mg);               /* 𝐆𝐊 = [𝐆߬₁ 𝐆߬₂] */
    }
    
    /* Solve triangular system 𝐂߬₁𝐲₁ = 𝐝 */
    for (i = 0; i < mc; i++) {
        diag = c[i + lc * i];
        if (fabs(diag) < EPS) {
            return -4;  /* 𝚛𝚊𝚗𝚔(𝐂) < mc (C is singular) */
        }
        x[i] = (d[i] - ddot(i, &c[i], lc, x, 1)) / diag;  /* 𝐲߮₁ = 𝐂߬₁⁻¹𝐝 */
    }
    
    /* First [mg] of working space store the multiplier returned by LDP */
    for (i = 0; i < mg; i++) {
        ws[i] = 0.0;
    }
    
    if (mc < n) {  /* 𝚛𝚊𝚗𝚔(𝐂) < n */
        /* 𝐟 - 𝐄߬₁𝐲߮₁ */
        for (i = 0; i < me; i++) {
            wf[i] = f[i] - ddot(mc, &e[i], le, x, 1);
        }
        
        if (l > 0) {
            /* Copy 𝐄߬₂ */
            for (i = 0; i < me; i++) {
                dcopy(l, &e[i + le * mc], le, &we[i], me);
            }
            /* Copy 𝐆߬₂ */
            for (i = 0; i < mg; i++) {
                dcopy(l, &g[i + lg * mc], lg, &wg[i], mg);
            }
        }
        
        if (mg > 0) {
            /* 𝐡 - 𝐆߬₁𝐲߮₁ */
            for (i = 0; i < mg; i++) {
                h[i] -= ddot(mc, &g[i], lg, x, 1);
            }
            
            /* Compute 𝐲߮₂ by solving LSI: 𝚖𝚒𝚗‖ 𝐄߬₂𝐲₂ - (𝐟 - 𝐄߬₂𝐲߮₁) ‖₂  𝚜.𝚝  𝐆߬₂𝐲₂ ≥ 𝐡 - 𝐆߬₁𝐲߮₁ */
            status = lsi(we, wf, wg, h, me, me, mg, mg, l, &x[mc], ws, jw, maxIter, norm);
            
            if (mc == 0) {
                /* Multipliers returned as 𝛌 = w[0:mg] */
                return status;
            }
            
            if (status != 0) {
                return status;
            }
            
            t = dnrm2(mc, x, 1);
            *norm = sqrt((*norm) * (*norm) + t * t);
        } else {
            /* Solve unconstrained: 𝚖𝚒𝚗‖ 𝐄߬₂𝐲₂ - (𝐟 - 𝐄߬₂𝐲߮₁) ‖₂ */
            k = (le > n) ? le : n;
            double nrm[1];
            
            rank = hfti(me, l, we, me, wf, k, 1, SQRT_EPS, nrm, w, &w[l], jw);
            *norm = nrm[0];
            dcopy(l, wf, 1, &x[mc], 1);
            
            if (rank != l) {
                return -5;  /* HFTI rank defect */
            }
        }
    }
    
    /* 𝐄ᵀ(𝐄𝐱 - 𝐟) */
    for (i = 0; i < me; i++) {
        f[i] = ddot(n, &e[i], le, x, 1) - f[i];
    }
    
    /* 𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐆ᵀ𝛌 */
    for (i = 0; i < mc; i++) {
        d[i] = ddot(me, &e[i * le], 1, f, 1) -
               ddot(mg, &g[i * lg], 1, ws, 1);
    }
    
    /* 𝐱߮ = 𝐊[𝐲߮₁ 𝐲߮₂]ᵀ */
    for (i = mc - 1; i >= 0; i--) {
        h2(i, i + 1, n, &c[i], lc, wp[i], x, 1, 1, 1);
    }
    
    /* 𝛍 = (𝐂ᵀ)⁻¹[𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐆ᵀ𝛌] */
    for (i = mc - 1; i >= 0; i--) {
        j = (i + 1 < lc - 1) ? i + 1 : lc - 1;
        w[i] = (d[i] - ddot(mc - i - 1, &c[j + lc * i], 1, &w[j], 1)) / c[i + lc * i];
    }
    
    /* Copy 𝛌 multipliers from ws to w[mc:mc+mg] */
    for (i = 0; i < mg; i++) {
        w[mc + i] = ws[i];
    }
    
    /* Multipliers returned as 𝛍 = w[0:mc] and 𝛌 = w[mc:mc+mg] */
    return 0;  /* Success */
}
