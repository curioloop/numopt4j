/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * L-BFGS-B BFGS Update Functions
 * Based on the Go implementation in update.go
 *
 * This file implements the L-BFGS-B matrix update routines:
 *   - updateCorrection (matupd): Updates correction matrices Sₖ and Yₖ
 *   - formT (formt): Forms and factorizes T = θSᵀS + LD⁻¹Lᵀ
 *   - formK (formk): Forms the LELᵀ factorization of the indefinite K matrix
 *
 * The L-BFGS-B algorithm maintains a limited-memory approximation to the
 * inverse Hessian using the most recent m correction pairs:
 *   Sₖ = [sₖ₋ₘ, ..., sₖ₋₁]  where sᵢ = xᵢ₊₁ - xᵢ
 *   Yₖ = [yₖ₋ₘ, ..., yₖ₋₁]  where yᵢ = gᵢ₊₁ - gᵢ
 *
 * Reference: Byrd, Lu, Nocedal, Zhu, "A Limited Memory Algorithm for Bound
 *            Constrained Optimization", SIAM J. Scientific Computing, 1995.
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/* External BLAS functions */
extern void dcopy(int n, const double* x, int incx, double* y, int incy);
extern void daxpy(int n, double a, const double* x, int incx, double* y, int incy);
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern void dscal(int n, double a, double* x, int incx);

/* External LINPACK functions */
extern int dpofa(double* a, int lda, int n);
extern int dtrsl(double* t, int ldt, int n, double* b, int job);

/* Constants */

/* Solve mode constants for dtrsl (matching Go linpack.go) */
#define SOLVE_LOWER_N 0   /* 0b00 - Solve L*x = b (lower triangular, no transpose) */
#define SOLVE_UPPER_N 1   /* 0b01 - Solve U*x = b (upper triangular, no transpose) */
#define SOLVE_LOWER_T 2   /* 0b10 - Solve L'*x = b (lower triangular, transpose) */
#define SOLVE_UPPER_T 3   /* 0b11 - Solve U'*x = b (upper triangular, transpose) */

/**
 * Subroutine updateCorrection (matupd)
 *
 * This subroutine updates the correction matrices Sₖ and Yₖ, and forms the
 * middle matrix components needed for the L-BFGS approximation.
 *
 * Given the new correction pair:
 *   sₖ = xₖ₊₁ - xₖ  (step vector)
 *   yₖ = gₖ₊₁ - gₖ  (gradient difference)
 *
 * The subroutine:
 *   1. Checks the curvature condition: sₖᵀyₖ > ε‖yₖ‖²
 *   2. Updates the circular buffer storing Sₖ and Yₖ
 *   3. Computes θ = yₖᵀyₖ / sₖᵀyₖ (scaling factor)
 *   4. Updates SᵀS (upper triangular) and SᵀY (lower triangular)
 *
 * The matrices SᵀY contains:
 *   - D = diag{sᵢᵀyᵢ} on the diagonal
 *   - L = {sᵢᵀyⱼ}ᵢ>ⱼ in the strictly lower triangular part
 *
 * @param n     Problem dimension
 * @param m     Maximum number of correction pairs
 * @param s     New step vector sₖ = xₖ₊₁ - xₖ
 * @param y     New gradient difference yₖ = gₖ₊₁ - gₖ
 * @param ws    Workspace containing correction matrices
 */
void update_correction(int n, int m, const double* s, const double* y,
                       LbfgsbWorkspace* ws) {
    int j, ptr;
    
    double* ws_arr = ws->ws;  /* Sₖ matrix (n × m) - stores correction vectors sᵢ */
    double* wy = ws->wy;      /* Yₖ matrix (n × m) - stores gradient differences yᵢ */
    double* sy = ws->sy;      /* SᵀY matrix (m × m) - contains D and L */
    double* ss = ws->ss;      /* SᵀS matrix (m × m) - upper triangular */
    
    int col = ws->col;
    int head = ws->head;
    int tail = ws->tail;
    int updates = ws->updates;

    /* Compute yₖᵀyₖ and sₖᵀyₖ */
    double rr = ddot(n, y, 1, y, 1);  /* yₖᵀyₖ = ‖yₖ‖² */
    double dr = ddot(n, s, 1, y, 1);  /* sₖᵀyₖ */
    
    /* Skip update when curvature condition sₖᵀyₖ ≤ ε‖yₖ‖² is not satisfied.
     * This ensures the BFGS approximation remains positive definite.
     * Matches Go: if dr <= spec.epsilon*y2 */
    if (dr <= EPS * rr) {
        ws->updated = 0;
        return;
    }
    
    ws->updated = 1;
    ws->updates++;
    updates = ws->updates;
    
    /* Update pointers for matrices S and Y
     * This matches Go:
     *   if ctx.updates <= m {
     *       ctx.col = ctx.updates
     *       ctx.tail = (ctx.head + ctx.updates - 1) % m
     *   } else {
     *       ctx.tail = (ctx.tail + 1) % m
     *       ctx.head = (ctx.head + 1) % m
     *   }
     */
    if (updates <= m) {
        col = updates;
        tail = (head + updates - 1) % m;
    } else {
        tail = (tail + 1) % m;
        head = (head + 1) % m;
    }
    
    ws->col = col;
    ws->head = head;
    ws->tail = tail;
    
    /* Update matrices Sₖ and Yₖ
     * Store sₖ in column 'tail' of Sₖ matrix
     * Store yₖ in column 'tail' of Yₖ matrix
     * Using strided copy: element (i, tail) = array[i * m + tail]
     * Matches Go:
     *   dcopy(n, d, 1, ws[ctx.tail:], m)
     *   dcopy(n, r, 1, wy[ctx.tail:], m)
     */
    dcopy(n, s, 1, ws_arr + tail, m);
    dcopy(n, y, 1, wy + tail, m);
    
    /* Update θ = yₖᵀyₖ / sₖᵀyₖ
     * θ is the scaling factor for the initial Hessian approximation H₀ = θI
     * Matches Go: ctx.theta = rr / dr
     */
    ws->theta = rr / dr;

    /* ========================================================================
     * Update the middle matrix in Bₖ
     * 
     * The L-BFGS approximation uses:
     *   Bₖ = θI - Wₖ Mₖ Wₖᵀ
     * where Wₖ = [Y, θS] and Mₖ involves SᵀS and SᵀY.
     *
     * Update the upper triangle of SᵀS and the lower triangle of SᵀY:
     *   - SᵀS[i,j] = sᵢᵀsⱼ (upper triangular)
     *   - SᵀY[i,j] = sᵢᵀyⱼ (lower triangular, with D = diag{sᵢᵀyᵢ} on diagonal)
     * ======================================================================== */
    
    /* Move old information when buffer is full (circular buffer management)
     * When updates > m, we shift the matrices to make room for new data.
     * Matches Go:
     *   if ctx.updates > m {
     *       for j := 0; j < col-1; j++ {
     *           dcopy(col-(j+1), ss[(j+1)*m+(j+1):], 1, ss[j*m+j:], 1) // SᵀS upper triangle
     *           dcopy(j+1, sy[(j+1)*m+1:], 1, sy[j*m:], 1)             // SᵀY lower triangle
     *       }
     *   }
     */
    if (updates > m) {
        /* Shift SᵀS upper triangle: move row j+1 to row j */
        for (j = 0; j < col - 1; j++) {
            /* SᵀS upper triangle: copy from row j+1 to row j
             * Elements SᵀS[(j+1)*m + (j+1)...(col-1)] → SᵀS[j*m + j...(col-2)]
             * Number of elements: col - (j+1) = col - j - 1
             */
            dcopy(col - (j + 1), ss + (j + 1) * m + (j + 1), 1, ss + j * m + j, 1);
            
            /* SᵀY lower triangle: copy from row j+1 to row j
             * Elements SᵀY[(j+1)*m + 0..j] → SᵀY[j*m + 0..(j-1)]
             * Number of elements: j + 1
             */
            dcopy(j + 1, sy + (j + 1) * m + 1, 1, sy + j * m, 1);
        }
    }
    
    /* Add new information: compute inner products for the new correction pair
     * Matches Go:
     *   ptr := ctx.head
     *   for j := 0; j < col-1; j++ {
     *       sy[(col-1)*m+j] = ddot(n, d, 1, wy[ptr:], m) // Last row of SᵀY
     *       ss[j*m+(col-1)] = ddot(n, ws[ptr:], m, d, 1) // Last column of SᵀS
     *       ptr = (ptr + 1) % m
     *   }
     */
    ptr = head;
    for (j = 0; j < col - 1; j++) {
        /* Last row of SᵀY: (SᵀY)[(col-1), j] = sₖᵀyⱼ */
        sy[(col - 1) * m + j] = ddot(n, s, 1, wy + ptr, m);
        
        /* Last column of SᵀS: (SᵀS)[j, (col-1)] = sⱼᵀsₖ */
        ss[j * m + (col - 1)] = ddot(n, ws_arr + ptr, m, s, 1);
        
        ptr = (ptr + 1) % m;
    }
    
    /* Update diagonal elements
     * D = diag{sᵢᵀyᵢ} is stored on the diagonal of SᵀY
     * Matches Go:
     *   sy[(col-1)*m+(col-1)] = dr        // sₖᵀyₖ
     *   ss[(col-1)*m+(col-1)] = ctx.dSqrt // sₖᵀsₖ
     */
    sy[(col - 1) * m + (col - 1)] = dr;                       /* sₖᵀyₖ (diagonal of D) */
    ss[(col - 1) * m + (col - 1)] = ddot(n, s, 1, s, 1);      /* sₖᵀsₖ = ‖sₖ‖² */
}


/**
 * Subroutine formT (formt)
 *
 * This subroutine computes the matrix T = θSᵀS + LD⁻¹Lᵀ and performs
 * Cholesky factorization T = JJᵀ with Jᵀ stored in the upper triangle of wt.
 *
 * The matrix T appears in the compact representation of the L-BFGS
 * inverse Hessian approximation. It is used in the BMV (B matrix-vector)
 * multiplication during the Cauchy point computation.
 *
 * Components:
 *   - θ: scaling factor = yₖᵀyₖ / sₖᵀyₖ
 *   - SᵀS: inner products of step vectors
 *   - D = diag{sᵢᵀyᵢ}: diagonal matrix from SᵀY
 *   - L = {sᵢᵀyⱼ}ᵢ>ⱼ: strictly lower triangular part of SᵀY
 *
 * @param m     Maximum number of correction pairs
 * @param ws    Workspace containing SᵀS, SᵀY, and output wt
 * @return      0 on success, -1 if T is not positive definite
 */
int form_t(int m, LbfgsbWorkspace* ws) {
    int i, j, k, kk;
    double ldl;
    
    int col = ws->col;
    double theta = ws->theta;
    
    double* wt = ws->wt;    /* m × m, Cholesky factor output (Jᵀ stored in upper triangle) */
    double* ss = ws->ss;    /* m × m, SᵀS matrix */
    double* sy = ws->sy;    /* m × m, SᵀY matrix (contains D on diagonal, L in lower triangle) */
    
    if (col == 0) {
        return 0;
    }
    
    /* ========================================================================
     * Form the upper half of T = θSᵀS + LD⁻¹Lᵀ
     * Store T in the upper triangle of the array wt.
     *
     * The (i,j) element of T is:
     *   T[i,j] = θ(SᵀS)[i,j] + Σₖ₌₀^{min(i,j)-1} L[i,k]L[j,k]/D[k,k]
     *
     * where L[i,k] = (SᵀY)[i,k] for i > k, and D[k,k] = (SᵀY)[k,k].
     *
     * Matches Go:
     *   for j := 0; j < col; j++ {
     *       wt[j] = theta * ss[j]
     *   }
     * ======================================================================== */
    
    /* First row: T[0,j] = θ × (SᵀS)[0,j] for j = 0,...,col-1
     * Note: For the first row, LD⁻¹Lᵀ contribution is zero since min(0,j) = 0 */
    for (j = 0; j < col; j++) {
        wt[j] = theta * ss[j];
    }
    
    /* Remaining rows: T[i,j] = θ(SᵀS)[i,j] + (LD⁻¹Lᵀ)ᵢⱼ for i = 1,...,col-1, j ≥ i
     * 
     * The (LD⁻¹Lᵀ)ᵢⱼ term is computed as:
     *   Σₖ₌₀^{min(i,j)-1} L[i,k] × L[j,k] / D[k,k]
     *
     * Matches Go:
     *   for i := 1; i < col; i++ {
     *       for j := i; j < col; j++ {
     *           ldl, kk := zero, min(i, j)
     *           for k := 0; k < kk; k++ {
     *               ldl += sy[i*m+k] * sy[j*m+k] / sy[k*m+k]
     *           }
     *           wt[i*m+j] = ldl + theta*ss[i*m+j]
     *       }
     *   }
     */
    for (i = 1; i < col; i++) {
        for (j = i; j < col; j++) {
            /* Compute (LD⁻¹Lᵀ)ᵢⱼ */
            ldl = ZERO;
            kk = (i < j) ? i : j;  /* min(i, j) */
            for (k = 0; k < kk; k++) {
                /* L[i,k] = (SᵀY)[i,k] (lower triangular part)
                 * L[j,k] = (SᵀY)[j,k]
                 * D[k,k] = (SᵀY)[k,k] (diagonal) */
                ldl += sy[i * m + k] * sy[j * m + k] / sy[k * m + k];
            }
            /* T[i,j] = (LD⁻¹Lᵀ)ᵢⱼ + θ(SᵀS)[i,j] */
            wt[i * m + j] = ldl + theta * ss[i * m + j];
        }
    }
    
    /* ========================================================================
     * Cholesky factorize T = JJᵀ with Jᵀ stored in the upper triangle of wt.
     * This factorization is used in BMV computations.
     *
     * Matches Go:
     *   if dpofa(wt, m, col) != 0 {
     *       info = errNotPosDefT
     *   }
     * ======================================================================== */
    if (dpofa(wt, m, col) != 0) {
        return -1;  /* Not positive definite */
    }
    
    return 0;
}


/**
 * Subroutine formK (formk)
 *
 * This subroutine forms the LELᵀ factorization of the indefinite matrix K:
 *
 *   K = [-D - YᵀZZᵀY/θ    Laᵀ - Rzᵀ]   where  E = [-I  0]
 *       [La - Rz          θSᵀAAᵀS  ]              [ 0  I]
 *
 * The matrix K can be shown to be equal to:
 *   - the matrix M⁻¹N occurring in section 5.1 of [1]
 *   - the matrix M̃⁻¹M̃ in section 5.3
 *
 * Notation:
 *   - Z: indices of free variables (not at bounds)
 *   - A: indices of active variables (at bounds)
 *   - D = diag{sᵢᵀyᵢ}: diagonal matrix
 *   - La: strictly lower triangular part of SᵀAAᵀY
 *   - Rz: upper triangular part of SᵀZZᵀY
 *
 * The workspace arrays:
 *   - wn (2m × 2m): On exit, stores the LELᵀ factorization of the 2×col × 2×col
 *                   indefinite matrix in the upper triangle
 *   - wn1 (2m × 2m): Stores inner products [YᵀZZᵀY, Laᵀ+Rzᵀ; La+Rz, SᵀAAᵀS]
 *                    for efficient incremental updates
 *
 * Reference: [1] Byrd, Lu, Nocedal, Zhu, "A Limited Memory Algorithm for Bound
 *                Constrained Optimization", SIAM J. Scientific Computing, 1995.
 *
 * @param n     Problem dimension
 * @param m     Maximum number of correction pairs
 * @param ws    Workspace containing matrices and index arrays
 * @return      0 on success, -1 if block (1,1) not positive definite,
 *              -2 if block (2,2) not positive definite
 */
int form_k(int n, int m, LbfgsbWorkspace* ws) {
    int i, k;
    int iy, is, jy, js;
    int is1, js1;
    int iptr, jptr;
    double temp1, temp2, temp3, temp4;
    
    int col = ws->col;
    int head = ws->head;
    int m2 = 2 * m;
    int col2 = 2 * col;
    
    /* 2m × 2m matrices */
    double* wn = ws->wn;    /* K matrix and its LELᵀ factorization */
    double* wn1 = ws->snd;  /* Inner products storage for incremental updates */
    
    double* ws_arr = ws->ws;  /* Sₖ matrix */
    double* wy = ws->wy;      /* Yₖ matrix */
    double* sy = ws->sy;      /* SᵀY matrix */
    
    /* index[0:free] are the indices of free variables (Z)
     * index[free:n] are the indices of bound variables (A) */
    int* inx = ws->index;
    
    /* index[n:n+enter] are variables entering the free set (Z → A)
     * index[n+leave:2n] are variables leaving the free set (A → Z) */
    int* inx2 = ws->index + n;
    
    int free = ws->free;
    int enter = ws->enter;
    int leave = ws->leave;
    int updated = ws->updated;
    int updates = ws->updates;
    double theta = ws->theta;
    
    if (col == 0) {
        return 0;
    }
    
    /* ========================================================================
     * Form the lower triangular part of WN1:
     *    WN1 = [YᵀZZᵀY   Laᵀ + Rzᵀ]
     *          [La + Rz   SᵀAAᵀS  ]
     *
     * where:
     *   - YᵀZZᵀY: inner products of Y vectors over free variables
     *   - SᵀAAᵀS: inner products of S vectors over active variables
     *   - La: strictly lower triangular part of SᵀAAᵀY (active variables)
     *   - Rz: upper triangular part of SᵀZZᵀY (free variables)
     * ======================================================================== */

    if (updated) {
        if (updates > m) {
            /* Shift old parts of WN1 (circular buffer management)
             * Matches Go:
             *   for jy := 0; jy < m-1; jy++ {
             *       js := m + jy
             *       y0, y1 := jy*m2, (jy+1)*m2+1
             *       dcopy(jy+1, wn1[y1:], 1, wn1[y0:], 1) // YᵀZZᵀY
             *       s0, s1 := js*m2+m, (js+1)*m2+1+m
             *       dcopy(jy+1, wn1[s1:], 1, wn1[s0:], 1) // SᵀAAᵀS
             *       r0, r1 := js*m2, (js+1)*m2+1
             *       dcopy(m-1, wn1[r1:], 1, wn1[r0:], 1) // La + Rz
             *   }
             */
            for (jy = 0; jy < m - 1; jy++) {
                js = m + jy;
                /* YᵀZZᵀY: shift rows */
                dcopy(jy + 1, wn1 + (jy + 1) * m2 + 1, 1, wn1 + jy * m2, 1);
                /* SᵀAAᵀS: shift rows */
                dcopy(jy + 1, wn1 + (js + 1) * m2 + 1 + m, 1, wn1 + js * m2 + m, 1);
                /* La + Rz: shift rows */
                dcopy(m - 1, wn1 + (js + 1) * m2 + 1, 1, wn1 + js * m2, 1);
            }
        }
        
        int pBeg = 0, pEnd = free;      /* free variables indices (Z) */
        int dBeg = free, dEnd = n;      /* active bounds indices (A) */
        
        /* Add new rows to blocks (1,1), (2,1), and (2,2)
         * Compute inner products for the newest correction pair.
         * Matches Go:
         *   iptr := (head + col - 1) % m
         *   jptr := head
         *   iy := wn1[(col-1)*m2:]   // last row of YᵀZZᵀY
         *   is := wn1[(m+col-1)*m2:] // last row of SᵀAAᵀS and La + Rz
         */
        iptr = (head + col - 1) % m;
        jptr = head;
        
        /* Last row of YᵀZZᵀY */
        double* iy_row = wn1 + (col - 1) * m2;
        /* Last row of SᵀAAᵀS and La + Rz */
        double* is_row = wn1 + (m + col - 1) * m2;
        
        for (jy = 0; jy < col; jy++) {
            js = m + jy;
            
            temp1 = ZERO;
            temp2 = ZERO;
            temp3 = ZERO;
            
            /* Sum over free variables (Z): YᵀZZᵀY = YᵀY restricted to Z */
            for (k = pBeg; k < pEnd; k++) {
                int k1 = inx[k];
                temp1 += wy[k1 * m + iptr] * wy[k1 * m + jptr];  /* yᵢᵀyⱼ over Z */
            }
            
            /* Sum over active bound variables (A) */
            for (k = dBeg; k < dEnd; k++) {
                int k1 = inx[k];
                temp2 += ws_arr[k1 * m + iptr] * ws_arr[k1 * m + jptr];  /* SᵀAAᵀS = sᵢᵀsⱼ over A */
                temp3 += ws_arr[k1 * m + iptr] * wy[k1 * m + jptr];      /* SᵀAAᵀY = sᵢᵀyⱼ over A (La) */
            }
            
            iy_row[jy] = temp1;      /* YᵀZZᵀY */
            is_row[js] = temp2;      /* SᵀAAᵀS */
            is_row[jy] = temp3;      /* La */
            
            jptr = (jptr + 1) % m;
        }

        /* Add new column to block (2,1) - Rz part
         * Rz is the upper triangular part of SᵀZZᵀY (free variables).
         * Matches Go:
         *   jptr = (head + col - 1) % m
         *   iptr = head
         *   jy := wn1[(m*m2)+col-1:] // last column of La + Rz
         */
        jptr = (head + col - 1) % m;
        iptr = head;
        
        /* Last column of La + Rz */
        double* jy_col = wn1 + m * m2 + (col - 1);
        
        for (i = 0; i < col; i++) {
            temp3 = ZERO;
            /* Sum over free variables (Z) for Rz: SᵀZZᵀY = sᵢᵀyⱼ over Z */
            for (k = pBeg; k < pEnd; k++) {
                int k1 = inx[k];
                temp3 += ws_arr[k1 * m + iptr] * wy[k1 * m + jptr];
            }
            jy_col[i * m2] = temp3;  /* Rz */
            iptr = (iptr + 1) % m;
        }
    }
    
    /* ========================================================================
     * Modify the old parts in blocks (1,1) and (2,2) due to changes
     * in the free/active variable sets.
     *
     * When variables move between free set Z and active set A:
     *   - Variables entering free set (Z → A): add to YᵀZZᵀY, subtract from SᵀAAᵀS
     *   - Variables leaving free set (A → Z): subtract from YᵀZZᵀY, add to SᵀAAᵀS
     * ======================================================================== */
    
    int nUpdate = col;
    if (updated) {
        nUpdate--;  /* Ignore last row and col */
    }
    
    iptr = head;
    for (iy = 0; iy < nUpdate; iy++) {
        is = m + iy;
        
        jptr = head;
        for (jy = 0; jy <= iy; jy++) {
            js = m + jy;
            
            temp1 = ZERO;
            temp2 = ZERO;
            temp3 = ZERO;
            temp4 = ZERO;
            
            /* Variables entering free set (from Z to A): update inner products */
            for (k = 0; k < enter; k++) {
                int k1 = inx2[k];
                temp1 += wy[k1 * m + iptr] * wy[k1 * m + jptr];      /* YᵀZZᵀY += yᵢᵀyⱼ */
                temp2 += ws_arr[k1 * m + iptr] * ws_arr[k1 * m + jptr];  /* SᵀAAᵀS -= sᵢᵀsⱼ */
            }
            
            /* Variables leaving free set (from A to Z): update inner products */
            for (k = leave; k < n; k++) {
                int k1 = inx2[k];
                temp3 += wy[k1 * m + iptr] * wy[k1 * m + jptr];      /* YᵀZZᵀY -= yᵢᵀyⱼ */
                temp4 += ws_arr[k1 * m + iptr] * ws_arr[k1 * m + jptr];  /* SᵀAAᵀS += sᵢᵀsⱼ */
            }
            
            wn1[iy * m2 + jy] += temp1 - temp3;  /* YᵀZZᵀY */
            wn1[is * m2 + js] += temp4 - temp2;  /* SᵀAAᵀS */
            
            jptr = (jptr + 1) % m;
        }
        iptr = (iptr + 1) % m;
    }

    /* Modify the old parts in block (2,1): La and Rz
     * La (strictly lower triangular): SᵀAAᵀY over active variables
     * Rz (upper triangular): SᵀZZᵀY over free variables */
    iptr = head;
    for (is = m; is < m + nUpdate; is++) {
        jptr = head;
        for (jy = 0; jy < nUpdate; jy++) {
            
            temp1 = ZERO;
            temp3 = ZERO;
            
            /* Variables entering free set (from Z to A): SᵀAAᵀY = sᵢᵀyⱼ */
            for (k = 0; k < enter; k++) {
                int k1 = inx2[k];
                temp1 += ws_arr[k1 * m + iptr] * wy[k1 * m + jptr];
            }
            
            /* Variables leaving free set (from A to Z): SᵀZZᵀY = sᵢᵀyⱼ */
            for (k = leave; k < n; k++) {
                int k1 = inx2[k];
                temp3 += ws_arr[k1 * m + iptr] * wy[k1 * m + jptr];
            }
            
            if (is - m <= jy) {
                /* Rz (upper triangular part) */
                wn1[is * m2 + jy] += temp1 - temp3;
            } else {
                /* La (strictly lower triangular, diagonal is zero) */
                wn1[is * m2 + jy] -= temp1 - temp3;
            }
            
            jptr = (jptr + 1) % m;
        }
        iptr = (iptr + 1) % m;
    }
    
    /* ========================================================================
     * Form the upper triangle of 2×col × 2×col indefinite matrix:
     *        [-D - YᵀZZᵀY/θ    -Laᵀ + Rzᵀ]
     *        [-La + Rz          θSᵀAAᵀS  ]
     * where
     *        D = diag{sᵢᵀyᵢ}ᵢ₌₁,...,ₙ
     *
     * This matrix appears in the reduced system for subspace minimization.
     * ======================================================================== */
    
    for (iy = 0; iy < col; iy++) {
        is = col + iy;
        is1 = m + iy;
        
        /* From WN1 lower triangle to WN upper triangle */
        for (jy = 0; jy <= iy; jy++) {
            js = col + jy;
            js1 = m + jy;
            wn[jy * m2 + iy] = wn1[iy * m2 + jy] / theta;   /* block (1,1) = (YᵀZZᵀY)ᵀ/θ */
            wn[js * m2 + is] = wn1[is1 * m2 + js1] * theta; /* block (2,2) = θ(SᵀAAᵀS)ᵀ */
        }
        
        /* From WN1 block (2,1) to WN block (1,2) */
        for (jy = 0; jy < iy; jy++) {
            wn[jy * m2 + is] = -wn1[is1 * m2 + jy];  /* block (2,1) = (-La)ᵀ */
        }
        for (jy = iy; jy < col; jy++) {
            wn[jy * m2 + is] = wn1[is1 * m2 + jy];   /* block (2,1) = +Rz */
        }
        
        /* Add D to diagonal of block (1,1): D = diag{sᵢᵀyᵢ} */
        wn[iy * m2 + iy] += sy[iy * m + iy];  /* += D[iy,iy] */
    }

    /* ========================================================================
     * Form the upper triangle of WN = [  LLᵀ           L⁻¹(-Laᵀ+Rzᵀ)]
     *                                 [(-La+Rz)L⁻ᵀ    θSᵀAAᵀS      ]
     *
     * This is the LELᵀ factorization where E = diag(-I, I).
     * ======================================================================== */
    
    /* First Cholesky factor (1,1) block of WN to get LLᵀ
     * with Lᵀ stored in the upper triangle of WN */
    if (dpofa(wn, m2, col) != 0) {
        return -1;  /* Not positive definite (block 1,1) */
    }
    
    /* Then solve Lx = (-Laᵀ+Rzᵀ) to form L⁻¹(-Laᵀ+Rzᵀ) in the (1,2) block of wn.
     * Since dtrsl expects contiguous b vector, we need to copy column by column.
     * Matches Go:
     *   for js := col; js < col2; js++ {
     *       dtrsl(wn, m2, col, wn[js:], m2, solveUpperT)
     *   }
     */
    double* temp_col = ws->wa;  /* Use workspace for temporary column storage */
    for (js = col; js < col2; js++) {
        /* Copy column js from wn to temp_col */
        for (i = 0; i < col; i++) {
            temp_col[i] = wn[i * m2 + js];
        }
        /* Solve Lᵀx = b where Lᵀ is upper triangular (job = SOLVE_UPPER_T) */
        dtrsl(wn, m2, col, temp_col, SOLVE_UPPER_T);
        /* Copy result back to wn */
        for (i = 0; i < col; i++) {
            wn[i * m2 + js] = temp_col[i];
        }
    }
    
    /* Form θSᵀAAᵀS + [L⁻¹(-Laᵀ+Rzᵀ)]ᵀ[L⁻¹(-Laᵀ+Rzᵀ)] in the upper triangle 
     * of (2,2) block of wn.
     * Matches Go:
     *   for is := col; is < col2; is++ {
     *       for js := is; js < col2; js++ {
     *           wn[is*m2+js] += ddot(col, wn[is:], m2, wn[js:], m2)
     *       }
     *   }
     */
    for (is = col; is < col2; is++) {
        for (js = is; js < col2; js++) {
            wn[is * m2 + js] += ddot(col, wn + is, m2, wn + js, m2);
        }
    }
    
    /* Cholesky factorization of (2,2) block of wn */
    if (dpofa(wn + col * m2 + col, m2, col) != 0) {
        return -2;  /* Not positive definite (block 2,2) */
    }
    
    return 0;
}


/**
 * Update BFGS matrices with new s and y vectors (complete update)
 *
 * This function integrates update_correction and form_t to perform a complete
 * BFGS update. It is a convenience wrapper that:
 *   1. Checks the curvature condition sₖᵀyₖ > ε‖yₖ‖²
 *   2. Updates correction matrices Sₖ and Yₖ
 *   3. Computes the Cholesky factor of T = θSᵀS + LD⁻¹Lᵀ
 *
 * @param n     Problem dimension
 * @param m     Maximum number of correction pairs
 * @param s     New step vector sₖ = xₖ₊₁ - xₖ
 * @param y     New gradient difference yₖ = gₖ₊₁ - gₖ
 * @param ws    Workspace containing correction matrices
 * @return      0 on success, -1 if curvature condition not satisfied,
 *              -2 if T matrix factorization failed
 */
int update_bfgs_full(int n, int m, const double* s, const double* y,
                     LbfgsbWorkspace* ws) {
    /* Compute yₖᵀyₖ and sₖᵀyₖ for curvature condition check */
    double yy = ddot(n, y, 1, y, 1);  /* yₖᵀyₖ = ‖yₖ‖² */
    double ys = ddot(n, s, 1, y, 1);  /* sₖᵀyₖ */
    
    /* Check curvature condition: sₖᵀyₖ > ε‖yₖ‖²
     * If not satisfied, skip the BFGS update to maintain positive definiteness */
    if (ys <= EPS * yy) {
        ws->updated = 0;
        return -1;  /* Curvature condition not satisfied */
    }
    
    /* Call update_correction to update Sₖ, Yₖ matrices and compute SᵀY, SᵀS, θ
     * Note: update_correction also checks curvature condition internally,
     * but we check it here first for early exit */
    update_correction(n, m, s, y, ws);
    
    /* Check if update_correction actually performed the update
     * (it may skip due to its own curvature check) */
    if (!ws->updated) {
        return -1;  /* Update was skipped */
    }
    
    /* Call form_t to compute the Cholesky factor of T = θSᵀS + LD⁻¹Lᵀ
     * This is needed for BMV computations in subsequent iterations */
    int info = form_t(m, ws);
    if (info != 0) {
        /* form_t failed - T matrix is not positive definite
         * This can happen in degenerate cases. We should handle this
         * by potentially resetting the BFGS approximation */
        ws->updated = 0;
        return -2;  /* form_t failed */
    }
    
    return 0;  /* Success */
}
