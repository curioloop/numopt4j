/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * 
 * LINPACK routines for linear algebra operations.
 * 
 * This file contains triangular system solvers and matrix factorization routines
 * used by the L-BFGS-B optimization algorithm. The implementations follow the
 * reference Go implementation in lbfgsb/linpack.go.
 * 
 * Functions:
 *   - dpofa: Cholesky factorization A = RᵀR for symmetric positive definite matrices
 *   - dtrsl: Triangular system solver for T*x = b or Tᵀ*x = b
 *   - compositeT: LDLᵀ factorization for rank-1 modified matrices (from slsqp/tool.go)
 * 
 * Reference: LINPACK Users' Guide, Dongarra et al., SIAM, 1979.
 */

#include "optimizer.h"
#include <math.h>

/*
 * Solve options for dtrsl (matching Go linpack.go constants)
 * 
 * The job parameter uses a 2-bit encoding:
 *   - Bit 0: 0 = lower triangular, 1 = upper triangular
 *   - Bit 1: 0 = no transpose, 1 = transpose
 */
#define SOLVE_LOWER_N  0   /* 0b00 - Solve T*x = b, T is lower triangular */
#define SOLVE_UPPER_N  1   /* 0b01 - Solve T*x = b, T is upper triangular */
#define SOLVE_LOWER_T  2   /* 0b10 - Solve Tᵀ*x = b, T is lower triangular */
#define SOLVE_UPPER_T  3   /* 0b11 - Solve Tᵀ*x = b, T is upper triangular */

/* External BLAS functions */
extern double ddot(int n, const double* x, int incx, const double* y, int incy);
extern void daxpy(int n, double a, const double* x, int incx, double* y, int incy);
extern void dscal(int n, double a, double* x, int incx);

/**
 * dpofa - Cholesky factorization of a symmetric positive definite matrix
 * 
 * Factors a double precision symmetric positive definite matrix A = Rᵀ * R.
 * 
 * On entry:
 *   a       double precision(n, lda)
 *           The symmetric matrix to be factored. Only the diagonal and upper
 *           triangle are used.
 *   
 *   lda     integer
 *           The leading dimension of the array a.
 *   
 *   n       integer
 *           The order of the matrix a.
 * 
 * On return:
 *   a       An upper triangular matrix R so that A = Rᵀ * R where trans(R)
 *           is the transpose. The strict lower triangle is unaltered.
 *           If info ≠ 0, the factorization is not complete.
 *   
 *   return  integer
 *           = 0  for normal return.
 *           = k  signals an error condition. The leading minor of order k
 *                is not positive definite.
 * 
 * Reference: LINPACK Users' Guide, Chapter 8.
 */
int dpofa(double* a, int lda, int n) {
    int j, k;
    double s, t;
    
    for (j = 0; j < n; j++) {
        s = 0.0;
        for (k = 0; k < j; k++) {
            t = a[k * lda + j] - ddot(k, &a[k], lda, &a[j], lda);
            t = t / a[k * lda + k];
            a[k * lda + j] = t;
            s += t * t;
        }
        s = a[j * lda + j] - s;
        if (s <= 0.0) {
            return j + 1;  /* Not positive definite */
        }
        a[j * lda + j] = sqrt(s);
    }
    return 0;
}

/**
 * dtrsl - Solve triangular system T*x = b or Tᵀ*x = b
 * 
 * Solves systems of the form T * x = b or Tᵀ * x = b where T is a triangular
 * matrix of order n.
 * 
 * On entry:
 *   t       double precision(n, ldt)
 *           t contains the matrix of the system. The zero elements of the
 *           matrix are not referenced, and the corresponding elements of
 *           the array can be used to store other information.
 *   
 *   ldt     integer
 *           ldt is the leading dimension of the array t.
 *   
 *   n       integer
 *           n is the order of the system.
 *   
 *   b       double precision(n)
 *           b contains the right hand side of the system.
 *   
 *   job     integer
 *           job specifies what kind of system is to be solved.
 *           if job is:
 *               00 (0)  solve T * x = b, T is lower triangular,
 *               01 (1)  solve T * x = b, T is upper triangular,
 *               10 (2)  solve Tᵀ * x = b, T is lower triangular,
 *               11 (3)  solve Tᵀ * x = b, T is upper triangular.
 * 
 * On return:
 *   b       b contains the solution, if info = 0.
 *           Otherwise b is unaltered.
 *   
 *   return  integer
 *           info contains zero if the system is nonsingular.
 *           Otherwise info contains the index of the first zero
 *           diagonal element of t.
 * 
 * Reference: LINPACK Users' Guide, Chapter 8.
 */
int dtrsl(double* t, int ldt, int n, double* b, int job) {
    int j, jj;
    double temp;
    /*
     * Job codes (matching Go linpack.go):
     *   0b00 = 0: solveLowerN - Solve T*x = b, T lower triangular
     *   0b01 = 1: solveUpperN - Solve T*x = b, T upper triangular
     *   0b10 = 2: solveLowerT - Solve Tᵀ*x = b, T lower triangular
     *   0b11 = 3: solveUpperT - Solve Tᵀ*x = b, T upper triangular
     */
    int upper = (job & 1);   /* bit 0: 0=lower, 1=upper */
    int trans = (job & 2);   /* bit 1: 0=no transpose, 2=transpose */
    
    /* Check for zero diagonal elements */
    for (j = 0; j < n; j++) {
        if (t[j * ldt + j] == 0.0) {
            return j + 1;
        }
    }
    
    if (!upper) {
        /* Lower triangular */
        if (!trans) {
            /* Solve T*x = b (solveLowerN) */
            b[0] /= t[0];
            for (j = 1; j < n; j++) {
                temp = -b[j - 1];
                daxpy(n - j, temp, &t[(j - 1) * ldt + j], 1, &b[j], 1);
                b[j] /= t[j * ldt + j];
            }
        } else {
            /* Solve Tᵀ*x = b (solveLowerT) */
            b[n - 1] /= t[(n - 1) * ldt + (n - 1)];
            for (jj = 1; jj < n; jj++) {
                j = n - 1 - jj;
                b[j] -= ddot(jj, &t[j * ldt + j + 1], 1, &b[j + 1], 1);
                b[j] /= t[j * ldt + j];
            }
        }
    } else {
        /* Upper triangular */
        if (!trans) {
            /* Solve T*x = b (solveUpperN) */
            b[n - 1] /= t[(n - 1) * ldt + (n - 1)];
            for (jj = 1; jj < n; jj++) {
                j = n - 1 - jj;
                temp = -b[j + 1];
                daxpy(j + 1, temp, &t[(j + 1) * ldt], 1, b, 1);
                b[j] /= t[j * ldt + j];
            }
        } else {
            /* Solve Tᵀ*x = b (solveUpperT) */
            b[0] /= t[0];
            for (j = 1; j < n; j++) {
                b[j] -= ddot(j, &t[j * ldt], 1, b, 1);
                b[j] /= t[j * ldt + j];
            }
        }
    }
    
    return 0;
}

/**
 * compositeT - Compute LDLᵀ factorization for a rank-1 modified matrix A' = A + σzzᵀ
 * 
 * Given:
 *   - A is n × n positive definite symmetric matrix
 *   - L = [l₁···lₙ] is lower triangle matrix with unit diagonal elements
 *   - D = (d₁···dₙ) is diagonal matrix with positive diagonal elements
 *   - A' is a positive definite matrix with rank-one modification
 *   - σ is scalar and z is a vector
 * 
 * The algorithm computes the LDLᵀ factorization of the modified matrix:
 *   A' = A + σzzᵀ = ∑ l'ᵢd'ᵢl'ᵢᵀ
 * 
 * The update formulas are:
 *   - tᵢ₊₁ = tᵢ + vᵢ²/dᵢ           (for σ > 0)
 *   - αᵢ = tᵢ₊₁ / tᵢ               (scaling factor)
 *   - d'ᵢ = αᵢ * dᵢ                (updated diagonal)
 *   - βᵢ = (vᵢ / dᵢ) / tᵢ          (update coefficient)
 *   - l'ᵢ = lᵢ + βᵢ * z⁽ⁱ⁺¹⁾ᵢ      (updated lower triangle, when α ≤ 4)
 *   - l'ᵢ = (tᵢ/tᵢ₊₁)lᵢ + βᵢz⁽ⁱ⁾ᵢ  (updated lower triangle, when α > 4)
 *   - z⁽ⁱ⁺¹⁾ = z⁽ⁱ⁾ - vᵢlᵢ         (updated z vector)
 * 
 * For σ < 0, an auxiliary vector w is used to handle the negative update:
 *   - w = z - L⁻¹z
 *   - tₙ = ε/σ if tₙ ≥ 0
 * 
 * Note: This function corresponds to compositeT in slsqp/tool.go, not linpack.go.
 * It is placed here for organizational convenience in the C implementation.
 * 
 * Reference: Dieter Kraft, 'A Software Package for Sequential Quadratic Programming', 1988.
 * Chapter 2.32.
 * 
 * @param n      Order of matrix
 * @param a      Lower triangular matrix in packed form (L and D stored together, modified)
 * @param z      Vector for rank-1 update (modified during computation)
 * @param sigma  Scalar multiplier for rank-1 update
 * @param w      Working vector (required if sigma ≤ 0, can be NULL if sigma > 0)
 */
void compositeT(int n, double* a, double* z, double sigma, double* w) {
    int i, j;
    int ij;
    double t, v, u_val, delta, tp, alpha, beta, gamma;
    
    /* if σ = 0 then terminate */
    if (sigma == 0.0) {
        return;
    }
    
    t = 1.0 / sigma;
    ij = 0;
    
    if (n <= 0) {
        return;
    }
    
    /* if σ < 0 construct w = z - L⁻¹z */
    if (sigma <= 0.0) {
        if (w == NULL) {
            return;
        }
        
        /* copy z to w */
        for (i = 0; i < n; i++) {
            w[i] = z[i];
        }
        
        /* solve Lv = z and update tᵢ₊₁ = tᵢ + vᵢ²/dᵢ */
        for (i = 0; i < n; i++) {
            v = w[i];
            t += v * v / a[ij];
            for (j = i + 1; j < n; j++) {
                ij++;
                w[j] -= v * a[ij];
            }
            ij++;
        }
        
        /* if tₙ ≥ 0 then set tₙ = ε/σ */
        if (t >= 0.0) {
            t = EPS / sigma;
        }
        
        /* recompute tᵢ₋₁ = tᵢ - vᵢ²/dᵢ */
        for (j = n - 1; j >= 0; j--) {
            u_val = w[j];
            w[j] = t;
            ij -= n - j;
            t -= u_val * u_val / a[ij];
        }
    }
    
    ij = 0;
    for (i = 0; i < n; i++) {
        v = z[i];
        delta = v / a[ij];
        
        if (sigma < 0.0) {
            tp = w[i];              /* tᵢ₊₁ = wᵢ₊₁ */
        } else {
            tp = t + delta * v;    /* tᵢ₊₁ = tᵢ + vᵢ²/dᵢ */
        }
        
        alpha = tp / t;            /* αᵢ = tᵢ₊₁ / tᵢ */
        a[ij] *= alpha;            /* d'ᵢ = αᵢ * dᵢ */
        
        if (i == n - 1) {
            break;
        }
        
        beta = delta / tp;         /* βᵢ = (vᵢ / dᵢ) / tᵢ */
        
        if (alpha > 4.0) {
            gamma = t / tp;
            for (j = i + 1; j < n; j++) {
                ij++;
                u_val = a[ij];                        /* lᵢ */
                a[ij] = gamma * u_val + beta * z[j]; /* l'ᵢ = (tᵢ / tᵢ₊₁)lᵢ + βᵢz⁽ⁱ⁾ᵢ */
                z[j] -= v * u_val;                    /* z⁽ⁱ⁺¹⁾ = z⁽ⁱ⁾ - vᵢlᵢ */
            }
        } else {
            for (j = i + 1; j < n; j++) {
                ij++;
                z[j] -= v * a[ij];                   /* z⁽ⁱ⁺¹⁾ = z⁽ⁱ⁾ - vᵢlᵢ */
                a[ij] += beta * z[j];                /* l'ᵢ = lᵢ + βᵢz⁽ⁱ⁺¹⁾ᵢ */
            }
        }
        ij++;
        t = tp;
    }
}
