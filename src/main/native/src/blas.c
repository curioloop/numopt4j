/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * Basic Linear Algebra Subprograms (BLAS) Level 1 implementation.
 * Standard C99 implementation for cross-platform compatibility.
 *
 * This file implements standard BLAS Level 1 operations for vector computations.
 * Reference: Go implementations in slsqp/blas.go and lbfgsb/linpack.go
 *
 * Mathematical Operations:
 *   dcopy: 𝐲 ← 𝐱           (vector copy)
 *   daxpy: 𝐲 ← α𝐱 + 𝐲      (scalar-vector multiply-add)
 *   ddot:  s ← 𝐱ᵀ𝐲         (dot product)
 *   dnrm2: s ← ‖𝐱‖₂        (Euclidean norm)
 *   dscal: 𝐱 ← α𝐱          (scalar-vector multiply)
 */

#include "optimizer.h"
#include <math.h>
#include <string.h>

/**
 * dcopy copies a vector, x, to a vector, y.
 *
 * Mathematical operation:
 *   𝐲 ← 𝐱
 *
 * For i = 0, 1, ..., n-1:
 *   y[i*incy] = x[i*incx]
 *
 * @param n    Number of elements to copy
 * @param x    Source vector (read-only)
 * @param incx Storage spacing between elements of x
 * @param y    Destination vector (modified)
 * @param incy Storage spacing between elements of y
 *
 * Note: When incx=1 and incy=1, uses memcpy for efficiency.
 *       Handles negative increments by adjusting starting indices.
 */
void dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        memcpy(y, x, (size_t)n * sizeof(double));
    } else {
        int ix = 0, iy = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        for (int i = 0; i < n; i++) {
            y[iy] = x[ix];
            ix += incx;
            iy += incy;
        }
    }
}

/**
 * daxpy performs constant times a vector plus a vector operation.
 *
 * Mathematical operation:
 *   𝐲 ← α𝐱 + 𝐲
 *
 * For i = 0, 1, ..., n-1:
 *   y[i*incy] = a * x[i*incx] + y[i*incy]
 *
 * @param n    Number of elements in vectors
 * @param a    Scalar multiplier α
 * @param x    Source vector (read-only)
 * @param incx Storage spacing between elements of x
 * @param y    Destination vector (modified in place)
 * @param incy Storage spacing between elements of y
 *
 * Note: Returns immediately if n ≤ 0 or a = 0.
 *       Handles negative increments by adjusting starting indices.
 */
void daxpy(int n, double a, const double* x, int incx, double* y, int incy) {
    if (n <= 0 || a == 0.0) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            y[i] += a * x[i];
        }
    } else {
        int ix = 0, iy = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        for (int i = 0; i < n; i++) {
            y[iy] += a * x[ix];
            ix += incx;
            iy += incy;
        }
    }
}

/**
 * ddot computes the dot product of two vectors.
 *
 * Mathematical operation:
 *   s ← 𝐱ᵀ𝐲 = Σᵢ xᵢyᵢ
 *
 * Computes:
 *   result = Σ(i=0 to n-1) x[i*incx] * y[i*incy]
 *
 * @param n    Number of elements in vectors
 * @param x    First vector (read-only)
 * @param incx Storage spacing between elements of x
 * @param y    Second vector (read-only)
 * @param incy Storage spacing between elements of y
 * @return     Dot product 𝐱ᵀ𝐲 (returns 0.0 if n ≤ 0)
 *
 * Note: Handles negative increments by adjusting starting indices.
 */
double ddot(int n, const double* x, int incx, const double* y, int incy) {
    if (n <= 0) return 0.0;
    
    double result = 0.0;
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            result += x[i] * y[i];
        }
    } else {
        int ix = 0, iy = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        for (int i = 0; i < n; i++) {
            result += x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return result;
}

/**
 * dnrm2 computes the Euclidean norm of a vector x.
 *
 * Mathematical operation:
 *   s ← ‖𝐱‖₂ = √(Σᵢ xᵢ²)
 *
 * Uses a numerically stable algorithm to avoid overflow/underflow:
 *   scale = max(|xᵢ|)
 *   ssq = Σᵢ (xᵢ/scale)²
 *   result = scale * √ssq
 *
 * @param n    Number of elements in vector
 * @param x    Input vector (read-only)
 * @param incx Storage spacing between elements of x
 * @return     Euclidean norm ‖𝐱‖₂ (returns 0.0 if n ≤ 0)
 *
 * Note: For n=1, returns |x[0]| directly.
 *       Handles negative increments by adjusting starting index.
 */
double dnrm2(int n, const double* x, int incx) {
    if (n <= 0) return 0.0;
    if (n == 1) return fabs(x[0]);
    
    double scale = 0.0;
    double ssq = 1.0;
    
    int ix = 0;
    if (incx < 0) ix = (-n + 1) * incx;
    
    for (int i = 0; i < n; i++) {
        double absxi = fabs(x[ix]);
        if (absxi > 0.0) {
            if (scale < absxi) {
                double temp = scale / absxi;
                ssq = 1.0 + ssq * temp * temp;
                scale = absxi;
            } else {
                double temp = absxi / scale;
                ssq += temp * temp;
            }
        }
        ix += incx;
    }
    return scale * sqrt(ssq);
}

/**
 * dscal scales a vector by a constant.
 *
 * Mathematical operation:
 *   𝐱 ← α𝐱
 *
 * For i = 0, 1, ..., n-1:
 *   x[i*incx] = a * x[i*incx]
 *
 * @param n    Number of elements in vector
 * @param a    Scalar multiplier α
 * @param x    Vector (modified in place)
 * @param incx Storage spacing between elements of x
 *
 * Note: Returns immediately if n ≤ 0.
 *       Handles negative increments by adjusting starting index.
 */
void dscal(int n, double a, double* x, int incx) {
    if (n <= 0) return;
    
    if (incx == 1) {
        for (int i = 0; i < n; i++) {
            x[i] *= a;
        }
    } else {
        int ix = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        for (int i = 0; i < n; i++) {
            x[ix] *= a;
            ix += incx;
        }
    }
}

/**
 * damax computes the infinity norm of a vector.
 *
 * Mathematical operation:
 *   s ← ‖𝐱‖∞ = maxᵢ |xᵢ|
 *
 * @param n    Number of elements in vector
 * @param x    Input vector (read-only)
 * @param incx Storage spacing between elements of x
 * @return     Infinity norm ‖𝐱‖∞ (returns 0.0 if n ≤ 0)
 *
 * Note: Handles negative increments by adjusting starting index.
 */
double damax(int n, const double* x, int incx) {
    if (n <= 0) return 0.0;
    
    double result = 0.0;
    int ix = 0;
    if (incx < 0) ix = (-n + 1) * incx;
    
    for (int i = 0; i < n; i++) {
        double absxi = fabs(x[ix]);
        if (absxi > result) {
            result = absxi;
        }
        ix += incx;
    }
    return result;
}

/**
 * idamax finds the index of the element with maximum absolute value.
 *
 * Mathematical operation:
 *   k ← argmaxᵢ |xᵢ|
 *
 * @param n    Number of elements in vector
 * @param x    Input vector (read-only)
 * @param incx Storage spacing between elements of x
 * @return     Index k of element with maximum |xₖ| (0-based, returns -1 if n ≤ 0)
 *
 * Note: Handles negative increments by adjusting starting index.
 */
int idamax(int n, const double* x, int incx) {
    if (n <= 0) return -1;
    if (n == 1) return 0;
    
    int result = 0;
    double dmax = fabs(x[0]);
    
    int ix = incx;
    if (incx < 0) ix = (-n + 1) * incx + incx;
    
    for (int i = 1; i < n; i++) {
        double absxi = fabs(x[ix]);
        if (absxi > dmax) {
            result = i;
            dmax = absxi;
        }
        ix += incx;
    }
    return result;
}

/**
 * dswap interchanges two vectors.
 *
 * Mathematical operation:
 *   𝐱 ↔ 𝐲
 *
 * For i = 0, 1, ..., n-1:
 *   swap(x[i*incx], y[i*incy])
 *
 * @param n    Number of elements in vectors
 * @param x    First vector (modified)
 * @param incx Storage spacing between elements of x
 * @param y    Second vector (modified)
 * @param incy Storage spacing between elements of y
 *
 * Note: Handles negative increments by adjusting starting indices.
 */
void dswap(int n, double* x, int incx, double* y, int incy) {
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            double temp = x[i];
            x[i] = y[i];
            y[i] = temp;
        }
    } else {
        int ix = 0, iy = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        for (int i = 0; i < n; i++) {
            double temp = x[ix];
            x[ix] = y[iy];
            y[iy] = temp;
            ix += incx;
            iy += incy;
        }
    }
}

/**
 * drot applies a plane rotation.
 *
 * Mathematical operation:
 *   [xᵢ]   [c  s] [xᵢ]
 *   [yᵢ] ← [-s c] [yᵢ]
 *
 * For i = 0, 1, ..., n-1:
 *   temp = c*x[i*incx] + s*y[i*incy]
 *   y[i*incy] = c*y[i*incy] - s*x[i*incx]
 *   x[i*incx] = temp
 *
 * @param n    Number of elements in vectors
 * @param x    First vector (modified)
 * @param incx Storage spacing between elements of x
 * @param y    Second vector (modified)
 * @param incy Storage spacing between elements of y
 * @param c    Cosine of rotation angle (cos θ)
 * @param s    Sine of rotation angle (sin θ)
 *
 * Note: Handles negative increments by adjusting starting indices.
 */
void drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    if (n <= 0) return;
    
    int ix = 0, iy = 0;
    if (incx < 0) ix = (-n + 1) * incx;
    if (incy < 0) iy = (-n + 1) * incy;
    
    for (int i = 0; i < n; i++) {
        double temp = c * x[ix] + s * y[iy];
        y[iy] = c * y[iy] - s * x[ix];
        x[ix] = temp;
        ix += incx;
        iy += incy;
    }
}

/**
 * drotg constructs a Givens plane rotation.
 *
 * Mathematical operation:
 *   Given scalars a and b, compute:
 *   r = ±√(a² + b²)
 *   c = a/r (cosine)
 *   s = b/r (sine)
 *
 *   Such that:
 *   [c  s] [a]   [r]
 *   [-s c] [b] = [0]
 *
 * @param a Input/output: on entry, x-coordinate; on exit, r = ±√(a² + b²)
 * @param b Input/output: on entry, y-coordinate; on exit, z (reconstruction info)
 * @param c Output: cosine of rotation angle
 * @param s Output: sine of rotation angle
 *
 * Note: The sign of r is chosen to match the sign of the larger of |a| or |b|.
 *       The value z stored in b allows reconstruction of c and s.
 */
void drotg(double* a, double* b, double* c, double* s) {
    double r, z;
    double roe = *b;
    double absA = fabs(*a);
    double absB = fabs(*b);
    
    if (absA > absB) {
        roe = *a;
    }
    
    double scale = absA + absB;
    if (scale == 0.0) {
        *c = 1.0;
        *s = 0.0;
        r = 0.0;
        z = 0.0;
    } else {
        double sA = *a / scale;
        double sB = *b / scale;
        r = scale * sqrt(sA * sA + sB * sB);
        if (roe < 0.0) r = -r;
        *c = *a / r;
        *s = *b / r;
        z = 1.0;
        if (absA > absB) {
            z = *s;
        } else if (*c != 0.0) {
            z = 1.0 / *c;
        }
    }
    *a = r;
    *b = z;
}

/**
 * dzero fills a vector with zeros.
 *
 * Mathematical operation:
 *   𝐱 ← 𝟎
 *
 * For i = 0, 1, ..., n-1:
 *   x[i*incx] = 0
 *
 * @param n    Number of elements in vector
 * @param x    Vector (modified)
 * @param incx Storage spacing between elements of x
 *
 * Note: When incx=1, uses memset for efficiency.
 *       Handles negative increments by adjusting starting index.
 *
 * Corresponds to Go function dzero in slsqp/blas.go (simplified interface).
 */
void dzero(int n, double* x, int incx) {
    if (n <= 0) return;
    
    if (incx == 1) {
        memset(x, 0, (size_t)n * sizeof(double));
    } else {
        int ix = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        for (int i = 0; i < n; i++) {
            x[ix] = 0.0;
            ix += incx;
        }
    }
}

/**
 * dset fills a vector with a constant value.
 *
 * Mathematical operation:
 *   𝐱 ← α𝟏
 *
 * For i = 0, 1, ..., n-1:
 *   x[i*incx] = a
 *
 * @param n    Number of elements in vector
 * @param a    Constant value α to fill
 * @param x    Vector (modified)
 * @param incx Storage spacing between elements of x
 *
 * Note: Handles negative increments by adjusting starting index.
 */
void dset(int n, double a, double* x, int incx) {
    if (n <= 0) return;
    
    int ix = 0;
    if (incx < 0) ix = (-n + 1) * incx;
    for (int i = 0; i < n; i++) {
        x[ix] = a;
        ix += incx;
    }
}
