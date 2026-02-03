/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * Householder transformation and Givens rotation utilities.
 * Based on Lawson & Hanson, "Solving Least Squares Problems"
 *
 * Reference:
 * C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
 * Prentice Hall, 1974. (revised 1995 edition)
 */

#include "optimizer.h"
#include <math.h>

/**
 * h1 - Compute Householder transformation vector
 *
 * Given m-vector v, construct m×m Householder vector u and scalar s for
 * transformation Qv ≡ y. The Householder matrix can be computed with:
 *
 *   Q = Iₘ - b⁻¹uuᵀ  where b = s·uₚ
 *
 * lₚ (pivot) is the index of the pivot element, which should satisfy 0 ≤ lₚ < l₁.
 * If l₁ < m, the transformation will be constructed to zero out elements
 * indexed from l₁ through m-1. If l₁ ≥ m, the subroutine does an identity
 * transformation.
 *
 * Algorithm:
 *   1. Find max(v) for scaling to avoid overflow
 *   2. Compute s = -σ(vₚ² + ∑vᵢ²)^(1/2) where σ = -sgn(vₚ)
 *   3. Set uₚ = vₚ - s and yₚ = s
 *
 * On input, u contains the pivot vector.
 * On output, u contains quantities defining the vector u of the Householder
 * transformation. The u[lₚ] element (uₚ) is returned separately.
 *
 * Reference: C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
 *            Prentice Hall, 1974. Chapter 10.
 *
 * @param pivot  Pivot index lₚ (must satisfy 0 ≤ pivot < start)
 * @param start  Start index l₁ for transformation (must satisfy pivot < start ≤ m-1)
 * @param m      Vector length
 * @param u      Vector to transform (modified in place)
 * @param inc    Storage increment between elements (ive)
 * @return       Pivot element uₚ for use in h2, or 0.0 if identity transformation
 */
double h1(int pivot, int start, int m, double* u, int inc) {
    double cl, sm;
    int i;
    
    /* Check bounds: 0 ≤ lₚ < l₁ ≤ m-1 */
    if (pivot < 0 || pivot >= start || start >= m) {
        return 0.0;
    }
    
    /* Find max|v| for numerical stability */
    cl = fabs(u[pivot * inc]);
    for (i = start; i < m; i++) {
        double v = fabs(u[i * inc]);
        if (v > cl) cl = v;
    }
    
    /* v is zero vector - identity transformation */
    if (cl <= 0.0) {
        return 0.0;
    }
    
    /* Compute (vₚ² + ∑vᵢ²)^(1/2) with normalized v to avoid overflow */
    double clinv = 1.0 / cl;
    sm = (u[pivot * inc] * clinv) * (u[pivot * inc] * clinv);
    for (i = start; i < m; i++) {
        sm += (u[i * inc] * clinv) * (u[i * inc] * clinv);
    }
    
    /* Compute s = -σ(vₚ² + ∑vᵢ²)^(1/2) where σ = -sgn(vₚ) */
    cl *= sqrt(sm);
    if (u[pivot * inc] > 0.0) {
        cl = -cl;
    }
    
    /* uₚ = vₚ - s, yₚ = s */
    double up = u[pivot * inc] - cl;
    u[pivot * inc] = cl;
    
    return up;
}

/**
 * h2 - Apply Householder transformation
 *
 * Apply m×m Householder transformation to columns of matrix C:
 *
 *   Qc = c + b⁻¹(uᵀc) × u
 *
 * where Q = Iₘ - b⁻¹uuᵀ and b = s·uₚ (computed from h1).
 *
 * On input, c contains a matrix which will be regarded as a set of vectors
 * to which the Householder transformation is to be applied.
 * On output, c contains the set of transformed vectors.
 *
 * Algorithm for each column j:
 *   1. Compute uᵀc = uₚcₚ + ∑cᵢuᵢ (l ≤ i < m)
 *   2. If uᵀc ≠ 0, update c = c + b⁻¹(uᵀc) × u
 *
 * Reference: C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
 *            Prentice Hall, 1974. Chapter 10.
 *
 * @param pivot  Pivot index lₚ (must satisfy 0 ≤ pivot < start)
 * @param start  Start index l₁ (must satisfy pivot < start ≤ m-1)
 * @param m      Number of rows
 * @param u      Householder vector from h1
 * @param incu   Storage increment for u (iue)
 * @param up     Pivot element uₚ from h1
 * @param c      Matrix to transform (column-major)
 * @param incc   Storage increment between elements of vector in c (ice)
 * @param mdc    Storage increment between vectors in c (icv)
 * @param nc     Number of vectors in c to transform (ncv)
 */
void h2(int pivot, int start, int m, double* u, int incu,
        double up, double* c, int incc, int mdc, int nc) {
    double b, sm;
    int i, j;
    
    /* Check bounds: 0 ≤ lₚ < l₁ ≤ m-1 and ncv > 0 */
    if (pivot < 0 || pivot >= start || start >= m || nc <= 0) {
        return;
    }
    
    /* Compute b = s·uₚ where s = u[pivot] (stored by h1) */
    b = up * u[pivot * incu];
    
    /* Q = Iₘ when b = s·uₚ ≥ 0 (identity transformation) */
    if (b >= 0.0) {
        return;
    }
    
    b = 1.0 / b;
    
    /* Apply transformation to each column */
    for (j = 0; j < nc; j++) {
        /* Compute uᵀc = uₚcₚ + ∑cᵢuᵢ (l ≤ i < m) */
        sm = c[j * mdc + pivot * incc] * up;
        for (i = start; i < m; i++) {
            sm += c[j * mdc + i * incc] * u[i * incu];
        }
        
        if (sm != 0.0) {
            /* c = c + b⁻¹(uᵀc) × u */
            sm *= b;
            c[j * mdc + pivot * incc] += sm * up;
            for (i = start; i < m; i++) {
                c[j * mdc + i * incc] += sm * u[i * incu];
            }
        }
    }
}

/**
 * g1 - Compute 2×2 Givens rotation matrix G
 *
 * Compute rotation matrix G such that:
 *
 *   G ⎡x₁⎤ ≡ ⎡ c  s⎤⎡x₁⎤ = ⎡(x₁²+x₂²)^(1/2)⎤ ≡ ⎡r⎤
 *     ⎣x₂⎦   ⎣-s  c⎦⎣x₂⎦   ⎣      0        ⎦   ⎣0⎦
 *
 * This is used for special form least squares Ax ≌ b where:
 *
 *           ⎡ Rₙₓₙ ⎤      ⎡ dₙₓ₁ ⎤
 *   A =     ⎢ 0₁ₓₙ ⎥, b = ⎢ e₁ₓ₁ ⎥  and R is upper triangular
 *           ⎣ y₁ₓₙ ⎦      ⎣ z₁ₓ₁ ⎦
 *
 * The rotation matrix is used to reduce the system to upper triangular form
 * and reduce the right side so that only first n+1 components are non-zero.
 *
 * Reference: C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
 *            Prentice Hall, 1974. Chapter 3.
 *
 * @param a    First element (x₁)
 * @param b    Second element (x₂)
 * @param c    Output: cosine of rotation angle
 * @param s    Output: sine of rotation angle
 * @param sig  Output: resulting value r = (x₁²+x₂²)^(1/2)
 */
void g1(double a, double b, double* c, double* s, double* sig) {
    double xr, yr;
    
    if (fabs(a) > fabs(b)) {
        /* |x₁| > |x₂|: compute via x₂/x₁ */
        xr = b / a;
        yr = sqrt(1.0 + xr * xr);
        *c = copysign(1.0 / yr, a);
        *s = (*c) * xr;
        *sig = fabs(a) * yr;
    } else if (b != 0.0) {
        /* |x₂| > 0: compute via x₁/x₂ */
        xr = a / b;
        yr = sqrt(1.0 + xr * xr);
        *s = copysign(1.0 / yr, b);
        *c = (*s) * xr;
        *sig = fabs(b) * yr;
    } else {
        /* Both zero: identity rotation */
        *sig = 0.0;
        *c = 0.0;
        *s = 1.0;
    }
}

/**
 * g2 - Apply Givens rotation
 *
 * Apply the Givens rotation matrix G computed by g1:
 *
 *   G ⎡z₁⎤ = ⎡ c  s⎤⎡z₁⎤ = ⎡ c·z₁ + s·z₂⎤
 *     ⎣z₂⎦   ⎣-s  c⎦⎣z₂⎦   ⎣-s·z₁ + c·z₂⎦
 *
 * @param c  Cosine from g1
 * @param s  Sine from g1
 * @param a  First element z₁ (modified in place to c·z₁ + s·z₂)
 * @param b  Second element z₂ (modified in place to -s·z₁ + c·z₂)
 */
void g2(double c, double s, double* a, double* b) {
    double xa = *a;
    double xb = *b;
    *a = c * xa + s * xb;
    *b = -s * xa + c * xb;
}
