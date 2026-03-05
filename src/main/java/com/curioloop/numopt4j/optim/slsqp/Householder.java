/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * Householder transformation and Givens rotation utilities.
 * Based on Lawson & Hanson, "Solving Least Squares Problems"
 *
 * Reference:
 * C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
 * Prentice Hall, 1974. (revised 1995 edition)
 */
package com.curioloop.numopt4j.optim.slsqp;

/**
 * Householder transformation and Givens rotation utilities.
 * <p>
 * This class provides methods for computing and applying Householder transformations
 * and Givens rotations, which are fundamental operations in QR decomposition and
 * least squares problems.
 * <p>
 * Strictly follows the implementation in householder.c.
 */
final class Householder {

    private Householder() {
        // Utility class - prevent instantiation
    }

    /**
     * h1 - Compute Householder transformation vector
     * <p>
     * Given m-vector v, construct m×m Householder vector u and scalar s for
     * transformation Qv ≡ y. The Householder matrix can be computed with:
     * <pre>
     *   Q = Iₘ - b⁻¹uuᵀ  where b = s·uₚ
     * </pre>
     * <p>
     * lₚ (pivot) is the index of the pivot element, which should satisfy 0 ≤ lₚ &lt; l₁.
     * If l₁ &lt; m, the transformation will be constructed to zero out elements
     * indexed from l₁ through m-1. If l₁ ≥ m, the subroutine does an identity
     * transformation.
     * <p>
     * Algorithm:
     * <ol>
     *   <li>Find max(v) for scaling to avoid overflow</li>
     *   <li>Compute s = -σ(vₚ² + ∑vᵢ²)^(1/2) where σ = -sgn(vₚ)</li>
     *   <li>Set uₚ = vₚ - s and yₚ = s</li>
     * </ol>
     * <p>
     * On input, u contains the pivot vector.
     * On output, u contains quantities defining the vector u of the Householder
     * transformation. The u[lₚ] element (uₚ) is returned separately.
     * <p>
     * Reference: C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
     *            Prentice Hall, 1974. Chapter 10.
     *
     * @param pivot  Pivot index lₚ (must satisfy 0 ≤ pivot &lt; start)
     * @param start  Start index l₁ for transformation (must satisfy pivot &lt; start ≤ m-1)
     * @param m      Vector length
     * @param u      Vector to transform (modified in place)
     * @param uOff   Offset into u array
     * @param inc    Storage increment between elements (ive)
     * @return       Pivot element uₚ for use in h2, or 0.0 if identity transformation
     */
    static double h1(int pivot, int start, int m, double[] u, int uOff, int inc) {
        double cl, sm;
        int i;

        /* Check bounds: 0 ≤ lₚ < l₁ ≤ m-1 */
        if (pivot < 0 || pivot >= start || start >= m) {
            return 0.0;
        }

        /* Find max|v| for numerical stability */
        cl = Math.abs(u[uOff + pivot * inc]);
        for (i = start; i < m; i++) {
            double v = Math.abs(u[uOff + i * inc]);
            if (v > cl) cl = v;
        }

        /* v is zero vector - identity transformation */
        if (cl <= 0.0) {
            return 0.0;
        }

        /* Compute (vₚ² + ∑vᵢ²)^(1/2) with normalized v to avoid overflow */
        double clinv = 1.0 / cl;
        sm = (u[uOff + pivot * inc] * clinv) * (u[uOff + pivot * inc] * clinv);
        for (i = start; i < m; i++) {
            sm += (u[uOff + i * inc] * clinv) * (u[uOff + i * inc] * clinv);
        }

        /* Compute s = -σ(vₚ² + ∑vᵢ²)^(1/2) where σ = -sgn(vₚ) */
        cl *= Math.sqrt(sm);
        if (u[uOff + pivot * inc] > 0.0) {
            cl = -cl;
        }

        /* uₚ = vₚ - s, yₚ = s */
        double up = u[uOff + pivot * inc] - cl;
        u[uOff + pivot * inc] = cl;

        return up;
    }

    /**
     * h2 - Apply Householder transformation
     * <p>
     * Apply m×m Householder transformation to columns of matrix C:
     * <pre>
     *   Qc = c + b⁻¹(uᵀc) × u
     * </pre>
     * where Q = Iₘ - b⁻¹uuᵀ and b = s·uₚ (computed from h1).
     * <p>
     * On input, c contains a matrix which will be regarded as a set of vectors
     * to which the Householder transformation is to be applied.
     * On output, c contains the set of transformed vectors.
     * <p>
     * Algorithm for each column j:
     * <ol>
     *   <li>Compute uᵀc = uₚcₚ + ∑cᵢuᵢ (l ≤ i &lt; m)</li>
     *   <li>If uᵀc ≠ 0, update c = c + b⁻¹(uᵀc) × u</li>
     * </ol>
     * <p>
     * Reference: C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
     *            Prentice Hall, 1974. Chapter 10.
     *
     * @param pivot  Pivot index lₚ (must satisfy 0 ≤ pivot &lt; start)
     * @param start  Start index l₁ (must satisfy pivot &lt; start ≤ m-1)
     * @param m      Number of rows
     * @param u      Householder vector from h1
     * @param uOff   Offset into u array
     * @param incu   Storage increment for u (iue)
     * @param up     Pivot element uₚ from h1
     * @param c      Matrix to transform (column-major)
     * @param cOff   Offset into c array
     * @param incc   Storage increment between elements of vector in c (ice)
     * @param mdc    Storage increment between vectors in c (icv)
     * @param nc     Number of vectors in c to transform (ncv)
     */
    static void h2(int pivot, int start, int m,
                   double[] u, int uOff, int incu,
                   double up,
                   double[] c, int cOff, int incc, int mdc, int nc) {
        double b, sm;
        int i, j;

        /* Check bounds: 0 ≤ lₚ < l₁ ≤ m-1 and ncv > 0 */
        if (pivot < 0 || pivot >= start || start >= m || nc <= 0) {
            return;
        }

        /* Compute b = s·uₚ where s = u[pivot] (stored by h1) */
        b = up * u[uOff + pivot * incu];

        /* Q = Iₘ when b = s·uₚ ≥ 0 (identity transformation) */
        if (b >= 0.0) {
            return;
        }

        b = 1.0 / b;

        /* Apply transformation to each column */
        for (j = 0; j < nc; j++) {
            /* Compute uᵀc = uₚcₚ + ∑cᵢuᵢ (l ≤ i < m) */
            sm = c[cOff + j * mdc + pivot * incc] * up;
            for (i = start; i < m; i++) {
                sm += c[cOff + j * mdc + i * incc] * u[uOff + i * incu];
            }

            if (sm != 0.0) {
                /* c = c + b⁻¹(uᵀc) × u */
                sm *= b;
                c[cOff + j * mdc + pivot * incc] += sm * up;
                for (i = start; i < m; i++) {
                    c[cOff + j * mdc + i * incc] += sm * u[uOff + i * incu];
                }
            }
        }
    }

    /**
     * g1 - Compute 2×2 Givens rotation matrix G
     * <p>
     * Compute rotation matrix G such that:
     * <pre>
     *   G ⎡x₁⎤ ≡ ⎡ c  s⎤⎡x₁⎤ = ⎡(x₁²+x₂²)^(1/2)⎤ ≡ ⎡r⎤
     *     ⎣x₂⎦   ⎣-s  c⎦⎣x₂⎦   ⎣      0        ⎦   ⎣0⎦
     * </pre>
     * <p>
     * This is used for special form least squares Ax ≌ b where:
     * <pre>
     *           ⎡ Rₙₓₙ ⎤      ⎡ dₙₓ₁ ⎤
     *   A =     ⎢ 0₁ₓₙ ⎥, b = ⎢ e₁ₓ₁ ⎥  and R is upper triangular
     *           ⎣ y₁ₓₙ ⎦      ⎣ z₁ₓ₁ ⎦
     * </pre>
     * <p>
     * The rotation matrix is used to reduce the system to upper triangular form
     * and reduce the right side so that only first n+1 components are non-zero.
     * <p>
     * Reference: C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
     *            Prentice Hall, 1974. Chapter 3.
     *
     * @param a      First element (x₁)
     * @param b      Second element (x₂)
     * @param result Output array of length 3: [c, s, sig] where
     *               c = cosine of rotation angle,
     *               s = sine of rotation angle,
     *               sig = resulting value r = (x₁²+x₂²)^(1/2)
     */
    static void g1(double a, double b, double[] result) {
        double xr, yr;

        if (Math.abs(a) > Math.abs(b)) {
            /* |x₁| > |x₂|: compute via x₂/x₁ */
            xr = b / a;
            yr = Math.sqrt(1.0 + xr * xr);
            result[0] = Math.copySign(1.0 / yr, a);  // c
            result[1] = result[0] * xr;              // s
            result[2] = Math.abs(a) * yr;            // sig
        } else if (b != 0.0) {
            /* |x₂| > 0: compute via x₁/x₂ */
            xr = a / b;
            yr = Math.sqrt(1.0 + xr * xr);
            result[1] = Math.copySign(1.0 / yr, b);  // s
            result[0] = result[1] * xr;              // c
            result[2] = Math.abs(b) * yr;            // sig
        } else {
            /* Both zero: identity rotation */
            result[2] = 0.0;  // sig
            result[0] = 0.0;  // c
            result[1] = 1.0;  // s
        }
    }

    /**
     * g2 - Apply Givens rotation
     * <p>
     * Apply the Givens rotation matrix G computed by g1:
     * <pre>
     *   G ⎡z₁⎤ = ⎡ c  s⎤⎡z₁⎤ = ⎡ c·z₁ + s·z₂⎤
     *     ⎣z₂⎦   ⎣-s  c⎦⎣z₂⎦   ⎣-s·z₁ + c·z₂⎦
     * </pre>
     *
     * @param c    Cosine from g1
     * @param s    Sine from g1
     * @param ab   Array containing both elements
     * @param aOff Offset for first element z₁ (modified in place to c·z₁ + s·z₂)
     * @param bOff Offset for second element z₂ (modified in place to -s·z₁ + c·z₂)
     */
    static void g2(double c, double s, double[] ab, int aOff, int bOff) {
        double xa = ab[aOff];
        double xb = ab[bOff];
        ab[aOff] = c * xa + s * xb;
        ab[bOff] = -s * xa + c * xb;
    }
}
