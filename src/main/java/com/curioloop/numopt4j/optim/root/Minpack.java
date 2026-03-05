package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.linalg.blas.BLAS;

/**
 * MINPACK algorithm utilities — math helpers, QR factorization, and Powell hybrid primitives.
 *
 * <p>All matrices use column-major flat {@code double[]} layout: {@code a[i + lda*j]}.</p>
 */
class Minpack {

    private Minpack() {}

    // ── Machine constants ─────────────────────────────────────────────────────

    /**
     * Returns machine-dependent constants (equivalent to Fortran {@code dpmpar}).
     *
     * @param i 1 = machine epsilon, 2 = smallest positive number, 3 = largest positive number
     */
    public static double dpmpar(int i) {
        switch (i) {
            case 1: return Math.ulp(1.0);
            case 2: return Double.MIN_VALUE;
            case 3: return Double.MAX_VALUE;
            default: throw new IllegalArgumentException("dpmpar: i must be 1, 2, or 3");
        }
    }

    /**
     * Computes the Euclidean norm of a vector (equivalent to Fortran {@code enorm}).
     */
    public static double enorm(int n, double[] x) {
        return BLAS.dnrm2(n, x, 0, 1);
    }

    public static double enorm(int n, double[] x, int offset) {
        return BLAS.dnrm2(n, x, offset, 1);
    }

    // ── QR factorization ──────────────────────────────────────────────────────

    /**
     * Computes the QR factorization of an m×n matrix A with optional column pivoting
     * (translation of Fortran {@code qrfac} from MINPACK-1).
     *
     * @param m      number of rows
     * @param n      number of columns
     * @param a      m×n matrix (column-major, lda=m); on output contains factored form
     * @param lda    leading dimension of a (>= m)
     * @param pivot  if true, use column pivoting
     * @param ipvt   output permutation array of length lipvt (0-indexed)
     * @param lipvt  length of ipvt; if pivot=false may be 1
     * @param rdiag  output array of length n: diagonal elements of R
     * @param acnorm output array of length n: column norms of input A
     * @param wa     work array of length n
     */
    public static void qrfac(int m, int n, double[] a, int lda, boolean pivot,
                              int[] ipvt, int lipvt, double[] rdiag, double[] acnorm, double[] wa) {
        final double epsmch = dpmpar(1);
        final double one = 1.0, p05 = 0.05, zero = 0.0;

        for (int j = 0; j < n; j++) {
            acnorm[j] = BLAS.dnrm2(m, a, j * lda, 1);
            rdiag[j] = acnorm[j];
            wa[j] = rdiag[j];
            if (pivot) ipvt[j] = j;
        }

        int minmn = Math.min(m, n);
        for (int j = 0; j < minmn; j++) {
            if (pivot) {
                int kmax = j;
                for (int k = j; k < n; k++) {
                    if (rdiag[k] > rdiag[kmax]) kmax = k;
                }
                if (kmax != j) {
                    for (int i = 0; i < m; i++) {
                        double temp = a[i + lda * j];
                        a[i + lda * j] = a[i + lda * kmax];
                        a[i + lda * kmax] = temp;
                    }
                    rdiag[kmax] = rdiag[j];
                    wa[kmax] = wa[j];
                    int k = ipvt[j]; ipvt[j] = ipvt[kmax]; ipvt[kmax] = k;
                }
            }

            double ajnorm = BLAS.dnrm2(m - j, a, j + lda * j, 1);
            if (ajnorm == zero) { rdiag[j] = -ajnorm; continue; }
            if (a[j + lda * j] < zero) ajnorm = -ajnorm;
            for (int i = j; i < m; i++) a[i + lda * j] /= ajnorm;
            a[j + lda * j] += one;

            for (int k = j + 1; k < n; k++) {
                double sum = zero;
                for (int i = j; i < m; i++) sum += a[i + lda * j] * a[i + lda * k];
                double temp = sum / a[j + lda * j];
                for (int i = j; i < m; i++) a[i + lda * k] -= temp * a[i + lda * j];
                if (pivot && rdiag[k] != zero) {
                    temp = a[j + lda * k] / rdiag[k];
                    rdiag[k] *= Math.sqrt(Math.max(zero, one - temp * temp));
                    if (p05 * (rdiag[k] / wa[k]) * (rdiag[k] / wa[k]) <= epsmch) {
                        rdiag[k] = BLAS.dnrm2(m - j - 1, a, (j + 1) + lda * k, 1);
                        wa[k] = rdiag[k];
                    }
                }
            }
            rdiag[j] = -ajnorm;
        }
    }

    /**
     * Solves the augmented least-squares system {@code [R; sqrt(par)*D]*x = [qtb; 0]}
     * (translation of Fortran {@code qrsolv} from MINPACK-1).
     *
     * @param n      order of R
     * @param r      n×n upper triangular matrix (column-major, ldr=n); modified in place
     * @param ldr    leading dimension of r (>= n)
     * @param ipvt   permutation array of length n (0-indexed)
     * @param diag   diagonal elements of D, length n
     * @param qtb    first n elements of Q^T * b
     * @param x      output solution vector, length n
     * @param sdiag  output diagonal elements of S, length n
     * @param wa     work array of length n
     */
    public static void qrsolv(int n, double[] r, int ldr, int[] ipvt, double[] diag,
                               double[] qtb, double[] x, double[] sdiag, double[] wa) {
        final double p5 = 0.5, p25 = 0.25, zero = 0.0;

        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) r[i + ldr * j] = r[j + ldr * i];
            x[j] = r[j + ldr * j];
            wa[j] = qtb[j];
        }

        for (int j = 0; j < n; j++) {
            int l = ipvt[j];
            if (diag[l] == zero) {
                sdiag[j] = r[j + ldr * j];
                r[j + ldr * j] = x[j];
                continue;
            }
            for (int k = j; k < n; k++) sdiag[k] = zero;
            sdiag[j] = diag[l];

            double qtbpj = zero;
            for (int k = j; k < n; k++) {
                if (sdiag[k] == zero) continue;
                double cos, sin;
                double rkk = r[k + ldr * k], sdk = sdiag[k];
                if (Math.abs(rkk) >= Math.abs(sdk)) {
                    double tan = sdk / rkk;
                    cos = p5 / Math.sqrt(p25 + p25 * tan * tan);
                    sin = cos * tan;
                } else {
                    double cotan = rkk / sdk;
                    sin = p5 / Math.sqrt(p25 + p25 * cotan * cotan);
                    cos = sin * cotan;
                }
                r[k + ldr * k] = cos * rkk + sin * sdk;
                double temp = cos * wa[k] + sin * qtbpj;
                qtbpj = -sin * wa[k] + cos * qtbpj;
                wa[k] = temp;
                for (int i = k + 1; i < n; i++) {
                    temp = cos * r[i + ldr * k] + sin * sdiag[i];
                    sdiag[i] = -sin * r[i + ldr * k] + cos * sdiag[i];
                    r[i + ldr * k] = temp;
                }
            }
            sdiag[j] = r[j + ldr * j];
            r[j + ldr * j] = x[j];
        }

        int nsing = n;
        for (int j = 0; j < n; j++) {
            if (sdiag[j] == zero && nsing == n) nsing = j;
            if (nsing < n) wa[j] = zero;
        }
        for (int k = 0; k < nsing; k++) {
            int j = nsing - 1 - k;
            double sum = zero;
            for (int i = j + 1; i < nsing; i++) sum += r[i + ldr * j] * wa[i];
            wa[j] = (wa[j] - sum) / sdiag[j];
        }
        for (int j = 0; j < n; j++) x[ipvt[j]] = wa[j];
    }

    /**
     * Accumulates the m×m orthogonal matrix Q from its Householder factored form
     * (translation of Fortran {@code qform} from MINPACK-1).
     *
     * @param m   number of rows of A and order of Q
     * @param n   number of columns of A
     * @param q   m×m matrix (column-major, ldq=m); input: factored form, output: Q
     * @param ldq leading dimension of q (>= m)
     * @param wa  work array of length m
     */
    public static void qform(int m, int n, double[] q, int ldq, double[] wa) {
        final double one = 1.0, zero = 0.0;
        int minmn = Math.min(m, n);

        for (int j = 1; j < minmn; j++)
            for (int i = 0; i < j; i++) q[i + ldq * j] = zero;

        for (int j = n; j < m; j++) {
            for (int i = 0; i < m; i++) q[i + ldq * j] = zero;
            q[j + ldq * j] = one;
        }

        for (int l = 0; l < minmn; l++) {
            int k = minmn - 1 - l;
            for (int i = k; i < m; i++) { wa[i] = q[i + ldq * k]; q[i + ldq * k] = zero; }
            q[k + ldq * k] = one;
            if (wa[k] == zero) continue;
            for (int j = k; j < m; j++) {
                double sum = zero;
                for (int i = k; i < m; i++) sum += q[i + ldq * j] * wa[i];
                double temp = sum / wa[k];
                for (int i = k; i < m; i++) q[i + ldq * j] -= temp * wa[i];
            }
        }
    }

    // ── Powell hybrid primitives ──────────────────────────────────────────────

    /**
     * Computes the dogleg step: convex combination of Gauss-Newton and scaled gradient directions
     * (translation of Fortran {@code dogleg} from MINPACK-1).
     *
     * @param n     order of R
     * @param r     upper triangular matrix stored by rows (packed), length lr
     * @param lr    length of r (>= n*(n+1)/2)
     * @param diag  diagonal elements of D, length n
     * @param qtb   first n elements of Q^T * b
     * @param delta positive upper bound on ||D*x||
     * @param x     output step vector, length n
     * @param wa1   work array of length n
     * @param wa2   work array of length n
     */
    public static void dogleg(int n, double[] r, int lr, double[] diag, double[] qtb,
                               double delta, double[] x, double[] wa1, double[] wa2) {
        final double epsmch = dpmpar(1);
        final double one = 1.0, zero = 0.0;

        // Fortran (1-indexed): jj = n*(n+1)/2 + 1, then jj -= k each iteration, r(jj) is diagonal
        // Java (0-indexed):    jj = n*(n+1)/2 - 1, then jj -= (k+1) each iteration, r[jj] is diagonal
        //                      off-diagonal elements in row j start at r[jj+1]
        int jj = (n * (n + 1)) / 2;
        for (int k = 0; k < n; k++) {
            int j = n - 1 - k;
            jj -= (k + 1);
            int l = jj + 1;
            double sum = zero;
            for (int i = j + 1; i < n; i++) { sum += r[l] * x[i]; l++; }
            double temp = r[jj];
            if (temp == zero) {
                // Scan column j of the packed upper-triangular matrix to estimate scale.
                // In 0-indexed packed-by-rows storage, R[i,j] is at index j*(j+1)/2 + i.
                // Step from R[i,j] to R[i+1,j] = j - i.
                l = j * (j + 1) / 2;
                for (int i = 0; i <= j; i++) { temp = Math.max(temp, Math.abs(r[l])); l += j - i; }
                temp = epsmch * temp;
                if (temp == zero) temp = epsmch;
            }
            x[j] = (qtb[j] - sum) / temp;
        }

        for (int j = 0; j < n; j++) { wa1[j] = zero; wa2[j] = diag[j] * x[j]; }
        double qnorm = enorm(n, wa2);
        if (qnorm <= delta) return;

        int l = 0;
        for (int j = 0; j < n; j++) {
            double temp = qtb[j];
            for (int i = j; i < n; i++) { wa1[i] += r[l] * temp; l++; }
            wa1[j] /= diag[j];
        }

        double gnorm = enorm(n, wa1);
        double sgnorm = zero, alpha = delta / qnorm;
        if (gnorm != zero) {
            for (int j = 0; j < n; j++) wa1[j] = (wa1[j] / gnorm) / diag[j];
            l = 0;
            for (int j = 0; j < n; j++) {
                double sum = zero;
                for (int i = j; i < n; i++) { sum += r[l] * wa1[i]; l++; }
                wa2[j] = sum;
            }
            double temp = enorm(n, wa2);
            sgnorm = (gnorm / temp) / temp;
            alpha = zero;
            if (sgnorm < delta) {
                double bnorm = enorm(n, qtb);
                temp = (bnorm / gnorm) * (bnorm / qnorm) * (sgnorm / delta);
                temp = temp - (delta / qnorm) * (sgnorm / delta) * (sgnorm / delta)
                        + Math.sqrt((temp - (delta / qnorm)) * (temp - (delta / qnorm))
                        + (one - (delta / qnorm) * (delta / qnorm))
                        * (one - (sgnorm / delta) * (sgnorm / delta)));
                alpha = ((delta / qnorm) * (one - (sgnorm / delta) * (sgnorm / delta))) / temp;
            }
        }

        double temp2 = (one - alpha) * Math.min(sgnorm, delta);
        for (int j = 0; j < n; j++) x[j] = temp2 * wa1[j] + alpha * x[j];
    }

    /**
     * Performs a rank-1 update of the lower trapezoidal matrix S
     * (translation of Fortran {@code r1updt} from MINPACK-1).
     *
     * @param m    number of rows of S
     * @param n    number of columns of S (n <= m)
     * @param s    lower trapezoidal matrix stored by columns, length ls; modified in place
     * @param ls   length of s (>= n*(2*m-n+1)/2)
     * @param u    input vector of length m
     * @param v    input/output vector of length n; on output contains Givens rotation info
     * @param w    output vector of length m; contains Givens rotation info
     * @param sing single-element boolean array; set true if any diagonal of output S is zero
     */
    public static void r1updt(int m, int n, double[] s, int ls, double[] u, double[] v,
                               double[] w, boolean[] sing) {
        final double giant = dpmpar(3);
        final double one = 1.0, p5 = 0.5, p25 = 0.25, zero = 0.0;

        int jj = (n * (2 * m - n + 1)) / 2 - (m - n) - 1;
        int l = jj;
        for (int i = n - 1; i < m; i++) { w[i] = s[l]; l++; }

        int nm1 = n - 1;
        for (int nmj = 1; nmj <= nm1; nmj++) {
            int j = n - 1 - nmj;
            jj -= (m - j);
            w[j] = zero;
            if (v[j] == zero) continue;
            double cos, sin, tau;
            if (Math.abs(v[n - 1]) >= Math.abs(v[j])) {
                double tan = v[j] / v[n - 1];
                cos = p5 / Math.sqrt(p25 + p25 * tan * tan); sin = cos * tan; tau = sin;
            } else {
                double cotan = v[n - 1] / v[j];
                sin = p5 / Math.sqrt(p25 + p25 * cotan * cotan); cos = sin * cotan; tau = one;
                if (Math.abs(cos) * giant > one) tau = one / cos;
            }
            v[n - 1] = sin * v[j] + cos * v[n - 1]; v[j] = tau;
            l = jj;
            for (int i = j; i < m; i++) {
                double temp = cos * s[l] - sin * w[i]; w[i] = sin * s[l] + cos * w[i]; s[l] = temp; l++;
            }
        }

        for (int i = 0; i < m; i++) w[i] += v[n - 1] * u[i];

        sing[0] = false;
        if (nm1 >= 1) {
            jj = 0;
            for (int j = 0; j < nm1; j++) {
                if (w[j] != zero) {
                    double cos, sin, tau;
                    if (Math.abs(s[jj]) >= Math.abs(w[j])) {
                        double tan = w[j] / s[jj];
                        cos = p5 / Math.sqrt(p25 + p25 * tan * tan); sin = cos * tan; tau = sin;
                    } else {
                        double cotan = s[jj] / w[j];
                        sin = p5 / Math.sqrt(p25 + p25 * cotan * cotan); cos = sin * cotan; tau = one;
                        if (Math.abs(cos) * giant > one) tau = one / cos;
                    }
                    l = jj;
                    for (int i = j; i < m; i++) {
                        double temp = cos * s[l] + sin * w[i]; w[i] = -sin * s[l] + cos * w[i]; s[l] = temp; l++;
                    }
                    w[j] = tau;
                }
                if (s[jj] == zero) sing[0] = true;
                jj += (m - j);
            }
        }

        l = jj;
        for (int i = n - 1; i < m; i++) { s[l] = w[i]; l++; }
        if (s[jj] == zero) sing[0] = true;
    }

    /**
     * Computes A*Q where Q is the product of Givens rotations from {@code r1updt}
     * (translation of Fortran {@code r1mpyq} from MINPACK-1).
     *
     * @param m   number of rows of A
     * @param n   number of columns of A
     * @param a   m×n matrix (column-major, lda=m); replaced by A*Q on output
     * @param lda leading dimension of a
     * @param v   Givens rotation info from r1updt, length n
     * @param w   Givens rotation info from r1updt, length n
     */
    public static void r1mpyq(int m, int n, double[] a, int lda, double[] v, double[] w) {
        final double one = 1.0;
        int nm1 = n - 1;
        if (nm1 < 1) return;

        for (int nmj = 1; nmj <= nm1; nmj++) {
            int j = n - 1 - nmj;
            double cos, sin;
            if (Math.abs(v[j]) > one) { cos = one / v[j]; sin = Math.sqrt(one - cos * cos); }
            else { sin = v[j]; cos = Math.sqrt(one - sin * sin); }
            for (int i = 0; i < m; i++) {
                double temp = cos * a[i + lda * j] - sin * a[i + lda * (n - 1)];
                a[i + lda * (n - 1)] = sin * a[i + lda * j] + cos * a[i + lda * (n - 1)];
                a[i + lda * j] = temp;
            }
        }

        for (int j = 0; j < nm1; j++) {
            double cos, sin;
            if (Math.abs(w[j]) > one) { cos = one / w[j]; sin = Math.sqrt(one - cos * cos); }
            else { sin = w[j]; cos = Math.sqrt(one - sin * sin); }
            for (int i = 0; i < m; i++) {
                double temp = cos * a[i + lda * j] + sin * a[i + lda * (n - 1)];
                a[i + lda * (n - 1)] = -sin * a[i + lda * j] + cos * a[i + lda * (n - 1)];
                a[i + lda * j] = temp;
            }
        }
    }

    /**
     * Updates the upper triangular matrix R by adding a row w, maintaining upper triangular structure
     * (translation of Fortran {@code rwupdt} from MINPACK-1).
     *
     * @param n     order of R
     * @param r     n×n upper triangular matrix (column-major, ldr=n); updated in place
     * @param ldr   leading dimension of r
     * @param w     row vector to add, length n
     * @param b     vector of length n; updated by Q^T * [b; alpha]
     * @param alpha single-element array: (n+1)-th element of c; updated on output
     * @param cos   output cosines of Givens rotations, length n
     * @param sin   output sines of Givens rotations, length n
     */
    public static void rwupdt(int n, double[] r, int ldr, double[] w, double[] b,
                               double[] alpha, double[] cos, double[] sin) {
        final double one = 1.0, p5 = 0.5, p25 = 0.25, zero = 0.0;

        for (int j = 0; j < n; j++) {
            double rowj = w[j];
            for (int i = 0; i < j; i++) {
                double temp = cos[i] * r[i + ldr * j] + sin[i] * rowj;
                rowj = -sin[i] * r[i + ldr * j] + cos[i] * rowj;
                r[i + ldr * j] = temp;
            }
            cos[j] = one; sin[j] = zero;
            if (rowj == zero) continue;
            if (Math.abs(r[j + ldr * j]) >= Math.abs(rowj)) {
                double tan = rowj / r[j + ldr * j];
                cos[j] = p5 / Math.sqrt(p25 + p25 * tan * tan); sin[j] = cos[j] * tan;
            } else {
                double cotan = r[j + ldr * j] / rowj;
                sin[j] = p5 / Math.sqrt(p25 + p25 * cotan * cotan); cos[j] = sin[j] * cotan;
            }
            r[j + ldr * j] = cos[j] * r[j + ldr * j] + sin[j] * rowj;
            double temp = cos[j] * b[j] + sin[j] * alpha[0];
            alpha[0] = -sin[j] * b[j] + cos[j] * alpha[0];
            b[j] = temp;
        }
    }
}
