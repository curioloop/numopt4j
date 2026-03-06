package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.linalg.blas.BLAS;

/**
 * Core numerical routines ported from MINPACK (Argonne National Laboratory).
 *
 * <p><b>Matrix layout:</b> all matrices are stored column-major in a flat {@code double[]}:
 * element A[i,j] is at index {@code i + lda*j}.</p>
 *
 * <p><b>Packed upper-triangular storage</b> (used by {@code dogleg} / {@code r1updt}):
 * R[i,j] (i&le;j) is stored at index {@code i*(2n-i-1)/2 + j}, i.e. rows are packed
 * left-to-right, top-to-bottom.</p>
 *
 * <p><b>Offset convention:</b> every array parameter that may reside inside a merged
 * workspace buffer carries a companion {@code *Off} offset.  The routine accesses
 * element {@code k} as {@code array[off + k]}, so callers can pass a sub-region of a
 * larger {@code double[]} without copying.  Standalone arrays (e.g. the solution vector
 * {@code x}) always use offset 0 and have no companion parameter.</p>
 */
class Minpack {

    private Minpack() {}

    // ── Machine constants ─────────────────────────────────────────────────────

    /**
     * Returns machine-dependent floating-point constants (equivalent to Fortran {@code dpmpar}).
     *
     * @param i selects the constant:
     *          1 → machine epsilon ε (smallest value such that 1+ε &gt; 1),
     *          2 → smallest positive normalised number,
     *          3 → largest finite number
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
     * Euclidean norm ‖x‖₂ of {@code x[0..n-1]}.
     */
    public static double enorm(int n, double[] x) {
        return BLAS.dnrm2(n, x, 0, 1);
    }

    /**
     * Euclidean norm ‖x‖₂ of {@code x[offset..offset+n-1]}.
     */    public static double enorm(int n, double[] x, int offset) {
        return BLAS.dnrm2(n, x, offset, 1);
    }

    // ── QR factorization ──────────────────────────────────────────────────────

    /**
     * QR factorization of an m×n matrix A with optional column pivoting
     * (translation of Fortran {@code qrfac} from MINPACK-1).
     *
     * <p>Computes the decomposition A·P = Q·R where Q is orthogonal, R is upper triangular,
     * and P is a column-permutation matrix chosen so that
     * |R[0,0]| &ge; |R[1,1]| &ge; … &ge; |R[n-1,n-1]|.</p>
     *
     * <p>On entry {@code a} holds A in column-major order.  On exit the upper triangle of
     * {@code a} contains R; the strict lower triangle contains the Householder vectors
     * (scaled so that the first non-zero component equals 1).</p>
     *
     * @param m         number of rows of A
     * @param n         number of columns of A
     * @param a         A in column-major layout ({@code a[i + lda*j]} = A[i,j]); overwritten on exit
     * @param lda       leading dimension of {@code a} (&ge; m)
     * @param pivot     if {@code true}, apply column pivoting to maximise diagonal of R
     * @param ipvt      on exit: 0-indexed column permutation P (only used when {@code pivot=true})
     * @param lipvt     length of {@code ipvt}; may be 1 when {@code pivot=false}
     * @param rdiag     on exit: diagonal of R, i.e. rdiag[j] = R[j,j];
     *                  accessed at {@code rdiag[rdiagOff .. rdiagOff+n-1]}
     * @param rdiagOff  start offset into {@code rdiag}
     * @param acnorm    on exit: Euclidean column norms of the <em>input</em> A (before pivoting);
     *                  accessed at {@code acnorm[acnormOff .. acnormOff+n-1]}
     * @param acnormOff start offset into {@code acnorm}
     * @param wa        work array of length n; accessed at {@code wa[waOff .. waOff+n-1]}
     * @param waOff     start offset into {@code wa}
     */
    public static void qrfac(int m, int n, double[] a, int lda, boolean pivot,
                              int[] ipvt, int lipvt,
                              double[] rdiag, int rdiagOff,
                              double[] acnorm, int acnormOff,
                              double[] wa, int waOff) {
        final double epsmch = dpmpar(1);
        final double one = 1.0, p05 = 0.05, zero = 0.0;

        for (int j = 0; j < n; j++) {
            acnorm[acnormOff + j] = BLAS.dnrm2(m, a, j * lda, 1);
            rdiag[rdiagOff + j] = acnorm[acnormOff + j];
            wa[waOff + j] = rdiag[rdiagOff + j];
            if (pivot) ipvt[j] = j;
        }

        int minmn = Math.min(m, n);
        for (int j = 0; j < minmn; j++) {
            if (pivot) {
                int kmax = j;
                for (int k = j; k < n; k++) {
                    if (rdiag[rdiagOff + k] > rdiag[rdiagOff + kmax]) kmax = k;
                }
                if (kmax != j) {
                    for (int i = 0; i < m; i++) {
                        double temp = a[i + lda * j];
                        a[i + lda * j] = a[i + lda * kmax];
                        a[i + lda * kmax] = temp;
                    }
                    rdiag[rdiagOff + kmax] = rdiag[rdiagOff + j];
                    wa[waOff + kmax] = wa[waOff + j];
                    int k = ipvt[j]; ipvt[j] = ipvt[kmax]; ipvt[kmax] = k;
                }
            }

            double ajnorm = BLAS.dnrm2(m - j, a, j + lda * j, 1);
            if (ajnorm == zero) { rdiag[rdiagOff + j] = -ajnorm; continue; }
            if (a[j + lda * j] < zero) ajnorm = -ajnorm;
            for (int i = j; i < m; i++) a[i + lda * j] /= ajnorm;
            a[j + lda * j] += one;

            for (int k = j + 1; k < n; k++) {
                double sum = zero;
                for (int i = j; i < m; i++) sum += a[i + lda * j] * a[i + lda * k];
                double temp = sum / a[j + lda * j];
                for (int i = j; i < m; i++) a[i + lda * k] -= temp * a[i + lda * j];
                if (pivot && rdiag[rdiagOff + k] != zero) {
                    temp = a[j + lda * k] / rdiag[rdiagOff + k];
                    rdiag[rdiagOff + k] *= Math.sqrt(Math.max(zero, one - temp * temp));
                    if (p05 * (rdiag[rdiagOff + k] / wa[waOff + k]) * (rdiag[rdiagOff + k] / wa[waOff + k]) <= epsmch) {
                        rdiag[rdiagOff + k] = BLAS.dnrm2(m - j - 1, a, (j + 1) + lda * k, 1);
                        wa[waOff + k] = rdiag[rdiagOff + k];
                    }
                }
            }
            rdiag[rdiagOff + j] = -ajnorm;
        }
    }

    /**
     * Solves the rank-deficient augmented least-squares system arising in the Levenberg–Marquardt
     * and Powell hybrid methods (translation of Fortran {@code qrsolv} from MINPACK-1).
     *
     * <p>Given the QR factorization A·P = Q·R and a diagonal scaling matrix D, finds x that
     * minimises ‖[R·Pᵀ; D]·x - [Qᵀ·b; 0]‖₂.  The solution is obtained by applying a sequence
     * of Givens rotations to eliminate D, producing a new upper-triangular factor S such that
     * Sᵀ·S = Rᵀ·R + Dᵀ·D.</p>
     *
     * @param n        order of R (number of unknowns)
     * @param r        upper-triangular R in column-major layout; accessed at {@code r[rOff..]};
     *                 <em>modified in place</em> (lower triangle is used as scratch)
     * @param rOff     start offset into {@code r}
     * @param ldr      leading dimension of {@code r} (&ge; n)
     * @param ipvt     0-indexed column permutation P from {@code qrfac}
     * @param diag     diagonal of D; accessed at {@code diag[diagOff .. diagOff+n-1]}
     * @param diagOff  start offset into {@code diag}
     * @param qtb      first n elements of Qᵀ·b; accessed at {@code qtb[qtbOff .. qtbOff+n-1]}
     * @param qtbOff   start offset into {@code qtb}
     * @param x        on exit: solution vector x (standalone, offset 0)
     * @param sdiag    on exit: diagonal of S (the augmented upper-triangular factor);
     *                 accessed at {@code sdiag[sdiagOff .. sdiagOff+n-1]}
     * @param sdiagOff start offset into {@code sdiag}
     * @param wa       work array of length n; accessed at {@code wa[waOff .. waOff+n-1]}
     * @param waOff    start offset into {@code wa}
     */
    public static void qrsolv(int n,
                               double[] r, int rOff, int ldr,
                               int[] ipvt,
                               double[] diag, int diagOff,
                               double[] qtb, int qtbOff,
                               double[] x,
                               double[] sdiag, int sdiagOff,
                               double[] wa, int waOff) {
        final double p5 = 0.5, p25 = 0.25, zero = 0.0;

        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) r[rOff + i + ldr * j] = r[rOff + j + ldr * i];
            x[j] = r[rOff + j + ldr * j];
            wa[waOff + j] = qtb[qtbOff + j];
        }

        for (int j = 0; j < n; j++) {
            int l = ipvt[j];
            if (diag[diagOff + l] == zero) {
                sdiag[sdiagOff + j] = r[rOff + j + ldr * j];
                r[rOff + j + ldr * j] = x[j];
                continue;
            }
            for (int k = j; k < n; k++) sdiag[sdiagOff + k] = zero;
            sdiag[sdiagOff + j] = diag[diagOff + l];

            double qtbpj = zero;
            for (int k = j; k < n; k++) {
                if (sdiag[sdiagOff + k] == zero) continue;
                double cos, sin;
                double rkk = r[rOff + k + ldr * k], sdk = sdiag[sdiagOff + k];
                if (Math.abs(rkk) >= Math.abs(sdk)) {
                    double tan = sdk / rkk;
                    cos = p5 / Math.sqrt(p25 + p25 * tan * tan);
                    sin = cos * tan;
                } else {
                    double cotan = rkk / sdk;
                    sin = p5 / Math.sqrt(p25 + p25 * cotan * cotan);
                    cos = sin * cotan;
                }
                r[rOff + k + ldr * k] = cos * rkk + sin * sdk;
                double temp = cos * wa[waOff + k] + sin * qtbpj;
                qtbpj = -sin * wa[waOff + k] + cos * qtbpj;
                wa[waOff + k] = temp;
                for (int i = k + 1; i < n; i++) {
                    temp = cos * r[rOff + i + ldr * k] + sin * sdiag[sdiagOff + i];
                    sdiag[sdiagOff + i] = -sin * r[rOff + i + ldr * k] + cos * sdiag[sdiagOff + i];
                    r[rOff + i + ldr * k] = temp;
                }
            }
            sdiag[sdiagOff + j] = r[rOff + j + ldr * j];
            r[rOff + j + ldr * j] = x[j];
        }

        int nsing = n;
        for (int j = 0; j < n; j++) {
            if (sdiag[sdiagOff + j] == zero && nsing == n) nsing = j;
            if (nsing < n) wa[waOff + j] = zero;
        }
        for (int k = 0; k < nsing; k++) {
            int j = nsing - 1 - k;
            double sum = zero;
            for (int i = j + 1; i < nsing; i++) sum += r[rOff + i + ldr * j] * wa[waOff + i];
            wa[waOff + j] = (wa[waOff + j] - sum) / sdiag[sdiagOff + j];
        }
        for (int j = 0; j < n; j++) x[ipvt[j]] = wa[waOff + j];
    }

    /**
     * Accumulates the full m×m orthogonal factor Q from the Householder vectors stored in
     * the lower triangle of {@code q} after a call to {@code qrfac}
     * (translation of Fortran {@code qform} from MINPACK-1).
     *
     * <p>On entry {@code q[0..m*n-1]} holds the first n Householder vectors (column-major).
     * On exit {@code q} is the full m×m orthogonal matrix Q such that A = Q·R.</p>
     *
     * @param m      number of rows (order of Q)
     * @param n      number of Householder vectors stored in {@code q}
     * @param q      on entry: Householder factored form; on exit: full Q (column-major, ldq=m)
     * @param ldq    leading dimension of {@code q} (&ge; m)
     * @param wa     work array of length m; accessed at {@code wa[waOff .. waOff+m-1]}
     * @param waOff  start offset into {@code wa}
     */
    public static void qform(int m, int n, double[] q, int ldq, double[] wa, int waOff) {
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
            for (int i = k; i < m; i++) { wa[waOff + i] = q[i + ldq * k]; q[i + ldq * k] = zero; }
            q[k + ldq * k] = one;
            if (wa[waOff + k] == zero) continue;
            for (int j = k; j < m; j++) {
                double sum = zero;
                for (int i = k; i < m; i++) sum += q[i + ldq * j] * wa[waOff + i];
                double temp = sum / wa[waOff + k];
                for (int i = k; i < m; i++) q[i + ldq * j] -= temp * wa[waOff + i];
            }
        }
    }

    // ── Powell hybrid primitives ──────────────────────────────────────────────

    /**
     * Computes the dogleg step for the Powell hybrid trust-region method
     * (translation of Fortran {@code dogleg} from MINPACK-1).
     *
     * <p>Given the QR factorization J = Q·R and the scaled diagonal D, the dogleg step p
     * is the convex combination of two candidate directions that minimises the quadratic
     * model m(p) = ½‖J·p + F‖₂² subject to ‖D·p‖₂ &le; Δ:</p>
     * <ul>
     *   <li><b>Gauss-Newton direction</b>: p_gn = −R⁻¹·Qᵀ·F (full Newton step)</li>
     *   <li><b>Scaled gradient direction</b>: p_sd = −(‖Rᵀ·Qᵀ·F‖₂² / ‖J·D⁻¹·Rᵀ·Qᵀ·F‖₂²)·D⁻²·Jᵀ·F</li>
     * </ul>
     * <p>If ‖D·p_gn‖₂ &le; Δ the full Gauss-Newton step is returned.
     * Otherwise the step lies on the dogleg path from p_sd to p_gn, clipped to the
     * trust-region boundary ‖D·p‖₂ = Δ.</p>
     *
     * @param n       dimension of the system
     * @param r       packed upper-triangular R (row-packed: R[i,j] at index i*(2n-i-1)/2+j);
     *                accessed at {@code r[rOff .. rOff+lr-1]}
     * @param rOff    start offset into {@code r}
     * @param lr      length of the packed R region (&ge; n*(n+1)/2)
     * @param diag    scaling diagonal D; accessed at {@code diag[diagOff .. diagOff+n-1]}
     * @param diagOff start offset into {@code diag}
     * @param qtb     Qᵀ·F (first n elements); accessed at {@code qtb[qtbOff .. qtbOff+n-1]}
     * @param qtbOff  start offset into {@code qtb}
     * @param delta   trust-region radius Δ (upper bound on ‖D·p‖₂)
     * @param x       on exit: dogleg step p; accessed at {@code x[xOff .. xOff+n-1]}
     * @param xOff    start offset into {@code x}
     * @param wa1     work array of length n (holds scaled gradient direction);
     *                accessed at {@code wa1[wa1Off .. wa1Off+n-1]}
     * @param wa1Off  start offset into {@code wa1}
     * @param wa2     work array of length n (holds D·p for norm computation; standalone, offset 0)
     */
    public static void dogleg(int n,
                               double[] r, int rOff, int lr,
                               double[] diag, int diagOff,
                               double[] qtb, int qtbOff,
                               double delta,
                               double[] x, int xOff,
                               double[] wa1, int wa1Off,
                               double[] wa2) {
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
            for (int i = j + 1; i < n; i++) { sum += r[rOff + l] * x[xOff + i]; l++; }
            double temp = r[rOff + jj];
            if (temp == zero) {
                // Scan column j of the packed upper-triangular matrix to estimate scale.
                // In 0-indexed packed-by-rows storage, R[i,j] is at index j*(j+1)/2 + i.
                // Step from R[i,j] to R[i+1,j] = j - i.
                l = j * (j + 1) / 2;
                for (int i = 0; i <= j; i++) { temp = Math.max(temp, Math.abs(r[rOff + l])); l += j - i; }
                temp = epsmch * temp;
                if (temp == zero) temp = epsmch;
            }
            x[xOff + j] = (qtb[qtbOff + j] - sum) / temp;
        }

        for (int j = 0; j < n; j++) { wa1[wa1Off + j] = zero; wa2[j] = diag[diagOff + j] * x[xOff + j]; }
        double qnorm = enorm(n, wa2);
        if (qnorm <= delta) return;

        int l = 0;
        for (int j = 0; j < n; j++) {
            double temp = qtb[qtbOff + j];
            for (int i = j; i < n; i++) { wa1[wa1Off + i] += r[rOff + l] * temp; l++; }
            wa1[wa1Off + j] /= diag[diagOff + j];
        }

        double gnorm = enorm(n, wa1, wa1Off);
        double sgnorm = zero, alpha = delta / qnorm;
        if (gnorm != zero) {
            for (int j = 0; j < n; j++) wa1[wa1Off + j] = (wa1[wa1Off + j] / gnorm) / diag[diagOff + j];
            l = 0;
            for (int j = 0; j < n; j++) {
                double sum = zero;
                for (int i = j; i < n; i++) { sum += r[rOff + l] * wa1[wa1Off + i]; l++; }
                wa2[j] = sum;
            }
            double temp = enorm(n, wa2);
            sgnorm = (gnorm / temp) / temp;
            alpha = zero;
            if (sgnorm < delta) {
                double bnorm = enorm(n, qtb, qtbOff);
                temp = (bnorm / gnorm) * (bnorm / qnorm) * (sgnorm / delta);
                temp = temp - (delta / qnorm) * (sgnorm / delta) * (sgnorm / delta)
                        + Math.sqrt((temp - (delta / qnorm)) * (temp - (delta / qnorm))
                        + (one - (delta / qnorm) * (delta / qnorm))
                        * (one - (sgnorm / delta) * (sgnorm / delta)));
                alpha = ((delta / qnorm) * (one - (sgnorm / delta) * (sgnorm / delta))) / temp;
            }
        }

        double temp2 = (one - alpha) * Math.min(sgnorm, delta);
        for (int j = 0; j < n; j++) x[xOff + j] = temp2 * wa1[wa1Off + j] + alpha * x[xOff + j];
    }

    /**
     * Rank-1 update of the lower-trapezoidal factor S arising in the Powell hybrid method
     * (translation of Fortran {@code r1updt} from MINPACK-1).
     *
     * <p>Given the current packed lower-trapezoidal matrix S and vectors u, v, computes
     * the factored form of S + u·vᵀ using a sequence of Givens rotations.  The rotations
     * are accumulated in {@code v} and {@code w} for subsequent use by {@code r1mpyq}.</p>
     *
     * <p>Packed storage: S[i,j] (i&ge;j) is stored at index j*(2m-j-1)/2 + i.</p>
     *
     * @param m     number of rows of S
     * @param n     number of columns of S (n &le; m)
     * @param s     packed lower-trapezoidal S; accessed at {@code s[sOff..]};
     *              <em>modified in place</em> to hold the updated factor
     * @param sOff  start offset into {@code s}
     * @param ls    length of the packed S region (&ge; n*(2m-n+1)/2)
     * @param u     rank-1 update vector of length m; accessed at {@code u[uOff .. uOff+m-1]}
     * @param uOff  start offset into {@code u}
     * @param v     on entry: rank-1 update vector of length n;
     *              on exit: cosine/sine encoding of the applied Givens rotations;
     *              accessed at {@code v[vOff .. vOff+n-1]}
     * @param vOff  start offset into {@code v}
     * @param w     on exit: additional Givens rotation info of length m;
     *              accessed at {@code w[wOff .. wOff+m-1]}
     * @param wOff  start offset into {@code w}
     * @param sing  {@code sing[0]} is set to {@code true} if any diagonal of the updated S is zero
     */
    public static void r1updt(int m, int n,
                               double[] s, int sOff, int ls,
                               double[] u, int uOff,
                               double[] v, int vOff,
                               double[] w, int wOff,
                               boolean[] sing) {
        final double giant = dpmpar(3);
        final double one = 1.0, p5 = 0.5, p25 = 0.25, zero = 0.0;

        int jj = (n * (2 * m - n + 1)) / 2 - (m - n) - 1;
        int l = jj;
        for (int i = n - 1; i < m; i++) { w[wOff + i] = s[sOff + l]; l++; }

        int nm1 = n - 1;
        for (int nmj = 1; nmj <= nm1; nmj++) {
            int j = n - 1 - nmj;
            jj -= (m - j);
            w[wOff + j] = zero;
            if (v[vOff + j] == zero) continue;
            double cos, sin, tau;
            if (Math.abs(v[vOff + n - 1]) >= Math.abs(v[vOff + j])) {
                double tan = v[vOff + j] / v[vOff + n - 1];
                cos = p5 / Math.sqrt(p25 + p25 * tan * tan); sin = cos * tan; tau = sin;
            } else {
                double cotan = v[vOff + n - 1] / v[vOff + j];
                sin = p5 / Math.sqrt(p25 + p25 * cotan * cotan); cos = sin * cotan; tau = one;
                if (Math.abs(cos) * giant > one) tau = one / cos;
            }
            v[vOff + n - 1] = sin * v[vOff + j] + cos * v[vOff + n - 1]; v[vOff + j] = tau;
            l = jj;
            for (int i = j; i < m; i++) {
                double temp = cos * s[sOff + l] - sin * w[wOff + i];
                w[wOff + i] = sin * s[sOff + l] + cos * w[wOff + i];
                s[sOff + l] = temp; l++;
            }
        }

        for (int i = 0; i < m; i++) w[wOff + i] += v[vOff + n - 1] * u[uOff + i];

        sing[0] = false;
        if (nm1 >= 1) {
            jj = 0;
            for (int j = 0; j < nm1; j++) {
                if (w[wOff + j] != zero) {
                    double cos, sin, tau;
                    if (Math.abs(s[sOff + jj]) >= Math.abs(w[wOff + j])) {
                        double tan = w[wOff + j] / s[sOff + jj];
                        cos = p5 / Math.sqrt(p25 + p25 * tan * tan); sin = cos * tan; tau = sin;
                    } else {
                        double cotan = s[sOff + jj] / w[wOff + j];
                        sin = p5 / Math.sqrt(p25 + p25 * cotan * cotan); cos = sin * cotan; tau = one;
                        if (Math.abs(cos) * giant > one) tau = one / cos;
                    }
                    l = jj;
                    for (int i = j; i < m; i++) {
                        double temp = cos * s[sOff + l] + sin * w[wOff + i];
                        w[wOff + i] = -sin * s[sOff + l] + cos * w[wOff + i];
                        s[sOff + l] = temp; l++;
                    }
                    w[wOff + j] = tau;
                }
                if (s[sOff + jj] == zero) sing[0] = true;
                jj += (m - j);
            }
        }

        l = jj;
        for (int i = n - 1; i < m; i++) { s[sOff + l] = w[wOff + i]; l++; }
        if (s[sOff + jj] == zero) sing[0] = true;
    }

    /**
     * Applies the accumulated Givens rotations from {@code r1updt} to a matrix A,
     * computing A·Q in place (translation of Fortran {@code r1mpyq} from MINPACK-1).
     *
     * <p>The rotation sequence Q is encoded in the vectors {@code v} and {@code w} produced
     * by a preceding call to {@code r1updt}.  This is used to keep the accumulated Q factor
     * of the Jacobian consistent after a rank-1 update of R.</p>
     *
     * @param m     number of rows of A
     * @param n     number of columns of A
     * @param a     A in column-major layout; accessed at {@code a[aOff..]};
     *              <em>replaced by A·Q on exit</em>
     * @param aOff  start offset into {@code a}
     * @param lda   leading dimension of {@code a}
     * @param v     Givens rotation info from {@code r1updt}, length n (standalone, offset 0)
     * @param w     Givens rotation info from {@code r1updt};
     *              accessed at {@code w[wOff .. wOff+n-1]}
     * @param wOff  start offset into {@code w}
     */
    public static void r1mpyq(int m, int n, double[] a, int aOff, int lda, double[] v, double[] w, int wOff) {
        final double one = 1.0;
        int nm1 = n - 1;
        if (nm1 < 1) return;

        for (int nmj = 1; nmj <= nm1; nmj++) {
            int j = n - 1 - nmj;
            double cos, sin;
            if (Math.abs(v[j]) > one) { cos = one / v[j]; sin = Math.sqrt(one - cos * cos); }
            else { sin = v[j]; cos = Math.sqrt(one - sin * sin); }
            for (int i = 0; i < m; i++) {
                double temp = cos * a[aOff + i + lda * j] - sin * a[aOff + i + lda * (n - 1)];
                a[aOff + i + lda * (n - 1)] = sin * a[aOff + i + lda * j] + cos * a[aOff + i + lda * (n - 1)];
                a[aOff + i + lda * j] = temp;
            }
        }

        for (int j = 0; j < nm1; j++) {
            double cos, sin;
            if (Math.abs(w[wOff + j]) > one) { cos = one / w[wOff + j]; sin = Math.sqrt(one - cos * cos); }
            else { sin = w[wOff + j]; cos = Math.sqrt(one - sin * sin); }
            for (int i = 0; i < m; i++) {
                double temp = cos * a[aOff + i + lda * j] + sin * a[aOff + i + lda * (n - 1)];
                a[aOff + i + lda * (n - 1)] = -sin * a[aOff + i + lda * j] + cos * a[aOff + i + lda * (n - 1)];
                a[aOff + i + lda * j] = temp;
            }
        }
    }

    /**
     * Updates the upper-triangular factor R by appending a new row w and retriangularising
     * via Givens rotations (translation of Fortran {@code rwupdt} from MINPACK-1).
     *
     * <p>Given R (n×n upper triangular) and a row vector wᵀ, computes the QR factorisation
     * of the augmented matrix [R; wᵀ] and simultaneously applies the same rotations to the
     * right-hand side vector [b; α], so that on exit b holds Qᵀ·[b; α] and R holds the
     * updated upper-triangular factor.</p>
     *
     * @param n     order of R
     * @param r     R in column-major layout ({@code r[i + ldr*j]} = R[i,j]);
     *              <em>updated in place</em>
     * @param ldr   leading dimension of {@code r}
     * @param w     new row to append (length n)
     * @param b     right-hand side vector of length n; updated to Qᵀ·[b; α] on exit
     * @param alpha on entry: scalar (n+1)-th element of the augmented RHS;
     *              on exit: updated value after applying all rotations
     * @param cos   on exit: cosines of the n Givens rotations applied
     * @param sin   on exit: sines of the n Givens rotations applied
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
