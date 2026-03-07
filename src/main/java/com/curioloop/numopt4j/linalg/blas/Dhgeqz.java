/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * LAPACK DHGEQZ: QZ iteration for generalized upper Hessenberg matrix pair (H, T).
 *
 * <p>Computes eigenvalues (alphar, alphai, beta) and optionally Schur vectors Q, Z such that:
 * <pre>
 *   Q^T * H * Z = S  (quasi-upper-triangular)
 *   Q^T * T * Z = P  (upper triangular)
 * </pre>
 * Eigenvalue i = (alphar[i] + i*alphai[i]) / beta[i].
 *
 * <p>Row-major storage: element (i,j) is at array[i*ld + j].
 * Fortran column-major indices (r,c) map to Java row-major [r-1][c-1] = [(r-1)*ld + (c-1)].
 */
interface Dhgeqz {

    /**
     * LAPACK DLAG2 (inlined): Computes eigenvalues of a 2×2 generalized eigenproblem (A, B)
     * with scaling to avoid over-/underflow.
     * Output: out = [s1, s2, wr, wr2, wi]  where eigenvalues = (wr ± wi*i)/s1 and wr2/s2.
     */
    static void dlag2(double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb,
                      double safmin, double[] out) {
        double a11 = A[aOff], a12 = A[aOff + 1];
        double a21 = A[aOff + lda], a22 = A[aOff + lda + 1];
        double b11 = B[bOff], b12 = B[bOff + 1], b22 = B[bOff + ldb + 1];

        double bscale = 1.0 / Math.max(safmin, Math.max(Math.abs(b11), Math.max(Math.abs(b12), Math.abs(b22))));
        double bs11 = b11 * bscale, bs22 = b22 * bscale;

        if (Math.abs(bs11) <= safmin && Math.abs(bs22) <= safmin) {
            out[0] = 0; out[1] = 0; out[2] = a22; out[3] = a11; out[4] = 0; return;
        }
        if (Math.abs(bs11) <= safmin) {
            out[0] = Math.abs(b22); out[1] = out[0]; out[2] = a22;
            out[3] = a11 - a12 * (b12 / b22); out[4] = 0; return;
        }
        if (Math.abs(bs22) <= safmin) {
            out[0] = Math.abs(b11); out[1] = out[0]; out[2] = a11; out[3] = a22; out[4] = 0; return;
        }

        double qa = b11 * b22;
        double qb = -(a11 * b22 + a22 * b11);
        double qc = a11 * a22 - a12 * a21;

        if (Math.abs(qa) < safmin) {
            if (Math.abs(qb) < safmin) { out[0]=0; out[1]=0; out[2]=0; out[3]=0; out[4]=0; }
            else { out[0]=1; out[1]=1; out[2]=-qc/qb; out[3]=0; out[4]=0; }
            return;
        }

        double disc = qb * qb - 4.0 * qa * qc;
        if (disc >= 0.0) {
            double sq = Math.sqrt(disc);
            double r1 = (-qb + sq) / (2.0 * qa), r2 = (-qb - sq) / (2.0 * qa);
            out[0] = 1; out[1] = 1; out[4] = 0;
            if (Math.abs(r1) >= Math.abs(r2)) { out[2] = r1; out[3] = r2; }
            else                               { out[2] = r2; out[3] = r1; }
        } else {
            out[0] = 1; out[1] = 1;
            out[2] = -qb / (2.0 * qa);
            out[4] = Math.sqrt(-disc) / (2.0 * Math.abs(qa));
            out[3] = out[2];
        }
    }

    /**
     * @param job    'E' eigenvalues only, 'S' Schur form
     * @param compq  'N' no Q, 'I' initialize Q=I, 'V' accumulate into Q
     * @param compz  'N' no Z, 'I' initialize Z=I, 'V' accumulate into Z
     * @param n      order of H and T
     * @param ilo    lower index of active submatrix (0-based)
     * @param ihi    upper index of active submatrix (0-based inclusive)
     * @param H      upper Hessenberg matrix (n×n, row-major), overwritten
     * @param hOff   offset into H
     * @param ldh    leading dimension of H
     * @param T      upper triangular matrix (n×n, row-major), overwritten
     * @param tOff   offset into T
     * @param ldt    leading dimension of T
     * @param alphar real parts of alpha (output, length n)
     * @param alphai imaginary parts of alpha (output, length n)
     * @param beta   beta values (output, length n)
     * @param Q      orthogonal Q (n×n, row-major); used if compq != 'N'
     * @param qOff   offset into Q
     * @param ldq    leading dimension of Q
     * @param Z      orthogonal Z (n×n, row-major); used if compz != 'N'
     * @param zOff   offset into Z
     * @param ldz    leading dimension of Z
     * @param work   workspace
     * @param workOff offset into work
     * @param lwork  workspace size; -1 for query
     * @return 0 on success; positive = number of eigenvalues not computed
     */
    static int dhgeqz(char job, char compq, char compz, int n, int ilo, int ihi,
                      double[] H, int hOff, int ldh,
                      double[] T, int tOff, int ldt,
                      double[] alphar, double[] alphai, double[] beta,
                      double[] Q, int qOff, int ldq,
                      double[] Z, int zOff, int ldz,
                      double[] work, int workOff, int lwork) {

        boolean ilschr = (job == 'S');
        boolean ilq    = (compq == 'V' || compq == 'I');
        boolean ilz    = (compz == 'V' || compz == 'I');

        if (lwork == -1) { work[workOff] = max(1, n); return 0; }
        if (n <= 0) return 0;

        if (compq == 'I') BLAS.dlaset('A', n, n, 0.0, 1.0, Q, qOff, ldq);
        if (compz == 'I') BLAS.dlaset('A', n, n, 0.0, 1.0, Z, zOff, ldz);

        double safmin = BLAS.safmin();
        double safmax = 1.0 / safmin;
        double ulp    = BLAS.dlamch('E') * BLAS.dlamch('B');
        double safety = 100.0;

        // Compute norms of active submatrix
        double anorm = 0.0, bnorm = 0.0;
        for (int i = ilo; i <= ihi; i++) {
            for (int j = max(ilo, i-1); j <= ihi; j++) anorm += H[hOff + i*ldh + j] * H[hOff + i*ldh + j];
            for (int j = i; j <= ihi; j++) bnorm += T[tOff + i*ldt + j] * T[tOff + i*ldt + j];
        }
        anorm = sqrt(anorm); bnorm = sqrt(bnorm);
        double atol = max(safmin, ulp * anorm);
        double btol = max(safmin, ulp * bnorm);
        double ascale = 1.0 / max(safmin, anorm);
        double bscale = 1.0 / max(safmin, bnorm);

        // Set eigenvalues for rows/cols > ihi (already in triangular form)
        for (int j = ihi + 1; j < n; j++) {
            double tval = T[tOff + j*ldt + j];
            if (tval < 0.0) {
                if (ilschr) {
                    for (int jr = 0; jr <= j; jr++) {
                        H[hOff + jr*ldh + j] = -H[hOff + jr*ldh + j];
                        T[tOff + jr*ldt + j] = -T[tOff + jr*ldt + j];
                    }
                } else {
                    H[hOff + j*ldh + j] = -H[hOff + j*ldh + j];
                    T[tOff + j*ldt + j] = -T[tOff + j*ldt + j];
                }
                if (ilz) for (int jr = 0; jr < n; jr++) Z[zOff + jr*ldz + j] = -Z[zOff + jr*ldz + j];
            }
            alphar[j] = H[hOff + j*ldh + j];
            alphai[j] = 0.0;
            beta[j]   = T[tOff + j*ldt + j];
        }

        if (ihi < ilo) {
            // Also handle rows < ilo
            for (int j = 0; j < ilo; j++) {
                double tval = T[tOff + j*ldt + j];
                if (tval < 0.0) {
                    if (ilschr) {
                        for (int jr = 0; jr <= j; jr++) {
                            H[hOff + jr*ldh + j] = -H[hOff + jr*ldh + j];
                            T[tOff + jr*ldt + j] = -T[tOff + jr*ldt + j];
                        }
                    } else {
                        H[hOff + j*ldh + j] = -H[hOff + j*ldh + j];
                        T[tOff + j*ldt + j] = -T[tOff + j*ldt + j];
                    }
                    if (ilz) for (int jr = 0; jr < n; jr++) Z[zOff + jr*ldz + j] = -Z[zOff + jr*ldz + j];
                }
                alphar[j] = H[hOff + j*ldh + j];
                alphai[j] = 0.0;
                beta[j]   = T[tOff + j*ldt + j];
            }
            work[workOff] = n;
            return 0;
        }

        // Main QZ iteration
        int ilast  = ihi;
        int ifrstm = ilschr ? 0 : ilo;
        int ilastm = ilschr ? n - 1 : ihi;
        int iiter  = 0;
        double eshift = 0.0;
        int maxit  = 30 * (ihi - ilo + 1);
        int ifirst = ilo; // will be set in loop

        double[] csr = new double[3]; // [c, s, r] — reused for all Givens rotations
        double[] lag2out = new double[5];
        double[] v = new double[3];

        mainLoop:
        for (int jiter = 1; jiter <= maxit; jiter++) {

            // --- Check for deflation at ilast ---
            if (ilast == ilo) {
                // 1x1 block: deflate
                deflate1x1(H, hOff, ldh, T, tOff, ldt, Q, qOff, ldq, Z, zOff, ldz,
                           alphar, alphai, beta, ilast, ifrstm, ilschr, ilq, ilz, n);
                ilast--;
                if (ilast < ilo) break mainLoop;
                iiter = 0; eshift = 0.0;
                if (!ilschr) { ilastm = ilast; if (ifrstm > ilast) ifrstm = ilo; }
                continue;
            }

            // Check H[ilast, ilast-1] for deflation
            if (abs(H[hOff + ilast*ldh + ilast-1]) <= atol) {
                H[hOff + ilast*ldh + ilast-1] = 0.0;
                deflate1x1(H, hOff, ldh, T, tOff, ldt, Q, qOff, ldq, Z, zOff, ldz,
                           alphar, alphai, beta, ilast, ifrstm, ilschr, ilq, ilz, n);
                ilast--;
                if (ilast < ilo) break mainLoop;
                iiter = 0; eshift = 0.0;
                if (!ilschr) { ilastm = ilast; if (ifrstm > ilast) ifrstm = ilo; }
                continue;
            }

            // Check T[ilast, ilast] for zero (infinite eigenvalue)
            if (abs(T[tOff + ilast*ldt + ilast]) <= btol) {
                T[tOff + ilast*ldt + ilast] = 0.0;
                // Chase the zero off the diagonal of T
                chaseZeroT(H, hOff, ldh, T, tOff, ldt, Q, qOff, ldq, Z, zOff, ldz,
                           ilast, ilo, ifrstm, ilastm, ilschr, ilq, ilz, n, csr);
                deflate1x1(H, hOff, ldh, T, tOff, ldt, Q, qOff, ldq, Z, zOff, ldz,
                           alphar, alphai, beta, ilast, ifrstm, ilschr, ilq, ilz, n);
                ilast--;
                if (ilast < ilo) break mainLoop;
                iiter = 0; eshift = 0.0;
                if (!ilschr) { ilastm = ilast; if (ifrstm > ilast) ifrstm = ilo; }
                continue;
            }

            // Scan for interior deflation
            boolean deflated = false;
            for (int j = ilast - 1; j >= ilo; j--) {
                boolean ilazro = (j == ilo) || (abs(H[hOff + j*ldh + j-1]) <= atol);
                if (ilazro && j > ilo && j < ilast - 1) H[hOff + j*ldh + j-1] = 0.0;

                if (abs(T[tOff + j*ldt + j]) < btol) {
                    T[tOff + j*ldt + j] = 0.0;
                    if (ilazro) {
                        // Chase zero in H column
                        for (int jch = j; jch <= ilast - 1; jch++) {
                            double temp = H[hOff + jch*ldh + jch];
                            Dlartg.dlartg(temp, H[hOff + (jch+1)*ldh + jch], csr, 0);
                            double c = csr[0], s = csr[1];
                            H[hOff + jch*ldh + jch] = c * temp + s * H[hOff + (jch+1)*ldh + jch];
                            H[hOff + (jch+1)*ldh + jch] = 0.0;
                            // Apply to rows jch and jch+1, columns jch+1..ilastm
                            for (int jc = jch + 1; jc <= ilastm; jc++) {
                                double h1 = H[hOff + jch*ldh + jc], h2 = H[hOff + (jch+1)*ldh + jc];
                                H[hOff + jch*ldh + jc]     =  c * h1 + s * h2;
                                H[hOff + (jch+1)*ldh + jc] = -s * h1 + c * h2;
                                double t1 = T[tOff + jch*ldt + jc], t2 = T[tOff + (jch+1)*ldt + jc];
                                T[tOff + jch*ldt + jc]     =  c * t1 + s * t2;
                                T[tOff + (jch+1)*ldt + jc] = -s * t1 + c * t2;
                            }
                            if (ilq) for (int jr = 0; jr < n; jr++) {
                                double q1 = Q[qOff + jr*ldq + jch], q2 = Q[qOff + jr*ldq + jch+1];
                                Q[qOff + jr*ldq + jch]   =  c * q1 + s * q2;
                                Q[qOff + jr*ldq + jch+1] = -s * q1 + c * q2;
                            }
                            if (abs(T[tOff + (jch+1)*ldt + jch+1]) >= btol) {
                                ifirst = jch + 1;
                                deflated = true;
                                break;
                            }
                            T[tOff + (jch+1)*ldt + jch+1] = 0.0;
                        }
                        if (!deflated) {
                            // All T diagonal zeros: infinite eigenvalue chain
                            deflate1x1(H, hOff, ldh, T, tOff, ldt, Q, qOff, ldq, Z, zOff, ldz,
                                       alphar, alphai, beta, ilast, ifrstm, ilschr, ilq, ilz, n);
                            ilast--;
                            if (ilast < ilo) break mainLoop;
                            iiter = 0; eshift = 0.0;
                            if (!ilschr) { ilastm = ilast; if (ifrstm > ilast) ifrstm = ilo; }
                            continue mainLoop;
                        }
                        break;
                    } else {
                        // Chase zero in T row
                        for (int jch = j; jch <= ilast - 1; jch++) {
                            double temp = T[tOff + jch*ldt + jch+1];
                            Dlartg.dlartg(temp, T[tOff + (jch+1)*ldt + jch+1], csr, 0);
                            double c = csr[0], s = csr[1];
                            T[tOff + jch*ldt + jch+1]     = c * temp + s * T[tOff + (jch+1)*ldt + jch+1];
                            T[tOff + (jch+1)*ldt + jch+1] = 0.0;
                            for (int jc = jch + 2; jc <= ilastm; jc++) {
                                double t1 = T[tOff + jch*ldt + jc], t2 = T[tOff + (jch+1)*ldt + jc];
                                T[tOff + jch*ldt + jc]     =  c * t1 + s * t2;
                                T[tOff + (jch+1)*ldt + jc] = -s * t1 + c * t2;
                            }
                            // Apply to H rows jch and jch+1, columns jch-1..ilastm
                            for (int jc = max(0, jch - 1); jc <= ilastm; jc++) {
                                double h1 = H[hOff + jch*ldh + jc], h2 = H[hOff + (jch+1)*ldh + jc];
                                H[hOff + jch*ldh + jc]     =  c * h1 + s * h2;
                                H[hOff + (jch+1)*ldh + jc] = -s * h1 + c * h2;
                            }
                            if (ilq) for (int jr = 0; jr < n; jr++) {
                                double q1 = Q[qOff + jr*ldq + jch], q2 = Q[qOff + jr*ldq + jch+1];
                                Q[qOff + jr*ldq + jch]   =  c * q1 + s * q2;
                                Q[qOff + jr*ldq + jch+1] = -s * q1 + c * q2;
                            }
                            // Now zero H[jch+1, jch] with a right rotation
                            double temp2 = H[hOff + (jch+1)*ldh + jch];
                            Dlartg.dlartg(temp2, H[hOff + (jch+1)*ldh + jch-1], csr, 0);
                            c = csr[0]; s = csr[1];
                            H[hOff + (jch+1)*ldh + jch]   = c * temp2 + s * H[hOff + (jch+1)*ldh + jch-1];
                            H[hOff + (jch+1)*ldh + jch-1] = 0.0;
                            for (int jr = ifrstm; jr <= jch + 1; jr++) {
                                double h1 = H[hOff + jr*ldh + jch], h2 = H[hOff + jr*ldh + jch-1];
                                H[hOff + jr*ldh + jch]   =  c * h1 + s * h2;
                                H[hOff + jr*ldh + jch-1] = -s * h1 + c * h2;
                            }
                            for (int jr = ifrstm; jr <= jch; jr++) {
                                double t1 = T[tOff + jr*ldt + jch], t2 = T[tOff + jr*ldt + jch-1];
                                T[tOff + jr*ldt + jch]   =  c * t1 + s * t2;
                                T[tOff + jr*ldt + jch-1] = -s * t1 + c * t2;
                            }
                            if (ilz) for (int jr = 0; jr < n; jr++) {
                                double z1 = Z[zOff + jr*ldz + jch], z2 = Z[zOff + jr*ldz + jch-1];
                                Z[zOff + jr*ldz + jch]   =  c * z1 + s * z2;
                                Z[zOff + jr*ldz + jch-1] = -s * z1 + c * z2;
                            }
                        }
                        deflate1x1(H, hOff, ldh, T, tOff, ldt, Q, qOff, ldq, Z, zOff, ldz,
                                   alphar, alphai, beta, ilast, ifrstm, ilschr, ilq, ilz, n);
                        ilast--;
                        if (ilast < ilo) break mainLoop;
                        iiter = 0; eshift = 0.0;
                        if (!ilschr) { ilastm = ilast; if (ifrstm > ilast) ifrstm = ilo; }
                        continue mainLoop;
                    }
                } else if (ilazro) {
                    ifirst = j;
                    deflated = true;
                    break;
                }
            }

            if (!deflated) {
                // No deflation found: convergence failure
                work[workOff] = n;
                return ilast + 1;
            }

            // --- Compute shift ---
            iiter++;
            if (!ilschr) ifrstm = ifirst;

            double s1, wr, wi, s2, wr2;
            if (iiter % 10 == 0) {
                // Exceptional shift
                double h_il_ilm1 = H[hOff + ilast*ldh + ilast-1];
                double t_ilm1    = T[tOff + (ilast-1)*ldt + ilast-1];
                if (abs((double) maxit * safmin * h_il_ilm1) < abs(t_ilm1)) {
                    eshift = h_il_ilm1 / t_ilm1;
                } else {
                    eshift += 1.0 / (safmin * maxit);
                }
                s1 = 1.0; wr = eshift; wi = 0.0; s2 = 1.0; wr2 = eshift;
            } else {
                // Normal double shift from bottom 2x2
                dlag2(H, hOff + (ilast-1)*ldh + ilast-1, ldh,
                      T, tOff + (ilast-1)*ldt + ilast-1, ldt,
                      safmin * safety, lag2out);
                s1 = lag2out[0]; s2 = lag2out[1]; wr = lag2out[2]; wr2 = lag2out[3]; wi = lag2out[4];

                // Choose shift closer to H[ilast,ilast]/T[ilast,ilast]
                double tii = T[tOff + ilast*ldt + ilast];
                double hii = H[hOff + ilast*ldh + ilast];
                if (abs(wr / s1 * tii - hii) > abs(wr2 / s2 * tii - hii)) {
                    double tmp = wr; wr = wr2; wr2 = tmp;
                    tmp = s1; s1 = s2; s2 = tmp;
                }
                // temp = max(s1, safmin * max(1.0, max(abs(wr), abs(wi)))) — threshold for complex shift
                if (wi != 0.0) {
                    // Complex shift: use double-shift Householder
                    ilast = applyDoubleShift(H, hOff, ldh, T, tOff, ldt, Q, qOff, ldq, Z, zOff, ldz,
                                     n, ilo, ifirst, ilast, ifrstm, ilastm,
                                     ascale, bscale, s1, s2, wr, wr2, wi,
                                     ilschr, ilq, ilz, safmin, safety, v, csr,
                                     lag2out, atol, btol, alphar, alphai, beta);
                    // After double shift, check if 2x2 block deflated
                    if (ilast < ilo) break mainLoop;
                    iiter = 0; eshift = 0.0;
                    if (!ilschr) { ilastm = ilast; if (ifrstm > ilast) ifrstm = ilo; }
                    continue;
                }
                // Scale real shift
                double scaleVal = min(1.0, min(1.0 / max(safmin, anorm), 0.5 * safmax) / max(1.0, s1));
                double scaleWr  = min(1.0, min(1.0 / max(safmin, bnorm), 0.5 * safmax) / max(1.0, abs(wr)));
                scaleVal = min(scaleVal, scaleWr);
                s1 *= scaleVal; wr *= scaleVal;
            }

            // --- Single real shift QZ step ---
            // Find istart: look for a small subdiagonal element
            int istart = ifirst;
            for (int j = ilast - 1; j >= ifirst + 1; j--) {
                double temp  = abs(s1 * H[hOff + j*ldh + j-1]);
                double temp2 = abs(s1 * H[hOff + j*ldh + j] - wr * T[tOff + j*ldt + j]);
                double tempr = max(temp, temp2);
                if (tempr < 1.0 && tempr != 0.0) { temp /= tempr; temp2 /= tempr; }
                if (abs(ascale * H[hOff + (j+1)*ldh + j]) * temp <= ascale * atol * temp2) {
                    istart = j;
                    break;
                }
            }

            // Compute initial rotation
            double temp  = s1 * H[hOff + istart*ldh + istart] - wr * T[tOff + istart*ldt + istart];
            double temp2 = s1 * H[hOff + (istart+1)*ldh + istart];
            Dlartg.dlartg(temp, temp2, csr, 0);
            double c = csr[0], s = csr[1];

            // Chase the bulge
            for (int j = istart; j <= ilast - 1; j++) {
                if (j > istart) {
                    temp = H[hOff + j*ldh + j-1];
                    Dlartg.dlartg(temp, H[hOff + (j+1)*ldh + j-1], csr, 0);
                    c = csr[0]; s = csr[1];
                    H[hOff + j*ldh + j-1] = c * temp + s * H[hOff + (j+1)*ldh + j-1];
                    H[hOff + (j+1)*ldh + j-1] = 0.0;
                }
                // Apply left rotation to rows j and j+1
                for (int jc = j; jc <= ilastm; jc++) {
                    double h1 = H[hOff + j*ldh + jc], h2 = H[hOff + (j+1)*ldh + jc];
                    H[hOff + j*ldh + jc]     =  c * h1 + s * h2;
                    H[hOff + (j+1)*ldh + jc] = -s * h1 + c * h2;
                    double t1 = T[tOff + j*ldt + jc], t2 = T[tOff + (j+1)*ldt + jc];
                    T[tOff + j*ldt + jc]     =  c * t1 + s * t2;
                    T[tOff + (j+1)*ldt + jc] = -s * t1 + c * t2;
                }
                if (ilq) for (int jr = 0; jr < n; jr++) {
                    double q1 = Q[qOff + jr*ldq + j], q2 = Q[qOff + jr*ldq + j+1];
                    Q[qOff + jr*ldq + j]   =  c * q1 + s * q2;
                    Q[qOff + jr*ldq + j+1] = -s * q1 + c * q2;
                }
                // Restore upper triangular T: zero T[j+1, j]
                temp = T[tOff + (j+1)*ldt + j+1];
                Dlartg.dlartg(temp, T[tOff + (j+1)*ldt + j], csr, 0);
                c = csr[0]; s = csr[1];
                T[tOff + (j+1)*ldt + j+1] = c * temp + s * T[tOff + (j+1)*ldt + j];
                T[tOff + (j+1)*ldt + j]   = 0.0;
                // Apply right rotation to columns j and j+1
                for (int jr = ifrstm; jr <= min(j + 2, ilast); jr++) {
                    double h1 = H[hOff + jr*ldh + j+1], h2 = H[hOff + jr*ldh + j];
                    H[hOff + jr*ldh + j+1] =  c * h1 + s * h2;
                    H[hOff + jr*ldh + j]   = -s * h1 + c * h2;
                }
                for (int jr = ifrstm; jr <= j; jr++) {
                    double t1 = T[tOff + jr*ldt + j+1], t2 = T[tOff + jr*ldt + j];
                    T[tOff + jr*ldt + j+1] =  c * t1 + s * t2;
                    T[tOff + jr*ldt + j]   = -s * t1 + c * t2;
                }
                if (ilz) for (int jr = 0; jr < n; jr++) {
                    double z1 = Z[zOff + jr*ldz + j+1], z2 = Z[zOff + jr*ldz + j];
                    Z[zOff + jr*ldz + j+1] =  c * z1 + s * z2;
                    Z[zOff + jr*ldz + j]   = -s * z1 + c * z2;
                }
            }
        } // end mainLoop

        // Handle rows < ilo
        for (int j = 0; j < ilo; j++) {
            double tval = T[tOff + j*ldt + j];
            if (tval < 0.0) {
                if (ilschr) {
                    for (int jr = 0; jr <= j; jr++) {
                        H[hOff + jr*ldh + j] = -H[hOff + jr*ldh + j];
                        T[tOff + jr*ldt + j] = -T[tOff + jr*ldt + j];
                    }
                } else {
                    H[hOff + j*ldh + j] = -H[hOff + j*ldh + j];
                    T[tOff + j*ldt + j] = -T[tOff + j*ldt + j];
                }
                if (ilz) for (int jr = 0; jr < n; jr++) Z[zOff + jr*ldz + j] = -Z[zOff + jr*ldz + j];
            }
            alphar[j] = H[hOff + j*ldh + j];
            alphai[j] = 0.0;
            beta[j]   = T[tOff + j*ldt + j];
        }

        work[workOff] = n;
        return 0;
    }

    /** Deflate a 1x1 block at position ilast: ensure T[ilast,ilast] >= 0, then record eigenvalue. */
    static void deflate1x1(double[] H, int hOff, int ldh,
                           double[] T, int tOff, int ldt,
                           double[] Q, int qOff, int ldq,
                           double[] Z, int zOff, int ldz,
                           double[] alphar, double[] alphai, double[] beta,
                           int ilast, int ifrstm, boolean ilschr, boolean ilq, boolean ilz, int n) {
        double tval = T[tOff + ilast*ldt + ilast];
        if (tval < 0.0) {
            if (ilschr) {
                for (int jr = ifrstm; jr <= ilast; jr++) {
                    H[hOff + jr*ldh + ilast] = -H[hOff + jr*ldh + ilast];
                    T[tOff + jr*ldt + ilast] = -T[tOff + jr*ldt + ilast];
                }
            } else {
                H[hOff + ilast*ldh + ilast] = -H[hOff + ilast*ldh + ilast];
                T[tOff + ilast*ldt + ilast] = -T[tOff + ilast*ldt + ilast];
            }
            if (ilz) for (int jr = 0; jr < n; jr++) Z[zOff + jr*ldz + ilast] = -Z[zOff + jr*ldz + ilast];
        }
        alphar[ilast] = H[hOff + ilast*ldh + ilast];
        alphai[ilast] = 0.0;
        beta[ilast]   = T[tOff + ilast*ldt + ilast];
    }

    /** Chase a zero off T[ilast,ilast] using a right rotation. */
    static void chaseZeroT(double[] H, int hOff, int ldh,
                           double[] T, int tOff, int ldt,
                           double[] Q, int qOff, int ldq,
                           double[] Z, int zOff, int ldz,
                           int ilast, int ilo, int ifrstm, int ilastm,
                           boolean ilschr, boolean ilq, boolean ilz, int n,
                           double[] csr) {
        double temp = H[hOff + ilast*ldh + ilast];
        Dlartg.dlartg(temp, H[hOff + ilast*ldh + ilast-1], csr, 0);
        double c = csr[0], s = csr[1];
        H[hOff + ilast*ldh + ilast]   = c * temp + s * H[hOff + ilast*ldh + ilast-1];
        H[hOff + ilast*ldh + ilast-1] = 0.0;
        for (int jr = ifrstm; jr < ilast; jr++) {
            double h1 = H[hOff + jr*ldh + ilast], h2 = H[hOff + jr*ldh + ilast-1];
            H[hOff + jr*ldh + ilast]   =  c * h1 + s * h2;
            H[hOff + jr*ldh + ilast-1] = -s * h1 + c * h2;
            double t1 = T[tOff + jr*ldt + ilast], t2 = T[tOff + jr*ldt + ilast-1];
            T[tOff + jr*ldt + ilast]   =  c * t1 + s * t2;
            T[tOff + jr*ldt + ilast-1] = -s * t1 + c * t2;
        }
        if (ilz) for (int jr = 0; jr < n; jr++) {
            double z1 = Z[zOff + jr*ldz + ilast], z2 = Z[zOff + jr*ldz + ilast-1];
            Z[zOff + jr*ldz + ilast]   =  c * z1 + s * z2;
            Z[zOff + jr*ldz + ilast-1] = -s * z1 + c * z2;
        }
    }

    /**
     * Apply double-shift (complex shift) QZ step using Householder reflectors.
     * Follows LAPACK dhgeqz lines 875-1313 (the complex-shift branch).
     * Returns the new value of ilast (decremented by 2 if a 2x2 block was deflated).
     */
    static int applyDoubleShift(
            double[] H, int hOff, int ldh, double[] T, int tOff, int ldt,
            double[] Q, int qOff, int ldq, double[] Z, int zOff, int ldz,
            int n, int ilo, int ifirst, int ilast, int ifrstm, int ilastm,
            double ascale, double bscale, double s1, double s2, double wr, double wr2, double wi,
            boolean ilschr, boolean ilq, boolean ilz, double safmin, double safety,
            double[] v, double[] csr, double[] lag2out,
            double atol, double btol,
            double[] alphar, double[] alphai, double[] beta) {

        if (ifirst + 1 == ilast) {
            // 2x2 block: use SVD-based approach (LAPACK lines 876-1063)
            double b11 = T[tOff + (ilast-1)*ldt + ilast-1];
            double b12 = T[tOff + (ilast-1)*ldt + ilast];
            double b22 = T[tOff + ilast*ldt + ilast];

            // Compute SVD of 2x2 T block
            double[] sv = new double[6];
            BLAS.dlasv2(b11, b12, b22, sv, 0);
            // sv = [ssmin, ssmax, snr, csr, snl, csl]
            double b22sv = sv[0], b11sv = sv[1];
            double sr = sv[2], cr = sv[3], sl = sv[4], cl = sv[5];

            if (b11sv < 0.0) { cr = -cr; sr = -sr; b11sv = -b11sv; b22sv = -b22sv; }

            // Apply left rotation (cl, sl) to rows ilast-1 and ilast
            BLAS.drot(ilastm - ifirst + 1,
                      H, hOff + (ilast-1)*ldh + ifirst, 1,
                      H, hOff + ilast*ldh + ifirst, 1, cl, sl);
            // Apply right rotation (cr, sr) to columns ilast-1 and ilast
            BLAS.drot(ilast - ifrstm + 1,
                      H, hOff + ifrstm*ldh + ilast-1, ldh,
                      H, hOff + ifrstm*ldh + ilast, ldh, cr, sr);

            if (ilast < ilastm)
                BLAS.drot(ilastm - ilast,
                          T, tOff + (ilast-1)*ldt + ilast+1, 1,
                          T, tOff + ilast*ldt + ilast+1, 1, cl, sl);
            if (ifrstm < ilast - 1)
                BLAS.drot(ifirst - ifrstm,
                          T, tOff + ifrstm*ldt + ilast-1, ldt,
                          T, tOff + ifrstm*ldt + ilast, ldt, cr, sr);

            if (ilq) BLAS.drot(n, Q, qOff + (ilast-1), ldq, Q, qOff + ilast, ldq, cl, sl);
            if (ilz) BLAS.drot(n, Z, zOff + (ilast-1), ldz, Z, zOff + ilast, ldz, cr, sr);

            T[tOff + (ilast-1)*ldt + ilast-1] = b11sv;
            T[tOff + (ilast-1)*ldt + ilast]   = 0.0;
            T[tOff + ilast*ldt + ilast-1]      = 0.0;
            T[tOff + ilast*ldt + ilast]        = b22sv;

            if (b22sv < 0.0) {
                for (int jr = ifrstm; jr <= ilast; jr++) {
                    H[hOff + jr*ldh + ilast] = -H[hOff + jr*ldh + ilast];
                    T[tOff + jr*ldt + ilast] = -T[tOff + jr*ldt + ilast];
                }
                if (ilz) for (int jr = 0; jr < n; jr++) Z[zOff + jr*ldz + ilast] = -Z[zOff + jr*ldz + ilast];
            }

            // Recompute eigenvalues of the 2x2 block
            dlag2(H, hOff + (ilast-1)*ldh + ilast-1, ldh,
                  T, tOff + (ilast-1)*ldt + ilast-1, ldt,
                  safmin * safety, lag2out);
            double ls1 = lag2out[0], ls2 = lag2out[1], lwr = lag2out[2], lwr2 = lag2out[3], lwi = lag2out[4];

            if (lwi == 0.0) {
                // Real eigenvalues after all
                alphar[ilast-1] = lwr / ls1 * T[tOff + (ilast-1)*ldt + ilast-1];
                alphai[ilast-1] = 0.0;
                beta[ilast-1]   = T[tOff + (ilast-1)*ldt + ilast-1];
                alphar[ilast]   = lwr2 / ls2 * T[tOff + ilast*ldt + ilast];
                alphai[ilast]   = 0.0;
                beta[ilast]     = T[tOff + ilast*ldt + ilast];
            } else {
                // Complex conjugate pair
                double s1inv = 1.0 / ls1;
                // Compute the complex rotation to diagonalize the 2x2 block
                double a11 = H[hOff + (ilast-1)*ldh + ilast-1];
                double a21 = H[hOff + ilast*ldh + ilast-1];
                double a12 = H[hOff + (ilast-1)*ldh + ilast];
                double a22 = H[hOff + ilast*ldh + ilast];
                double tb11 = T[tOff + (ilast-1)*ldt + ilast-1];
                double tb22 = T[tOff + ilast*ldt + ilast];

                double c11r = ls1 * a11 - lwr * tb11;
                double c11i = -lwi * tb11;
                double c12  = ls1 * a12;
                double c21  = ls1 * a21;
                double c22r = ls1 * a22 - lwr * tb22;
                double c22i = -lwi * tb22;

                double cz, szr, szi;
                if (abs(c11r) + abs(c11i) + abs(c12) > abs(c21) + abs(c22r) + abs(c22i)) {
                    double t1 = hypot(c12, hypot(c11r, c11i));
                    cz = c12 / t1; szr = -c11r / t1; szi = -c11i / t1;
                } else {
                    cz = hypot(c22r, c22i);
                    if (cz <= safmin) { cz = 0.0; szr = 1.0; szi = 0.0; }
                    else {
                        double tempr = c22r / cz, tempi = c22i / cz;
                        double t1 = hypot(cz, c21);
                        cz = cz / t1; szr = -c21 * tempr / t1; szi = c21 * tempi / t1;
                    }
                }

                double an = abs(a11) + abs(a12) + abs(a21) + abs(a22);
                double bn = abs(tb11) + abs(tb22);
                double wabs = abs(lwr) + abs(lwi);
                double cq, sqr, sqi;
                if (ls1 * an > wabs * bn) {
                    cq = cz * tb11; sqr = szr * tb22; sqi = -szi * tb22;
                } else {
                    double a1r = cz * a11 + szr * a12, a1i = szi * a12;
                    double a2r = cz * a21 + szr * a22, a2i = szi * a22;
                    cq = hypot(a1r, a1i);
                    if (cq <= safmin) { cq = 0.0; sqr = 1.0; sqi = 0.0; }
                    else {
                        double tempr = a1r / cq, tempi = a1i / cq;
                        sqr = tempr * a2r + tempi * a2i;
                        sqi = tempi * a2r - tempr * a2i;
                    }
                }
                double t1 = hypot(cq, hypot(sqr, sqi));
                cq /= t1; sqr /= t1; sqi /= t1;

                double tempr = sqr * szr - sqi * szi;
                double tempi = sqr * szi + sqi * szr;
                double b1r = cq * cz * tb11 + tempr * tb22;
                double b1i = tempi * tb22;
                double b1a = hypot(b1r, b1i);
                double b2r = cq * cz * tb22 + tempr * tb11;
                double b2i = -tempi * tb11;
                double b2a = hypot(b2r, b2i);

                beta[ilast-1]   = b1a;
                beta[ilast]     = b2a;
                alphar[ilast-1] = (lwr * b1a) * s1inv;
                alphai[ilast-1] = (lwi * b1a) * s1inv;
                alphar[ilast]   = (lwr * b2a) * s1inv;
                alphai[ilast]   = -(lwi * b2a) * s1inv;
            }
            // Deflate 2 eigenvalues: return new ilast (decremented by 2)
            return ilast - 2;
        }

        // General case: block size > 2, use 3-vector Householder
        // Compute the shift vector v[0..2]
        double ad11 = (ascale * H[hOff + (ilast-1)*ldh + ilast-1]) / (bscale * T[tOff + (ilast-1)*ldt + ilast-1]);
        double ad21 = (ascale * H[hOff + ilast*ldh + ilast-1])     / (bscale * T[tOff + (ilast-1)*ldt + ilast-1]);
        double ad12 = (ascale * H[hOff + (ilast-1)*ldh + ilast])   / (bscale * T[tOff + ilast*ldt + ilast]);
        double ad22 = (ascale * H[hOff + ilast*ldh + ilast])       / (bscale * T[tOff + ilast*ldt + ilast]);
        double u12  = T[tOff + (ilast-1)*ldt + ilast] / T[tOff + ilast*ldt + ilast];
        double ad11l = (ascale * H[hOff + ifirst*ldh + ifirst])     / (bscale * T[tOff + ifirst*ldt + ifirst]);
        double ad21l = (ascale * H[hOff + (ifirst+1)*ldh + ifirst]) / (bscale * T[tOff + ifirst*ldt + ifirst]);
        double ad12l = (ascale * H[hOff + ifirst*ldh + ifirst+1])   / (bscale * T[tOff + (ifirst+1)*ldt + ifirst+1]);
        double ad22l = (ascale * H[hOff + (ifirst+1)*ldh + ifirst+1]) / (bscale * T[tOff + (ifirst+1)*ldt + ifirst+1]);
        double ad32l = (ascale * H[hOff + (ifirst+2)*ldh + ifirst+1]) / (bscale * T[tOff + (ifirst+1)*ldt + ifirst+1]);
        double u12l  = T[tOff + ifirst*ldt + ifirst+1] / T[tOff + (ifirst+1)*ldt + ifirst+1];

        v[0] = (ad11 - ad11l) * (ad22 - ad11l) - ad12 * ad21 + ad21 * u12 * ad11l + (ad12l - ad11l * u12l) * ad21l;
        v[1] = ((ad22l - ad11l) - ad21l * u12l - (ad11 - ad11l) - (ad22 - ad11l) + ad21 * u12) * ad21l;
        v[2] = ad32l * ad21l;

        // Compute Householder reflector for v
        double[] tau = new double[1];
        double beta0 = BLAS.dlarfg(3, v[0], v, 1, 1, tau, 0);
        v[0] = 1.0;

        // Chase the bulge
        for (int j = ifirst; j <= ilast - 2; j++) {
            if (j > ifirst) {
                v[0] = H[hOff + j*ldh + j-1];
                v[1] = H[hOff + (j+1)*ldh + j-1];
                v[2] = H[hOff + (j+2)*ldh + j-1];
                beta0 = BLAS.dlarfg(3, v[0], v, 1, 1, tau, 0);
                v[0] = 1.0;
                H[hOff + j*ldh + j-1]     = beta0;
                H[hOff + (j+1)*ldh + j-1] = 0.0;
                H[hOff + (j+2)*ldh + j-1] = 0.0;
            }

            // Apply Householder from left to H and T
            double tauVal = tau[0];
            for (int jc = j; jc <= ilastm; jc++) {
                double sum = H[hOff + j*ldh + jc] + v[1] * H[hOff + (j+1)*ldh + jc] + v[2] * H[hOff + (j+2)*ldh + jc];
                sum *= tauVal;
                H[hOff + j*ldh + jc]     -= sum;
                H[hOff + (j+1)*ldh + jc] -= sum * v[1];
                H[hOff + (j+2)*ldh + jc] -= sum * v[2];
                double sum2 = T[tOff + j*ldt + jc] + v[1] * T[tOff + (j+1)*ldt + jc] + v[2] * T[tOff + (j+2)*ldt + jc];
                sum2 *= tauVal;
                T[tOff + j*ldt + jc]     -= sum2;
                T[tOff + (j+1)*ldt + jc] -= sum2 * v[1];
                T[tOff + (j+2)*ldt + jc] -= sum2 * v[2];
            }
            if (ilq) for (int jr = 0; jr < n; jr++) {
                double sum = Q[qOff + jr*ldq + j] + v[1] * Q[qOff + jr*ldq + j+1] + v[2] * Q[qOff + jr*ldq + j+2];
                sum *= tauVal;
                Q[qOff + jr*ldq + j]   -= sum;
                Q[qOff + jr*ldq + j+1] -= sum * v[1];
                Q[qOff + jr*ldq + j+2] -= sum * v[2];
            }

            // Restore upper triangular T: zero T[j+1,j] and T[j+2,j]
            // Use pivoted elimination
            double w11, w21, w12, w22, u1, u2;
            boolean ilpivt = false;
            double temp  = Math.max(abs(T[tOff + (j+1)*ldt + j+1]), abs(T[tOff + (j+1)*ldt + j+2]));
            double temp2 = Math.max(abs(T[tOff + (j+2)*ldt + j+1]), abs(T[tOff + (j+2)*ldt + j+2]));
            double scale;
            if (Math.max(temp, temp2) < safmin) {
                scale = 0.0; u1 = 1.0; u2 = 0.0;
            } else {
                if (temp >= temp2) {
                    w11 = T[tOff + (j+1)*ldt + j+1]; w21 = T[tOff + (j+2)*ldt + j+1];
                    w12 = T[tOff + (j+1)*ldt + j+2]; w22 = T[tOff + (j+2)*ldt + j+2];
                    u1  = T[tOff + (j+1)*ldt + j];   u2  = T[tOff + (j+2)*ldt + j];
                } else {
                    w21 = T[tOff + (j+1)*ldt + j+1]; w11 = T[tOff + (j+2)*ldt + j+1];
                    w22 = T[tOff + (j+1)*ldt + j+2]; w12 = T[tOff + (j+2)*ldt + j+2];
                    u2  = T[tOff + (j+1)*ldt + j];   u1  = T[tOff + (j+2)*ldt + j];
                }
                if (abs(w12) > abs(w11)) {
                    ilpivt = true;
                    double tmp = w12; w12 = w11; w11 = tmp;
                    tmp = w22; w22 = w21; w21 = tmp;
                }
                double tmpElim = w21 / w11;
                u2 -= tmpElim * u1; w22 -= tmpElim * w12; w21 = 0.0;
                scale = 1.0;
                if (abs(w22) < safmin) {
                    scale = 0.0; u2 = 1.0; u1 = -w12 / w11;
                } else {
                    if (abs(w22) < abs(u2)) scale = abs(w22 / u2);
                    if (abs(w11) < abs(u1)) scale = Math.min(scale, abs(w11 / u1));
                    u2 = (scale * u2) / w22;
                    u1 = (scale * u1 - w12 * u2) / w11;
                }
                if (ilpivt) { double tmp = u2; u2 = u1; u1 = tmp; }
            }

            double t1 = Math.sqrt(scale * scale + u1 * u1 + u2 * u2);
            tauVal = 1.0 + scale / t1;
            double vs = -1.0 / (scale + t1);
            v[0] = 1.0; v[1] = vs * u1; v[2] = vs * u2;

            // Apply Householder from right to H and T
            for (int jr = ifrstm; jr <= Math.min(j + 3, ilast); jr++) {
                double sum = H[hOff + jr*ldh + j] + v[1] * H[hOff + jr*ldh + j+1] + v[2] * H[hOff + jr*ldh + j+2];
                sum *= tauVal;
                H[hOff + jr*ldh + j]   -= sum;
                H[hOff + jr*ldh + j+1] -= sum * v[1];
                H[hOff + jr*ldh + j+2] -= sum * v[2];
            }
            for (int jr = ifrstm; jr <= j + 2; jr++) {
                double sum = T[tOff + jr*ldt + j] + v[1] * T[tOff + jr*ldt + j+1] + v[2] * T[tOff + jr*ldt + j+2];
                sum *= tauVal;
                T[tOff + jr*ldt + j]   -= sum;
                T[tOff + jr*ldt + j+1] -= sum * v[1];
                T[tOff + jr*ldt + j+2] -= sum * v[2];
            }
            if (ilz) for (int jr = 0; jr < n; jr++) {
                double sum = Z[zOff + jr*ldz + j] + v[1] * Z[zOff + jr*ldz + j+1] + v[2] * Z[zOff + jr*ldz + j+2];
                sum *= tauVal;
                Z[zOff + jr*ldz + j]   -= sum;
                Z[zOff + jr*ldz + j+1] -= sum * v[1];
                Z[zOff + jr*ldz + j+2] -= sum * v[2];
            }
            T[tOff + (j+1)*ldt + j] = 0.0;
            T[tOff + (j+2)*ldt + j] = 0.0;
        }

        // Final 2x2 step at j = ilast-1
        int j = ilast - 1;
        double temp = H[hOff + j*ldh + j-1];
        Dlartg.dlartg(temp, H[hOff + (j+1)*ldh + j-1], csr, 0);
        double c = csr[0], s = csr[1];
        H[hOff + j*ldh + j-1]     = c * temp + s * H[hOff + (j+1)*ldh + j-1];
        H[hOff + (j+1)*ldh + j-1] = 0.0;
        for (int jc = j; jc <= ilastm; jc++) {
            double h1 = H[hOff + j*ldh + jc], h2 = H[hOff + (j+1)*ldh + jc];
            H[hOff + j*ldh + jc]     =  c * h1 + s * h2;
            H[hOff + (j+1)*ldh + jc] = -s * h1 + c * h2;
            double t1 = T[tOff + j*ldt + jc], t2 = T[tOff + (j+1)*ldt + jc];
            T[tOff + j*ldt + jc]     =  c * t1 + s * t2;
            T[tOff + (j+1)*ldt + jc] = -s * t1 + c * t2;
        }
        if (ilq) for (int jr = 0; jr < n; jr++) {
            double q1 = Q[qOff + jr*ldq + j], q2 = Q[qOff + jr*ldq + j+1];
            Q[qOff + jr*ldq + j]   =  c * q1 + s * q2;
            Q[qOff + jr*ldq + j+1] = -s * q1 + c * q2;
        }
        temp = T[tOff + (j+1)*ldt + j+1];
        Dlartg.dlartg(temp, T[tOff + (j+1)*ldt + j], csr, 0);
        c = csr[0]; s = csr[1];
        T[tOff + (j+1)*ldt + j+1] = c * temp + s * T[tOff + (j+1)*ldt + j];
        T[tOff + (j+1)*ldt + j]   = 0.0;
        for (int jr = ifrstm; jr <= ilast; jr++) {
            double h1 = H[hOff + jr*ldh + j+1], h2 = H[hOff + jr*ldh + j];
            H[hOff + jr*ldh + j+1] =  c * h1 + s * h2;
            H[hOff + jr*ldh + j]   = -s * h1 + c * h2;
        }
        for (int jr = ifrstm; jr <= ilast - 1; jr++) {
            double t1 = T[tOff + jr*ldt + j+1], t2 = T[tOff + jr*ldt + j];
            T[tOff + jr*ldt + j+1] =  c * t1 + s * t2;
            T[tOff + jr*ldt + j]   = -s * t1 + c * t2;
        }
        if (ilz) for (int jr = 0; jr < n; jr++) {
            double z1 = Z[zOff + jr*ldz + j+1], z2 = Z[zOff + jr*ldz + j];
            Z[zOff + jr*ldz + j+1] =  c * z1 + s * z2;
            Z[zOff + jr*ldz + j]   = -s * z1 + c * z2;
        }
        return ilast; // no deflation in general case
    }
}
