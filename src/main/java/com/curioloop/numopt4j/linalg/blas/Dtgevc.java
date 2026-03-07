/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * LAPACK DTGEVC: Computes right and/or left generalized eigenvectors of a
 * pair of real upper quasi-triangular matrices (S, P).
 *
 * <p>The right eigenvector x and left eigenvector y satisfy:
 * <pre>
 *   S * x = lambda * P * x
 *   y^H * S = lambda * y^H * P
 * </pre>
 *
 * <p>Row-major storage: element (i,j) is at array[i*ld + j].
 */
interface Dtgevc {

    /**
     * Computes generalized eigenvectors.
     *
     * @param side   'R' right only, 'L' left only, 'B' both
     * @param howmny 'A' all, 'B' backtransform (VR/VL contain initial vectors), 'S' selected
     * @param select boolean array (length n); used if howmny='S'
     * @param n      order of S and P
     * @param S      quasi-upper-triangular matrix (n×n, row-major)
     * @param sOff   offset into S
     * @param lds    leading dimension of S
     * @param P      upper triangular matrix (n×n, row-major)
     * @param pOff   offset into P
     * @param ldp    leading dimension of P
     * @param VL     left eigenvectors (n×mm, row-major); used if side='L' or 'B'
     * @param vlOff  offset into VL
     * @param ldvl   leading dimension of VL
     * @param VR     right eigenvectors (n×mm, row-major); used if side='R' or 'B'
     * @param vrOff  offset into VR
     * @param ldvr   leading dimension of VR
     * @param mm     number of columns in VL/VR (>= number of selected eigenvectors)
     * @param work   workspace (length 6*n)
     * @param workOff offset into work
     * @return number of eigenvectors computed (m)
     */
    static int dtgevc(char side, char howmny, boolean[] select, int n,
                      double[] S, int sOff, int lds,
                      double[] P, int pOff, int ldp,
                      double[] VL, int vlOff, int ldvl,
                      double[] VR, int vrOff, int ldvr,
                      int mm, double[] work, int workOff) {

        boolean compl = (side == 'L' || side == 'B');
        boolean compr = (side == 'R' || side == 'B');
        boolean ilall = (howmny == 'A' || howmny == 'B');
        boolean ilback = (howmny == 'B');

        double safmin = BLAS.safmin();
        double eps    = BLAS.eps();
        double smlnum = safmin * n / eps;

        // Count eigenvectors to compute
        int m = 0;
        if (ilall) {
            // Count: real eigenvalues count 1, complex pairs count 2
            int j = 0;
            while (j < n) {
                if (j == n - 1 || S[sOff + (j+1)*lds + j] == 0.0) {
                    m++; j++;
                } else {
                    m += 2; j += 2;
                }
            }
        } else {
            int j = 0;
            while (j < n) {
                if (j == n - 1 || S[sOff + (j+1)*lds + j] == 0.0) {
                    if (select[j]) m++;
                    j++;
                } else {
                    if (select[j] || select[j+1]) m += 2;
                    j += 2;
                }
            }
        }

        // Initialize VL/VR if backtransform
        if (ilback) {
            // VL/VR already contain Q/Z; we'll multiply eigvecs by them after computing
        } else if (ilall) {
            if (compl) BLAS.dlaset('A', n, n, 0.0, 1.0, VL, vlOff, ldvl);
            if (compr) BLAS.dlaset('A', n, n, 0.0, 1.0, VR, vrOff, ldvr);
        }

        // Work array layout: work[0..n-1] = real part, work[n..2n-1] = imag part
        // work[2n..3n-1] = temp, work[3n..4n-1] = temp2, etc.
        int wOff1 = workOff;
        int wOff2 = workOff + n;

        // Process each eigenvector
        // Store eigenvector for eigenvalue j in column j (not je) to maintain correspondence
        int je = 0; // column counter (used for howmny='B' backtransform ordering)
        int j = n - 1;
        while (j >= 0) {
            // Determine if this is a real or complex eigenvalue
            boolean ilcplx = (j > 0 && S[sOff + j*lds + j-1] != 0.0);

            if (!ilall && !ilcplx && !select[j]) { j--; continue; }
            if (!ilall && ilcplx && !select[j] && !select[j-1]) { j -= 2; continue; }

            // For howmny='A', store in column j (matching eigenvalue index)
            // For howmny='B', store in column je (sequential)
            int col = ilback ? je : j;

            if (!ilcplx) {
                // Real eigenvalue: compute right eigenvector
                if (compr) {
                    computeRealRightEigvecToWork(S, sOff, lds, P, pOff, ldp, n, j,
                                                 work, wOff1, safmin, smlnum);
                    if (ilback) {
                        for (int i = 0; i < n; i++) {
                            double sum = 0.0;
                            for (int k = 0; k < n; k++) sum += VR[vrOff + i*ldvr + k] * work[wOff1 + k];
                            work[wOff2 + i] = sum;
                        }
                        for (int i = 0; i < n; i++) VR[vrOff + i*ldvr + col] = work[wOff2 + i];
                    } else {
                        for (int i = 0; i < n; i++) VR[vrOff + i*ldvr + col] = work[wOff1 + i];
                    }
                }
                if (compl) {
                    computeRealLeftEigvecToWork(S, sOff, lds, P, pOff, ldp, n, j,
                                                work, wOff1, safmin, smlnum);
                    if (ilback) {
                        for (int i = 0; i < n; i++) {
                            double sum = 0.0;
                            for (int k = 0; k < n; k++) sum += VL[vlOff + i*ldvl + k] * work[wOff1 + k];
                            work[wOff2 + i] = sum;
                        }
                        for (int i = 0; i < n; i++) VL[vlOff + i*ldvl + col] = work[wOff2 + i];
                    } else {
                        for (int i = 0; i < n; i++) VL[vlOff + i*ldvl + col] = work[wOff1 + i];
                    }
                }
                je++;
                j--;
            } else {
                // Complex eigenvalue pair at (j-1, j): store in columns (j-1, j)
                int col2 = ilback ? je : j - 1;
                int wOff3 = workOff + 2*n, wOff4 = workOff + 3*n;
                if (compr) {
                    computeComplexRightEigvecToWork(S, sOff, lds, P, pOff, ldp, n, j-1,
                                                    work, wOff1, wOff2, safmin, smlnum);
                    if (ilback) {
                        for (int i = 0; i < n; i++) {
                            double sumr = 0.0, sumi = 0.0;
                            for (int k = 0; k < n; k++) {
                                sumr += VR[vrOff + i*ldvr + k] * work[wOff1 + k];
                                sumi += VR[vrOff + i*ldvr + k] * work[wOff2 + k];
                            }
                            work[wOff3 + i] = sumr;
                            work[wOff4 + i] = sumi;
                        }
                        for (int i = 0; i < n; i++) {
                            VR[vrOff + i*ldvr + col]   = work[wOff3 + i];
                            VR[vrOff + i*ldvr + col+1] = work[wOff4 + i];
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            VR[vrOff + i*ldvr + col2]   = work[wOff1 + i];
                            VR[vrOff + i*ldvr + col2+1] = work[wOff2 + i];
                        }
                    }
                }
                if (compl) {
                    computeComplexLeftEigvecToWork(S, sOff, lds, P, pOff, ldp, n, j-1,
                                                   work, wOff1, wOff2, safmin, smlnum);
                    if (ilback) {
                        for (int i = 0; i < n; i++) {
                            double sumr = 0.0, sumi = 0.0;
                            for (int k = 0; k < n; k++) {
                                sumr += VL[vlOff + i*ldvl + k] * work[wOff1 + k];
                                sumi += VL[vlOff + i*ldvl + k] * work[wOff2 + k];
                            }
                            work[wOff3 + i] = sumr;
                            work[wOff4 + i] = sumi;
                        }
                        for (int i = 0; i < n; i++) {
                            VL[vlOff + i*ldvl + col]   = work[wOff3 + i];
                            VL[vlOff + i*ldvl + col+1] = work[wOff4 + i];
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            VL[vlOff + i*ldvl + col2]   = work[wOff1 + i];
                            VL[vlOff + i*ldvl + col2+1] = work[wOff2 + i];
                        }
                    }
                }
                je += 2;
                j -= 2;
            }
        }

        return m;
    }

    /** Compute real right eigenvector into work[wOff..wOff+n-1] (normalized). */
    static void computeRealRightEigvecToWork(double[] S, int sOff, int lds,
                                              double[] P, int pOff, int ldp,
                                              int n, int j,
                                              double[] work, int wOff,
                                              double safmin, double smlnum) {
        for (int i = 0; i < n; i++) work[wOff + i] = 0.0;
        work[wOff + j] = 1.0;
        double lambda_r = (abs(P[pOff + j*ldp + j]) > safmin) ?
                          S[sOff + j*lds + j] / P[pOff + j*ldp + j] : 0.0;
        for (int i = j - 1; i >= 0; i--) {
            if (i > 0 && S[sOff + i*lds + i-1] != 0.0) {
                double s11 = S[sOff + (i-1)*lds + i-1] - lambda_r * P[pOff + (i-1)*ldp + i-1];
                double s12 = S[sOff + (i-1)*lds + i]   - lambda_r * P[pOff + (i-1)*ldp + i];
                double s21 = S[sOff + i*lds + i-1]     - lambda_r * P[pOff + i*ldp + i-1];
                double s22 = S[sOff + i*lds + i]       - lambda_r * P[pOff + i*ldp + i];
                double r1 = 0.0, r2 = 0.0;
                for (int k = i + 1; k <= j; k++) {
                    r1 -= (S[sOff + (i-1)*lds + k] - lambda_r * P[pOff + (i-1)*ldp + k]) * work[wOff + k];
                    r2 -= (S[sOff + i*lds + k]     - lambda_r * P[pOff + i*ldp + k]) * work[wOff + k];
                }
                double det = s11 * s22 - s12 * s21;
                if (abs(det) < smlnum) det = smlnum;
                work[wOff + i-1] = (s22 * r1 - s12 * r2) / det;
                work[wOff + i]   = (s11 * r2 - s21 * r1) / det;
                i--;
            } else {
                double sii = S[sOff + i*lds + i] - lambda_r * P[pOff + i*ldp + i];
                double rhs = 0.0;
                for (int k = i + 1; k <= j; k++)
                    rhs -= (S[sOff + i*lds + k] - lambda_r * P[pOff + i*ldp + k]) * work[wOff + k];
                if (abs(sii) < smlnum) sii = smlnum;
                work[wOff + i] = rhs / sii;
            }
        }
        double norm = 0.0;
        for (int i = 0; i <= j; i++) norm = max(norm, abs(work[wOff + i]));
        if (norm > 0.0) for (int i = 0; i <= j; i++) work[wOff + i] /= norm;
    }

    /** Compute real left eigenvector into work[wOff..wOff+n-1] (normalized). */
    static void computeRealLeftEigvecToWork(double[] S, int sOff, int lds,
                                             double[] P, int pOff, int ldp,
                                             int n, int j,
                                             double[] work, int wOff,
                                             double safmin, double smlnum) {
        for (int i = 0; i < n; i++) work[wOff + i] = 0.0;
        work[wOff + j] = 1.0;
        double lambda_r = (abs(P[pOff + j*ldp + j]) > safmin) ?
                          S[sOff + j*lds + j] / P[pOff + j*ldp + j] : 0.0;
        for (int i = j + 1; i < n; i++) {
            double sii = S[sOff + i*lds + i] - lambda_r * P[pOff + i*ldp + i];
            double rhs = 0.0;
            for (int k = j; k < i; k++)
                rhs -= (S[sOff + k*lds + i] - lambda_r * P[pOff + k*ldp + i]) * work[wOff + k];
            if (abs(sii) < smlnum) sii = smlnum;
            work[wOff + i] = rhs / sii;
        }
        double norm = 0.0;
        for (int i = j; i < n; i++) norm = max(norm, abs(work[wOff + i]));
        if (norm > 0.0) for (int i = j; i < n; i++) work[wOff + i] /= norm;
    }

    /** Compute complex right eigenvec into work[wOff1..] (real) and work[wOff2..] (imag). */
    static void computeComplexRightEigvecToWork(double[] S, int sOff, int lds,
                                                 double[] P, int pOff, int ldp,
                                                 int n, int j,
                                                 double[] work, int wOff1, int wOff2,
                                                 double safmin, double smlnum) {
        for (int i = 0; i < n; i++) { work[wOff1 + i] = 0.0; work[wOff2 + i] = 0.0; }
        double s11 = S[sOff + j*lds + j], s12 = S[sOff + j*lds + j+1];
        double s21 = S[sOff + (j+1)*lds + j], s22 = S[sOff + (j+1)*lds + j+1];
        double p11 = P[pOff + j*ldp + j], p22 = P[pOff + (j+1)*ldp + j+1];
        double tr = 0.5 * (s11 / p11 + s22 / p22);
        double disc = (s11 / p11 - s22 / p22) * (s11 / p11 - s22 / p22) * 0.25 + s12 * s21 / (p11 * p22);
        double wi = (disc < 0.0) ? sqrt(-disc) : 0.0;
        double wr = tr;
        work[wOff1 + j] = 1.0; work[wOff2 + j] = 0.0;
        work[wOff1 + j+1] = 0.0; work[wOff2 + j+1] = 1.0;
        for (int i = j - 1; i >= 0; i--) {
            double sii = S[sOff + i*lds + i], pii = P[pOff + i*ldp + i];
            double ar = sii - wr * pii, ai = -wi * pii;
            double rr = 0.0, ri = 0.0;
            for (int k = i + 1; k <= j + 1; k++) {
                double sik = S[sOff + i*lds + k], pik = P[pOff + i*ldp + k];
                rr -= (sik - wr * pik) * work[wOff1 + k] + wi * pik * work[wOff2 + k];
                ri -= (sik - wr * pik) * work[wOff2 + k] - wi * pik * work[wOff1 + k];
            }
            double denom = ar * ar + ai * ai;
            if (denom < smlnum) denom = smlnum;
            work[wOff1 + i] = (ar * rr + ai * ri) / denom;
            work[wOff2 + i] = (ar * ri - ai * rr) / denom;
        }
        double norm = 0.0;
        for (int i = 0; i <= j + 1; i++) norm = max(norm, abs(work[wOff1 + i]) + abs(work[wOff2 + i]));
        if (norm > 0.0) for (int i = 0; i <= j+1; i++) { work[wOff1 + i] /= norm; work[wOff2 + i] /= norm; }
    }

    /** Compute complex left eigenvec into work[wOff1..] (real) and work[wOff2..] (imag). */
    static void computeComplexLeftEigvecToWork(double[] S, int sOff, int lds,
                                                double[] P, int pOff, int ldp,
                                                int n, int j,
                                                double[] work, int wOff1, int wOff2,
                                                double safmin, double smlnum) {
        for (int i = 0; i < n; i++) { work[wOff1 + i] = 0.0; work[wOff2 + i] = 0.0; }
        double s11 = S[sOff + j*lds + j], s22 = S[sOff + (j+1)*lds + j+1];
        double p11 = P[pOff + j*ldp + j], p22 = P[pOff + (j+1)*ldp + j+1];
        double tr = 0.5 * (s11 / p11 + s22 / p22);
        double disc = (s11 / p11 - s22 / p22) * (s11 / p11 - s22 / p22) * 0.25
                    + S[sOff + j*lds + j+1] * S[sOff + (j+1)*lds + j] / (p11 * p22);
        double wi = (disc < 0.0) ? sqrt(-disc) : 0.0;
        double wr = tr;
        work[wOff1 + j] = 1.0; work[wOff2 + j] = 0.0;
        work[wOff1 + j+1] = 0.0; work[wOff2 + j+1] = 1.0;
        for (int i = j + 2; i < n; i++) {
            double sii = S[sOff + i*lds + i], pii = P[pOff + i*ldp + i];
            double ar = sii - wr * pii, ai = -wi * pii;
            double rr = 0.0, ri = 0.0;
            for (int k = j; k < i; k++) {
                double ski = S[sOff + k*lds + i], pki = P[pOff + k*ldp + i];
                rr -= (ski - wr * pki) * work[wOff1 + k] + wi * pki * work[wOff2 + k];
                ri -= (ski - wr * pki) * work[wOff2 + k] - wi * pki * work[wOff1 + k];
            }
            double denom = ar * ar + ai * ai;
            if (denom < smlnum) denom = smlnum;
            work[wOff1 + i] = (ar * rr + ai * ri) / denom;
            work[wOff2 + i] = (ar * ri - ai * rr) / denom;
        }
        double norm = 0.0;
        for (int i = j; i < n; i++) norm = max(norm, abs(work[wOff1 + i]) + abs(work[wOff2 + i]));
        if (norm > 0.0) for (int i = j; i < n; i++) { work[wOff1 + i] /= norm; work[wOff2 + i] /= norm; }
    }
}
