/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import java.util.Arrays;

interface Dlamv {

    static final double SAFMIN = 0x1p-1022;

    static void dcopy(int n, double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY) {
        if (n <= 0) return;

        if (incX == 1 && incY == 1) {
            System.arraycopy(x, xOff, y, yOff, n);
        } else {
            int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));
            int iy = yOff + ((incY > 0) ? 0 : (n - 1) * (-incY));
            for (int i = 0; i < n; i++) {
                y[iy] = x[ix];
                ix += incX;
                iy += incY;
            }
        }
    }

    static void dset(int n, double a, double[] x, int xOff, int incX) {
        if (n <= 0) return;

        if (incX == 1) {
            Arrays.fill(x, xOff, xOff + n, a);
        } else {
            int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));
            for (int i = 0; i < n; i++) {
                x[ix] = a;
                ix += incX;
            }
        }
    }

    static void dzero(int n, double[] x, int xOff, int incX) {
        dset(n, 0.0, x, xOff, incX);
    }

    static void dswap(int n, double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY) {
        if (n <= 0) return;

        if (incX == 1 && incY == 1) {
            int k = 0;
            for (; k + 3 < n; k += 4) {
                int i0 = xOff + k, j0 = yOff + k;
                double t0 = x[i0];     x[i0] = y[j0];         y[j0] = t0;
                double t1 = x[i0 + 1]; x[i0 + 1] = y[j0 + 1]; y[j0 + 1] = t1;
                double t2 = x[i0 + 2]; x[i0 + 2] = y[j0 + 2]; y[j0 + 2] = t2;
                double t3 = x[i0 + 3]; x[i0 + 3] = y[j0 + 3]; y[j0 + 3] = t3;
            }
            for (; k < n; k++) {
                int i = xOff + k, j = yOff + k;
                double temp = x[i];
                x[i] = y[j];
                y[j] = temp;
            }
        } else {
            int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));
            int iy = yOff + ((incY > 0) ? 0 : (n - 1) * (-incY));
            for (int i = 0; i < n; i++) {
                double temp = x[ix];
                x[ix] = y[iy];
                y[iy] = temp;
                ix += incX;
                iy += incY;
            }
        }
    }

    static void dlaswp(int n, double[] A, int aOff, int lda, int k1, int k2, int[] ipiv, int ipivOff, int incX) {
        if (n <= 0) return;

        if (incX == 1) {
            for (int k = k1; k <= k2; k++) {
                int p = ipiv[ipivOff + k];
                if (p != k) {
                    dswap(n, A, aOff + k * lda, 1, A, aOff + p * lda, 1);
                }
            }
        } else {
            for (int k = k2; k >= k1; k--) {
                int p = ipiv[ipivOff + k];
                if (p != k) {
                    dswap(n, A, aOff + k * lda, 1, A, aOff + p * lda, 1);
                }
            }
        }
    }

    static void dlacpy(char uplo, int m, int n, double[] A, int aOff, int lda,
                       double[] B, int bOff, int ldb) {
        if (m == 0 || n == 0) return;

        char u = Character.toUpperCase(uplo);

        switch (u) {
            case 'U':
                for (int i = 0; i < m; i++) {
                    System.arraycopy(A, aOff + i * lda + i, B, bOff + i * ldb + i, n - i);
                }
                break;
            case 'L':
                for (int i = 0; i < m; i++) {
                    int maxJ = Math.min(i + 1, n);
                    System.arraycopy(A, aOff + i * lda, B, bOff + i * ldb, maxJ);
                }
                break;
            default:
                for (int i = 0; i < m; i++) {
                    System.arraycopy(A, aOff + i * lda, B, bOff + i * ldb, n);
                }
                break;
        }
    }

    static void dlaset(char uplo, int m, int n, double alpha, double beta,
                       double[] A, int aOff, int lda) {
        if (m == 0 || n == 0) return;

        char u = Character.toUpperCase(uplo);
        int minMN = Math.min(m, n);

        switch (u) {
            case 'U':
                for (int i = 0; i < m; i++) {
                    int rowOff = aOff + i * lda;
                    for (int j = i + 1; j < n; j++) {
                        A[rowOff + j] = alpha;
                    }
                }
                break;
            case 'L':
                for (int i = 0; i < m; i++) {
                    int rowOff = aOff + i * lda;
                    for (int j = 0; j < Math.min(i, n); j++) {
                        A[rowOff + j] = alpha;
                    }
                }
                break;
            default:
                for (int i = 0; i < m; i++) {
                    int rowOff = aOff + i * lda;
                    for (int j = 0; j < n; j++) {
                        A[rowOff + j] = alpha;
                    }
                }
                break;
        }

        for (int i = 0; i < minMN; i++) {
            A[aOff + i * lda + i] = beta;
        }
    }

    static void dlascl(char kind, int kl, int ku, double cfrom, double cto,
                       int m, int n, double[] a, int aOff, int lda) {
        if (m == 0 || n == 0) return;
        if (cfrom == 0.0) throw new ArithmeticException("cfrom is zero");

        char k = Character.toUpperCase(kind);
        double smlnum = SAFMIN;
        double bignum = 1.0 / smlnum;
        double cfromc = cfrom;
        double ctoc = cto;
        double cfrom1 = cfromc * smlnum;

        while (true) {
            double mul;
            boolean done;

            if (cfrom1 == cfromc) {
                mul = ctoc / cfromc;
                done = true;
            } else {
                double ctol = ctoc / bignum;
                if (ctol == ctoc) {
                    mul = ctoc;
                    done = true;
                    cfromc = 1.0;
                } else if (Math.abs(cfrom1) > Math.abs(ctoc) && ctoc != 0.0) {
                    mul = smlnum;
                    done = false;
                    cfromc = cfrom1;
                } else if (Math.abs(ctol) > Math.abs(cfromc)) {
                    mul = bignum;
                    done = false;
                    ctoc = ctol;
                } else {
                    mul = ctoc / cfromc;
                    done = true;
                    if (mul == 1.0) return;
                }
            }

            switch (k) {
                case 'U':
                    for (int i = 0, rowOff = aOff; i < m; i++, rowOff += lda) {
                        for (int j = i; j < n; j++) a[rowOff + j] *= mul;
                    }
                    break;
                case 'L':
                    for (int i = 0, rowOff = aOff; i < m; i++, rowOff += lda) {
                        int jEnd = Math.min(i, n - 1);
                        for (int j = 0; j <= jEnd; j++) a[rowOff + j] *= mul;
                    }
                    break;
                case 'G':
                    for (int i = 0, rowOff = aOff; i < m; i++, rowOff += lda) {
                        for (int j = 0; j < n; j++) a[rowOff + j] *= mul;
                    }
                    break;
                case 'H':
                    for (int i = 0, rowOff = aOff; i < m; i++, rowOff += lda) {
                        int jStart = Math.max(i - 1, 0);
                        for (int j = jStart; j < n; j++) a[rowOff + j] *= mul;
                    }
                    break;
                case 'B':
                    for (int i = 0, rowOff = aOff; i < m; i++, rowOff += lda) {
                        int jStart = Math.max(i - kl, 0);
                        int jEnd = Math.min(i, n - 1);
                        for (int j = jStart; j <= jEnd; j++) a[rowOff + j] *= mul;
                    }
                    break;
                case 'Q':
                    for (int i = 0, rowOff = aOff; i < m; i++, rowOff += lda) {
                        int jEnd = Math.min(i + ku, n - 1);
                        for (int j = i; j <= jEnd; j++) a[rowOff + j] *= mul;
                    }
                    break;
                case 'Z':
                    for (int i = 0, rowOff = aOff; i < m; i++, rowOff += lda) {
                        int jStart = Math.max(i - kl, 0);
                        int jEnd = Math.min(i + ku, n - 1);
                        for (int j = jStart; j <= jEnd; j++) a[rowOff + j] *= mul;
                    }
                    break;
                default:
                    throw new IllegalArgumentException("unknown matrix type: " + kind);
            }

            if (done) break;
        }
    }

    static void dlascl(char kind, double cfrom, double cto,
                       int n, double[] a, int off, int inc) {
        if (n == 0) return;
        if (cfrom == 0.0) throw new ArithmeticException("cfrom is zero");

        double smlnum = SAFMIN;
        double bignum = 1.0 / smlnum;
        double cfromc = cfrom;
        double ctoc = cto;
        double cfrom1 = cfromc * smlnum;

        while (true) {
            double mul;
            boolean done;

            if (cfrom1 == cfromc) {
                mul = ctoc / cfromc;
                done = true;
            } else {
                double ctol = ctoc / bignum;
                if (ctol == ctoc) {
                    mul = ctoc;
                    done = true;
                    cfromc = 1.0;
                } else if (Math.abs(cfrom1) > Math.abs(ctol) && ctol != 0.0) {
                    mul = smlnum;
                    done = false;
                    cfromc = cfrom1;
                } else if (Math.abs(ctol) > Math.abs(cfromc)) {
                    mul = bignum;
                    done = false;
                    ctoc = ctol;
                } else {
                    mul = ctoc / cfromc;
                    done = true;
                    if (mul == 1.0) return;
                }
            }

            for (int i = 0, idx = off; i < n; i++, idx += inc) {
                a[idx] *= mul;
            }

            if (done) break;
        }
    }
}
