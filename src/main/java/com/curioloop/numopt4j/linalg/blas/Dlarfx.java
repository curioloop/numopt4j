/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Applies an elementary reflector H to a real m×n matrix C, with loop unrolling
 * for small reflector orders (1–10).
 * LAPACK DLARFX algorithm (adapted for row-major storage).
 *
 * <p>H is represented as H = I - tau * v * v^T. For reflector order &lt; 11,
 * the inner loop is unrolled for performance. For larger orders, falls back
 * to {@link Dlarf#dlarf}.</p>
 *
 * <p>If tau = 0, H is the identity and C is unchanged.</p>
 */
interface Dlarfx {

    /**
     * Applies an elementary reflector H to matrix C.
     *
     * @param left    true to apply from left (H*C), false to apply from right (C*H)
     * @param m       number of rows of C
     * @param n       number of columns of C
     * @param v       the Householder vector v (length m if left, n if right)
     * @param vOff    offset into v
     * @param tau     the scalar tau; if 0, H is identity
     * @param c       the matrix C (m × n, row-major), overwritten with result
     * @param cOff    offset into c
     * @param ldc     leading dimension of c
     * @param work    workspace array (length &ge; n if left, &ge; m if right); only used when order &ge; 11
     * @param workOff offset into work
     */
    static void dlarfx(boolean left, int m, int n,
                       double[] v, int vOff, double tau,
                       double[] c, int cOff, int ldc, double[] work, int workOff) {
        if (m == 0 || n == 0) {
            return;
        }

        if (tau == 0) {
            return;
        }

        if (left) {
            switch (m) {
                case 0:
                    return;
                case 1: {
                    double t0 = 1 - tau * v[vOff] * v[vOff];
                    for (int j = 0; j < n; j++) {
                        c[cOff + j] *= t0;
                    }
                    return;
                }
                case 2: {
                    double v0 = v[vOff], t0 = tau * v0;
                    double v1 = v[vOff + 1], t1 = tau * v1;
                    for (int j = 0; j < n; j++) {
                        double sum = v0 * c[cOff + j] + v1 * c[cOff + ldc + j];
                        c[cOff + j] -= sum * t0;
                        c[cOff + ldc + j] -= sum * t1;
                    }
                    return;
                }
                case 3: {
                    double v0 = v[vOff], t0 = tau * v0;
                    double v1 = v[vOff + 1], t1 = tau * v1;
                    double v2 = v[vOff + 2], t2 = tau * v2;
                    for (int j = 0; j < n; j++) {
                        double sum = v0 * c[cOff + j] + v1 * c[cOff + ldc + j] + v2 * c[cOff + 2 * ldc + j];
                        c[cOff + j] -= sum * t0;
                        c[cOff + ldc + j] -= sum * t1;
                        c[cOff + 2 * ldc + j] -= sum * t2;
                    }
                    return;
                }
                case 4: {
                    double v0 = v[vOff], t0 = tau * v0;
                    double v1 = v[vOff + 1], t1 = tau * v1;
                    double v2 = v[vOff + 2], t2 = tau * v2;
                    double v3 = v[vOff + 3], t3 = tau * v3;
                    for (int j = 0; j < n; j++) {
                        double sum = v0 * c[cOff + j] + v1 * c[cOff + ldc + j] + v2 * c[cOff + 2 * ldc + j] + v3 * c[cOff + 3 * ldc + j];
                        c[cOff + j] -= sum * t0;
                        c[cOff + ldc + j] -= sum * t1;
                        c[cOff + 2 * ldc + j] -= sum * t2;
                        c[cOff + 3 * ldc + j] -= sum * t3;
                    }
                    return;
                }
                case 5: {
                    double v0 = v[vOff], t0 = tau * v0;
                    double v1 = v[vOff + 1], t1 = tau * v1;
                    double v2 = v[vOff + 2], t2 = tau * v2;
                    double v3 = v[vOff + 3], t3 = tau * v3;
                    double v4 = v[vOff + 4], t4 = tau * v4;
                    for (int j = 0; j < n; j++) {
                        double sum = v0 * c[cOff + j] + v1 * c[cOff + ldc + j] + v2 * c[cOff + 2 * ldc + j] + v3 * c[cOff + 3 * ldc + j] + v4 * c[cOff + 4 * ldc + j];
                        c[cOff + j] -= sum * t0;
                        c[cOff + ldc + j] -= sum * t1;
                        c[cOff + 2 * ldc + j] -= sum * t2;
                        c[cOff + 3 * ldc + j] -= sum * t3;
                        c[cOff + 4 * ldc + j] -= sum * t4;
                    }
                    return;
                }
                case 6: {
                    double v0 = v[vOff], t0 = tau * v0;
                    double v1 = v[vOff + 1], t1 = tau * v1;
                    double v2 = v[vOff + 2], t2 = tau * v2;
                    double v3 = v[vOff + 3], t3 = tau * v3;
                    double v4 = v[vOff + 4], t4 = tau * v4;
                    double v5 = v[vOff + 5], t5 = tau * v5;
                    for (int j = 0; j < n; j++) {
                        double sum = v0 * c[cOff + j] + v1 * c[cOff + ldc + j] + v2 * c[cOff + 2 * ldc + j] + v3 * c[cOff + 3 * ldc + j] + v4 * c[cOff + 4 * ldc + j] + v5 * c[cOff + 5 * ldc + j];
                        c[cOff + j] -= sum * t0;
                        c[cOff + ldc + j] -= sum * t1;
                        c[cOff + 2 * ldc + j] -= sum * t2;
                        c[cOff + 3 * ldc + j] -= sum * t3;
                        c[cOff + 4 * ldc + j] -= sum * t4;
                        c[cOff + 5 * ldc + j] -= sum * t5;
                    }
                    return;
                }
                case 7: {
                    double v0 = v[vOff], t0 = tau * v0;
                    double v1 = v[vOff + 1], t1 = tau * v1;
                    double v2 = v[vOff + 2], t2 = tau * v2;
                    double v3 = v[vOff + 3], t3 = tau * v3;
                    double v4 = v[vOff + 4], t4 = tau * v4;
                    double v5 = v[vOff + 5], t5 = tau * v5;
                    double v6 = v[vOff + 6], t6 = tau * v6;
                    for (int j = 0; j < n; j++) {
                        double sum = v0 * c[cOff + j] + v1 * c[cOff + ldc + j] + v2 * c[cOff + 2 * ldc + j] + v3 * c[cOff + 3 * ldc + j] + v4 * c[cOff + 4 * ldc + j] + v5 * c[cOff + 5 * ldc + j] + v6 * c[cOff + 6 * ldc + j];
                        c[cOff + j] -= sum * t0;
                        c[cOff + ldc + j] -= sum * t1;
                        c[cOff + 2 * ldc + j] -= sum * t2;
                        c[cOff + 3 * ldc + j] -= sum * t3;
                        c[cOff + 4 * ldc + j] -= sum * t4;
                        c[cOff + 5 * ldc + j] -= sum * t5;
                        c[cOff + 6 * ldc + j] -= sum * t6;
                    }
                    return;
                }
                case 8: {
                    double v0 = v[vOff], t0 = tau * v0;
                    double v1 = v[vOff + 1], t1 = tau * v1;
                    double v2 = v[vOff + 2], t2 = tau * v2;
                    double v3 = v[vOff + 3], t3 = tau * v3;
                    double v4 = v[vOff + 4], t4 = tau * v4;
                    double v5 = v[vOff + 5], t5 = tau * v5;
                    double v6 = v[vOff + 6], t6 = tau * v6;
                    double v7 = v[vOff + 7], t7 = tau * v7;
                    for (int j = 0; j < n; j++) {
                        double sum = v0 * c[cOff + j] + v1 * c[cOff + ldc + j] + v2 * c[cOff + 2 * ldc + j] + v3 * c[cOff + 3 * ldc + j] + v4 * c[cOff + 4 * ldc + j] + v5 * c[cOff + 5 * ldc + j] + v6 * c[cOff + 6 * ldc + j] + v7 * c[cOff + 7 * ldc + j];
                        c[cOff + j] -= sum * t0;
                        c[cOff + ldc + j] -= sum * t1;
                        c[cOff + 2 * ldc + j] -= sum * t2;
                        c[cOff + 3 * ldc + j] -= sum * t3;
                        c[cOff + 4 * ldc + j] -= sum * t4;
                        c[cOff + 5 * ldc + j] -= sum * t5;
                        c[cOff + 6 * ldc + j] -= sum * t6;
                        c[cOff + 7 * ldc + j] -= sum * t7;
                    }
                    return;
                }
                case 9: {
                    double v0 = v[vOff], t0 = tau * v0;
                    double v1 = v[vOff + 1], t1 = tau * v1;
                    double v2 = v[vOff + 2], t2 = tau * v2;
                    double v3 = v[vOff + 3], t3 = tau * v3;
                    double v4 = v[vOff + 4], t4 = tau * v4;
                    double v5 = v[vOff + 5], t5 = tau * v5;
                    double v6 = v[vOff + 6], t6 = tau * v6;
                    double v7 = v[vOff + 7], t7 = tau * v7;
                    double v8 = v[vOff + 8], t8 = tau * v8;
                    for (int j = 0; j < n; j++) {
                        double sum = v0 * c[cOff + j] + v1 * c[cOff + ldc + j] + v2 * c[cOff + 2 * ldc + j] + v3 * c[cOff + 3 * ldc + j] + v4 * c[cOff + 4 * ldc + j] + v5 * c[cOff + 5 * ldc + j] + v6 * c[cOff + 6 * ldc + j] + v7 * c[cOff + 7 * ldc + j] + v8 * c[cOff + 8 * ldc + j];
                        c[cOff + j] -= sum * t0;
                        c[cOff + ldc + j] -= sum * t1;
                        c[cOff + 2 * ldc + j] -= sum * t2;
                        c[cOff + 3 * ldc + j] -= sum * t3;
                        c[cOff + 4 * ldc + j] -= sum * t4;
                        c[cOff + 5 * ldc + j] -= sum * t5;
                        c[cOff + 6 * ldc + j] -= sum * t6;
                        c[cOff + 7 * ldc + j] -= sum * t7;
                        c[cOff + 8 * ldc + j] -= sum * t8;
                    }
                    return;
                }
                case 10: {
                    double v0 = v[vOff], t0 = tau * v0;
                    double v1 = v[vOff + 1], t1 = tau * v1;
                    double v2 = v[vOff + 2], t2 = tau * v2;
                    double v3 = v[vOff + 3], t3 = tau * v3;
                    double v4 = v[vOff + 4], t4 = tau * v4;
                    double v5 = v[vOff + 5], t5 = tau * v5;
                    double v6 = v[vOff + 6], t6 = tau * v6;
                    double v7 = v[vOff + 7], t7 = tau * v7;
                    double v8 = v[vOff + 8], t8 = tau * v8;
                    double v9 = v[vOff + 9], t9 = tau * v9;
                    for (int j = 0; j < n; j++) {
                        double sum = v0 * c[cOff + j] + v1 * c[cOff + ldc + j] + v2 * c[cOff + 2 * ldc + j] + v3 * c[cOff + 3 * ldc + j] + v4 * c[cOff + 4 * ldc + j] + v5 * c[cOff + 5 * ldc + j] + v6 * c[cOff + 6 * ldc + j] + v7 * c[cOff + 7 * ldc + j] + v8 * c[cOff + 8 * ldc + j] + v9 * c[cOff + 9 * ldc + j];
                        c[cOff + j] -= sum * t0;
                        c[cOff + ldc + j] -= sum * t1;
                        c[cOff + 2 * ldc + j] -= sum * t2;
                        c[cOff + 3 * ldc + j] -= sum * t3;
                        c[cOff + 4 * ldc + j] -= sum * t4;
                        c[cOff + 5 * ldc + j] -= sum * t5;
                        c[cOff + 6 * ldc + j] -= sum * t6;
                        c[cOff + 7 * ldc + j] -= sum * t7;
                        c[cOff + 8 * ldc + j] -= sum * t8;
                        c[cOff + 9 * ldc + j] -= sum * t9;
                    }
                    return;
                }
                default:
                    Dlarf.dlarf(BLAS.Side.Left, m, n, v, vOff, 1, tau, c, cOff, ldc, work, workOff);
                    return;
            }
        }

        switch (n) {
            case 0:
                return;
            case 1: {
                double t0 = 1 - tau * v[vOff] * v[vOff];
                for (int j = 0; j < m; j++) {
                    c[cOff + j * ldc] *= t0;
                }
                return;
            }
            case 2: {
                double v0 = v[vOff], t0 = tau * v0;
                double v1 = v[vOff + 1], t1 = tau * v1;
                for (int j = 0; j < m; j++) {
                    int rowOff = cOff + j * ldc;
                    double sum = v0 * c[rowOff] + v1 * c[rowOff + 1];
                    c[rowOff] -= sum * t0;
                    c[rowOff + 1] -= sum * t1;
                }
                return;
            }
            case 3: {
                double v0 = v[vOff], t0 = tau * v0;
                double v1 = v[vOff + 1], t1 = tau * v1;
                double v2 = v[vOff + 2], t2 = tau * v2;
                for (int j = 0; j < m; j++) {
                    int rowOff = cOff + j * ldc;
                    double sum = v0 * c[rowOff] + v1 * c[rowOff + 1] + v2 * c[rowOff + 2];
                    c[rowOff] -= sum * t0;
                    c[rowOff + 1] -= sum * t1;
                    c[rowOff + 2] -= sum * t2;
                }
                return;
            }
            case 4: {
                double v0 = v[vOff], t0 = tau * v0;
                double v1 = v[vOff + 1], t1 = tau * v1;
                double v2 = v[vOff + 2], t2 = tau * v2;
                double v3 = v[vOff + 3], t3 = tau * v3;
                for (int j = 0; j < m; j++) {
                    int rowOff = cOff + j * ldc;
                    double sum = v0 * c[rowOff] + v1 * c[rowOff + 1] + v2 * c[rowOff + 2] + v3 * c[rowOff + 3];
                    c[rowOff] -= sum * t0;
                    c[rowOff + 1] -= sum * t1;
                    c[rowOff + 2] -= sum * t2;
                    c[rowOff + 3] -= sum * t3;
                }
                return;
            }
            case 5: {
                double v0 = v[vOff], t0 = tau * v0;
                double v1 = v[vOff + 1], t1 = tau * v1;
                double v2 = v[vOff + 2], t2 = tau * v2;
                double v3 = v[vOff + 3], t3 = tau * v3;
                double v4 = v[vOff + 4], t4 = tau * v4;
                for (int j = 0; j < m; j++) {
                    int rowOff = cOff + j * ldc;
                    double sum = v0 * c[rowOff] + v1 * c[rowOff + 1] + v2 * c[rowOff + 2] + v3 * c[rowOff + 3] + v4 * c[rowOff + 4];
                    c[rowOff] -= sum * t0;
                    c[rowOff + 1] -= sum * t1;
                    c[rowOff + 2] -= sum * t2;
                    c[rowOff + 3] -= sum * t3;
                    c[rowOff + 4] -= sum * t4;
                }
                return;
            }
            case 6: {
                double v0 = v[vOff], t0 = tau * v0;
                double v1 = v[vOff + 1], t1 = tau * v1;
                double v2 = v[vOff + 2], t2 = tau * v2;
                double v3 = v[vOff + 3], t3 = tau * v3;
                double v4 = v[vOff + 4], t4 = tau * v4;
                double v5 = v[vOff + 5], t5 = tau * v5;
                for (int j = 0; j < m; j++) {
                    int rowOff = cOff + j * ldc;
                    double sum = v0 * c[rowOff] + v1 * c[rowOff + 1] + v2 * c[rowOff + 2] + v3 * c[rowOff + 3] + v4 * c[rowOff + 4] + v5 * c[rowOff + 5];
                    c[rowOff] -= sum * t0;
                    c[rowOff + 1] -= sum * t1;
                    c[rowOff + 2] -= sum * t2;
                    c[rowOff + 3] -= sum * t3;
                    c[rowOff + 4] -= sum * t4;
                    c[rowOff + 5] -= sum * t5;
                }
                return;
            }
            case 7: {
                double v0 = v[vOff], t0 = tau * v0;
                double v1 = v[vOff + 1], t1 = tau * v1;
                double v2 = v[vOff + 2], t2 = tau * v2;
                double v3 = v[vOff + 3], t3 = tau * v3;
                double v4 = v[vOff + 4], t4 = tau * v4;
                double v5 = v[vOff + 5], t5 = tau * v5;
                double v6 = v[vOff + 6], t6 = tau * v6;
                for (int j = 0; j < m; j++) {
                    int rowOff = cOff + j * ldc;
                    double sum = v0 * c[rowOff] + v1 * c[rowOff + 1] + v2 * c[rowOff + 2] + v3 * c[rowOff + 3] + v4 * c[rowOff + 4] + v5 * c[rowOff + 5] + v6 * c[rowOff + 6];
                    c[rowOff] -= sum * t0;
                    c[rowOff + 1] -= sum * t1;
                    c[rowOff + 2] -= sum * t2;
                    c[rowOff + 3] -= sum * t3;
                    c[rowOff + 4] -= sum * t4;
                    c[rowOff + 5] -= sum * t5;
                    c[rowOff + 6] -= sum * t6;
                }
                return;
            }
            case 8: {
                double v0 = v[vOff], t0 = tau * v0;
                double v1 = v[vOff + 1], t1 = tau * v1;
                double v2 = v[vOff + 2], t2 = tau * v2;
                double v3 = v[vOff + 3], t3 = tau * v3;
                double v4 = v[vOff + 4], t4 = tau * v4;
                double v5 = v[vOff + 5], t5 = tau * v5;
                double v6 = v[vOff + 6], t6 = tau * v6;
                double v7 = v[vOff + 7], t7 = tau * v7;
                for (int j = 0; j < m; j++) {
                    int rowOff = cOff + j * ldc;
                    double sum = v0 * c[rowOff] + v1 * c[rowOff + 1] + v2 * c[rowOff + 2] + v3 * c[rowOff + 3] + v4 * c[rowOff + 4] + v5 * c[rowOff + 5] + v6 * c[rowOff + 6] + v7 * c[rowOff + 7];
                    c[rowOff] -= sum * t0;
                    c[rowOff + 1] -= sum * t1;
                    c[rowOff + 2] -= sum * t2;
                    c[rowOff + 3] -= sum * t3;
                    c[rowOff + 4] -= sum * t4;
                    c[rowOff + 5] -= sum * t5;
                    c[rowOff + 6] -= sum * t6;
                    c[rowOff + 7] -= sum * t7;
                }
                return;
            }
            case 9: {
                double v0 = v[vOff], t0 = tau * v0;
                double v1 = v[vOff + 1], t1 = tau * v1;
                double v2 = v[vOff + 2], t2 = tau * v2;
                double v3 = v[vOff + 3], t3 = tau * v3;
                double v4 = v[vOff + 4], t4 = tau * v4;
                double v5 = v[vOff + 5], t5 = tau * v5;
                double v6 = v[vOff + 6], t6 = tau * v6;
                double v7 = v[vOff + 7], t7 = tau * v7;
                double v8 = v[vOff + 8], t8 = tau * v8;
                for (int j = 0; j < m; j++) {
                    int rowOff = cOff + j * ldc;
                    double sum = v0 * c[rowOff] + v1 * c[rowOff + 1] + v2 * c[rowOff + 2] + v3 * c[rowOff + 3] + v4 * c[rowOff + 4] + v5 * c[rowOff + 5] + v6 * c[rowOff + 6] + v7 * c[rowOff + 7] + v8 * c[rowOff + 8];
                    c[rowOff] -= sum * t0;
                    c[rowOff + 1] -= sum * t1;
                    c[rowOff + 2] -= sum * t2;
                    c[rowOff + 3] -= sum * t3;
                    c[rowOff + 4] -= sum * t4;
                    c[rowOff + 5] -= sum * t5;
                    c[rowOff + 6] -= sum * t6;
                    c[rowOff + 7] -= sum * t7;
                    c[rowOff + 8] -= sum * t8;
                }
                return;
            }
            case 10: {
                double v0 = v[vOff], t0 = tau * v0;
                double v1 = v[vOff + 1], t1 = tau * v1;
                double v2 = v[vOff + 2], t2 = tau * v2;
                double v3 = v[vOff + 3], t3 = tau * v3;
                double v4 = v[vOff + 4], t4 = tau * v4;
                double v5 = v[vOff + 5], t5 = tau * v5;
                double v6 = v[vOff + 6], t6 = tau * v6;
                double v7 = v[vOff + 7], t7 = tau * v7;
                double v8 = v[vOff + 8], t8 = tau * v8;
                double v9 = v[vOff + 9], t9 = tau * v9;
                for (int j = 0; j < m; j++) {
                    int rowOff = cOff + j * ldc;
                    double sum = v0 * c[rowOff] + v1 * c[rowOff + 1] + v2 * c[rowOff + 2] + v3 * c[rowOff + 3] + v4 * c[rowOff + 4] + v5 * c[rowOff + 5] + v6 * c[rowOff + 6] + v7 * c[rowOff + 7] + v8 * c[rowOff + 8] + v9 * c[rowOff + 9];
                    c[rowOff] -= sum * t0;
                    c[rowOff + 1] -= sum * t1;
                    c[rowOff + 2] -= sum * t2;
                    c[rowOff + 3] -= sum * t3;
                    c[rowOff + 4] -= sum * t4;
                    c[rowOff + 5] -= sum * t5;
                    c[rowOff + 6] -= sum * t6;
                    c[rowOff + 7] -= sum * t7;
                    c[rowOff + 8] -= sum * t8;
                    c[rowOff + 9] -= sum * t9;
                }
                return;
            }
            default:
                Dlarf.dlarf(BLAS.Side.Right, m, n, v, vOff, 1, tau, c, cOff, ldc, work, workOff);
                return;
        }
    }
}
