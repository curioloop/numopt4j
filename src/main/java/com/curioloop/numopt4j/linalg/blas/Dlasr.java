/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DLASR: Applies a sequence of plane rotations to a matrix.
 * 
 * <p>Note: This implementation uses row-major storage convention:
 * a[row*lda+col] = A[row][col]</p>
 */
interface Dlasr {

    static void dlasr(BLAS.Side side, char pivot, char direct, int m, int n,
                      double[] c, int cOff, double[] s, int sOff,
                      double[] a, int aOff, int lda) {
        if (m == 0 || n == 0) {
            return;
        }

        char pivotU = Character.toUpperCase(pivot);
        char directU = Character.toUpperCase(direct);

        if (side == BLAS.Side.Left) {
            applyLeft(pivotU, directU, m, n, c, cOff, s, sOff, a, aOff, lda);
            return;
        }

        applyRight(pivotU, directU, m, n, c, cOff, s, sOff, a, aOff, lda);
    }

    private static void applyLeft(char pivot, char direct, int m, int n,
                                  double[] c, int cOff, double[] s, int sOff,
                                  double[] a, int aOff, int lda) {
        switch (pivot) {
            case 'V' -> {
                if (direct == 'F') {
                    int rowAOff = aOff;
                    int rowBOff = aOff + lda;
                    for (int j = 0; j < m - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateRows(n, ctmp, stmp, a, rowAOff, rowBOff);
                        }
                        rowAOff += lda;
                        rowBOff += lda;
                    }
                    return;
                }
                if (direct == 'B') {
                    int rowAOff = aOff + (m - 2) * lda;
                    int rowBOff = rowAOff + lda;
                    for (int j = m - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateRows(n, ctmp, stmp, a, rowAOff, rowBOff);
                        }
                        rowAOff -= lda;
                        rowBOff -= lda;
                    }
                }
            }
            case 'T' -> {
                int topOff = aOff;
                if (direct == 'F') {
                    int rowOff = aOff + lda;
                    for (int j = 1; j < m; j++) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateRows(n, ctmp, stmp, a, topOff, rowOff);
                        }
                        rowOff += lda;
                    }
                    return;
                }
                if (direct == 'B') {
                    int rowOff = aOff + (m - 1) * lda;
                    for (int j = m - 1; j >= 1; j--) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateRows(n, ctmp, stmp, a, topOff, rowOff);
                        }
                        rowOff -= lda;
                    }
                }
            }
            case 'B' -> {
                int bottomOff = aOff + (m - 1) * lda;
                if (direct == 'F') {
                    int rowOff = aOff;
                    for (int j = 0; j < m - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateRows(n, ctmp, stmp, a, rowOff, bottomOff);
                        }
                        rowOff += lda;
                    }
                    return;
                }
                if (direct == 'B') {
                    int rowOff = aOff + (m - 2) * lda;
                    for (int j = m - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateRows(n, ctmp, stmp, a, rowOff, bottomOff);
                        }
                        rowOff -= lda;
                    }
                }
            }
            default -> {
            }
        }
    }

    private static void applyRight(char pivot, char direct, int m, int n,
                                   double[] c, int cOff, double[] s, int sOff,
                                   double[] a, int aOff, int lda) {
        switch (pivot) {
            case 'V' -> {
                if (direct == 'F') {
                    for (int j = 0; j < n - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateColumns(m, ctmp, stmp, a, aOff, lda, j, j + 1);
                        }
                    }
                    return;
                }
                if (direct == 'B') {
                    for (int j = n - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateColumns(m, ctmp, stmp, a, aOff, lda, j, j + 1);
                        }
                    }
                }
            }
            case 'T' -> {
                if (direct == 'F') {
                    for (int j = 1; j < n; j++) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateColumns(m, ctmp, stmp, a, aOff, lda, 0, j);
                        }
                    }
                    return;
                }
                if (direct == 'B') {
                    for (int j = n - 1; j >= 1; j--) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateColumns(m, ctmp, stmp, a, aOff, lda, 0, j);
                        }
                    }
                }
            }
            case 'B' -> {
                int bottom = n - 1;
                if (direct == 'F') {
                    for (int j = 0; j < n - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateColumns(m, ctmp, stmp, a, aOff, lda, j, bottom);
                        }
                    }
                    return;
                }
                if (direct == 'B') {
                    for (int j = n - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (!isIdentityRotation(ctmp, stmp)) {
                            rotateColumns(m, ctmp, stmp, a, aOff, lda, j, bottom);
                        }
                    }
                }
            }
            default -> {
            }
        }
    }

    private static boolean isIdentityRotation(double c, double s) {
        return c == 1.0 && s == 0.0;
    }

    private static void rotateRows(int n, double c, double s, double[] a, int rowAOff, int rowBOff) {
        int col = 0;
        int unrolled = n & ~3;
        for (; col < unrolled; col += 4) {
            double a0 = a[rowAOff + col];
            double b0 = a[rowBOff + col];
            a[rowBOff + col] = c * b0 - s * a0;
            a[rowAOff + col] = s * b0 + c * a0;

            double a1 = a[rowAOff + col + 1];
            double b1 = a[rowBOff + col + 1];
            a[rowBOff + col + 1] = c * b1 - s * a1;
            a[rowAOff + col + 1] = s * b1 + c * a1;

            double a2 = a[rowAOff + col + 2];
            double b2 = a[rowBOff + col + 2];
            a[rowBOff + col + 2] = c * b2 - s * a2;
            a[rowAOff + col + 2] = s * b2 + c * a2;

            double a3 = a[rowAOff + col + 3];
            double b3 = a[rowBOff + col + 3];
            a[rowBOff + col + 3] = c * b3 - s * a3;
            a[rowAOff + col + 3] = s * b3 + c * a3;
        }

        for (; col < n; col++) {
            double aVal = a[rowAOff + col];
            double bVal = a[rowBOff + col];
            a[rowBOff + col] = c * bVal - s * aVal;
            a[rowAOff + col] = s * bVal + c * aVal;
        }
    }

    private static void rotateColumns(int m, double c, double s, double[] a, int aOff, int lda,
                                      int colA, int colB) {
        int offA = aOff + colA;
        int offB = aOff + colB;
        int row = 0;
        int unrolled = m & ~3;
        for (; row < unrolled; row += 4) {
            rotateColumnPair(c, s, a, offA, offB);
            rotateColumnPair(c, s, a, offA + lda, offB + lda);
            rotateColumnPair(c, s, a, offA + 2 * lda, offB + 2 * lda);
            rotateColumnPair(c, s, a, offA + 3 * lda, offB + 3 * lda);
            offA += 4 * lda;
            offB += 4 * lda;
        }

        for (; row < m; row++) {
            rotateColumnPair(c, s, a, offA, offB);
            offA += lda;
            offB += lda;
        }
    }

    private static void rotateColumnPair(double c, double s, double[] a, int offA, int offB) {
        double aVal = a[offA];
        double bVal = a[offB];
        a[offB] = c * bVal - s * aVal;
        a[offA] = s * bVal + c * aVal;
    }
}
