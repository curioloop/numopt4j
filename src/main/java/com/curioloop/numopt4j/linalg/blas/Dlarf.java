/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Applies Householder reflections.
 * LAPACK DLARF algorithm - adapted for row-major storage.
 *
 * <p>Applies an elementary reflector H to a matrix C:</p>
 * <ul>
 *   <li>side = 'L': H * C (apply from left)</li>
 *   <li>side = 'R': C * H (apply from right)</li>
 * </ul>
 *
 * <p>Where H = I - tau * v * v^T</p>
 *
 * <p>This implements the LAPACK DLARF algorithm adapted for row-major storage.</p>
 */
interface Dlarf {

    /**
     * Applies a Householder reflection to a matrix.
     * Adapted for row-major storage.
     *
     * <p>If side = 'L': C := H * C where H = I - tau * v * v^T</p>
     * <p>If side = 'R': C := C * H where H = I - tau * v * v^T</p>
     *
     * @param side   Side.Left to apply from left, Side.Right to apply from right
     * @param m      number of rows of C
     * @param n      number of columns of C
     * @param v      the Householder vector v
     * @param vOff   offset into v
     * @param incv   increment between elements of v
     * @param tau    the scalar tau
     * @param C      the matrix C (m × n, row-major)
     * @param cOff   offset into C
     * @param ldc    leading dimension of C
     * @param work   workspace array (size max(m, n)), must not be null
     * @param workOff offset into work array
     */
    static void dlarf(BLAS.Side side, int m, int n,
                      double[] v, int vOff, int incv, double tau,
                      double[] C, int cOff, int ldc, double[] work, int workOff) {
        if (m == 0 || n == 0) {
            return;
        }

        boolean applyLeft = (side == BLAS.Side.Left);

        int lastv = -1;
        int lastc = -1;

        if (tau != 0.0) {
            if (applyLeft) {
                lastv = m - 1;
            } else {
                lastv = n - 1;
            }

            int i = lastv;
            if (incv > 0) {
                i = lastv * incv;
            }
            while (lastv >= 0 && v[vOff + i] == 0) {
                lastv--;
                i -= incv;
            }

            if (applyLeft) {
                if (lastv >= 0) {
                    lastc = Ilaux.iladlc(lastv + 1, n, C, cOff, ldc);
                }
            } else {
                if (lastv >= 0) {
                    lastc = Ilaux.iladlr(m, lastv + 1, C, cOff, ldc);
                }
            }
        }

        if (lastv == -1 || lastc == -1) {
            return;
        }

        if (applyLeft) {
            // Form H * C where H = I - tau * v * v^T
            // C = C - tau * v * (v^T * C)
            // w = C^T * v, then C = C - tau * v * w^T
            // w[j] = sum_i C[i,j] * v[i] for j = 0..lastc
            // This is dgemv('T', lastv+1, lastc+1, ...) with C being (lastv+1) x (lastc+1)
            Dgemv.dgemv(BLAS.Trans.Trans, lastv + 1, lastc + 1, 1.0,
                        C, cOff, ldc, 
                        v, vOff, incv, 
                        0.0, work, workOff, 1);
            
            // C = C - tau * v * w^T
            // C[i,j] = C[i,j] - tau * v[i] * w[j]
            Dger.dger(lastv + 1, lastc + 1, -tau,
                      v, vOff, incv,
                      work, workOff, 1,
                      C, cOff, ldc);
        } else {
            // Form C * H where H = I - tau * v * v^T
            // C = C - tau * (C * v) * v^T
            // w = C * v, then C = C - tau * w * v^T
            // w[i] = sum_j C[i,j] * v[j] for i = 0..lastc
            // This is dgemv('N', lastc+1, lastv+1, ...) with C being (lastc+1) x (lastv+1)
            Dgemv.dgemv(BLAS.Trans.NoTrans, lastc + 1, lastv + 1, 1.0,
                        C, cOff, ldc,
                        v, vOff, incv,
                        0.0, work, workOff, 1);
            
            // C = C - tau * w * v^T
            // C[i,j] = C[i,j] - tau * w[i] * v[j]
            Dger.dger(lastc + 1, lastv + 1, -tau,
                      work, workOff, 1,
                      v, vOff, incv,
                      C, cOff, ldc);
        }
    }
}
