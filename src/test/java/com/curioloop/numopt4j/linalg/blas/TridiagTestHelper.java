/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import java.util.Random;

/** Shared test utilities for tridiagonal routines. */
class TridiagTestHelper {

    /** Generate a random diagonally dominant SPD tridiagonal matrix. Returns {d, e}. */
    static double[][] newRandomSymTridiag(int n, Random rnd) {
        if (n == 0) return new double[][]{new double[0], new double[0]};
        double[] d = new double[n];
        double[] e = n > 1 ? new double[n-1] : new double[0];
        for (int i = 0; i < n; i++) d[i] = rnd.nextDouble();
        for (int i = 0; i < n-1; i++) e[i] = rnd.nextDouble() * 2 - 1;
        d[0] += n > 1 ? Math.abs(e[0]) : 0;
        for (int i = 1; i < n-1; i++) d[i] += Math.abs(e[i]) + Math.abs(e[i-1]);
        if (n > 1) d[n-1] += Math.abs(e[n-2]);
        return new double[][]{d, e};
    }

    /** Compute C = A*B where A is symmetric tridiagonal (d, e), B and C are n x nrhs row-major. */
    static void dstmm(int n, int nrhs, double[] d, double[] e,
                      double[] B, int ldb, double[] C, int ldc) {
        if (n == 0 || nrhs == 0) return;
        if (n == 1) {
            for (int j = 0; j < nrhs; j++) C[j] = d[0] * B[j];
            return;
        }
        for (int j = 0; j < nrhs; j++) C[j] = d[0] * B[j] + e[0] * B[ldb + j];
        for (int i = 1; i < n-1; i++)
            for (int j = 0; j < nrhs; j++)
                C[i*ldc+j] = e[i-1]*B[(i-1)*ldb+j] + d[i]*B[i*ldb+j] + e[i]*B[(i+1)*ldb+j];
        for (int j = 0; j < nrhs; j++)
            C[(n-1)*ldc+j] = e[n-2]*B[(n-2)*ldb+j] + d[n-1]*B[(n-1)*ldb+j];
    }

    /** |xGot - xWant|_1 / n (max column sum of difference). */
    static double residual(int n, int nrhs, double[] xGot, double[] xWant, int ld) {
        double maxCol = 0;
        for (int j = 0; j < nrhs; j++) {
            double col = 0;
            for (int i = 0; i < n; i++) col += Math.abs(xGot[i*ld+j] - xWant[i*ld+j]);
            maxCol = Math.max(maxCol, col);
        }
        return n == 0 ? 0 : maxCol / n;
    }

    /** Compute residual norm(L*D*Lt - A) / (n * norm(A)) for Dpttrf verification. */
    static double pttrf_residual(int n, double[] d, double[] e, double[] dFac, double[] eFac) {
        if (n == 0) return 0;
        double[] dDiff = new double[n];
        double[] eDiff = n > 1 ? new double[n-1] : new double[0];
        dDiff[0] = dFac[0] - d[0];
        for (int i = 0; i < n-1; i++) {
            double de = dFac[i] * eFac[i];
            dDiff[i+1] = de * eFac[i] + dFac[i+1] - d[i+1];
            eDiff[i]   = de - e[i];
        }
        double resid;
        if (n == 1) {
            resid = Math.abs(dDiff[0]);
        } else {
            resid = Math.max(Math.abs(dDiff[0]) + Math.abs(eDiff[0]),
                             Math.abs(dDiff[n-1]) + Math.abs(eDiff[n-2]));
            for (int i = 1; i < n-1; i++)
                resid = Math.max(resid, Math.abs(dDiff[i]) + Math.abs(eDiff[i-1]) + Math.abs(eDiff[i]));
        }
        double anorm;
        if (n == 1) {
            anorm = Math.abs(d[0]);
        } else {
            anorm = Math.max(Math.abs(d[0]) + Math.abs(e[0]),
                             Math.abs(d[n-1]) + Math.abs(e[n-2]));
            for (int i = 1; i < n-1; i++)
                anorm = Math.max(anorm, Math.abs(e[i-1]) + Math.abs(d[i]) + Math.abs(e[i]));
        }
        if (anorm == 0) return resid == 0 ? 0 : Double.POSITIVE_INFINITY;
        return resid / n / anorm;
    }
}
