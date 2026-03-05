/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.abs;

public interface Dlacn2 {

    public static double dlacn2(int n, double[] v, int vOff,
                         double[] x, int xOff,
                         int[] isgn, int isgnOff,
                         double est, int[] kase, int kaseOff,
                         int[] isave, int isaveOff) {
        final int itmax = 5;

        if (kase[kaseOff] == 0) {
            for (int i = 0; i < n; i++) {
                x[xOff + i] = 1.0 / n;
            }
            kase[kaseOff] = 1;
            isave[isaveOff] = 1;
            return est;
        }

        switch (isave[isaveOff]) {
            case 1: {
                if (n == 1) {
                    v[vOff] = x[xOff];
                    est = abs(v[vOff]);
                    kase[kaseOff] = 0;
                    return est;
                }
                est = BLAS.dasum(n, x, xOff, 1);
                for (int i = 0; i < n; i++) {
                    x[xOff + i] = Math.copySign(1.0, x[xOff + i]);
                    isgn[isgnOff + i] = (int) x[xOff + i];
                }
                kase[kaseOff] = 2;
                isave[isaveOff] = 2;
                return est;
            }
            case 2: {
                isave[isaveOff + 1] = BLAS.idamax(n, x, xOff, 1);
                isave[isaveOff + 2] = 2;
                for (int i = 0; i < n; i++) {
                    x[xOff + i] = 0;
                }
                x[xOff + isave[isaveOff + 1]] = 1;
                kase[kaseOff] = 1;
                isave[isaveOff] = 3;
                return est;
            }
            case 3: {
                BLAS.dcopy(n, x, xOff, 1, v, vOff, 1);
                double estold = est;
                est = BLAS.dasum(n, v, vOff, 1);
                boolean sameSigns = true;
                for (int i = 0; i < n; i++) {
                    if ((int) Math.copySign(1.0, x[xOff + i]) != isgn[isgnOff + i]) {
                        sameSigns = false;
                        break;
                    }
                }
                if (!sameSigns && est > estold) {
                    for (int i = 0; i < n; i++) {
                        x[xOff + i] = Math.copySign(1.0, x[xOff + i]);
                        isgn[isgnOff + i] = (int) x[xOff + i];
                    }
                    kase[kaseOff] = 2;
                    isave[isaveOff] = 4;
                    return est;
                }
                break;
            }
            case 4: {
                int jlast = isave[isaveOff + 1];
                isave[isaveOff + 1] = BLAS.idamax(n, x, xOff, 1);
                if (x[xOff + jlast] != abs(x[xOff + isave[isaveOff + 1]]) && isave[isaveOff + 2] < itmax) {
                    isave[isaveOff + 2]++;
                    for (int i = 0; i < n; i++) {
                        x[xOff + i] = 0;
                    }
                    x[xOff + isave[isaveOff + 1]] = 1;
                    kase[kaseOff] = 1;
                    isave[isaveOff] = 3;
                    return est;
                }
                break;
            }
            case 5: {
                double tmp = 2.0 * BLAS.dasum(n, x, xOff, 1) / (3.0 * n);
                if (tmp > est) {
                    BLAS.dcopy(n, x, xOff, 1, v, vOff, 1);
                    est = tmp;
                }
                kase[kaseOff] = 0;
                return est;
            }
        }

        double altsgn = 1.0;
        for (int i = 0; i < n; i++) {
            x[xOff + i] = altsgn * (1.0 + (double) i / (n - 1));
            altsgn *= -1;
        }
        kase[kaseOff] = 1;
        isave[isaveOff] = 5;
        return est;
    }
}
