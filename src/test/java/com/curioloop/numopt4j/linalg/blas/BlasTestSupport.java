/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

final class BlasTestSupport {

    private BlasTestSupport() {
    }

    static double scalarDasum(int n, double[] x, int xOff, int incX) {
        if (n <= 0) {
            return 0.0;
        }

        double result = 0.0;
        int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));
        for (int i = 0; i < n; i++) {
            result += Math.abs(x[ix]);
            ix += incX;
        }
        return result;
    }

    static void scalarDgemm(BLAS.Trans transA, BLAS.Trans transB, int m, int n, int k,
                            double alpha, double[] a, int aOff, int lda,
                            double[] b, int bOff, int ldb,
                            double beta, double[] c, int cOff, int ldc) {
        boolean transAFlag = transA == BLAS.Trans.Trans || transA == BLAS.Trans.Conj;
        boolean transBFlag = transB == BLAS.Trans.Trans || transB == BLAS.Trans.Conj;

        if (m <= 0 || n <= 0 || ((alpha == 0.0 || k <= 0) && beta == 1.0)) {
            return;
        }

        for (int row = 0; row < m; row++) {
            int cRow = cOff + row * ldc;
            if (beta == 0.0) {
                for (int col = 0; col < n; col++) {
                    c[cRow + col] = 0.0;
                }
            } else if (beta != 1.0) {
                for (int col = 0; col < n; col++) {
                    c[cRow + col] *= beta;
                }
            }
        }

        if (alpha == 0.0 || k <= 0) {
            return;
        }

        for (int row = 0; row < m; row++) {
            int cRow = cOff + row * ldc;
            for (int col = 0; col < n; col++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    double aVal = transAFlag ? a[aOff + p * lda + row] : a[aOff + row * lda + p];
                    double bVal = transBFlag ? b[bOff + col * ldb + p] : b[bOff + p * ldb + col];
                    sum = Math.fma(aVal, bVal, sum);
                }
                c[cRow + col] = Math.fma(alpha, sum, c[cRow + col]);
            }
        }
    }

    static void assertUsedRegionClose(String label, double[] expected, double[] actual,
                                      int rows, int cols, int ld, double tolerance) {
        for (int row = 0; row < rows; row++) {
            int rowOff = row * ld;
            for (int col = 0; col < cols; col++) {
                double e = expected[rowOff + col];
                double a = actual[rowOff + col];
                double scale = Math.max(1.0, Math.max(Math.abs(e), Math.abs(a)));
                if (Math.abs(e - a) > tolerance * scale) {
                    throw new IllegalStateException(label + " mismatch at (" + row + "," + col + ")"
                        + ": expected=" + e + ", actual=" + a);
                }
            }
        }
    }

    static void scalarDrot(int n, double[] x, int xOff, int incX,
                           double[] y, int yOff, int incY, double c, double s) {
        if (n <= 0) {
            return;
        }

        int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));
        int iy = yOff + ((incY > 0) ? 0 : (n - 1) * (-incY));
        for (int i = 0; i < n; i++) {
            double temp = Math.fma(c, x[ix], s * y[iy]);
            y[iy] = Math.fma(c, y[iy], -s * x[ix]);
            x[ix] = temp;
            ix += incX;
            iy += incY;
        }
    }

    static void scalarDlasr(BLAS.Side side, char pivot, char direct, int m, int n,
                            double[] c, int cOff, double[] s, int sOff,
                            double[] a, int aOff, int lda) {
        if (m == 0 || n == 0) {
            return;
        }

        char pivotU = Character.toUpperCase(pivot);
        char directU = Character.toUpperCase(direct);

        if (side == BLAS.Side.Left) {
            if (pivotU == 'V') {
                if (directU == 'F') {
                    for (int j = 0; j < m - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp2 = a[aOff + j * lda + i];
                                double tmp = a[aOff + (j + 1) * lda + i];
                                a[aOff + (j + 1) * lda + i] = ctmp * tmp - stmp * tmp2;
                                a[aOff + j * lda + i] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                    return;
                }
                if (directU == 'B') {
                    for (int j = m - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp2 = a[aOff + j * lda + i];
                                double tmp = a[aOff + (j + 1) * lda + i];
                                a[aOff + (j + 1) * lda + i] = ctmp * tmp - stmp * tmp2;
                                a[aOff + j * lda + i] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                    return;
                }
            }

            if (pivotU == 'T') {
                if (directU == 'F') {
                    for (int j = 1; j < m; j++) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp = a[aOff + j * lda + i];
                                double tmp2 = a[aOff + i];
                                a[aOff + j * lda + i] = ctmp * tmp - stmp * tmp2;
                                a[aOff + i] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                    return;
                }
                if (directU == 'B') {
                    for (int j = m - 1; j >= 1; j--) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp = a[aOff + j * lda + i];
                                double tmp2 = a[aOff + i];
                                a[aOff + j * lda + i] = ctmp * tmp - stmp * tmp2;
                                a[aOff + i] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                    return;
                }
            }

            if (pivotU == 'B') {
                if (directU == 'F') {
                    for (int j = 0; j < m - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp = a[aOff + j * lda + i];
                                double tmp2 = a[aOff + (m - 1) * lda + i];
                                a[aOff + j * lda + i] = stmp * tmp2 + ctmp * tmp;
                                a[aOff + (m - 1) * lda + i] = ctmp * tmp2 - stmp * tmp;
                            }
                        }
                    }
                    return;
                }
                if (directU == 'B') {
                    for (int j = m - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp = a[aOff + j * lda + i];
                                double tmp2 = a[aOff + (m - 1) * lda + i];
                                a[aOff + j * lda + i] = stmp * tmp2 + ctmp * tmp;
                                a[aOff + (m - 1) * lda + i] = ctmp * tmp2 - stmp * tmp;
                            }
                        }
                    }
                }
            }
            return;
        }

        if (pivotU == 'V') {
            if (directU == 'F') {
                for (int j = 0; j < n - 1; j++) {
                    double ctmp = c[cOff + j];
                    double stmp = s[sOff + j];
                    if (ctmp != 1.0 || stmp != 0.0) {
                        for (int i = 0; i < m; i++) {
                            double tmp = a[aOff + i * lda + j + 1];
                            double tmp2 = a[aOff + i * lda + j];
                            a[aOff + i * lda + j + 1] = ctmp * tmp - stmp * tmp2;
                            a[aOff + i * lda + j] = stmp * tmp + ctmp * tmp2;
                        }
                    }
                }
                return;
            }
            if (directU == 'B') {
                for (int j = n - 2; j >= 0; j--) {
                    double ctmp = c[cOff + j];
                    double stmp = s[sOff + j];
                    if (ctmp != 1.0 || stmp != 0.0) {
                        for (int i = 0; i < m; i++) {
                            double tmp = a[aOff + i * lda + j + 1];
                            double tmp2 = a[aOff + i * lda + j];
                            a[aOff + i * lda + j + 1] = ctmp * tmp - stmp * tmp2;
                            a[aOff + i * lda + j] = stmp * tmp + ctmp * tmp2;
                        }
                    }
                }
                return;
            }
        }

        if (pivotU == 'T') {
            if (directU == 'F') {
                for (int j = 1; j < n; j++) {
                    double ctmp = c[cOff + j - 1];
                    double stmp = s[sOff + j - 1];
                    if (ctmp != 1.0 || stmp != 0.0) {
                        for (int i = 0; i < m; i++) {
                            double tmp = a[aOff + i * lda + j];
                            double tmp2 = a[aOff + i * lda];
                            a[aOff + i * lda + j] = ctmp * tmp - stmp * tmp2;
                            a[aOff + i * lda] = stmp * tmp + ctmp * tmp2;
                        }
                    }
                }
                return;
            }
            if (directU == 'B') {
                for (int j = n - 1; j >= 1; j--) {
                    double ctmp = c[cOff + j - 1];
                    double stmp = s[sOff + j - 1];
                    if (ctmp != 1.0 || stmp != 0.0) {
                        for (int i = 0; i < m; i++) {
                            double tmp = a[aOff + i * lda + j];
                            double tmp2 = a[aOff + i * lda];
                            a[aOff + i * lda + j] = ctmp * tmp - stmp * tmp2;
                            a[aOff + i * lda] = stmp * tmp + ctmp * tmp2;
                        }
                    }
                }
                return;
            }
        }

        if (pivotU == 'B') {
            if (directU == 'F') {
                for (int j = 0; j < n - 1; j++) {
                    double ctmp = c[cOff + j];
                    double stmp = s[sOff + j];
                    if (ctmp != 1.0 || stmp != 0.0) {
                        for (int i = 0; i < m; i++) {
                            double tmp = a[aOff + i * lda + j];
                            double tmp2 = a[aOff + i * lda + n - 1];
                            a[aOff + i * lda + j] = stmp * tmp2 + ctmp * tmp;
                            a[aOff + i * lda + n - 1] = ctmp * tmp2 - stmp * tmp;
                        }
                    }
                }
                return;
            }
            if (directU == 'B') {
                for (int j = n - 2; j >= 0; j--) {
                    double ctmp = c[cOff + j];
                    double stmp = s[sOff + j];
                    if (ctmp != 1.0 || stmp != 0.0) {
                        for (int i = 0; i < m; i++) {
                            double tmp = a[aOff + i * lda + j];
                            double tmp2 = a[aOff + i * lda + n - 1];
                            a[aOff + i * lda + j] = stmp * tmp2 + ctmp * tmp;
                            a[aOff + i * lda + n - 1] = ctmp * tmp2 - stmp * tmp;
                        }
                    }
                }
            }
        }
    }

    static void dtrsmScalar(BLAS.Side side, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                            int m, int n,
                            double[] a, int aOff, int lda,
                            double[] b, int bOff, int ldb) {
        if (side == BLAS.Side.Right) {
            dtrsmScalarRight(uplo, trans, diag, m, n, a, aOff, lda, b, bOff, ldb);
            return;
        }

        boolean transA = trans == BLAS.Trans.Trans || trans == BLAS.Trans.Conj;
        boolean unitDiag = diag == BLAS.Diag.Unit;

        if (!transA && uplo == BLAS.Uplo.Upper) {
            for (int i = m - 1; i >= 0; i--) {
                int aRow = aOff + i * lda;
                int bRow = bOff + i * ldb;
                for (int k = i + 1; k < m; k++) {
                    scalarDaxpy(n, -a[aRow + k], b, bOff + k * ldb, 1, b, bRow, 1);
                }
                if (!unitDiag) {
                    scalarDscal(n, 1.0 / a[aRow + i], b, bRow, 1);
                }
            }
            return;
        }

        if (!transA) {
            for (int i = 0; i < m; i++) {
                int aRow = aOff + i * lda;
                int bRow = bOff + i * ldb;
                for (int k = 0; k < i; k++) {
                    scalarDaxpy(n, -a[aRow + k], b, bOff + k * ldb, 1, b, bRow, 1);
                }
                if (!unitDiag) {
                    scalarDscal(n, 1.0 / a[aRow + i], b, bRow, 1);
                }
            }
            return;
        }

        if (uplo == BLAS.Uplo.Upper) {
            for (int i = 0; i < m; i++) {
                int bRow = bOff + i * ldb;
                for (int k = 0; k < i; k++) {
                    scalarDaxpy(n, -a[aOff + k * lda + i], b, bOff + k * ldb, 1, b, bRow, 1);
                }
                if (!unitDiag) {
                    scalarDscal(n, 1.0 / a[aOff + i * lda + i], b, bRow, 1);
                }
            }
            return;
        }

        for (int i = m - 1; i >= 0; i--) {
            int bRow = bOff + i * ldb;
            for (int k = i + 1; k < m; k++) {
                scalarDaxpy(n, -a[aOff + k * lda + i], b, bOff + k * ldb, 1, b, bRow, 1);
            }
            if (!unitDiag) {
                scalarDscal(n, 1.0 / a[aOff + i * lda + i], b, bRow, 1);
            }
        }
    }

    static void scalarDger(int m, int n, double alpha, double[] x, int xOff, int incX,
                           double[] y, int yOff, int incY, double[] a, int aOff, int lda) {
        if (m == 0 || n == 0 || alpha == 0.0) {
            return;
        }

        int ky = (incY < 0) ? (-(n - 1) * incY) : 0;
        int kx = (incX < 0) ? (-(m - 1) * incX) : 0;

        int ix = kx;
        for (int i = 0; i < m; i++) {
            double xi = alpha * x[xOff + ix];
            int rowOff = aOff + i * lda;
            int iy = ky;
            for (int j = 0; j < n; j++) {
                a[rowOff + j] = Math.fma(xi, y[yOff + iy], a[rowOff + j]);
                iy += incY;
            }
            ix += incX;
        }
    }

    static int scalarDtrsl(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff, BLAS.Uplo uplo, BLAS.Trans trans) {
        return scalarDtrsl(t, tOff, ldt, n, b, bOff, uplo, trans, BLAS.Diag.NonUnit);
    }

    static int scalarDtrsl(double[] t, int tOff, int ldt, int n,
                           double[] b, int bOff, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean transpose = trans == BLAS.Trans.Trans || trans == BLAS.Trans.Conj;
        boolean nounit = diag == BLAS.Diag.NonUnit;

        if (!upper) {
            return transpose ? scalarSolveLowerT(t, tOff, ldt, n, b, bOff, nounit) : scalarSolveLowerN(t, tOff, ldt, n, b, bOff, nounit);
        }
        return transpose ? scalarSolveUpperT(t, tOff, ldt, n, b, bOff, nounit) : scalarSolveUpperN(t, tOff, ldt, n, b, bOff, nounit);
    }

    static double scalarDgecon(char norm, int n, double[] a, int lda, double anorm,
                               double[] work, int[] iwork) {
        if (n == 0) {
            return 1.0;
        }
        if (anorm == 0.0) {
            return 0.0;
        }
        if (Double.isNaN(anorm)) {
            return anorm;
        }
        if (Double.isInfinite(anorm)) {
            return 0.0;
        }

        double smlnum = Dlamch.dlamch('S');
        double rcond = 0.0;
        double ainvnm = 0.0;
        boolean normin = false;
        boolean onenrm = norm == '1' || norm == 'O' || norm == 'o';
        int kase1 = onenrm ? 1 : 2;
        int[] isave = new int[3];

        while (true) {
            ainvnm = Dlacn2.dlacn2(n, work, n, work, 0, iwork, 0, ainvnm, isave, 2, isave, 0);

            if (isave[2] == 0) {
                if (ainvnm != 0.0) {
                    rcond = (1.0 / ainvnm) / anorm;
                }
                return rcond;
            }

            double sl;
            double su;
            if (isave[2] == kase1) {
                sl = scalarDlatrs(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, normin, n, a, lda, work, 0, work, 2 * n);
                su = scalarDlatrs(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, normin, n, a, lda, work, 0, work, 3 * n);
            } else {
                su = scalarDlatrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, normin, n, a, lda, work, 0, work, 3 * n);
                sl = scalarDlatrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, normin, n, a, lda, work, 0, work, 2 * n);
            }

            double scale = sl * su;
            normin = true;

            if (scale != 1.0) {
                int ix = 0;
                double xmax = Math.abs(work[0]);
                for (int i = 1; i < n; i++) {
                    if (Math.abs(work[i]) > xmax) {
                        xmax = Math.abs(work[i]);
                        ix = i;
                    }
                }
                if (scale == 0.0 || scale < Math.abs(work[ix]) * smlnum) {
                    return rcond;
                }
                Drscl.drscl(n, scale, work, 0, 1);
            }
        }
    }

    private static void dtrsmScalarRight(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                                         int m, int n,
                                         double[] a, int aOff, int lda,
                                         double[] b, int bOff, int ldb) {
        boolean transA = trans == BLAS.Trans.Trans || trans == BLAS.Trans.Conj;
        boolean unitDiag = diag == BLAS.Diag.Unit;

        if (!transA && uplo == BLAS.Uplo.Upper) {
            for (int row = 0; row < m; row++) {
                int bRow = bOff + row * ldb;
                for (int k = 0; k < n; k++) {
                    double bk = b[bRow + k];
                    if (bk == 0.0) {
                        continue;
                    }
                    if (!unitDiag) {
                        bk /= a[aOff + k * lda + k];
                        b[bRow + k] = bk;
                    }
                    for (int j = k + 1; j < n; j++) {
                        b[bRow + j] = Math.fma(-bk, a[aOff + k * lda + j], b[bRow + j]);
                    }
                }
            }
            return;
        }

        if (!transA) {
            for (int row = 0; row < m; row++) {
                int bRow = bOff + row * ldb;
                for (int k = n - 1; k >= 0; k--) {
                    double bk = b[bRow + k];
                    if (bk == 0.0) {
                        continue;
                    }
                    if (!unitDiag) {
                        bk /= a[aOff + k * lda + k];
                        b[bRow + k] = bk;
                    }
                    for (int j = 0; j < k; j++) {
                        b[bRow + j] = Math.fma(-bk, a[aOff + k * lda + j], b[bRow + j]);
                    }
                }
            }
            return;
        }

        if (uplo == BLAS.Uplo.Upper) {
            for (int row = 0; row < m; row++) {
                int bRow = bOff + row * ldb;
                for (int j = n - 1; j >= 0; j--) {
                    double sum = b[bRow + j];
                    for (int k = j + 1; k < n; k++) {
                        sum = Math.fma(-b[bRow + k], a[aOff + j * lda + k], sum);
                    }
                    if (!unitDiag) {
                        sum /= a[aOff + j * lda + j];
                    }
                    b[bRow + j] = sum;
                }
            }
            return;
        }

        for (int row = 0; row < m; row++) {
            int bRow = bOff + row * ldb;
            for (int j = 0; j < n; j++) {
                double sum = b[bRow + j];
                for (int k = 0; k < j; k++) {
                    sum = Math.fma(-b[bRow + k], a[aOff + j * lda + k], sum);
                }
                if (!unitDiag) {
                    sum /= a[aOff + j * lda + j];
                }
                b[bRow + j] = sum;
            }
        }
    }

    private static void scalarDaxpy(int n, double alpha,
                                    double[] x, int xOff, int incX,
                                    double[] y, int yOff, int incY) {
        if (n <= 0 || alpha == 0.0) {
            return;
        }
        int xi = xOff;
        int yi = yOff;
        for (int i = 0; i < n; i++) {
            y[yi] = Math.fma(alpha, x[xi], y[yi]);
            xi += incX;
            yi += incY;
        }
    }

    private static void scalarDscal(int n, double alpha, double[] x, int xOff, int incX) {
        int xi = xOff;
        for (int i = 0; i < n; i++) {
            x[xi] *= alpha;
            xi += incX;
        }
    }

    private static int scalarSolveLowerN(double[] t, int tOff, int ldt, int n, double[] b, int bOff, boolean nounit) {
        if (n <= 0) {
            return 0;
        }

        double diag = 1.0;
        if (nounit) {
            diag = t[tOff];
            if (diag == 0.0) {
                return 1;
            }
            b[bOff] /= diag;
        }

        for (int j = 1; j < n; j++) {
            int jCol = tOff + j * ldt;
            if (nounit) {
                diag = t[jCol + j];
                if (diag == 0.0) {
                    return j + 1;
                }
            }

            double scale = -b[bOff + j - 1];
            for (int i = 0; i < n - j; i++) {
                b[bOff + j + i] = Math.fma(scale, t[jCol + (j - 1) + i * ldt], b[bOff + j + i]);
            }
            if (nounit) {
                b[bOff + j] /= diag;
            }
        }
        return 0;
    }

    private static int scalarSolveLowerT(double[] t, int tOff, int ldt, int n, double[] b, int bOff, boolean nounit) {
        if (n <= 0) {
            return 0;
        }

        int lastCol = tOff + (n - 1) * ldt;
        double diag = 1.0;
        if (nounit) {
            diag = t[lastCol + (n - 1)];
            if (diag == 0.0) {
                return n;
            }
            b[bOff + n - 1] /= diag;
        }

        for (int jj = 1; jj < n; jj++) {
            int j = n - 1 - jj;
            int jCol = tOff + j * ldt;
            if (nounit) {
                diag = t[jCol + j];
                if (diag == 0.0) {
                    return j + 1;
                }
            }

            double dot = 0.0;
            for (int i = 0; i < jj; i++) {
                dot = Math.fma(t[tOff + (j + 1) * ldt + j + i * ldt], b[bOff + j + 1 + i], dot);
            }
            if (nounit) {
                b[bOff + j] = (b[bOff + j] - dot) / diag;
            } else {
                b[bOff + j] -= dot;
            }
        }
        return 0;
    }

    private static int scalarSolveUpperN(double[] t, int tOff, int ldt, int n, double[] b, int bOff, boolean nounit) {
        if (n <= 0) {
            return 0;
        }

        int lastCol = tOff + (n - 1) * ldt;
        double diag = 1.0;
        if (nounit) {
            diag = t[lastCol + (n - 1)];
            if (diag == 0.0) {
                return n;
            }
            b[bOff + n - 1] /= diag;
        }

        for (int jj = 1; jj < n; jj++) {
            int j = n - 1 - jj;
            int jCol = tOff + j * ldt;
            if (nounit) {
                diag = t[jCol + j];
                if (diag == 0.0) {
                    return j + 1;
                }
            }

            double scale = -b[bOff + j + 1];
            for (int i = 0; i <= j; i++) {
                b[bOff + i] = Math.fma(scale, t[tOff + j + 1 + i * ldt], b[bOff + i]);
            }
            if (nounit) {
                b[bOff + j] /= diag;
            }
        }
        return 0;
    }

    private static int scalarSolveUpperT(double[] t, int tOff, int ldt, int n, double[] b, int bOff, boolean nounit) {
        if (n <= 0) {
            return 0;
        }

        double diag = 1.0;
        if (nounit) {
            diag = t[tOff];
            if (diag == 0.0) {
                return 1;
            }
            b[bOff] /= diag;
        }

        for (int j = 1; j < n; j++) {
            int jCol = tOff + j * ldt;
            if (nounit) {
                diag = t[jCol + j];
                if (diag == 0.0) {
                    return j + 1;
                }
            }

            double dot = 0.0;
            for (int i = 0; i < j; i++) {
                dot = Math.fma(t[tOff + j + i * ldt], b[bOff + i], dot);
            }
            if (nounit) {
                b[bOff + j] = (b[bOff + j] - dot) / diag;
            } else {
                b[bOff + j] -= dot;
            }
        }
        return 0;
    }

    private static double scalarDlatrs(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, boolean normin,
                                       int n, double[] a, int lda, double[] x, int xOff, double[] cnorm, int cnormOff) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean notrans = trans == BLAS.Trans.NoTrans;
        boolean nounit = diag == BLAS.Diag.NonUnit;
        if (n == 0) {
            return 1.0;
        }

        double smlnum = Dlamch.dlamch('S') / Dlamch.dlamch('P');
        double bignum = 1.0 / smlnum;
        double scale = 1.0;

        if (!normin) {
            if (upper) {
                cnorm[cnormOff] = 0.0;
                for (int j = 1; j < n; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < j; i++) {
                        sum += Math.abs(a[i * lda + j]);
                    }
                    cnorm[cnormOff + j] = sum;
                }
            } else {
                for (int j = 0; j < n - 1; j++) {
                    double sum = 0.0;
                    for (int i = j + 1; i < n; i++) {
                        sum += Math.abs(a[i * lda + j]);
                    }
                    cnorm[cnormOff + j] = sum;
                }
                cnorm[cnormOff + n - 1] = 0.0;
            }
        }

        int imax = 0;
        double tmax = cnorm[cnormOff];
        for (int i = 1; i < n; i++) {
            if (cnorm[cnormOff + i] > tmax) {
                tmax = cnorm[cnormOff + i];
                imax = i;
            }
        }

        double tscal = 1.0;
        if (cnorm[cnormOff + imax] > bignum) {
            if (tmax <= Double.MAX_VALUE) {
                tscal = 1.0 / (smlnum * tmax);
                for (int j = 0; j < n; j++) {
                    cnorm[cnormOff + j] *= tscal;
                }
            } else {
                tmax = 0.0;
                if (upper) {
                    for (int j = 1; j < n; j++) {
                        for (int i = 0; i < j; i++) {
                            tmax = Math.max(tmax, Math.abs(a[i * lda + j]));
                        }
                    }
                } else {
                    for (int j = 0; j < n - 1; j++) {
                        for (int i = j + 1; i < n; i++) {
                            tmax = Math.max(tmax, Math.abs(a[i * lda + j]));
                        }
                    }
                }
                if (tmax <= Double.MAX_VALUE) {
                    tscal = 1.0 / (smlnum * tmax);
                    for (int j = 0; j < n; j++) {
                        if (cnorm[cnormOff + j] <= Double.MAX_VALUE) {
                            cnorm[cnormOff + j] *= tscal;
                        } else {
                            cnorm[cnormOff + j] = 0.0;
                            if (upper) {
                                for (int i = 0; i < j; i++) {
                                    cnorm[cnormOff + j] += tscal * Math.abs(a[i * lda + j]);
                                }
                            } else {
                                for (int i = j + 1; i < n; i++) {
                                    cnorm[cnormOff + j] += tscal * Math.abs(a[i * lda + j]);
                                }
                            }
                        }
                    }
                } else {
                    scalarDtrsl(a, 0, lda, n, x, xOff,
                        upper ? BLAS.Uplo.Upper : BLAS.Uplo.Lower,
                        notrans ? BLAS.Trans.NoTrans : BLAS.Trans.Trans,
                        diag);
                    return scale;
                }
            }
        }

        int jfirst;
        int jlast;
        int jinc;
        if (notrans) {
            if (upper) {
                jfirst = n - 1;
                jlast = -1;
                jinc = -1;
            } else {
                jfirst = 0;
                jlast = n;
                jinc = 1;
            }
        } else if (upper) {
            jfirst = 0;
            jlast = n;
            jinc = 1;
        } else {
            jfirst = n - 1;
            jlast = -1;
            jinc = -1;
        }

        double xmax = Math.abs(x[xOff]);
        for (int i = 1; i < n; i++) {
            if (Math.abs(x[xOff + i]) > xmax) {
                xmax = Math.abs(x[xOff + i]);
            }
        }
        double xbnd = xmax;

        double grow = 0.0;
        if (tscal == 1.0) {
            if (nounit) {
                grow = 1.0 / Math.max(xbnd, smlnum);
                xbnd = grow;
                for (int j = jfirst; j != jlast; j += jinc) {
                    if (grow <= smlnum) {
                        break;
                    }
                    double tjj = Math.abs(a[j * lda + j]);
                    xbnd = Math.min(xbnd, Math.min(1.0, tjj) * grow);
                    if (tjj + cnorm[cnormOff + j] >= smlnum) {
                        grow *= tjj / (tjj + cnorm[cnormOff + j]);
                    } else {
                        grow = 0.0;
                    }
                }
                grow = xbnd;
            } else {
                grow = Math.min(1.0, 1.0 / Math.max(xbnd, smlnum));
                for (int j = jfirst; j != jlast; j += jinc) {
                    if (grow <= smlnum) {
                        break;
                    }
                    grow *= 1.0 / (1.0 + cnorm[cnormOff + j]);
                }
            }
        }

        if (grow * tscal > smlnum) {
            scalarDtrsl(a, 0, lda, n, x, xOff,
                upper ? BLAS.Uplo.Upper : BLAS.Uplo.Lower,
                notrans ? BLAS.Trans.NoTrans : BLAS.Trans.Trans,
                diag);
            if (tscal != 1.0) {
                for (int j = 0; j < n; j++) {
                    cnorm[cnormOff + j] /= tscal;
                }
            }
            return scale;
        }

        if (xmax > bignum) {
            scale = bignum / xmax;
            BLAS.dscal(n, scale, x, xOff, 1);
            xmax = bignum;
        }

        if (notrans) {
            for (int j = jfirst; j != jlast; j += jinc) {
                double xj = Math.abs(x[xOff + j]);
                double tjjs = nounit ? a[j * lda + j] * tscal : tscal;
                double tjj = Math.abs(tjjs);

                if (tjj > smlnum) {
                    if (tjj < 1.0 && xj > tjj * bignum) {
                        double rec = 1.0 / xj;
                        BLAS.dscal(n, rec, x, xOff, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                    x[xOff + j] /= tjjs;
                    xj = Math.abs(x[xOff + j]);
                } else if (tjj > 0.0) {
                    if (xj > tjj * bignum) {
                        double rec = (tjj * bignum) / xj;
                        if (cnorm[cnormOff + j] > 1.0) {
                            rec /= cnorm[cnormOff + j];
                        }
                        BLAS.dscal(n, rec, x, xOff, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                    x[xOff + j] /= tjjs;
                    xj = Math.abs(x[xOff + j]);
                } else {
                    for (int i = 0; i < n; i++) {
                        x[xOff + i] = 0.0;
                    }
                    x[xOff + j] = 1.0;
                    xj = 1.0;
                    scale = 0.0;
                    xmax = 0.0;
                }

                if (xj > 1.0) {
                    double rec = 1.0 / xj;
                    if (cnorm[cnormOff + j] > (bignum - xmax) * rec) {
                        rec *= 0.5;
                        BLAS.dscal(n, rec, x, xOff, 1);
                        scale *= rec;
                    }
                } else if (xj * cnorm[cnormOff + j] > bignum - xmax) {
                    BLAS.dscal(n, 0.5, x, xOff, 1);
                    scale *= 0.5;
                }

                if (upper) {
                    if (j > 0) {
                        BLAS.daxpy(j, -x[xOff + j] * tscal, a, j, lda, x, xOff, 1);
                        xmax = Math.abs(x[xOff]);
                        for (int i = 1; i < j; i++) {
                            if (Math.abs(x[xOff + i]) > xmax) {
                                xmax = Math.abs(x[xOff + i]);
                            }
                        }
                    }
                } else if (j < n - 1) {
                    BLAS.daxpy(n - j - 1, -x[xOff + j] * tscal, a, (j + 1) * lda + j, lda, x, xOff + j + 1, 1);
                    xmax = Math.abs(x[xOff + j + 1]);
                    for (int i = j + 2; i < n; i++) {
                        if (Math.abs(x[xOff + i]) > xmax) {
                            xmax = Math.abs(x[xOff + i]);
                        }
                    }
                }
            }
        } else {
            for (int j = jfirst; j != jlast; j += jinc) {
                double xj = Math.abs(x[xOff + j]);
                double uscal = tscal;
                double rec = 1.0 / Math.max(xmax, 1.0);

                if (cnorm[cnormOff + j] > (bignum - xj) * rec) {
                    rec *= 0.5;
                    double tjjs = nounit ? a[j * lda + j] * tscal : tscal;
                    double tjj = Math.abs(tjjs);
                    if (tjj > 1.0) {
                        rec = Math.min(1.0, rec * tjj);
                        uscal /= tjjs;
                    }
                    if (rec < 1.0) {
                        BLAS.dscal(n, rec, x, xOff, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                }

                double sumj = 0.0;
                if (uscal == 1.0) {
                    if (upper) {
                        for (int i = 0; i < j; i++) {
                            sumj += a[i * lda + j] * x[xOff + i];
                        }
                    } else if (j < n - 1) {
                        for (int i = j + 1; i < n; i++) {
                            sumj += a[i * lda + j] * x[xOff + i];
                        }
                    }
                } else if (upper) {
                    for (int i = 0; i < j; i++) {
                        sumj += (a[i * lda + j] * uscal) * x[xOff + i];
                    }
                } else {
                    for (int i = j + 1; i < n; i++) {
                        sumj += (a[i * lda + j] * uscal) * x[xOff + i];
                    }
                }

                if (uscal == tscal) {
                    x[xOff + j] -= sumj;
                    xj = Math.abs(x[xOff + j]);
                    double tjjs = nounit ? a[j * lda + j] * tscal : tscal;
                    double tjj = Math.abs(tjjs);

                    if (tjj > smlnum) {
                        if (tjj < 1.0 && xj > tjj * bignum) {
                            rec = 1.0 / xj;
                            BLAS.dscal(n, rec, x, xOff, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[xOff + j] /= tjjs;
                    } else if (tjj > 0.0) {
                        if (xj > tjj * bignum) {
                            rec = (tjj * bignum) / xj;
                            BLAS.dscal(n, rec, x, xOff, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[xOff + j] /= tjjs;
                    } else {
                        for (int i = 0; i < n; i++) {
                            x[xOff + i] = 0.0;
                        }
                        x[xOff + j] = 1.0;
                        scale = 0.0;
                        xmax = 0.0;
                    }
                } else {
                    double tjjs = nounit ? a[j * lda + j] * tscal : tscal;
                    x[xOff + j] = x[xOff + j] / tjjs - sumj;
                }

                xmax = Math.max(xmax, Math.abs(x[xOff + j]));
            }
        }

        scale /= tscal;
        if (tscal != 1.0) {
            for (int j = 0; j < n; j++) {
                cnorm[cnormOff + j] /= tscal;
            }
        }
        return scale;
    }
}