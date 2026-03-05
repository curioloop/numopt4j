/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;

public interface Dlatrs {

    static double dlatrs(BLAS.Uplo uplo, BLAS.Transpose trans, BLAS.Diag diag, boolean normin,
                         int n, double[] A, int lda, double[] x, int xOff, double[] cnorm, int cnormOff) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean notrans = trans == BLAS.Transpose.NoTrans;
        boolean nounit = diag == BLAS.Diag.NonUnit;
        if (n == 0) return 1.0;

        double smlnum = Dlamch.dlamch('S') / Dlamch.dlamch('P');
        double bignum = 1.0 / smlnum;
        double scale = 1.0;

        if (!normin) {
            if (upper) {
                cnorm[cnormOff] = 0;
                for (int j = 1; j < n; j++) {
                    double sum = 0;
                    for (int i = 0; i < j; i++) {
                        sum += abs(A[i * lda + j]);
                    }
                    cnorm[cnormOff + j] = sum;
                }
            } else {
                for (int j = 0; j < n - 1; j++) {
                    double sum = 0;
                    for (int i = j + 1; i < n; i++) {
                        sum += abs(A[i * lda + j]);
                    }
                    cnorm[cnormOff + j] = sum;
                }
                cnorm[cnormOff + n - 1] = 0;
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
                tmax = 0;
                if (upper) {
                    for (int j = 1; j < n; j++) {
                        for (int i = 0; i < j; i++) {
                            tmax = max(tmax, abs(A[i * lda + j]));
                        }
                    }
                } else {
                    for (int j = 0; j < n - 1; j++) {
                        for (int i = j + 1; i < n; i++) {
                            tmax = max(tmax, abs(A[i * lda + j]));
                        }
                    }
                }
                if (tmax <= Double.MAX_VALUE) {
                    tscal = 1.0 / (smlnum * tmax);
                    for (int j = 0; j < n; j++) {
                        if (cnorm[cnormOff + j] <= Double.MAX_VALUE) {
                            cnorm[cnormOff + j] *= tscal;
                        } else {
                            cnorm[cnormOff + j] = 0;
                            if (upper) {
                                for (int i = 0; i < j; i++) {
                                    cnorm[cnormOff + j] += tscal * abs(A[i * lda + j]);
                                }
                            } else {
                                for (int i = j + 1; i < n; i++) {
                                    cnorm[cnormOff + j] += tscal * abs(A[i * lda + j]);
                                }
                            }
                        }
                    }
                } else {
                    Dtrsl.dtrsl(A, 0, lda, n, x, xOff, upper ? BLAS.Uplo.Upper : BLAS.Uplo.Lower, notrans ? BLAS.Transpose.NoTrans : BLAS.Transpose.Trans);
                    return scale;
                }
            }
        }

        int jfirst, jlast, jinc;
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
        } else {
            if (upper) {
                jfirst = 0;
                jlast = n;
                jinc = 1;
            } else {
                jfirst = n - 1;
                jlast = -1;
                jinc = -1;
            }
        }

        int jmax = 0;
        double xmax = abs(x[xOff]);
        for (int i = 1; i < n; i++) {
            if (abs(x[xOff + i]) > xmax) {
                xmax = abs(x[xOff + i]);
                jmax = i;
            }
        }
        double xbnd = xmax;

        double grow = 0;
        if (tscal == 1.0) {
            if (nounit) {
                grow = 1.0 / max(xbnd, smlnum);
                xbnd = grow;
                for (int j = jfirst; j != jlast; j += jinc) {
                    if (grow <= smlnum) break;
                    double tjj = abs(A[j * lda + j]);
                    xbnd = min(xbnd, min(1.0, tjj) * grow);
                    if (tjj + cnorm[cnormOff + j] >= smlnum) {
                        grow *= tjj / (tjj + cnorm[cnormOff + j]);
                    } else {
                        grow = 0;
                    }
                }
                grow = xbnd;
            } else {
                grow = min(1.0, 1.0 / max(xbnd, smlnum));
                for (int j = jfirst; j != jlast; j += jinc) {
                    if (grow <= smlnum) break;
                    grow *= 1.0 / (1.0 + cnorm[cnormOff + j]);
                }
            }
        }

        if (grow * tscal > smlnum) {
            Dtrsl.dtrsl(A, 0, lda, n, x, xOff, upper ? BLAS.Uplo.Upper : BLAS.Uplo.Lower, notrans ? BLAS.Transpose.NoTrans : BLAS.Transpose.Trans);
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
                double xj = abs(x[xOff + j]);
                double tjjs = nounit ? A[j * lda + j] * tscal : tscal;
                double tjj = abs(tjjs);

                if (tjj > smlnum) {
                    if (tjj < 1 && xj > tjj * bignum) {
                        double rec = 1.0 / xj;
                        BLAS.dscal(n, rec, x, xOff, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                    x[xOff + j] /= tjjs;
                    xj = abs(x[xOff + j]);
                } else if (tjj > 0) {
                    if (xj > tjj * bignum) {
                        double rec = (tjj * bignum) / xj;
                        if (cnorm[cnormOff + j] > 1) rec /= cnorm[cnormOff + j];
                        BLAS.dscal(n, rec, x, xOff, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                    x[xOff + j] /= tjjs;
                    xj = abs(x[xOff + j]);
                } else {
                    for (int i = 0; i < n; i++) x[xOff + i] = 0;
                    x[xOff + j] = 1;
                    xj = 1;
                    scale = 0;
                    xmax = 0;
                }

                if (xj > 1) {
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
                        BLAS.daxpy(j, -x[xOff + j] * tscal, A, j, lda, x, xOff, 1);
                        int im = 0;
                        double xm = abs(x[xOff]);
                        for (int i = 1; i < j; i++) {
                            if (abs(x[xOff + i]) > xm) {
                                xm = abs(x[xOff + i]);
                                im = i;
                            }
                        }
                        xmax = abs(x[xOff + im]);
                    }
                } else {
                    if (j < n - 1) {
                        BLAS.daxpy(n - j - 1, -x[xOff + j] * tscal, A, (j + 1) * lda + j, lda, x, xOff + j + 1, 1);
                        int im = j + 1;
                        double xm = abs(x[xOff + j + 1]);
                        for (int i = j + 2; i < n; i++) {
                            if (abs(x[xOff + i]) > xm) {
                                xm = abs(x[xOff + i]);
                                im = i;
                            }
                        }
                        xmax = abs(x[xOff + im]);
                    }
                }
            }
        } else {
            for (int j = jfirst; j != jlast; j += jinc) {
                double xj = abs(x[xOff + j]);
                double uscal = tscal;
                double rec = 1.0 / max(xmax, 1.0);

                if (cnorm[cnormOff + j] > (bignum - xj) * rec) {
                    rec *= 0.5;
                    double tjjs = nounit ? A[j * lda + j] * tscal : tscal;
                    double tjj = abs(tjjs);
                    if (tjj > 1) {
                        rec = min(1.0, rec * tjj);
                        uscal /= tjjs;
                    }
                    if (rec < 1) {
                        BLAS.dscal(n, rec, x, xOff, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                }

                double sumj = 0;
                if (uscal == 1.0) {
                    if (upper) {
                        for (int i = 0; i < j; i++) {
                            sumj += A[i * lda + j] * x[xOff + i];
                        }
                    } else if (j < n - 1) {
                        for (int i = j + 1; i < n; i++) {
                            sumj += A[i * lda + j] * x[xOff + i];
                        }
                    }
                } else {
                    if (upper) {
                        for (int i = 0; i < j; i++) {
                            sumj += (A[i * lda + j] * uscal) * x[xOff + i];
                        }
                    } else if (j < n) {
                        for (int i = j + 1; i < n; i++) {
                            sumj += (A[i * lda + j] * uscal) * x[xOff + i];
                        }
                    }
                }

                if (uscal == tscal) {
                    x[xOff + j] -= sumj;
                    xj = abs(x[xOff + j]);
                    double tjjs = nounit ? A[j * lda + j] * tscal : tscal;
                    double tjj = abs(tjjs);

                    if (tjj > smlnum) {
                        if (tjj < 1 && xj > tjj * bignum) {
                            rec = 1.0 / xj;
                            BLAS.dscal(n, rec, x, xOff, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[xOff + j] /= tjjs;
                    } else if (tjj > 0) {
                        if (xj > tjj * bignum) {
                            rec = (tjj * bignum) / xj;
                            BLAS.dscal(n, rec, x, xOff, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[xOff + j] /= tjjs;
                    } else {
                        for (int i = 0; i < n; i++) x[xOff + i] = 0;
                        x[xOff + j] = 1;
                        scale = 0;
                        xmax = 0;
                    }
                } else {
                    double tjjs = nounit ? A[j * lda + j] * tscal : tscal;
                    x[xOff + j] = x[xOff + j] / tjjs - sumj;
                }

                xmax = max(xmax, abs(x[xOff + j]));
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
