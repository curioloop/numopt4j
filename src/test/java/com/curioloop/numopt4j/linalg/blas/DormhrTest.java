/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class DormhrTest {

    private static final double EPSILON = 1e-10;

    @Test
    void testDormhr_WorkspaceQuery() {
        int m = 4, n = 3;
        int ilo = 0, ihi = 3;

        double[] a = new double[m * m];
        double[] tau = new double[m - 1];
        double[] c = new double[m * n];
        double[] work = new double[1];

        Dgehrd.dormhr(BLAS.Side.Left, BLAS.Transpose.NoTrans, m, n, ilo, ihi,
                      a, 0, m, tau, 0, c, 0, n, work, 0, -1);

        assertThat(work[0]).isGreaterThanOrEqualTo(1.0);
    }

    @Test
    void testDormhr_ZeroMatrix() {
        int m = 4, n = 3;
        int ilo = 0, ihi = 3;

        double[] a = new double[m * m];
        double[] tau = new double[m - 1];
        double[] c = new double[m * n];
        double[] work = new double[100];

        Dgehrd.dormhr(BLAS.Side.Left, BLAS.Transpose.NoTrans, m, n, ilo, ihi,
                      a, 0, m, tau, 0, c, 0, n, work, 0, 100);

        for (int i = 0; i < c.length; i++) {
            assertThat(c[i]).isCloseTo(0.0, within(EPSILON));
        }
    }

    @Test
    void testDormhr_NhZero() {
        int m = 4, n = 3;
        int ilo = 2, ihi = 2;

        double[] a = new double[m * m];
        for (int i = 0; i < m; i++) {
            a[i * m + i] = 1.0;
        }
        double[] tau = new double[m - 1];
        double[] work = new double[100];

        double[] c = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i * n + j] = i + j + 1;
            }
        }
        double[] cOrig = c.clone();

        Dgehrd.dormhr(BLAS.Side.Left, BLAS.Transpose.NoTrans, m, n, ilo, ihi,
                      a, 0, m, tau, 0, c, 0, n, work, 0, 100);

        for (int i = 0; i < c.length; i++) {
            assertThat(c[i]).isCloseTo(cOrig[i], within(EPSILON));
        }
    }

    @Test
    void testDormhr_IdentityMatrix() {
        int m = 4, n = 3;
        int ilo = 0, ihi = 3;

        double[] a = new double[m * m];
        for (int i = 0; i < m; i++) {
            a[i * m + i] = 1.0;
        }
        double[] tau = new double[m - 1];
        double[] work = new double[100];

        double[] c = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i * n + j] = i + j + 1;
            }
        }
        double[] cOrig = c.clone();

        Dgehrd.dormhr(BLAS.Side.Left, BLAS.Transpose.NoTrans, m, n, ilo, ihi,
                      a, 0, m, tau, 0, c, 0, n, work, 0, 100);

        for (int i = 0; i < c.length; i++) {
            assertThat(c[i]).isCloseTo(cOrig[i], within(EPSILON));
        }
    }

    @Test
    void testDormhr_EmptyMatrix() {
        int m = 0, n = 3;
        double[] a = new double[0];
        double[] tau = new double[0];
        double[] c = new double[0];
        double[] work = new double[1];

        Dgehrd.dormhr(BLAS.Side.Left, BLAS.Transpose.NoTrans, m, n, 0, 0,
                      a, 0, 1, tau, 0, c, 0, n, work, 0, 1);

        assertThat(work[0]).isEqualTo(1.0);
    }
}
