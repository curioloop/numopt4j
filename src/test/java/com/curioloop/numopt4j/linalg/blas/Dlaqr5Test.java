/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class Dlaqr5Test {

    private static final double EPSILON = 1e-10;

    @Test
    void testDlaqr5_Simple() {
        int n = 4;
        double[] h = {
            4.0, 1.0, 0.0, 0.0,
            2.0, 3.0, 1.0, 0.0,
            0.0, 2.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 1.0
        };

        int nshfts = 2;
        double[] sr = {3.0, 2.0};
        double[] si = {0.0, 0.0};

        double[] v = new double[3 * n];
        double[] u = new double[4 * n];
        double[] wv = new double[n * n];
        double[] wh = new double[n * n];

        Dlaqr.dlaqr5(false, false, 0, n, 0, n - 1, nshfts,
                      sr, 0, si, 0,
                      h, 0, n, 0, n - 1, null, 0, n,
                      v, 0, 3, u, 0, 4, n, wv, 0, n, n, wh, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j < i - 1) {
                    assertThat(h[i * n + j]).isCloseTo(0.0, within(EPSILON));
                }
            }
        }
    }

    @Test
    void testDlaqr5_ComplexShifts() {
        int n = 4;
        double[] h = {
            4.0, 1.0, 0.0, 0.0,
            2.0, 3.0, 1.0, 0.0,
            0.0, 2.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 1.0
        };

        int nshfts = 4;
        double[] sr = {3.0, 3.0, 2.0, 2.0};
        double[] si = {1.0, -1.0, 0.5, -0.5};

        double[] v = new double[3 * n];
        double[] u = new double[4 * n];
        double[] wv = new double[n * n];
        double[] wh = new double[n * n];

        Dlaqr.dlaqr5(false, false, 0, n, 0, n - 1, nshfts,
                      sr, 0, si, 0,
                      h, 0, n, 0, n - 1, null, 0, n,
                      v, 0, 3, u, 0, 4, n, wv, 0, n, n, wh, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j < i - 1) {
                    assertThat(h[i * n + j]).isCloseTo(0.0, within(EPSILON));
                }
            }
        }
    }

    @Test
    void testDlaqr5_WantT() {
        int n = 4;
        double[] h = {
            4.0, 1.0, 2.0, 3.0,
            2.0, 3.0, 1.0, 4.0,
            0.0, 2.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 1.0
        };

        int nshfts = 2;
        double[] sr = {3.0, 2.0};
        double[] si = {0.0, 0.0};

        double[] v = new double[3 * n];
        double[] u = new double[4 * n];
        double[] wv = new double[n * n];
        double[] wh = new double[n * n];

        Dlaqr.dlaqr5(true, false, 0, n, 0, n - 1, nshfts,
                      sr, 0, si, 0,
                      h, 0, n, 0, n - 1, null, 0, n,
                      v, 0, 3, u, 0, 4, n, wv, 0, n, n, wh, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j < i - 1) {
                    assertThat(h[i * n + j]).isCloseTo(0.0, within(EPSILON));
                }
            }
        }
    }

    @Test
    void testDlaqr5_NoShifts() {
        int n = 4;
        double[] h = {
            4.0, 1.0, 0.0, 0.0,
            2.0, 3.0, 1.0, 0.0,
            0.0, 2.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 1.0
        };
        double[] hOrig = h.clone();

        int nshfts = 0;
        double[] sr = new double[0];
        double[] si = new double[0];

        double[] v = new double[3 * n];
        double[] u = new double[4 * n];
        double[] wv = new double[n * n];
        double[] wh = new double[n * n];

        Dlaqr.dlaqr5(false, false, 0, n, 0, n - 1, nshfts,
                      sr, 0, si, 0,
                      h, 0, n, 0, n - 1, null, 0, n,
                      v, 0, 3, u, 0, 4, n, wv, 0, n, n, wh, 0, n);

        for (int i = 0; i < h.length; i++) {
            assertThat(h[i]).isCloseTo(hOrig[i], within(EPSILON));
        }
    }

    @Test
    void testDlaqr5_SingleShift() {
        int n = 4;
        double[] h = {
            4.0, 1.0, 0.0, 0.0,
            2.0, 3.0, 1.0, 0.0,
            0.0, 2.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 1.0
        };
        double[] hOrig = h.clone();

        int nshfts = 1;
        double[] sr = {3.0};
        double[] si = {0.0};

        double[] v = new double[3 * n];
        double[] u = new double[4 * n];
        double[] wv = new double[n * n];
        double[] wh = new double[n * n];

        Dlaqr.dlaqr5(false, false, 0, n, 0, n - 1, nshfts,
                      sr, 0, si, 0,
                      h, 0, n, 0, n - 1, null, 0, n,
                      v, 0, 3, u, 0, 4, n, wv, 0, n, n, wh, 0, n);

        for (int i = 0; i < h.length; i++) {
            assertThat(h[i]).isCloseTo(hOrig[i], within(EPSILON));
        }
    }
}
