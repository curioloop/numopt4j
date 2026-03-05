/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class Dlaqr1Test {

    private static final double EPSILON = 1e-12;

    @Test
    void testDlaqr1_2x2() {
        double[] h = {
            4.0, 1.0,
            2.0, 3.0
        };
        double sr1 = 5.0, si1 = 0.0;
        double sr2 = 2.0, si2 = 0.0;
        double[] v = new double[2];

        Dlaqr.dlaqr1(2, h, 0, 2, sr1, si1, sr2, si2, v, 0);

        assertThat(v[0]).isFinite();
        assertThat(v[1]).isFinite();
    }

    @Test
    void testDlaqr1_3x3() {
        double[] h = {
            4.0, 1.0, 1.0,
            2.0, 3.0, 1.0,
            0.0, 2.0, 2.0
        };
        double sr1 = 5.0, si1 = 0.0;
        double sr2 = 2.0, si2 = 0.0;
        double[] v = new double[3];

        Dlaqr.dlaqr1(3, h, 0, 3, sr1, si1, sr2, si2, v, 0);

        assertThat(v[0]).isFinite();
        assertThat(v[1]).isFinite();
        assertThat(v[2]).isFinite();
    }

    @Test
    void testDlaqr1_ComplexShifts() {
        double[] h = {
            4.0, 1.0, 1.0,
            2.0, 3.0, 1.0,
            0.0, 2.0, 2.0
        };
        double sr1 = 3.0, si1 = 1.0;
        double sr2 = 3.0, si2 = -1.0;
        double[] v = new double[3];

        Dlaqr.dlaqr1(3, h, 0, 3, sr1, si1, sr2, si2, v, 0);

        assertThat(v[0]).isNotEqualTo(0.0);
        assertThat(v[1]).isNotEqualTo(0.0);
        assertThat(v[2]).isNotEqualTo(0.0);
    }

    @Test
    void testDlaqr1_ZeroSubdiagonal() {
        double[] h = {
            4.0, 1.0,
            0.0, 3.0
        };
        double sr1 = 5.0, si1 = 0.0;
        double sr2 = 2.0, si2 = 0.0;
        double[] v = new double[2];

        Dlaqr.dlaqr1(2, h, 0, 2, sr1, si1, sr2, si2, v, 0);

        assertThat(v[0]).isFinite();
        assertThat(v[1]).isFinite();
    }
}
