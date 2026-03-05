/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition.Part;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class GEVDTest {

    private static final double EPSILON = 1e-8;

    @Test
    void testType1Basic() {
        double[] A = {
            2.0, 1.0,
            1.0, 2.0
        };
        double[] B = {
            1.0, 0.0,
            0.0, 1.0
        };
        
        GEVD eg = GEVD.decompose(A, B, 2, 'L');
        
        assertThat(eg.ok()).isTrue();
        assertThat(eg.extract(Part.S).data).hasSize(2);
        assertThat(eg.extract(Part.S).data[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(eg.extract(Part.S).data[1]).isCloseTo(3.0, offset(EPSILON));
    }

    @Test
    void testType1WithNonIdentityB() {
        double[] A = {
            2.0, 1.0,
            1.0, 2.0
        };
        double[] B = {
            4.0, 0.0,
            0.0, 1.0
        };
        
        GEVD eg = GEVD.decompose(A, B, 2, 'L');
        
        assertThat(eg.ok()).isTrue();
        assertThat(eg.extract(Part.S).data).hasSize(2);
        
        double[] w = eg.extract(Part.S).data;
        double[] v = eg.extract(Part.Q).data;
        
        for (int i = 0; i < 2; i++) {
            double[] Av = new double[2];
            double[] Bv = new double[2];
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    double[] Aorig = {2.0, 1.0, 1.0, 2.0};
                    double[] Borig = {4.0, 0.0, 0.0, 1.0};
                    Av[j] += Aorig[j * 2 + k] * v[k * 2 + i];
                    Bv[j] += Borig[j * 2 + k] * v[k * 2 + i];
                }
            }
            
            for (int j = 0; j < 2; j++) {
                assertThat(Av[j]).isCloseTo(w[i] * Bv[j], offset(1e-6));
            }
        }
    }

    @Test
    void testType2() {
        double[] A = {
            2.0, 1.0,
            1.0, 2.0
        };
        double[] B = {
            1.0, 0.0,
            0.0, 1.0
        };
        
        GEVD eg = GEVD.decompose(A, B, 2, 'L', 2, null);
        
        assertThat(eg.ok()).isTrue();
        assertThat(eg.type()).isEqualTo(2);
    }

    @Test
    void testType3() {
        double[] A = {
            2.0, 1.0,
            1.0, 2.0
        };
        double[] B = {
            1.0, 0.0,
            0.0, 1.0
        };
        
        GEVD eg = GEVD.decompose(A, B, 2, 'L', 3, null);
        
        assertThat(eg.ok()).isTrue();
        assertThat(eg.type()).isEqualTo(3);
    }

    @Test
    void testConditionNumber() {
        double[] A = {
            2.0, 1.0,
            1.0, 2.0
        };
        double[] B = {
            1.0, 0.0,
            0.0, 1.0
        };
        
        GEVD eg = GEVD.decompose(A, B, 2, 'L');
        
        assertThat(eg.ok()).isTrue();
        assertThat(eg.cond()).isCloseTo(3.0, offset(EPSILON));
    }

    @Test
    void testLargerMatrix() {
        double[] A = {
            6.0, 3.0, 1.0,
            3.0, 5.0, 2.0,
            1.0, 2.0, 4.0
        };
        double[] B = {
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0
        };
        
        GEVD eg = GEVD.decompose(A, B, 3, 'L');
        
        assertThat(eg.ok()).isTrue();
        assertThat(eg.extract(Part.S).data).hasSize(3);
        
        double[] eigenvalues = eg.extract(Part.S).data;
        for (int i = 1; i < 3; i++) {
            assertThat(eigenvalues[i]).isGreaterThanOrEqualTo(eigenvalues[i-1]);
        }
    }
}
