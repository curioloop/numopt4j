/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class SVDConditionTest {

    private static final double EPSILON = 1e-10;

    @Test
    void testConditionNumber() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        
        SVD svd = SVD.decompose(A, 2, 2);
        
        assertThat(svd.ok()).isTrue();
        assertThat(svd.cond()).isCloseTo(14.933034373659254, offset(EPSILON));
    }

    @Test
    void testNorm2() {
        double[] A = {
            3.0, 0.0,
            0.0, 4.0
        };
        
        SVD svd = SVD.decompose(A, 2, 2);
        
        assertThat(svd.ok()).isTrue();
        assertThat(svd.norm2()).isCloseTo(4.0, offset(EPSILON));
    }

    @Test
    void testSingularMatrix() {
        double[] A = {
            1.0, 2.0,
            2.0, 4.0
        };
        
        SVD svd = SVD.decompose(A, 2, 2);
        
        assertThat(svd.ok()).isTrue();
        assertThat(svd.cond()).isGreaterThan(1e15);
    }

    @Test
    void testWellConditioned() {
        double[] A = {
            1.0, 0.0,
            0.0, 1.0
        };
        
        SVD svd = SVD.decompose(A, 2, 2);
        
        assertThat(svd.ok()).isTrue();
        assertThat(svd.cond()).isCloseTo(1.0, offset(EPSILON));
    }

    @Test
    void testIllConditioned() {
        double[] A = {
            1.0, 1.0,
            1.0, 1.0 + 1e-10
        };
        
        SVD svd = SVD.decompose(A, 2, 2);
        
        assertThat(svd.ok()).isTrue();
        assertThat(svd.cond()).isGreaterThan(1e9);
    }
}
