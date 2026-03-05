/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlamchTest {

    @Test
    void testEpsilon() {
        double eps = Dlamch.dlamch('E');
        assertTrue(eps > 0);
        assertTrue(eps < 1);
    }

    @Test
    void testSafeMin() {
        double smin = Dlamch.dlamch('S');
        assertTrue(smin > 0);
    }

    @Test
    void testPrecision() {
        double prec = Dlamch.dlamch('P');
        assertTrue(prec > 0);
    }
}
