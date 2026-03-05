/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DormqrTest {

    private static final double TOL = 1e-10;

    @Test
    void testEmpty() {
        int info = Dormqr.dormqr(BLAS.Side.Left, BLAS.Transpose.NoTrans, 0, 0, 0, new double[0], 0, 0, new double[0], 0, new double[0], 0, 0, new double[1], 0);
        assertEquals(0, info);
    }
}
