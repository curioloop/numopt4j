/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DgebakTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        int n = 3;
        double[] V = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        double[] scale = {1, 1, 1};

        Dgebak.dgebak('B', BLAS.Side.Right, n, 0, n - 1, scale, 0, n, V, n);

        assertTrue(Double.isFinite(V[0]));
    }

    @Test
    void testEmpty() {
        Dgebak.dgebak('N', BLAS.Side.Right, 0, 0, -1, new double[0], 0, 0, new double[0], 0);
    }

    @Test
    void testNoOp() {
        double[] V = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        double[] VOrig = V.clone();
        double[] scale = {1, 1, 1};

        Dgebak.dgebak('N', BLAS.Side.Right, 3, 0, 2, scale, 0, 3, V, 3);

        for (int i = 0; i < V.length; i++) {
            assertEquals(VOrig[i], V[i], TOL);
        }
    }
}
