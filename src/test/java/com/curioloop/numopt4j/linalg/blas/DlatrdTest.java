/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlatrdTest {

    @Test
    void testBasic() {
        double[] A = {
            1, 2, 3, 4,
            0, 5, 6, 7,
            0, 0, 8, 9,
            0, 0, 0, 10
        };
        int n = 4, nb = 2;
        double[] e = new double[n - 1];
        double[] tau = new double[n - 1];
        double[] w = new double[n * nb];

        Dlatrd.dlatrd(BLAS.Uplo.Upper, n, nb, A, n, e, tau, w, nb);

        assertTrue(Double.isFinite(e[0]));
    }

    @Test
    void testEmpty() {
        Dlatrd.dlatrd(BLAS.Uplo.Upper, 0, 0, new double[0], 0, new double[0], new double[0], new double[0], 0);
    }
}
