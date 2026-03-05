/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.blas.BLAS;

import java.lang.reflect.Method;

public class CholeskyTestAccess {

    public static void dtrti2(char uplo, char diag, int n, double[] A, int aOff, int lda) {
        try {
            Class<?> dtrtriClass = Class.forName("com.curioloop.numopt4j.linalg.blas.Dtrtri");
            Method method = dtrtriClass.getDeclaredMethod("dtrti2", BLAS.Uplo.class, BLAS.Diag.class, int.class, double[].class, int.class, int.class);
            method.setAccessible(true);
            BLAS.Uplo uploEnum = (uplo == 'U' || uplo == 'u') ? BLAS.Uplo.Upper : BLAS.Uplo.Lower;
            BLAS.Diag diagEnum = (diag == 'U' || diag == 'u') ? BLAS.Diag.Unit : BLAS.Diag.NonUnit;
            method.invoke(null, uploEnum, diagEnum, n, A, aOff, lda);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static void dlauu2(char uplo, int n, double[] A, int aOff, int lda) {
        try {
            Class<?> dlauumClass = Class.forName("com.curioloop.numopt4j.linalg.blas.Dlauum");
            Method method = dlauumClass.getDeclaredMethod("dlauu2", BLAS.Uplo.class, int.class, double[].class, int.class, int.class);
            method.setAccessible(true);
            BLAS.Uplo uploEnum = (uplo == 'U' || uplo == 'u') ? BLAS.Uplo.Upper : BLAS.Uplo.Lower;
            method.invoke(null, uploEnum, n, A, aOff, lda);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
