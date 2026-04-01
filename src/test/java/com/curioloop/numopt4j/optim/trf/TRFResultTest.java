/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.Optimization;


import org.junit.jupiter.api.Test;

import java.util.function.BiConsumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Tests for {@link Optimization} fields returned by TRF.
 */
class TRFResultTest {

    private static final int M = 5;
    private static final double[] X_DATA = {0, 1, 2, 3, 4};
    private static final double[] Y_DATA = {1, 3, 5, 7, 9};

    private Optimization runLinearFit() {
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < M; i++) r[i] = Y_DATA[i] - (c[0] + c[1] * X_DATA[i]);
        };
        return new TRFProblem()
            .residuals(fn, M)
            .initialPoint(0.0, 1.0)
            .gradientTolerance(1e-10).parameterTolerance(1e-10).maxEvaluations(1000)
            .solve();
    }

    @Test
    void convergenceStatusIsSet() {
        Optimization r = runLinearFit();
        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getStatus().isConverged()).isTrue();
    }

    @Test
    void chiSquaredIsNearZeroForExactFit() {
        Optimization r = runLinearFit();
        assertThat(r.getCost()).isGreaterThanOrEqualTo(0.0);
        assertThat(r.getCost()).isLessThan(1e-10);
    }

    @Test
    void solutionContainsCorrectCoefficients() {
        Optimization r = runLinearFit();
        double[] sol = r.getSolution();
        assertThat(sol).hasSize(2);
        assertThat(sol[0]).isCloseTo(1.0, within(1e-6)); // intercept
        assertThat(sol[1]).isCloseTo(2.0, within(1e-6)); // slope
    }

    @Test
    void evaluationAndIterationCountsArePositive() {
        Optimization r = runLinearFit();
        assertThat(r.getEvaluations()).isGreaterThan(0);
        assertThat(r.getIterations()).isGreaterThanOrEqualTo(0);
    }

    @Test
    void toStringContainsClassName() {
        assertThat(runLinearFit().toString()).contains("Optimization {");
    }

    @Test
    void maxEvaluationsStatusIsReported() {
        final int m = 10;
        double[] xd = {0,1,2,3,4,5,6,7,8,9};
        double[] yd = {1,3,5,7,9,11,13,15,17,19};
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) r[i] = yd[i] - (c[0] + c[1] * xd[i]);
        };
        Optimization r = new TRFProblem()
            .residuals(fn, m)
            .initialPoint(0.0, 1.0)
            .maxEvaluations(2)
            .solve();

        assertThat(r.getStatus()).isEqualTo(Optimization.Status.MAX_EVALUATIONS_REACHED);
        assertThat(r.getEvaluations()).isLessThanOrEqualTo(2);
    }
}
