/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Multivariate;
import com.curioloop.numopt4j.optim.Optimization;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Correctness tests for TRFOptimizer.
 */
class TRFCorrectnessTest {

    private static TRFProblem trf(int m, int n, Multivariate.Objective fn,
                                   Bound[] bounds, int maxfev) {
        TRFProblem p = new TRFProblem()
            .residuals(fn, m)
            .functionTolerance(1e-10).parameterTolerance(1e-10).gradientTolerance(1e-10)
            .maxEvaluations(maxfev);
        if (bounds != null) p.bounds(bounds);
        return p;
    }

    private static void assertBoundsRespected(double[] x, Bound[] bounds) {
        if (bounds == null) return;
        for (int i = 0; i < x.length; i++) {
            Bound b = bounds[i];
            if (b == null || b.isUnbounded()) continue;
            if (b.hasLower()) assertThat(x[i]).isGreaterThanOrEqualTo(b.getLower() - 1e-9);
            if (b.hasUpper()) assertThat(x[i]).isLessThanOrEqualTo(b.getUpper() + 1e-9);
        }
    }

    @Test
    void linearLeastSquares_unbounded() {
        int m = 3, n = 2;
        Multivariate.Objective fn = (x, xn, r, rm) -> {
            r[0] = 2*x[0] + x[1] - 5;
            r[1] = x[0] + 3*x[1] - 10;
            r[2] = x[1] - 3;
        };
        Optimization r = trf(m, n, fn, null, 200).initialPoint(0.0, 0.0).solve();
        double[] sol = r.getSolution();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(sol[0]).isCloseTo(1.0, within(1e-6));
        assertThat(sol[1]).isCloseTo(3.0, within(1e-6));
        assertThat(r.getCost()).isCloseTo(0.0, within(1e-10));
    }

    @Test
    void rosenbrock_unbounded() {
        int m = 2, n = 2;
        Multivariate.Objective fn = (x, xn, r, rm) -> {
            r[0] = 10 * (x[1] - x[0]*x[0]);
            r[1] = 1 - x[0];
        };
        Optimization r = trf(m, n, fn, null, 2000).initialPoint(-1.2, 1.0).solve();
        double[] sol = r.getSolution();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(sol[0]).isCloseTo(1.0, within(1e-5));
        assertThat(sol[1]).isCloseTo(1.0, within(1e-5));
    }

    @Test
    void rosenbrock_bounded_solutionInside() {
        int m = 2, n = 2;
        Multivariate.Objective fn = (x, xn, r, rm) -> {
            r[0] = 10 * (x[1] - x[0]*x[0]);
            r[1] = 1 - x[0];
        };
        Bound[] bounds = {Bound.between(-2, 2), Bound.between(-2, 2)};
        Optimization r = trf(m, n, fn, bounds, 2000).initialPoint(-1.2, 1.0).solve();
        double[] sol = r.getSolution();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(sol[0]).isCloseTo(1.0, within(1e-5));
        assertThat(sol[1]).isCloseTo(1.0, within(1e-5));
        assertBoundsRespected(sol, bounds);
    }

    @Test
    void rosenbrock_bounded_solutionOnBoundary() {
        int m = 2, n = 2;
        Multivariate.Objective fn = (x, xn, r, rm) -> {
            r[0] = 10 * (x[1] - x[0]*x[0]);
            r[1] = 1 - x[0];
        };
        Bound[] bounds = {Bound.between(-2, 0.5), Bound.between(-2, 2)};
        Optimization r = trf(m, n, fn, bounds, 2000).initialPoint(-1.2, 1.0).solve();
        double[] sol = r.getSolution();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(sol[0]).isLessThanOrEqualTo(0.5 + 1e-6);
        assertBoundsRespected(sol, bounds);
        assertThat(r.getCost()).isGreaterThan(0).isLessThan(0.3);
    }

    @Test
    void exponentialDecay_unbounded() {
        int m = 20, n = 2;
        double[] t = new double[m], y = new double[m];
        for (int i = 0; i < m; i++) { t[i] = i * 0.2; y[i] = 2.0 * Math.exp(-0.5 * t[i]); }
        Multivariate.Objective fn = (x, xn, r, rm) -> {
            for (int i = 0; i < m; i++) r[i] = y[i] - x[0] * Math.exp(-x[1] * t[i]);
        };
        Optimization r = trf(m, n, fn, null, 500).initialPoint(1.0, 1.0).solve();
        double[] sol = r.getSolution();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(sol[0]).isCloseTo(2.0, within(1e-6));
        assertThat(sol[1]).isCloseTo(0.5, within(1e-6));
    }

    @Test
    void exponentialDecay_bounded() {
        int m = 20, n = 2;
        double[] t = new double[m], y = new double[m];
        for (int i = 0; i < m; i++) { t[i] = i * 0.2; y[i] = 2.0 * Math.exp(-0.5 * t[i]); }
        Multivariate.Objective fn = (x, xn, r, rm) -> {
            for (int i = 0; i < m; i++) r[i] = y[i] - x[0] * Math.exp(-x[1] * t[i]);
        };
        Bound[] bounds = {Bound.atLeast(0), Bound.atLeast(0)};
        Optimization r = trf(m, n, fn, bounds, 500).initialPoint(1.0, 1.0).solve();
        double[] sol = r.getSolution();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(sol[0]).isCloseTo(2.0, within(1e-5));
        assertThat(sol[1]).isCloseTo(0.5, within(1e-5));
        assertBoundsRespected(sol, bounds);
    }

    @Test
    void nonNegativeLeastSquares() {
        int m = 3, n = 2;
        Multivariate.Objective fn = (x, xn, r, rm) -> {
            r[0] = x[0] - (-1.0);
            r[1] = x[1] - (-1.0);
            r[2] = x[0] + x[1] - 0.0;
        };
        Bound[] bounds = {Bound.atLeast(0), Bound.atLeast(0)};
        Optimization r = trf(m, n, fn, bounds, 500).initialPoint(1.0, 1.0).solve();
        double[] sol = r.getSolution();

        assertThat(r.isSuccessful()).isTrue();
        assertBoundsRespected(sol, bounds);
        assertThat(sol[0]).isGreaterThanOrEqualTo(-1e-9);
        assertThat(sol[1]).isGreaterThanOrEqualTo(-1e-9);
    }

    @Test
    void boundedStepConsistency() {
        int m = 1, n = 1;
        Multivariate.Objective fn = (x, xn, r, rm) -> r[0] = x[0] - 5.0;
        Bound[] bounds = {Bound.between(0, 3)};
        Optimization r = trf(m, n, fn, bounds, 200).initialPoint(1.0).solve();
        double[] sol = r.getSolution();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(sol[0]).isCloseTo(3.0, within(1e-6));
        assertBoundsRespected(sol, bounds);
    }
}
