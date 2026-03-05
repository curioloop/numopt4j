/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.OptimizationResult;
import com.curioloop.numopt4j.optim.OptimizationStatus;
import org.junit.jupiter.api.Test;

import java.util.function.BiConsumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Convergence tests for TRF on standard optimization test functions.
 *
 * <p>Covers: linear least squares, exponential decay, Rosenbrock (multiple starts),
 * Beale, Powell Singular, and composite exponential/polynomial examples.</p>
 */
class TRFStandardFunctionsTest {

    // ── Linear least squares ──────────────────────────────────────────────────

    @Test
    void linearLeastSquaresConvergesToExactSolution() {
        final int m = 10;
        double[] xd = {0,1,2,3,4,5,6,7,8,9};
        double[] yd = {1,3,5,7,9,11,13,15,17,19};
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) r[i] = yd[i] - (c[0] + c[1] * xd[i]);
        };
        OptimizationResult r = TRFProblem.create()
            .residuals(fn, m).initialPoint(0.0, 1.0)
            .gradientTolerance(1e-10).coefficientTolerance(1e-10).maxEvaluations(1000)
            .solve();

        assertThat(r.isConverged()).isTrue();
        assertThat(r.getObjectiveValue()).isLessThan(1e-10);
        double[] sol = r.getSolution();
        assertThat(sol[0]).isCloseTo(1.0, within(1e-6)); // intercept
        assertThat(sol[1]).isCloseTo(2.0, within(1e-6)); // slope
    }

    // ── Exponential decay ─────────────────────────────────────────────────────

    @Test
    void exponentialDecayConvergesToTrueParameters() {
        final int m = 20;
        double trueA = 3.0, trueK = 0.5;
        double[] td = new double[m], yd = new double[m];
        for (int i = 0; i < m; i++) { td[i] = i * 0.2; yd[i] = trueA * Math.exp(-trueK * td[i]); }
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) r[i] = yd[i] - c[0] * Math.exp(-c[1] * td[i]);
        };
        OptimizationResult r = TRFProblem.create()
            .residuals(fn, m).initialPoint(2.0, 0.3)
            .gradientTolerance(1e-10).coefficientTolerance(1e-10).maxEvaluations(1000)
            .solve();

        assertThat(r.isConverged()).isTrue();
        assertThat(r.getObjectiveValue()).isLessThan(1e-10);
        double[] sol = r.getSolution();
        assertThat(sol[0]).isCloseTo(trueA, within(1e-6));
        assertThat(sol[1]).isCloseTo(trueK, within(1e-6));
    }

    // ── Rosenbrock ────────────────────────────────────────────────────────────

    @Test
    void rosenbrockConvergesFromClassicStart() {
        final int m = 2;
        BiConsumer<double[], double[]> fn = (c, r) -> {
            r[0] = 10.0 * (c[1] - c[0]*c[0]);
            r[1] = 1.0 - c[0];
        };
        OptimizationResult r = TRFProblem.create()
            .residuals(fn, m).initialPoint(-1.2, 1.0)
            .gradientTolerance(1e-10).coefficientTolerance(1e-10).maxEvaluations(10000)
            .solve();

        assertThat(r.isConverged()).isTrue();
        assertThat(r.getObjectiveValue()).isLessThan(1e-10);
    }

    @Test
    void rosenbrockConvergesFromMultipleStartingPoints() {
        final int m = 3; // padded with r[2]=0 to satisfy m >= n
        BiConsumer<double[], double[]> fn = (c, r) -> {
            r[0] = 10.0 * (c[1] - c[0]*c[0]);
            r[1] = 1.0 - c[0];
            r[2] = 0.0;
        };
        double[][] starts = {
            {-1.2, 1.0}, {0.0, 0.0}, {2.0, 2.0}, {-2.0, -2.0}, {1.5, 1.5}, {-0.5, 0.5}
        };
        for (double[] start : starts) {
            OptimizationResult r = TRFProblem.create()
                .residuals(fn, m).initialPoint(start.clone())
                .gradientTolerance(1e-8).functionTolerance(1e-10).maxEvaluations(10000)
                .solve();
            assertThat(r.getObjectiveValue())
                .as("Rosenbrock from %s", java.util.Arrays.toString(start))
                .isLessThan(1.0);
        }
    }

    // ── Beale ─────────────────────────────────────────────────────────────────

    @Test
    void bealeConvergesFromStandardStart() {
        final int m = 4;
        BiConsumer<double[], double[]> fn = (c, r) -> {
            r[0] = 1.5   - c[0] * (1.0 - c[1]);
            r[1] = 2.25  - c[0] * (1.0 - c[1]*c[1]);
            r[2] = 2.625 - c[0] * (1.0 - c[1]*c[1]*c[1]);
            r[3] = 0.0;
        };
        OptimizationResult r = TRFProblem.create()
            .residuals(fn, m).initialPoint(2.0, 0.3)
            .gradientTolerance(1e-8).coefficientTolerance(1e-8).maxEvaluations(5000)
            .solve();

        assertThat(r.isConverged()).isTrue();
        assertThat(r.getObjectiveValue()).isLessThan(0.2);
    }

    @Test
    void bealeConvergesFromNearSolution() {
        final int m = 4;
        BiConsumer<double[], double[]> fn = (c, r) -> {
            r[0] = 1.5   - c[0] * (1.0 - c[1]);
            r[1] = 2.25  - c[0] * (1.0 - c[1]*c[1]);
            r[2] = 2.625 - c[0] * (1.0 - c[1]*c[1]*c[1]);
            r[3] = 0.0;
        };
        OptimizationResult r = TRFProblem.create()
            .residuals(fn, m).initialPoint(2.8, 0.45)
            .gradientTolerance(1e-12).maxEvaluations(2000)
            .solve();

        assertThat(r.getObjectiveValue()).isLessThan(0.1);
    }

    // ── Powell Singular ───────────────────────────────────────────────────────

    @Test
    void powellSingularConvergesOrReachesLimit() {
        final int m = 5;
        BiConsumer<double[], double[]> fn = (c, r) -> {
            r[0] = c[0] + 10.0 * c[1];
            r[1] = Math.sqrt(5.0) * (c[2] - c[3]);
            r[2] = (c[1] - 2.0*c[2]) * (c[1] - 2.0*c[2]);
            r[3] = Math.sqrt(10.0) * (c[0] - c[3]) * (c[0] - c[3]);
            r[4] = 0.0;
        };
        OptimizationResult r = TRFProblem.create()
            .residuals(fn, m).initialPoint(3.0, -1.0, 0.0, 1.0)
            .gradientTolerance(1e-8).coefficientTolerance(1e-8).maxEvaluations(10000)
            .solve();

        assertThat(r.isConverged() || r.getStatus() == OptimizationStatus.MAX_EVALUATIONS_REACHED).isTrue();
        assertThat(r.getObjectiveValue()).isLessThan(1.0);
    }

    // ── Composite examples ────────────────────────────────────────────────────

    @Test
    void compositeExponentialFitConverges() {
        final int m = 20;
        double[] td = new double[m], yd = new double[m];
        double[] tc = {5.0, 2.0, 3.0, 4.0};
        for (int i = 0; i < m; i++) {
            td[i] = 0.5 * i;
            double t = td[i];
            yd[i] = tc[0]*Math.exp(-t/tc[1]) + tc[2]*t*Math.exp(-t/tc[3])
                  + 0.01 * (Math.random() - 0.5);
        }
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) {
                double t = td[i];
                r[i] = yd[i] - (c[0]*Math.exp(-t/c[1]) + c[2]*t*Math.exp(-t/c[3]));
            }
        };
        OptimizationResult r = TRFProblem.create()
            .residuals(fn, m).initialPoint(4.0, 2.5, 2.5, 3.5)
            .gradientTolerance(1e-8).maxEvaluations(2000)
            .solve();

        assertThat(r.isConverged()).isTrue();
        assertThat(r.getObjectiveValue()).isLessThan(0.1);
    }

    @Test
    void polynomialFitConvergesOrReachesLimit() {
        final int m = 30;
        final double mt = 15.0;
        double[] td = new double[m], yd = new double[m];
        double[] tc = {1.0, -2.0, 3.0, -1.0};
        for (int i = 0; i < m; i++) {
            td[i] = i + 1;
            double x = td[i] / mt;
            yd[i] = tc[0]*x + tc[1]*x*x + tc[2]*x*x*x + tc[3]*x*x*x*x
                  + 0.01 * (Math.random() - 0.5);
        }
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) {
                double x = td[i] / mt;
                r[i] = yd[i] - (c[0]*x + c[1]*x*x + c[2]*x*x*x + c[3]*x*x*x*x);
            }
        };
        OptimizationResult r = TRFProblem.create()
            .residuals(fn, m).initialPoint(0.5, -1.0, 2.0, -0.5)
            .gradientTolerance(1e-8).maxEvaluations(2000)
            .solve();

        assertThat(r.isConverged() || r.getStatus() == OptimizationStatus.MAX_EVALUATIONS_REACHED).isTrue();
        assertThat(r.getObjectiveValue()).isLessThan(0.5);
    }
}
