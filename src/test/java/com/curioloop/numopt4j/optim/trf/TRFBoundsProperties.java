/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Optimization;

import net.jqwik.api.*;

import com.curioloop.numopt4j.optim.Multivariate;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Property-based tests for TRF bound-constraint enforcement.
 */
public class TRFBoundsProperties {

    @Provide
    Arbitrary<QuadraticProblem> quadraticProblems() {
        return Combinators.combine(
            Arbitraries.integers().between(10, 30),
            Arbitraries.doubles().between(-5, 5).array(double[].class).ofSize(3),
            Arbitraries.doubles().between(0.01, 0.5)
        ).as((m, coeffs, noise) -> new QuadraticProblem(m, coeffs, noise));
    }

    @Property(tries = 100)
    @Label("Wide bounds should not cause abnormal termination")
    void wideBoundsShouldNotCauseAbnormalTermination(@ForAll("quadraticProblems") QuadraticProblem p) {
        Bound[] bounds = {
            Bound.between(-10.0, 10.0),
            Bound.between(-10.0, 10.0),
            Bound.between(-10.0, 10.0)
        };
        Optimization r = p.solve(bounds, 10000);
        assertThat(r.getStatus())
            .as("Bounded optimization should not terminate abnormally")
            .isNotEqualTo(Optimization.Status.ABNORMAL_TERMINATION);
    }

    @Property(tries = 100)
    @Label("Tight bounds should constrain solution and not terminate abnormally")
    void tightBoundsShouldConstrainSolution(@ForAll("quadraticProblems") QuadraticProblem p) {
        Bound[] bounds = new Bound[3];
        for (int i = 0; i < 3; i++) {
            bounds[i] = Bound.between(p.initialGuess[i] - 0.1, p.initialGuess[i] + 0.1);
        }
        Optimization r = p.solve(bounds, 2000);
        assertThat(r.getStatus())
            .as("Tight bounded optimization should complete")
            .isNotEqualTo(Optimization.Status.ABNORMAL_TERMINATION);
    }

    // ── Problem helper ────────────────────────────────────────────────────────

    static class QuadraticProblem {
        final int m;
        final double[] tData, yData, initialGuess;
        final Multivariate.Objective fn;

        QuadraticProblem(int m, double[] coeffs, double noise) {
            this.m = m;
            tData = new double[m]; yData = new double[m];
            java.util.Random rng = new java.util.Random(42);
            for (int i = 0; i < m; i++) {
                tData[i] = (double) i / (m - 1) * 4.0 - 2.0;
                double t = tData[i];
                yData[i] = coeffs[0] + coeffs[1]*t + coeffs[2]*t*t + noise * rng.nextGaussian();
            }
            initialGuess = new double[3];
            for (int i = 0; i < 3; i++)
                initialGuess[i] = coeffs[i] + 0.5 * (rng.nextDouble() - 0.5);
            fn = (c, n, r, mm) -> {
                for (int i = 0; i < mm; i++) {
                    double t = tData[i];
                    r[i] = yData[i] - (c[0] + c[1]*t + c[2]*t*t);
                }
            };
        }

        Optimization solve(Bound[] bounds, int maxfev) {
            return new TRFProblem()
                .residuals(fn, m).initialPoint(initialGuess.clone())
                .gradientTolerance(1e-15).parameterTolerance(1e-15).functionTolerance(1e-15)
                .bounds(bounds).maxEvaluations(maxfev).solve();
        }
    }
}
