/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.Optimization;

import net.jqwik.api.*;

import java.util.function.BiConsumer;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Property-based tests for TRF convergence behaviour.
 *
 * <p>Covers chi-squared reduction, convergence criteria (gradient / coefficient /
 * chi-squared tolerances), evaluation-limit enforcement, and status accuracy.</p>
 */
public class TRFConvergenceProperties {

    // ── Problem generators ────────────────────────────────────────────────────

    @Provide
    Arbitrary<QuadraticProblem> quadraticProblems() {
        return Combinators.combine(
            Arbitraries.integers().between(10, 30),
            Arbitraries.doubles().between(-5, 5).array(double[].class).ofSize(3),
            Arbitraries.doubles().between(0.01, 0.5)
        ).as((m, coeffs, noise) -> new QuadraticProblem(m, coeffs, noise));
    }

    @Provide
    Arbitrary<ExponentialProblem> exponentialProblems() {
        return Combinators.combine(
            Arbitraries.integers().between(10, 30),
            Arbitraries.doubles().between(0.5, 5.0),
            Arbitraries.doubles().between(0.1, 2.0),
            Arbitraries.doubles().between(0.01, 0.2)
        ).as(ExponentialProblem::new);
    }

    // ── Chi-squared reduction ─────────────────────────────────────────────────

    @Property(tries = 100)
    @Label("Chi-squared must not increase after optimization (quadratic)")
    void chiSquaredShouldDecreaseOrStayConstant(@ForAll("quadraticProblems") QuadraticProblem p) {
        double initialChi2 = p.initialChi2();
        Optimization r = p.solve(1e-8, 1e-8, 1e-8, 2000);
        assertThat(r.getCost())
            .as("Final χ² should be ≤ initial χ²")
            .isLessThanOrEqualTo(initialChi2 + 1e-10);
    }

    @Property(tries = 100)
    @Label("Chi-squared must not increase after optimization (exponential)")
    void chiSquaredShouldDecreaseForExponentialProblems(@ForAll("exponentialProblems") ExponentialProblem p) {
        double initialChi2 = p.initialChi2();
        Optimization r = p.solve(1e-8, 1e-8, 1e-8, 2000);
        assertThat(r.getCost())
            .as("Final χ² should be ≤ initial χ²")
            .isLessThanOrEqualTo(initialChi2 + 1e-10);
    }

    // ── Convergence ───────────────────────────────────────────────────────────

    @Property(tries = 100)
    @Label("Optimizer should converge or reach limit on quadratic problems")
    void optimizerShouldConvergeOnQuadraticProblems(@ForAll("quadraticProblems") QuadraticProblem p) {
        Optimization r = p.solve(1e-8, 1e-8, 1e-8, 3000);
        assertThat(r.isSuccessful() || r.getStatus() == Optimization.Status.MAX_EVALUATIONS_REACHED)
            .as("Should converge or reach limit, not fail abnormally")
            .isTrue();
    }

    // ── Convergence criteria ──────────────────────────────────────────────────

    @Property(tries = 100)
    @Label("GRADIENT_TOLERANCE_REACHED implies isConverged()")
    void gradientConvergenceImpliesConvergedStatus(@ForAll("quadraticProblems") QuadraticProblem p) {
        Optimization r = p.solve(1e-6, 1e-12, 1e-12, 5000);
        if (r.getStatus() == Optimization.Status.GRADIENT_TOLERANCE_REACHED) {
            assertThat(r.isSuccessful()).isTrue();
        }
    }

    @Property(tries = 100)
    @Label("COEFFICIENT_TOLERANCE_REACHED implies isConverged()")
    void coefficientConvergenceImpliesConvergedStatus(@ForAll("quadraticProblems") QuadraticProblem p) {
        Optimization r = p.solve(1e-12, 1e-4, 1e-12, 5000);
        if (r.getStatus() == Optimization.Status.COEFFICIENT_TOLERANCE_REACHED) {
            assertThat(r.isSuccessful()).isTrue();
        }
    }

    @Property(tries = 100)
    @Label("CHI_SQUARED_TOLERANCE_REACHED implies isConverged()")
    void chiSquaredConvergenceImpliesConvergedStatus(@ForAll("quadraticProblems") QuadraticProblem p) {
        Optimization r = p.solve(1e-12, 1e-12, 0.5, 5000);
        if (r.getStatus() == Optimization.Status.CHI_SQUARED_TOLERANCE_REACHED) {
            assertThat(r.isSuccessful()).isTrue();
        }
    }

    // ── Evaluation limit ──────────────────────────────────────────────────────

    @Property(tries = 100)
    @Label("Evaluation count must not exceed maxEvaluations")
    void evaluationLimitShouldBeEnforced(@ForAll("quadraticProblems") QuadraticProblem p) {
        int limit = 10;
        Optimization r = p.solve(1e-15, 1e-15, 1e-15, limit);
        assertThat(r.getEvaluations())
            .as("Evaluations should not exceed limit")
            .isLessThanOrEqualTo(limit);
    }

    // ── Status completeness ───────────────────────────────────────────────────

    @Property(tries = 100)
    @Label("Status must be either converged or limit-reached")
    void statusMustBeConvergedOrLimitReached(@ForAll("quadraticProblems") QuadraticProblem p) {
        Optimization r = p.solve(1e-8, 1e-8, 1e-8, 2000);
        assertThat(r.getStatus().isConverged() || r.getStatus().isLimitReached())
            .as("Status should be converged or limit reached, not abnormal")
            .isTrue();
    }

    // ── Problem helpers ───────────────────────────────────────────────────────

    static class QuadraticProblem {
        final int m;
        final double[] tData, yData, initialGuess;
        final BiConsumer<double[], double[]> fn;

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
            fn = (c, r) -> {
                for (int i = 0; i < m; i++) {
                    double t = tData[i];
                    r[i] = yData[i] - (c[0] + c[1]*t + c[2]*t*t);
                }
            };
        }

        double initialChi2() {
            double[] r = new double[m];
            fn.accept(initialGuess, r);
            double s = 0; for (double v : r) s += v*v;
            return s;
        }

        Optimization solve(double gtol, double xtol, double ftol, int maxfev) {
            return new TRFProblem()
                .residuals(fn, m).initialPoint(initialGuess.clone())
                .gradientTolerance(gtol).parameterTolerance(xtol).functionTolerance(ftol)
                .maxEvaluations(maxfev).solve();
        }
    }

    static class ExponentialProblem {
        final int m;
        final double[] tData, yData, initialGuess;
        final BiConsumer<double[], double[]> fn;

        ExponentialProblem(int m, double a0, double a1, double noise) {
            this.m = m;
            tData = new double[m]; yData = new double[m];
            java.util.Random rng = new java.util.Random(42);
            for (int i = 0; i < m; i++) {
                tData[i] = (double) i / (m - 1) * 3.0;
                yData[i] = a0 * Math.exp(-a1 * tData[i]) + noise * rng.nextGaussian();
            }
            initialGuess = new double[]{a0 * 1.2, a1 * 0.8};
            fn = (c, r) -> {
                for (int i = 0; i < m; i++)
                    r[i] = yData[i] - c[0] * Math.exp(-c[1] * tData[i]);
            };
        }

        double initialChi2() {
            double[] r = new double[m];
            fn.accept(initialGuess, r);
            double s = 0; for (double v : r) s += v*v;
            return s;
        }

        Optimization solve(double gtol, double xtol, double ftol, int maxfev) {
            return new TRFProblem()
                .residuals(fn, m).initialPoint(initialGuess.clone())
                .gradientTolerance(gtol).parameterTolerance(xtol).functionTolerance(ftol)
                .maxEvaluations(maxfev).solve();
        }
    }
}
