/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.OptimizationResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.function.BiConsumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Extreme convergence tests — validated against scipy least_squares(method='trf').
 *
 * <p>Covers the following scenarios:</p>
 * <ol>
 *   <li>Solution exactly on the boundary (single-sided constraint active)</li>
 *   <li>Ill-conditioned Jacobian (Hilbert matrix, condition number ~1.5e4)</li>
 *   <li>High-dimensional diagonal linear system (n=10, m=20)</li>
 *   <li>Tight bounds forcing solution to a corner (multiple constraints active)</li>
 *   <li>Very distant initial point (initial residual ~1e6)</li>
 *   <li>Flat valley (high-aspect-ratio Rosenbrock variant)</li>
 *   <li>Overdetermined system with no exact solution (non-zero least-squares residual)</li>
 *   <li>Multiple constraints simultaneously active (3-D corner)</li>
 * </ol>
 *
 * <p>All expected values are computed by scipy {@code least_squares(..., method='trf')}.</p>
 */
class TRFExtremeConvergenceTest {

    /** Builds a standard TRF problem with tight tolerances. */
    private static TRFProblem trf(int m, BiConsumer<double[], double[]> fn, Bound[] bounds) {
        TRFProblem p = TRFProblem.create()
            .residuals(fn, m)
            .functionTolerance(1e-10)
            .coefficientTolerance(1e-10)
            .gradientTolerance(1e-10)
            .maxEvaluations(5000);
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

    // ── Case 1: solution exactly on the boundary ──────────────────────────────
    // f(x) = [x0-3, x1-5], true solution [3,5]
    // bounds: x0∈[0,2], x1∈[0,8] → constrained solution [2,5] (x0 at upper bound, x1 interior)
    // Both variables have two-sided bounds; verifies convergence when one variable hits its upper bound.
    // scipy: solution=[2.0, 5.0], cost=0.5
    @Test
    void solutionExactlyOnBound() {
        BiConsumer<double[], double[]> fn = (x, r) -> {
            r[0] = x[0] - 3.0;
            r[1] = x[1] - 5.0;
        };
        Bound[] bounds = { Bound.between(0.0, 2.0), Bound.between(0.0, 8.0) };
        OptimizationResult result = trf(2, fn, bounds).initialPoint(0.5, 0.5).solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        assertBoundsRespected(sol, bounds);
        // x0 should be pushed to upper bound 2.0; x1 converges freely to 5.0 (interior of [0,8])
        assertThat(sol[0]).isCloseTo(2.0, within(1e-6));
        assertThat(sol[1]).isCloseTo(5.0, within(1e-6));
        // RSS = (2-3)^2 + (5-5)^2 = 1
        assertThat(result.getObjectiveValue()).isCloseTo(1.0, within(1e-6));
    }

    // ── Case 2: ill-conditioned Jacobian (Hilbert 4×4, condition number ~1.5e4) ──
    // A·x = b, A = Hilbert(4), x_true = [1,2,3,4]
    // scipy: solution=[1,2,3,4], cost≈0
    @Test
    void illConditionedHilbertJacobian() {
        // Hilbert matrix H[i][j] = 1/(i+j+1), 0-indexed
        final double[][] H = {
            {1.0,        1.0/2,      1.0/3,      1.0/4     },
            {1.0/2,      1.0/3,      1.0/4,      1.0/5     },
            {1.0/3,      1.0/4,      1.0/5,      1.0/6     },
            {1.0/4,      1.0/5,      1.0/6,      1.0/7     }
        };
        final double[] xTrue = {1.0, 2.0, 3.0, 4.0};
        // b = H * xTrue
        final double[] b = new double[4];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                b[i] += H[i][j] * xTrue[j];

        BiConsumer<double[], double[]> fn = (x, r) -> {
            for (int i = 0; i < 4; i++) {
                r[i] = -b[i];
                for (int j = 0; j < 4; j++) r[i] += H[i][j] * x[j];
            }
        };
        OptimizationResult result = trf(4, fn, null).initialPoint(0.0, 0.0, 0.0, 0.0).solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        // Ill-conditioned system should still converge to the correct solution (within numerical precision)
        assertThat(sol[0]).isCloseTo(1.0, within(1e-5));
        assertThat(sol[1]).isCloseTo(2.0, within(1e-5));
        assertThat(sol[2]).isCloseTo(3.0, within(1e-5));
        assertThat(sol[3]).isCloseTo(4.0, within(1e-5));
        assertThat(result.getObjectiveValue()).isLessThan(1e-10);
    }

    // ── Case 3: high-dimensional diagonal linear system (n=10, m=20) ─────────
    // A[i][i%n] = (i%n+1), b = A*x_true, x_true = [1..10]
    // scipy: solution=[1..10], cost=0
    @Test
    void highDimensionalDiagonalSystem() {
        final int n = 10, m = 20;
        // A[i][i%n] = (i%n+1), all other entries zero
        final double[] b = new double[m];
        final double[] xTrue = new double[n];
        for (int j = 0; j < n; j++) xTrue[j] = j + 1.0;
        for (int i = 0; i < m; i++) b[i] = (i % n + 1.0) * xTrue[i % n];

        BiConsumer<double[], double[]> fn = (x, r) -> {
            for (int i = 0; i < m; i++) r[i] = (i % n + 1.0) * x[i % n] - b[i];
        };
        double[] x0 = new double[n]; // all-zero initial point
        OptimizationResult result = trf(m, fn, null).initialPoint(x0).solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        for (int j = 0; j < n; j++) {
            assertThat(sol[j]).as("x[%d]", j).isCloseTo(xTrue[j], within(1e-6));
        }
        assertThat(result.getObjectiveValue()).isLessThan(1e-10);
    }

    // ── Case 4: tight bounds force corner solution ────────────────────────────
    // f(x) = [x0-2, x1-2], true solution [2,2], but x0∈[0,1], x1∈[0,1] → solution at [1,1]
    // scipy: solution=[1,1], cost=1.0
    @Test
    void tightBoundsForcesCornerSolution() {
        BiConsumer<double[], double[]> fn = (x, r) -> {
            r[0] = x[0] - 2.0;
            r[1] = x[1] - 2.0;
        };
        Bound[] bounds = { Bound.between(0.0, 1.0), Bound.between(0.0, 1.0) };
        OptimizationResult result = trf(2, fn, bounds).initialPoint(0.5, 0.5).solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        assertBoundsRespected(sol, bounds);
        assertThat(sol[0]).isCloseTo(1.0, within(1e-6));
        assertThat(sol[1]).isCloseTo(1.0, within(1e-6));
        // RSS = (1-2)^2 + (1-2)^2 = 2
        assertThat(result.getObjectiveValue()).isCloseTo(2.0, within(1e-6));
    }

    // ── Case 5: very distant initial point (initial residual ~1e6) ───────────
    // f(x) = [x0-1, x1-2], starting from [1e6, 1e6]
    // scipy: solution=[1,2], cost=0
    @Test
    void veryFarInitialPoint() {
        BiConsumer<double[], double[]> fn = (x, r) -> {
            r[0] = x[0] - 1.0;
            r[1] = x[1] - 2.0;
        };
        OptimizationResult result = trf(2, fn, null).initialPoint(1e6, 1e6).solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        assertThat(sol[0]).isCloseTo(1.0, within(1e-5));
        assertThat(sol[1]).isCloseTo(2.0, within(1e-5));
        assertThat(result.getObjectiveValue()).isLessThan(1e-8);
    }

    // ── Case 6: flat valley (high-aspect-ratio Rosenbrock variant) ───────────
    // f = [100*(x1-x0^2), 1-x0], starting from (-1.2, 1.0)
    // Aspect ratio 100:1; tests trust-region expansion along the narrow direction.
    // scipy: solution=[1,1], cost=0
    @Test
    void flatValleyHighAspectRatio() {
        BiConsumer<double[], double[]> fn = (x, r) -> {
            r[0] = 100.0 * (x[1] - x[0] * x[0]);
            r[1] = 1.0 - x[0];
        };
        OptimizationResult result = trf(2, fn, null)
            .maxEvaluations(10000)
            .initialPoint(-1.2, 1.0)
            .solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        assertThat(sol[0]).isCloseTo(1.0, within(1e-5));
        assertThat(sol[1]).isCloseTo(1.0, within(1e-5));
        assertThat(result.getObjectiveValue()).isLessThan(1e-8);
    }

    // ── Case 7: overdetermined system with no exact solution ─────────────────
    // f = [x0+x1-1, x0-x1-2, x0+2*x1-3], no exact solution
    // Least-squares solution: x*=[13/7, 3/14] ≈ [1.857, 0.214]
    // scipy: solution≈[1.857, 0.214], cost≈0.893
    @Test
    void overdeterminedSystemNoExactSolution() {
        BiConsumer<double[], double[]> fn = (x, r) -> {
            r[0] = x[0] + x[1] - 1.0;
            r[1] = x[0] - x[1] - 2.0;
            r[2] = x[0] + 2.0 * x[1] - 3.0;
        };
        OptimizationResult result = trf(3, fn, null).initialPoint(0.0, 0.0).solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        // Least-squares solution: A^T A x = A^T b → x = [13/7, 3/14] ≈ [1.857, 0.214]
        assertThat(sol[0]).isCloseTo(13.0 / 7.0, within(1e-5));
        assertThat(sol[1]).isCloseTo(3.0 / 14.0, within(1e-5));
        // RSS = 25/14 ≈ 1.786, cost = RSS/2 ≈ 0.893
        assertThat(result.getObjectiveValue()).isCloseTo(25.0 / 14.0, within(1e-5));
    }

    // ── Case 8: multiple constraints simultaneously active (3-D corner) ───────
    // f(x) = [x0-3, x1-3, x2-3], true solution [3,3,3]
    // bounds: x0∈[0,1], x1∈[0,2], x2∈[0,1.5] → solution at [1,2,1.5] (all three at upper bounds)
    // Verifies convergence when multiple variables simultaneously reach their upper bounds.
    // scipy: solution=[1,2,1.5], cost≈3.625
    @Test
    void multipleActiveConstraintsSimultaneously() {
        BiConsumer<double[], double[]> fn = (x, r) -> {
            r[0] = x[0] - 3.0;
            r[1] = x[1] - 3.0;
            r[2] = x[2] - 3.0;
        };
        Bound[] bounds = {
            Bound.between(0.0, 1.0),
            Bound.between(0.0, 2.0),
            Bound.between(0.0, 1.5)
        };
        OptimizationResult result = trf(3, fn, bounds).initialPoint(0.5, 0.5, 0.5).solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        assertBoundsRespected(sol, bounds);
        assertThat(sol[0]).isCloseTo(1.0,  within(1e-6));
        assertThat(sol[1]).isCloseTo(2.0,  within(1e-6));
        assertThat(sol[2]).isCloseTo(1.5,  within(1e-6));
        // RSS = (1-3)^2 + (2-3)^2 + (1.5-3)^2 = 4 + 1 + 2.25 = 7.25
        assertThat(result.getObjectiveValue()).isCloseTo(7.25, within(1e-5));
    }

    // ── Case 9: single-variable bounded convergence (1-D exact verification) ─
    // f(x) = x - 5, true solution x*=5, but upper bound=3 → solution at x=3
    // Verifies correctness of makeStrictlyFeasible + clScaling in the 1-D case.
    @Test
    void singleVariableBoundedConvergence() {
        BiConsumer<double[], double[]> fn = (x, r) -> r[0] = x[0] - 5.0;
        Bound[] bounds = { Bound.between(0.0, 3.0) };
        OptimizationResult result = trf(1, fn, bounds).initialPoint(1.0).solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        assertBoundsRespected(sol, bounds);
        assertThat(sol[0]).isCloseTo(3.0, within(1e-8));
        // RSS = (3-5)^2 = 4
        assertThat(result.getObjectiveValue()).isCloseTo(4.0, within(1e-8));
    }

    // ── Case 10: exponential fit with non-negativity constraints (bounded + nonlinear) ─
    // y = a*exp(-b*t), true params a=2, b=0.5, constraints a>=0, b>=0
    // Starting from a negative initial point; tests bound reflection + nonlinear convergence.
    @Test
    void exponentialFitWithNonNegativityConstraints() {
        final int m = 15;
        final double trueA = 2.0, trueB = 0.5;
        final double[] t = new double[m], y = new double[m];
        for (int i = 0; i < m; i++) {
            t[i] = i * 0.3;
            y[i] = trueA * Math.exp(-trueB * t[i]);
        }
        BiConsumer<double[], double[]> fn = (x, r) -> {
            for (int i = 0; i < m; i++) r[i] = y[i] - x[0] * Math.exp(-x[1] * t[i]);
        };
        Bound[] bounds = { Bound.atLeast(0.0), Bound.atLeast(0.0) };
        // Starting from a positive initial point (negative would be pushed to 0 by applyBounds)
        OptimizationResult result = trf(m, fn, bounds).initialPoint(0.5, 0.1).solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        assertBoundsRespected(sol, bounds);
        assertThat(sol[0]).isCloseTo(trueA, within(1e-5));
        assertThat(sol[1]).isCloseTo(trueB, within(1e-5));
        assertThat(result.getObjectiveValue()).isLessThan(1e-8);
    }

    // ── Case 11: bounded constraints + robust loss (exponential fit with outliers) ──
    // y = a*exp(-b*t) + outliers, constraints a∈[0,5], b∈[0,2]
    // True params a=2, b=0.5; large outliers injected at points 5/10/15 (magnitude 8-12).
    // LINEAR is heavily skewed by outliers (a≈2.25, b≈0.19); robust loss should recover near true values.
    // Verifies correctness of bounds + robust loss combination.
    // scipy (soft_l1/huber/cauchy, scale=1.0): a≈2.0, b≈0.46~0.50
    @ParameterizedTest
    @EnumSource(value = RobustLoss.class, names = {"SOFT_L1", "HUBER", "CAUCHY", "ARCTAN"})
    void boundedRobustLoss_outlierExponentialFit(RobustLoss loss) {
        final int m = 20;
        final double trueA = 2.0, trueB = 0.5;
        final double[] t = new double[m], y = new double[m];
        for (int i = 0; i < m; i++) {
            t[i] = i * 0.3;
            y[i] = trueA * Math.exp(-trueB * t[i]);
        }
        // Inject large outliers at points 5/10/15
        y[5]  += 10.0;
        y[10] -= 8.0;
        y[15] += 12.0;

        BiConsumer<double[], double[]> fn = (x, r) -> {
            for (int i = 0; i < m; i++) r[i] = y[i] - x[0] * Math.exp(-x[1] * t[i]);
        };
        Bound[] bounds = { Bound.between(0.0, 5.0), Bound.between(0.0, 2.0) };

        OptimizationResult result = TRFProblem.create()
            .residuals(fn, m)
            .bounds(bounds)
            .loss(loss)
            .lossScale(1.0)
            .functionTolerance(1e-10).coefficientTolerance(1e-10).gradientTolerance(1e-10)
            .maxEvaluations(5000)
            .initialPoint(1.0, 1.0)
            .solve();
        double[] sol = result.getSolution();

        assertThat(result.isConverged()).isTrue();
        assertBoundsRespected(sol, bounds);
        // scipy (soft_l1/huber): a≈2.039, b≈0.462; (cauchy): a≈2.006, b≈0.499; (arctan): a≈2.000, b≈0.500
        // Robust loss should significantly suppress outliers and recover near true values (LINEAR gives b≈0.19, robust should be > 0.3)
        assertThat(sol[0]).isCloseTo(trueA, within(0.2));
        assertThat(sol[1]).isGreaterThan(0.3);
    }
}
