/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.subplex;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;


import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Tests for the Subplex optimizer.
 */
class SubplexProblemTest {

    // ── Classic test functions (N ≤ 5, direct NM path) ───────────────────

    @Test
    void rosenbrock2D() {
        Optimization r = Minimizer.subplex()
                .objective(x -> 100 * Math.pow(x[1] - x[0] * x[0], 2) + Math.pow(1 - x[0], 2))
                .initialPoint(-1.0, 1.0)
                .functionTolerance(1e-8)
                .parameterTolerance(1e-8)
                .maxEvaluations(5000)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(1.0, within(1e-3));
        assertThat(r.getSolution()[1]).isCloseTo(1.0, within(1e-3));
    }

    @Test
    void sphere3D() {
        Optimization r = Minimizer.subplex()
                .objective(x -> x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
                .initialPoint(3.0, -4.0, 5.0)
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        for (double v : r.getSolution()) {
            assertThat(v).isCloseTo(0.0, within(1e-4));
        }
        assertThat(r.getCost()).isCloseTo(0.0, within(1e-8));
    }

    @Test
    void beale() {
        Optimization r = Minimizer.subplex()
                .objective(x -> {
                    double x0 = x[0], x1 = x[1];
                    double t1 = 1.5 - x0 + x0 * x1;
                    double t2 = 2.25 - x0 + x0 * x1 * x1;
                    double t3 = 2.625 - x0 + x0 * x1 * x1 * x1;
                    return t1 * t1 + t2 * t2 + t3 * t3;
                })
                .initialPoint(1.0, 1.0)
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(3.0, within(1e-3));
        assertThat(r.getSolution()[1]).isCloseTo(0.5, within(1e-3));
    }

    @Test
    void booth() {
        Optimization r = Minimizer.subplex()
                .objective(x -> {
                    double t1 = x[0] + 2 * x[1] - 7;
                    double t2 = 2 * x[0] + x[1] - 5;
                    return t1 * t1 + t2 * t2;
                })
                .initialPoint(0.0, 0.0)
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(1.0, within(1e-5));
        assertThat(r.getSolution()[1]).isCloseTo(3.0, within(1e-5));
        assertThat(r.getCost()).isCloseTo(0.0, within(1e-8));
    }

    @Test
    void matyas() {
        Optimization r = Minimizer.subplex()
                .objective(x -> 0.26 * (x[0] * x[0] + x[1] * x[1]) - 0.48 * x[0] * x[1])
                .initialPoint(5.0, -5.0)
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(0.0, within(1e-4));
        assertThat(r.getSolution()[1]).isCloseTo(0.0, within(1e-4));
    }

    @Test
    void oneDimensional() {
        Optimization r = Minimizer.subplex()
                .objective(x -> (x[0] - 3) * (x[0] - 3))
                .initialPoint(0.0)
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(3.0, within(1e-5));
    }

    // ── High-dimensional (where Subplex shines) ───────────────────────────

    @Test
    void sphere20D() {
        double[] x0 = new double[20];
        for (int i = 0; i < 20; i++) x0[i] = i + 1;

        Optimization r = Minimizer.subplex()
                .objective(x -> {
                    double s = 0;
                    for (double v : x) s += v * v;
                    return s;
                })
                .initialPoint(x0)
                .functionTolerance(1e-8)
                .parameterTolerance(1e-8)
                .maxEvaluations(50000)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        for (double v : r.getSolution()) {
            assertThat(v).isCloseTo(0.0, within(1e-2));
        }
    }

    @Test
    void rosenbrock10D() {
        double[] x0 = new double[10];
        for (int i = 0; i < 10; i++) x0[i] = (i % 2 == 0) ? -1.2 : 1.0;

        Optimization r = Minimizer.subplex()
                .objective(x -> {
                    double s = 0;
                    for (int i = 0; i < x.length - 1; i++) {
                        s += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(1 - x[i], 2);
                    }
                    return s;
                })
                .initialPoint(x0)
                .functionTolerance(1e-8)
                .parameterTolerance(1e-8)
                .maxEvaluations(200000)
                .solve();

        // Rosenbrock 10D is very hard; verify reasonable cost, not convergence status
        assertThat(r.getCost()).isLessThan(1.0);
    }

    // ── Bounded ────────────────────────────────────────────────────────────

    @Test
    void bounded3D() {
        Optimization r = Minimizer.subplex()
                .objective(x -> (x[0] - 0.5) * (x[0] - 0.5)
                        + (x[1] - 1.5) * (x[1] - 1.5)
                        + (x[2] - 2.5) * (x[2] - 2.5))
                .initialPoint(0.0, 0.0, 0.0)
                .bounds(Bound.between(0, 1), Bound.between(0, 2), Bound.between(0, 3))
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(0.5, within(1e-3));
        assertThat(r.getSolution()[1]).isCloseTo(1.5, within(1e-3));
        assertThat(r.getSolution()[2]).isCloseTo(2.5, within(1e-3));
    }

    // ── Bounded (N ≤ 5) ─────────────────────────────────────────────────

    @Test
    void boundedRosenbrock() {
        Optimization r = Minimizer.subplex()
                .objective(x -> 100 * Math.pow(x[1] - x[0] * x[0], 2) + Math.pow(1 - x[0], 2))
                .initialPoint(0.0, 0.0)
                .bounds(Bound.between(-2, 2), Bound.between(-2, 2))
                .functionTolerance(1e-8)
                .parameterTolerance(1e-8)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(1.0, within(1e-3));
        assertThat(r.getSolution()[1]).isCloseTo(1.0, within(1e-3));
    }

    @Test
    void boundedMinimumOnBoundary() {
        Optimization r = Minimizer.subplex()
                .objective(x -> x[0] * x[0])
                .initialPoint(4.0)
                .bounds(Bound.atLeast(2.0))
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.getSolution()[0]).isCloseTo(2.0, within(1e-4));
    }

    @Test
    void startAtBoundCorner() {
        Optimization r = Minimizer.subplex()
                .objective(x -> (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5))
                .initialPoint(0.0, 0.0)
                .bounds(Bound.between(0, 1), Bound.between(0, 1))
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(0.5, within(1e-3));
        assertThat(r.getSolution()[1]).isCloseTo(0.5, within(1e-3));
    }

    @Test
    void tightBounds() {
        Optimization r = Minimizer.subplex()
                .objective(x -> (x[0] - 1) * (x[0] - 1) + (x[1] - 2) * (x[1] - 2))
                .initialPoint(0.99, 1.99)
                .bounds(Bound.between(0.9, 1.1), Bound.between(1.9, 2.1))
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(1.0, within(1e-3));
        assertThat(r.getSolution()[1]).isCloseTo(2.0, within(1e-3));
    }

    // ── NaN protection ──────────────────────────────────────────────────

    @Test
    void nanInObjective() {
        Optimization r = Minimizer.subplex()
                .objective(x -> x[0] < 0 ? Double.NaN : (x[0] - 2) * (x[0] - 2))
                .initialPoint(5.0)
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(2.0, within(1e-3));
    }

    // ── Workspace reuse ─────────────────────────────────────────────────

    @Test
    void workspaceReuse() {
        SubplexProblem p = Minimizer.subplex()
                .objective(x -> x[0] * x[0] + x[1] * x[1])
                .initialPoint(5.0, 5.0)
                .functionTolerance(1e-8)
                .parameterTolerance(1e-8);

        SubplexWorkspace ws = p.alloc();

        Optimization r1 = p.solve(ws);
        assertThat(r1.isSuccessful()).isTrue();

        Optimization r2 = p.initialPoint(-3.0, 4.0).solve(ws);
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r2.getSolution()[0]).isCloseTo(0.0, within(1e-4));
        assertThat(r2.getSolution()[1]).isCloseTo(0.0, within(1e-4));
    }

    // ── Custom step sizes ──────────────────────────────────────────────────

    @Test
    void customStepSizes() {
        // Different scales per dimension
        Optimization r = Minimizer.subplex()
                .objective(x -> (x[0] - 1) * (x[0] - 1) + (x[1] - 1000) * (x[1] - 1000))
                .initialPoint(0.5, 500.0)
                .initialStep(0.1, 100.0)
                .functionTolerance(1e-4)
                .parameterTolerance(1e-4)
                .maxEvaluations(10000)
                .solve();

        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(1.0, within(0.5));
        assertThat(r.getSolution()[1]).isCloseTo(1000.0, within(50.0));
    }

    // ── Max evaluations limit ──────────────────────────────────────────────

    @Test
    void maxEvaluationsReached() {
        Optimization r = Minimizer.subplex()
                .objective(x -> x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
                .initialPoint(10.0, 10.0, 10.0)
                .maxEvaluations(5)
                .solve();

        assertThat(r.getStatus()).isEqualTo(Optimization.Status.MAX_EVALUATIONS_REACHED);
        assertThat(r.isSuccessful()).isFalse();
    }
}
