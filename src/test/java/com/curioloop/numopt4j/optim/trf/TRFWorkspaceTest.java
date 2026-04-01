/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.Optimization;
import net.jqwik.api.*;

import org.junit.jupiter.api.Test;

import java.util.function.BiConsumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Tests for {@link TRFWorkspace} allocation and reuse.
 *
 * <p>Covers both unit-level allocation checks and property-based reuse correctness.</p>
 */
class TRFWorkspaceTest {

    // ── Workspace allocation ──────────────────────────────────────────────────

    @Test
    void allocatedWorkspaceHasCorrectDimensions() {
        final int m = 10, n = 3;
        TRFWorkspace ws = new TRFWorkspace(m, n);

        assertThat(ws.fvec).hasSize(m);
        assertThat(ws.fjac).hasSize(m * n);
        assertThat(ws.work).hasSize(n * n + 12 * n);
        assertThat(ws.getN()).isEqualTo(n);
        assertThat(ws.wa2).hasSize(n);
        assertThat(ws.wa4).hasSize(m);
        assertThat(ws.ipvt).hasSize(n);
    }

    @Test
    void allocViaOptimizerMatchesDirectAlloc() {
        final int m = 10, n = 3;
        BiConsumer<double[], double[]> fn = (c, r) -> {};
        TRFProblem p = new TRFProblem().residuals(fn, m).initialPoint(0.0, 0.0, 0.0);
        TRFWorkspace ws = p.alloc();
        assertThat(ws.fvec).hasSize(m);
        assertThat(ws.getN()).isEqualTo(n);
    }

    @Test
    void allocViaProblemMatchesDirectAlloc() {
        final int m = 5;
        BiConsumer<double[], double[]> fn = (c, r) -> {};
        TRFProblem p = new TRFProblem().residuals(fn, m).initialPoint(0.0, 0.0);
        TRFWorkspace ws = p.alloc();
        assertThat(ws.fvec).hasSize(m);
        assertThat(ws.getN()).isEqualTo(2);
    }

    // ── Workspace reuse via TRFProblem ───────────────────────────────────────

    @Test
    void workspaceReuseProducesIdenticalResults() {
        final int m = 5, n = 2;
        double[] xd = {0, 1, 2, 3, 4}, yd = {1, 3, 5, 7, 9};
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) r[i] = yd[i] - (c[0] + c[1] * xd[i]);
        };
        TRFProblem p = new TRFProblem()
            .residuals(fn, m).initialPoint(0.0, 1.0)
            .gradientTolerance(1e-10).parameterTolerance(1e-10);
        TRFWorkspace ws = p.alloc();
        double[] x0 = {0.0, 1.0};

        Optimization r1 = p.solve(ws);
        p.initialPoint(x0.clone());
        Optimization r2 = p.solve(ws);

        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(1e-10));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    @Test
    void withAndWithoutWorkspaceProduceSameResult() {
        final int m = 8, n = 2;
        double[] xd = {0,1,2,3,4,5,6,7}, yd = {2,4,6,8,10,12,14,16};
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) r[i] = yd[i] - (c[0] + c[1] * xd[i]);
        };
        TRFProblem p = new TRFProblem()
            .residuals(fn, m).initialPoint(1.0, 1.5)
            .gradientTolerance(1e-10).parameterTolerance(1e-10);

        Optimization r1 = p.solve();
        Optimization r2 = p.solve(p.alloc());

        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(1e-10));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    // ── Workspace reuse via TRFProblem ────────────────────────────────────────

    @Test
    void problemWorkspaceReuseProducesIdenticalResults() {
        final int m = 5;
        double[] xd = {0, 1, 2, 3, 4}, yd = {1, 3, 5, 7, 9};
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) r[i] = yd[i] - (c[0] + c[1] * xd[i]);
        };
        TRFProblem p = new TRFProblem()
            .residuals(fn, m).initialPoint(0.0, 1.0)
            .gradientTolerance(1e-10).parameterTolerance(1e-10);
        TRFWorkspace ws = p.alloc();

        Optimization r1 = p.solve(ws);
        Optimization r2 = p.solve(ws);

        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(1e-10));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    // ── Property-based: workspace equivalence ────────────────────────────────

    @Provide
    Arbitrary<double[][]> linearProblems() {
        return Combinators.combine(
            Arbitraries.integers().between(5, 20),
            Arbitraries.doubles().between(-5, 5),
            Arbitraries.doubles().between(0.5, 5)
        ).as((m, intercept, slope) -> {
            double[] xd = new double[m], yd = new double[m];
            for (int i = 0; i < m; i++) { xd[i] = i; yd[i] = intercept + slope * i; }
            return new double[][]{xd, yd};
        });
    }

    @Property(tries = 100)
    @Label("Workspace equivalence: with vs without workspace")
    void workspaceShouldProduceIdenticalResults(@ForAll("linearProblems") double[][] data) {
        double[] xd = data[0], yd = data[1];
        int m = xd.length;
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) r[i] = yd[i] - (c[0] + c[1] * xd[i]);
        };
        TRFProblem p = new TRFProblem()
            .residuals(fn, m).initialPoint(0.0, 1.0)
            .gradientTolerance(1e-15).parameterTolerance(1e-15).functionTolerance(1e-15)
            .maxEvaluations(10000);

        Optimization r1 = p.solve();
        Optimization r2 = p.solve(p.alloc());

        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(1e-10));
        assertThat(r1.getIterations()).isEqualTo(r2.getIterations());
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    @Property(tries = 100)
    @Label("Workspace reusability: same result on repeated calls")
    void workspaceShouldBeReusable(@ForAll("linearProblems") double[][] data) {
        double[] xd = data[0], yd = data[1];
        int m = xd.length;
        BiConsumer<double[], double[]> fn = (c, r) -> {
            for (int i = 0; i < m; i++) r[i] = yd[i] - (c[0] + c[1] * xd[i]);
        };
        TRFProblem p = new TRFProblem()
            .residuals(fn, m).initialPoint(0.0, 1.0)
            .gradientTolerance(1e-15).parameterTolerance(1e-15).functionTolerance(1e-15)
            .maxEvaluations(10000);
        TRFWorkspace ws = p.alloc();

        Optimization r1 = p.solve(ws);
        Optimization r2 = p.solve(ws);

        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(1e-10));
    }
}
