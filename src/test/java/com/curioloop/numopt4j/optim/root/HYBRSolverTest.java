package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.Multivariate;
import com.curioloop.numopt4j.optim.NumericalJacobian;
import com.curioloop.numopt4j.optim.Optimization;


import net.jqwik.api.*;
import net.jqwik.api.constraints.DoubleRange;
import org.junit.jupiter.api.Test;

import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link HYBRSolver} (col-major variant).
 *
 * Covers:
 *  1. Powell's badly-scaled problem (n=2) — numerical Jacobian via NumericalJacobian.FORWARD
 *  2. Rosenbrock system (n=2) — analytical Jacobian via Multivariate
 *  3. Helical valley (n=3) — numerical Jacobian
 *  4. RootFinder.HYBR fluent API
 *  5. NaN/Inf guard on initial point
 *  6. NaN/Inf guard on function output
 *  7. fnorm==0 fast-return
 *  8. Property: solution satisfies ||F(x)||<tol for random 2-D linear systems
 */
class HYBRSolverTest {

    private static final double TOL = 1e-8;

    // ── 1. Powell badly-scaled — numerical Jacobian ───────────────────────────

    @Test
    void powellBadlyScaled_numericalJacobian() {
        BiConsumer<double[], double[]> fn = (x, f) -> {
            f[0] = x[0] * x[1] - 1e-4;
            f[1] = Math.exp(-x[0]) + Math.exp(-x[1]) - 1.0001;
        };
        Multivariate eval = NumericalJacobian.FORWARD.wrap(fn, 2, 2, true);
        HYBRWorkspace ws = new HYBRWorkspace(2);
        Optimization res = HYBRSolver.solve(eval, new double[]{0.0, 1.0}, TOL, 2000, ws);

        assertTrue(res.isSuccessful(), "status=" + res.getStatus());
        double[] f = new double[2];
        fn.accept(res.getSolution(), f);
        assertTrue(Math.abs(f[0]) < 1e-7 && Math.abs(f[1]) < 1e-7,
                "residual too large: f=" + f[0] + ", " + f[1]);
    }

    // ── 2. Rosenbrock system — analytical Jacobian ────────────────────────────

    @Test
    void rosenbrock_analyticalJacobian() {
        Multivariate eval = (x, f, jac) -> {
            f[0] = 1.0 - x[0];
            f[1] = 10.0 * (x[1] - x[0] * x[0]);
            if (jac != null) {
                jac[0 + 2*0] = -1.0;
                jac[1 + 2*0] = -20.0 * x[0];
                jac[0 + 2*1] = 0.0;
                jac[1 + 2*1] = 10.0;
            }
        };
        HYBRWorkspace ws = new HYBRWorkspace(2);
        Optimization res = HYBRSolver.solve(eval, new double[]{-1.2, 1.0}, TOL, 1000, ws);

        assertTrue(res.isSuccessful(), "status=" + res.getStatus());
        double[] sol = res.getSolution();
        assertEquals(1.0, sol[0], 1e-7);
        assertEquals(1.0, sol[1], 1e-7);
    }

    // ── 3. Helical valley (n=3) ───────────────────────────────────────────────

    @Test
    void helicalValley_n3() {
        BiConsumer<double[], double[]> fn = (x, f) -> {
            double theta = (x[0] >= 0)
                    ? Math.atan2(x[1], x[0]) / (2 * Math.PI)
                    : Math.atan2(x[1], x[0]) / (2 * Math.PI) + 0.5;
            f[0] = 10.0 * (x[2] - 10.0 * theta);
            f[1] = 10.0 * (Math.sqrt(x[0]*x[0] + x[1]*x[1]) - 1.0);
            f[2] = x[2];
        };
        Multivariate eval = NumericalJacobian.FORWARD.wrap(fn, 3, 3, true);
        HYBRWorkspace ws = new HYBRWorkspace(3);
        Optimization res = HYBRSolver.solve(eval, new double[]{-1.0, 0.0, 0.0}, TOL, 1000, ws);

        assertTrue(res.isSuccessful(), "status=" + res.getStatus());
        double[] f = new double[3];
        fn.accept(res.getSolution(), f);
        for (int i = 0; i < 3; i++) assertTrue(Math.abs(f[i]) < 1e-7, "f[" + i + "]=" + f[i]);
    }

    // ── 4. RootFinder.HYBR fluent API ────────────────────────────────────────

    @Test
    void rootFinder_hybr() {
        Optimization res = new HYBRProblem()
                .equations((x, f) -> { f[0] = x[0] - 2.0; f[1] = x[1] + 3.0; }, 2)
                .initialPoint(0.0, 0.0)
                .solve();

        assertTrue(res.isSuccessful());
        double[] sol = res.getSolution();
        assertEquals(2.0, sol[0], 1e-8);
        assertEquals(-3.0, sol[1], 1e-8);
    }

    // ── 5. NaN guard on initial point ────────────────────────────────────────

    @Test
    void nanInitialPoint_throws() {
        Multivariate eval = (x, f, jac) -> f[0] = x[0];
        HYBRWorkspace ws = new HYBRWorkspace(1);
        assertThrows(IllegalArgumentException.class,
                () -> HYBRSolver.solve(eval, new double[]{Double.NaN}, TOL, 100, ws));
    }

    // ── 6. NaN guard on function output ──────────────────────────────────────

    @Test
    void nanFunctionOutput_returnsAbnormal() {
        Multivariate eval = (x, f, jac) -> f[0] = Double.NaN;
        HYBRWorkspace ws = new HYBRWorkspace(1);
        Optimization res = HYBRSolver.solve(eval, new double[]{1.0}, TOL, 100, ws);
        assertEquals(Optimization.Status.INVALID_INPUT, res.getStatus());
    }

    // ── 7. fnorm==0 fast-return ───────────────────────────────────────────────

    @Test
    void fnormZero_fastReturn() {
        Multivariate eval = (x, f, jac) -> f[0] = 0.0;
        HYBRWorkspace ws = new HYBRWorkspace(1);
        Optimization res = HYBRSolver.solve(eval, new double[]{5.0}, TOL, 100, ws);
        assertTrue(res.isSuccessful());
        assertEquals(1, res.getEvaluations());
    }

    // ── 8. Property: random 2-D linear system ────────────────────────────────

    @Property(tries = 50)
    void randomLinearSystem_converges(
            @ForAll @DoubleRange(min = -5, max = 5) double a00,
            @ForAll @DoubleRange(min = -5, max = 5) double a01,
            @ForAll @DoubleRange(min = -5, max = 5) double a10,
            @ForAll @DoubleRange(min = -5, max = 5) double a11,
            @ForAll @DoubleRange(min = -3, max = 3) double b0,
            @ForAll @DoubleRange(min = -3, max = 3) double b1) {

        double det = a00 * a11 - a01 * a10;
        Assume.that(Math.abs(det) > 0.5);

        BiConsumer<double[], double[]> fn = (x, f) -> {
            f[0] = a00 * x[0] + a01 * x[1] - b0;
            f[1] = a10 * x[0] + a11 * x[1] - b1;
        };
        Multivariate eval = NumericalJacobian.FORWARD.wrap(fn, 2, 2, true);
        HYBRWorkspace ws = new HYBRWorkspace(2);
        Optimization res = HYBRSolver.solve(eval, new double[]{0.0, 0.0}, TOL, 500, ws);

        assertTrue(res.isSuccessful(), "det=" + det + " status=" + res.getStatus());
        double[] f = new double[2];
        fn.accept(res.getSolution(), f);
        assertTrue(Math.abs(f[0]) < 1e-6 && Math.abs(f[1]) < 1e-6);
    }
}
