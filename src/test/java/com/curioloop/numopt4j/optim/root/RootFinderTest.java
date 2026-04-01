package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Optimization;

import net.jqwik.api.*;
import net.jqwik.api.constraints.DoubleRange;
import net.jqwik.api.constraints.IntRange;
import org.junit.jupiter.api.Test;

import java.util.function.BiConsumer;
import java.util.function.DoubleUnaryOperator;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for RootFinder concrete finder APIs.
 */
class RootFinderTest {

    // ========================================================================
    // Missing required parameter exceptions
    // ========================================================================

    /**
     * Property 5a: Brentq without function → IllegalStateException mentioning "function"
     */
    @Property(tries = 20)
    void property5a_brentqMissingFunction(
            @ForAll @DoubleRange(min = -10, max = 0) double a,
            @ForAll @DoubleRange(min = 0.01, max = 10) double b) {
        IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
            new BrentqProblem()
                .bracket(Bound.between(a, b))
                .solve()
        );
        assertTrue(ex.getMessage().toLowerCase().contains("function"),
            "Message should mention 'function': " + ex.getMessage());
    }

    /**
     * Property 5b: Brentq without bracket → IllegalStateException mentioning "bracket"
     */
    @Property(tries = 20)
    void property5b_brentqMissingBracket(
            @ForAll @DoubleRange(min = -5, max = 5) double c) {
        DoubleUnaryOperator f = x -> x - c;
        IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
            new BrentqProblem()
                .function(f)
                .solve()
        );
        assertTrue(ex.getMessage().toLowerCase().contains("bracket"),
            "Message should mention 'bracket': " + ex.getMessage());
    }

    /**
     * Property 5c: HYBR without equations → IllegalStateException mentioning "equations"
     */
    @Property(tries = 20)
    void property5c_hybrMissingEquations(
            @ForAll @DoubleRange(min = -5, max = 5) double x0) {
        IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
            new HYBRProblem()
                .initialPoint(x0, x0)
                .solve()
        );
        assertTrue(ex.getMessage().toLowerCase().contains("equations"),
            "Message should mention 'equations': " + ex.getMessage());
    }

    /**
     * Property 5d: HYBR without initialPoint → IllegalStateException mentioning "initialPoint"
     */
    @Property(tries = 20)
    void property5d_hybrMissingInitialPoint(
            @ForAll @IntRange(min = 1, max = 4) int n) {
        IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
            new HYBRProblem()
                .equations((x, f) -> { for (int i = 0; i < x.length; i++) f[i] = x[i]; }, n)
                .solve()
        );
        assertTrue(ex.getMessage().toLowerCase().contains("initialpoint"),
            "Message should mention 'initialPoint': " + ex.getMessage());
    }

    /**
     * Property 5e: Broyden without initialPoint → IllegalStateException mentioning "initialPoint"
     */
    @Property(tries = 20)
    void property5e_broydenMissingInitialPoint(
            @ForAll @IntRange(min = 1, max = 4) int n) {
        IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
            new BroydenProblem()
                .equations((x, f) -> { for (int i = 0; i < x.length; i++) f[i] = x[i]; }, n)
                .solve()
        );
        assertTrue(ex.getMessage().toLowerCase().contains("initialpoint"),
            "Message should mention 'initialPoint': " + ex.getMessage());
    }

    // ========================================================================
    // Result field consistency
    // ========================================================================

    /**
     * 1-D result consistency: getCost() == |f(getRoot())|
     */
    @Property(tries = 100)
    void property6a_scalarResultConsistency(
            @ForAll @DoubleRange(min = -5, max = 5) double root,
            @ForAll @DoubleRange(min = 0.1, max = 3) double halfWidth) {
        DoubleUnaryOperator f = x -> x - root;
        double a = root - halfWidth;
        double b = root + halfWidth;

        Optimization result = new BrentqProblem()
            .function(f)
            .bracket(Bound.between(a, b))
            .solve();

        assertTrue(result.isSuccessful(), "Should converge: " + result.getSummary());
        double actualResidual = Math.abs(f.applyAsDouble(result.getRoot()));
        assertEquals(actualResidual, result.getCost(), 1e-14,
            "getCost() should equal |f(getRoot())|");
    }

    /**
     * N-D result consistency: getCost() == max|F(getSolution())|
     */
    @Property(tries = 50)
    void property6b_vectorResultConsistency(
            @ForAll @DoubleRange(min = -3, max = 3) double r1,
            @ForAll @DoubleRange(min = -3, max = 3) double r2) {
        BiConsumer<double[], double[]> fn = (x, f) -> {
            f[0] = x[0] - r1;
            f[1] = x[1] - r2;
        };

        Optimization result = new HYBRProblem()
            .equations(fn, 2)
            .initialPoint(0.0, 0.0)
            .solve();

        assertTrue(result.isSuccessful(), "Should converge: " + result.getSummary());

        double[] sol = result.getSolution();
        double[] fSol = new double[2];
        fn.accept(sol, fSol);
        double expectedResidual = Math.max(Math.abs(fSol[0]), Math.abs(fSol[1]));
        assertEquals(expectedResidual, result.getCost(), 1e-10,
            "getCost() should equal max|F(solution)| (max-norm)");
    }

    // ========================================================================
    // Integration tests
    // ========================================================================

    @Test
    void integrationTest_brentqViaSolve() {
        Optimization result = new BrentqProblem()
            .function(Math::sin)
            .bracket(Bound.between(3.0, 4.0))
            .solve();

        assertTrue(result.isSuccessful());
        assertEquals(Math.PI, result.getRoot(), 1e-10);
    }

    @Test
    void integrationTest_hybrViaSolve() {
        BiConsumer<double[], double[]> fn = (x, f) -> {
            f[0] = 1.0 - x[0];
            f[1] = 10.0 * (x[1] - x[0] * x[0]);
        };

        Optimization result = new HYBRProblem()
            .equations(fn, 2)
            .initialPoint(-1.0, 1.0)
            .solve();

        assertTrue(result.isSuccessful(), result.getSummary());
        double[] sol = result.getSolution();
        assertEquals(1.0, sol[0], 1e-6);
        assertEquals(1.0, sol[1], 1e-6);
    }

    @Test
    void integrationTest_broydenViaSolve() {
        BiConsumer<double[], double[]> fn = (x, f) -> {
            f[0] = 1.0 - x[0];
            f[1] = 10.0 * (x[1] - x[0] * x[0]);
        };

        Optimization result = new BroydenProblem()
            .equations(fn, 2)
            .initialPoint(0.5, 0.5)
            .solve();

        assertTrue(result.isSuccessful(), result.getSummary());
        double[] sol = result.getSolution();
        assertEquals(1.0, sol[0], 1e-4);
        assertEquals(1.0, sol[1], 1e-4);
    }

    @Test
    void integrationTest_workspaceReuse() {
        BiConsumer<double[], double[]> fn = (x, f) -> {
            f[0] = x[0] - 3.0;
            f[1] = x[1] - 7.0;
        };

        HYBRProblem finder = new HYBRProblem()
            .equations(fn, 2)
            .initialPoint(0.0, 0.0);

        HYBRWorkspace ws = finder.alloc();

        Optimization r1 = finder.solve(ws);
        Optimization r2 = finder.solve(ws);

        assertTrue(r1.isSuccessful());
        assertTrue(r2.isSuccessful());

        double[] s1 = r1.getSolution();
        double[] s2 = r2.getSolution();
        assertEquals(s1[0], s2[0], 1e-10, "Workspace reuse should give same result");
        assertEquals(s1[1], s2[1], 1e-10, "Workspace reuse should give same result");
    }

    @Test
    void integrationTest_autoSelectHybrWhenEquationsSet() {
        Optimization result = new HYBRProblem()
            .equations((x, f) -> { f[0] = x[0] - 5.0; }, 1)
            .initialPoint(0.0)
            .solve();

        assertTrue(result.isSuccessful());
        assertEquals(5.0, result.getSolution()[0], 1e-6);
    }
}
