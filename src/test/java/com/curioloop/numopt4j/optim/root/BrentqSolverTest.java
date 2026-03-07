/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.OptimizationResult;
import com.curioloop.numopt4j.optim.OptimizationStatus;
import net.jqwik.api.*;
import net.jqwik.api.constraints.IntRange;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleUnaryOperator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Tests for {@link BrentqSolver}.
 *
 * <p>Covers:</p>
 * <ul>
 *   <li>Property 1: Brentq convergence</li>
 *   <li>Property 2: invalid bracket exception</li>
 *   <li>Property 11: abnormal termination on NaN/Infinity function values</li>
 *   <li>Unit tests: standard functions, endpoint roots, max-iterations limit</li>
 * </ul>
 */
@SuppressWarnings("unused")
public class BrentqSolverTest {

    // ========================================================================
    // Property 1: Brentq convergence
    // ========================================================================

    /**
     * For any random cubic polynomial {@code f(x) = a*x³ + b*x² + c*x + d}
     * with a valid bracket {@code [xa, xb]} satisfying {@code f(xa)*f(xb) <= 0},
     * the solver SHALL return a root {@code x} with {@code |f(x)| <= xtol + rtol*|x|}
     * and iterations {@code <= 100}.
     */
    @Property(tries = 100)
    void brentqConvergence(
            @ForAll("coefficients") double a,
            @ForAll("coefficients") double b,
            @ForAll("coefficients") double c,
            @ForAll("coefficients") double d,
            @ForAll("bracketLeft") double xa,
            @ForAll("bracketWidth") double width) {

        double xb = xa + width;
        DoubleUnaryOperator f = x -> a * x * x * x + b * x * x + c * x + d;

        double fpre = f.applyAsDouble(xa);
        double fcur = f.applyAsDouble(xb);

        // Skip if bracket is invalid (same sign)
        Assume.that(fpre * fcur <= 0);

        OptimizationResult result = BrentqSolver.solve(f, xa, xb,
                BrentqSolver.DEFAULT_XTOL, BrentqSolver.DEFAULT_RTOL, BrentqSolver.DEFAULT_MAXITER);

        assertThat(result.isSuccessful())
                .as("Solver must converge for valid bracket")
                .isTrue();

        double root = result.getRoot();
        double fRoot = Math.abs(f.applyAsDouble(root));
        // The convergence check is based on bracket size (|sbis| < delta),
        // which guarantees the root is within delta of xcur.
        // We allow a small multiplier to account for function derivative effects.
        double tol = 10 * (BrentqSolver.DEFAULT_XTOL + BrentqSolver.DEFAULT_RTOL * Math.abs(root));

        assertThat(fRoot)
                .as("|f(root)| must be near zero, got |f(%s)| = %s, tol = %s",
                        root, fRoot, tol)
                .isLessThanOrEqualTo(tol);

        assertThat(result.getIterations())
                .as("Iterations must not exceed maxIterations=100")
                .isLessThanOrEqualTo(100);
    }

    @Provide
    Arbitrary<Double> coefficients() {
        return Arbitraries.doubles().between(-5.0, 5.0);
    }

    @Provide
    Arbitrary<Double> bracketLeft() {
        return Arbitraries.doubles().between(-10.0, 9.0);
    }

    @Provide
    Arbitrary<Double> bracketWidth() {
        // Width in [0.01, 10] to avoid degenerate brackets
        return Arbitraries.doubles().between(0.01, 10.0);
    }

    // ========================================================================
    // Property 2: invalid bracket exception
    // ========================================================================

    /**
     * For any interval {@code [a, b]} where {@code f(a)*f(b) > 0}
     * (using {@code f(x) = x² + 1}, which is always positive), the solver
     * SHALL throw {@link IllegalArgumentException}.
     */
    @Property(tries = 100)
    void invalidBracketThrowsException(
            @ForAll("anyDouble") double a,
            @ForAll("positiveWidth") double width) {

        double b = a + width;
        // f(x) = x² + 1 is always positive, so f(a)*f(b) > 0 always
        DoubleUnaryOperator f = x -> x * x + 1.0;

        double fa = f.applyAsDouble(a);
        double fb = f.applyAsDouble(b);

        // Confirm the bracket is indeed invalid
        Assume.that(fa * fb > 0);

        assertThatThrownBy(() ->
                BrentqSolver.solve(f, a, b, BrentqSolver.DEFAULT_XTOL, BrentqSolver.DEFAULT_RTOL, BrentqSolver.DEFAULT_MAXITER))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Provide
    Arbitrary<Double> anyDouble() {
        return Arbitraries.doubles().between(-100.0, 100.0);
    }

    @Provide
    Arbitrary<Double> positiveWidth() {
        return Arbitraries.doubles().between(0.01, 100.0);
    }

    // ========================================================================
    // Property 11: abnormal termination on NaN/Infinity function values
    // ========================================================================

    /**
     * For any function that returns {@code NaN} on the {@code k}-th call
     * (k randomly chosen between 1 and 5), the solver SHALL stop and return
     * {@link OptimizationStatus#ABNORMAL_TERMINATION} rather than throwing an exception
     * or looping infinitely.
     */
    @Property(tries = 50)
    void nanFunctionValueCausesAbnormalTermination(
            @ForAll @IntRange(min = 1, max = 5) int nanOnCall) {

        // Base function: sin(x) on [3, 4] has a root at π
        AtomicInteger callCount = new AtomicInteger(0);
        DoubleUnaryOperator f = x -> {
            int call = callCount.incrementAndGet();
            // The first 2 calls are for xa and xb in initialization;
            // subsequent calls are iteration steps
            if (call == nanOnCall + 2) {
                return Double.NaN;
            }
            return Math.sin(x);
        };

        OptimizationResult result = BrentqSolver.solve(
                f, 3.0, 4.0,
                BrentqSolver.DEFAULT_XTOL, BrentqSolver.DEFAULT_RTOL,
                BrentqSolver.DEFAULT_MAXITER);

        assertThat(result.getStatus())
                .as("Solver must return ABNORMAL_TERMINATION when function returns NaN")
                .isEqualTo(OptimizationStatus.ABNORMAL_TERMINATION);
    }

    // ========================================================================
    // Unit tests (@Example)
    // ========================================================================

    /**
     * sin(x) on [3, 4] has a root at π ≈ 3.14159265358979...
     * Verifies |root - π| < 1e-10.
     */
    @Example
    void sinRootApproximatesPi() {
        OptimizationResult result = BrentqSolver.solve(
                Math::sin, 3.0, 4.0,
                BrentqSolver.DEFAULT_XTOL, BrentqSolver.DEFAULT_RTOL,
                BrentqSolver.DEFAULT_MAXITER);

        assertThat(result.isSuccessful()).isTrue();
        assertThat(Math.abs(result.getRoot() - Math.PI))
                .as("|root - π| must be < 1e-10")
                .isLessThan(1e-10);
    }

    /**
     * x³ - x - 2 on [1, 2] has a root at ≈ 1.5213797068045676.
     * Verifies precision < 1e-10.
     */
    @Example
    void cubicRootPrecision() {
        DoubleUnaryOperator f = x -> x * x * x - x - 2.0;
        OptimizationResult result = BrentqSolver.solve(
                f, 1.0, 2.0,
                BrentqSolver.DEFAULT_XTOL, BrentqSolver.DEFAULT_RTOL,
                BrentqSolver.DEFAULT_MAXITER);

        double expected = 1.5213797068045676;
        assertThat(result.isSuccessful()).isTrue();
        assertThat(Math.abs(result.getRoot() - expected))
                .as("|root - 1.5213797...| must be < 1e-10")
                .isLessThan(1e-10);
    }

    /**
     * exp(x) - 2 on [0, 1] has a root at ln(2) ≈ 0.6931471805599453.
     * Verifies precision < 1e-10.
     */
    @Example
    void expMinusTwoRoot() {
        DoubleUnaryOperator f = x -> Math.exp(x) - 2.0;
        OptimizationResult result = BrentqSolver.solve(
                f, 0.0, 1.0,
                BrentqSolver.DEFAULT_XTOL, BrentqSolver.DEFAULT_RTOL,
                BrentqSolver.DEFAULT_MAXITER);

        assertThat(result.isSuccessful()).isTrue();
        assertThat(Math.abs(result.getRoot() - Math.log(2.0)))
                .as("|root - ln(2)| must be < 1e-10")
                .isLessThan(1e-10);
    }

    /**
     * When f(a) = 0, the solver SHALL return the left endpoint directly
     * with iterations == 0.
     * f(x) = x - 1.5, bracket [1.5, 3]: f(1.5) = 0.
     */
    @Example
    void endpointRootReturnedDirectly() {
        DoubleUnaryOperator f = x -> x - 1.5;
        OptimizationResult result = BrentqSolver.solve(
                f, 1.5, 3.0,
                BrentqSolver.DEFAULT_XTOL, BrentqSolver.DEFAULT_RTOL,
                BrentqSolver.DEFAULT_MAXITER);

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getRoot())
                .as("Root must equal the endpoint 1.5")
                .isEqualTo(1.5);
        assertThat(result.getIterations())
                .as("Iterations must be 0 when endpoint is already a root")
                .isEqualTo(0);
    }

    /**
     * When maxIterations = 1, the solver SHALL return MAX_ITERATIONS_REACHED
     * for sin(x) on [3, 4] (which needs more than 1 iteration).
     */
    @Example
    void maxIterationsOneReturnsMaxIterationsReached() {
        OptimizationResult result = BrentqSolver.solve(
                Math::sin, 3.0, 4.0,
                BrentqSolver.DEFAULT_XTOL, BrentqSolver.DEFAULT_RTOL,
                1);

        assertThat(result.getStatus())
                .as("Status must be MAX_ITERATIONS_REACHED when maxIterations=1")
                .isEqualTo(OptimizationStatus.MAX_ITERATIONS_REACHED);
    }
}
