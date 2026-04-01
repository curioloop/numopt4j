/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.Optimization;

import net.jqwik.api.*;
import net.jqwik.api.constraints.IntRange;

import java.util.function.BiConsumer;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Tests for {@link BroydenSolver}.
 *
 * <p>Covers:</p>
 * <ul>
 *   <li>Property: convergence on diagonally dominant linear systems</li>
 *   <li>Unit tests: Rosenbrock system, maxIterations exceeded</li>
 * </ul>
 */
@SuppressWarnings("unused")
public class BroydenSolverTest {

    // ========================================================================
    // Convergence on diagonally dominant linear systems
    // ========================================================================

    /**
     * For any random n×n diagonally dominant linear system {@code F(x) = A*x - b}
     * with a known solution {@code x*}, starting from a random initial point near {@code x*}
     * (offset in [-1, 1]), Broyden SHALL return a solution with
     * {@code max|F(x)| <= ftol} (max-norm, matching scipy's tol_norm=maxnorm)
     * and status converged.
     */
    @Property(tries = 100)
    void broydenConvergesOnLinearSystem(
            @ForAll @IntRange(min = 1, max = 4) int n,
            @ForAll("smallDoubles") double[] rawCoeffs,
            @ForAll("smallDoubles") double[] rawSolution,
            @ForAll("unitOffsets") double[] rawOffsets) {

        // Build known solution x*
        double[] xStar = new double[n];
        for (int i = 0; i < n; i++) {
            xStar[i] = rawSolution[i % rawSolution.length];
        }

        // Build diagonally dominant A (n×n): diagonal = sum(|off-diagonal|) + 1
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) {
            double rowSum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    A[i][j] = rawCoeffs[(i * n + j) % rawCoeffs.length] * 0.5;
                    rowSum += Math.abs(A[i][j]);
                }
            }
            A[i][i] = rowSum + 1.0;
        }

        // Compute b = A * x*
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i] += A[i][j] * xStar[j];
            }
        }

        // F(x) = A*x - b
        final double[][] Afinal = A;
        BiConsumer<double[], double[]> fn = (x, fx) -> {
            for (int i = 0; i < n; i++) {
                fx[i] = -b[i];
                for (int j = 0; j < n; j++) fx[i] += Afinal[i][j] * x[j];
            }
        };

        // Random initial point near x* with offset in [-1, 1]
        double[] x0 = new double[n];
        for (int i = 0; i < n; i++) {
            x0[i] = xStar[i] + rawOffsets[i % rawOffsets.length];
        }

        double ftol = BroydenSolver.DEFAULT_FTOL;
        int maxiter = Math.max(50 * n, 200);
        BroydenWorkspace ws = new BroydenWorkspace(n);

        Optimization result = BroydenSolver.solve(fn, x0, ftol, maxiter, ws);

        assertThat(result.isSuccessful())
                .as("Broyden must converge for diagonally dominant linear system (n=%d)", n)
                .isTrue();

        // getCost() returns max-norm (scipy convention)
        assertThat(result.getCost())
                .as("max|F(x)| must be <= ftol, got %s", result.getCost())
                .isLessThanOrEqualTo(ftol);
    }

    @Provide
    Arbitrary<double[]> smallDoubles() {
        return Arbitraries.doubles().between(-2.0, 2.0)
                .array(double[].class).ofMinSize(16).ofMaxSize(16);
    }

    @Provide
    Arbitrary<double[]> unitOffsets() {
        return Arbitraries.doubles().between(-1.0, 1.0)
                .array(double[].class).ofMinSize(4).ofMaxSize(4);
    }

    // ========================================================================
    // Unit tests (@Example)
    // ========================================================================

    /**
     * Rosenbrock system (n=2):
     *   F₁ = 1 - x₁
     *   F₂ = 10*(x₂ - x₁²)
     * Starting from (0.5, 0.5), solution is (1, 1).
     * Verifies convergence and max-norm residual <= ftol.
     */
    @Example
    void rosenbrockSystemConverges() {
        BiConsumer<double[], double[]> fn = (x, fx) -> {
            fx[0] = 1.0 - x[0];
            fx[1] = 10.0 * (x[1] - x[0] * x[0]);
        };

        double[] x0 = {0.5, 0.5};
        int n = 2;
        int maxiter = 500;
        BroydenWorkspace ws = new BroydenWorkspace(n);

        Optimization result = BroydenSolver.solve(fn, x0, BroydenSolver.DEFAULT_FTOL, maxiter, ws);

        assertThat(result.isSuccessful())
                .as("Broyden must converge on Rosenbrock system from (0.5, 0.5)")
                .isTrue();

        // getCost() returns max-norm (scipy convention)
        assertThat(result.getCost())
                .as("max|F(x)| must be <= ftol")
                .isLessThanOrEqualTo(BroydenSolver.DEFAULT_FTOL);
    }

    /**
     * When maxIterations is too small, the solver SHALL return MAX_ITERATIONS_REACHED.
     * F(x) = [x₁² - 2, x₂² - 3] (solutions at ±√2, ±√3), starting far from solution,
     * maxiter=2 forces early termination.
     */
    @Example
    void maxIterationsExceededReturnsMaxIterationsReached() {
        BiConsumer<double[], double[]> fn = (x, fx) -> {
            fx[0] = x[0] * x[0] - 2.0;
            fx[1] = x[1] * x[1] - 3.0;
        };

        // Start far from any solution; with maxiter=2 Broyden cannot converge
        double[] x0 = {10.0, 10.0};
        int n = 2;
        BroydenWorkspace ws = new BroydenWorkspace(n);

        Optimization result = BroydenSolver.solve(fn, x0, BroydenSolver.DEFAULT_FTOL, 2, ws);

        assertThat(result.getStatus())
                .as("Status must be MAX_ITERATIONS_REACHED when maxiter=2 and solution is far")
                .isEqualTo(Optimization.Status.MAX_ITERATIONS_REACHED);
    }
}
