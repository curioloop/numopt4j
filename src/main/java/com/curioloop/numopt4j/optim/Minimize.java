/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.numopt4j.optim.slsqp.SLSQPProblem;
import com.curioloop.numopt4j.optim.trf.TRFProblem;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.ToDoubleFunction;

/**
 * Unified optimization entry point that automatically selects the best algorithm.
 *
 * <p>Algorithm selection rules:</p>
 * <ul>
 *   <li>Objective function only + optional bounds → L-BFGS-B (bounded quasi-Newton)</li>
 *   <li>Equality or inequality constraints → SLSQP (sequential least squares programming)</li>
 *   <li>Residual function (least squares) → TRF (trust region reflective)</li>
 * </ul>
 *
 * <h2>Examples</h2>
 * <pre>{@code
 * // Unconstrained optimization (auto-selects L-BFGS-B)
 * OptimizationResult r = Minimize.objective(x -> x[0]*x[0] + x[1]*x[1])
 *     .startingFrom(0.5, 0.5)
 *     .run();
 *
 * // Bound-constrained optimization (auto-selects L-BFGS-B)
 * OptimizationResult r = Minimize.objective(x -> x[0]*x[0] + x[1]*x[1])
 *     .bounds(Bound.atLeast(0), Bound.atLeast(0))
 *     .startingFrom(0.5, 0.5)
 *     .run();
 *
 * // Constrained optimization (auto-selects SLSQP)
 * OptimizationResult r = Minimize.objective(x -> x[0] + x[1])
 *     .subjectTo(x -> x[0]*x[0] + x[1]*x[1] - 1)
 *     .startingFrom(0.5, 0.5)
 *     .run();
 *
 * // Least squares (auto-selects TRF)
 * OptimizationResult r = Minimize.leastSquares((x, res) -> {
 *         res[0] = x[0] - 1; res[1] = x[1] - 2;
 *     }, 2)
 *     .startingFrom(0.0, 0.0)
 *     .run();
 * }</pre>
 *
 * @see LBFGSBProblem
 * @see SLSQPProblem
 * @see TRFProblem
 */
public final class Minimize {

    private Minimize() {}

    /**
     * Creates an objective builder for minimization with a simple function (recommended for AI use).
     * <p>Uses {@link NumericalGradient#CENTRAL} for automatic gradient computation.</p>
     *
     * @param f objective function ℝⁿ → ℝ
     * @return builder for configuring and running the optimization
     */
    public static ObjectiveBuilder objective(ToDoubleFunction<double[]> f) {
        if (f == null) throw new IllegalArgumentException("objective function must not be null");
        return new ObjectiveBuilder(f, null);
    }

    /**
     * Creates an objective builder with an analytical gradient function.
     *
     * @param f objective function with gradient
     * @return builder for configuring and running the optimization
     */
    public static ObjectiveBuilder objective(Univariate f) {
        if (f == null) throw new IllegalArgumentException("objective function must not be null");
        return new ObjectiveBuilder(null, f);
    }

    /**
     * Creates a least-squares builder using a residual function (auto-selects TRF).
     *
     * @param residuals function computing residuals: (x, r) -> void
     * @param m         number of residuals
     * @return builder for configuring and running the optimization
     */
    public static LeastSquaresBuilder leastSquares(BiConsumer<double[], double[]> residuals, int m) {
        if (residuals == null) throw new IllegalArgumentException("residuals function must not be null");
        return new LeastSquaresBuilder(residuals, null, m);
    }

    /**
     * Creates a least-squares builder using a multivariate function with analytical Jacobian.
     *
     * @param f multivariate function computing residuals and Jacobian
     * @param m number of residuals
     * @return builder for configuring and running the optimization
     */
    public static LeastSquaresBuilder leastSquares(Multivariate f, int m) {
        if (f == null) throw new IllegalArgumentException("residuals function must not be null");
        return new LeastSquaresBuilder(null, f, m);
    }

    /**
     * Builder for objective-based optimization (L-BFGS-B or SLSQP).
     */
    public static final class ObjectiveBuilder {
        private final ToDoubleFunction<double[]> func;
        private final Univariate univariate;
        private Bound[] bounds;
        private final List<ToDoubleFunction<double[]>> ineqConstraints = new ArrayList<>();
        private final List<ToDoubleFunction<double[]>> eqConstraints = new ArrayList<>();
        private double[] x0;

        private ObjectiveBuilder(ToDoubleFunction<double[]> func, Univariate univariate) {
            this.func = func;
            this.univariate = univariate;
        }

        /** Sets variable bounds (uses L-BFGS-B if no other constraints). */
        public ObjectiveBuilder bounds(Bound... bounds) {
            this.bounds = bounds;
            return this;
        }

        /** Adds inequality constraints c(x) >= 0 (triggers SLSQP selection). */
        @SafeVarargs
        public final ObjectiveBuilder subjectTo(ToDoubleFunction<double[]>... constraints) {
            for (ToDoubleFunction<double[]> c : constraints) ineqConstraints.add(c);
            return this;
        }

        /** Adds equality constraints c(x) = 0 (triggers SLSQP selection). */
        @SafeVarargs
        public final ObjectiveBuilder equalTo(ToDoubleFunction<double[]>... constraints) {
            for (ToDoubleFunction<double[]> c : constraints) eqConstraints.add(c);
            return this;
        }

        /** Sets the initial point. */
        public ObjectiveBuilder startingFrom(double... x0) {
            this.x0 = x0;
            return this;
        }

        /**
         * Runs the optimization, automatically selecting L-BFGS-B or SLSQP.
         *
         * @return optimization result
         */
        public OptimizationResult run() {
            boolean hasConstraints = !ineqConstraints.isEmpty() || !eqConstraints.isEmpty();
            if (hasConstraints) {
                // Use SLSQP for constrained optimization
                SLSQPProblem problem = SLSQPProblem.create();
                if (func != null) {
                    problem.objective(func);
                } else {
                    problem.objective(univariate);
                }
                if (x0 != null) problem.initialPoint(x0);
                if (bounds != null) problem.bounds(bounds);
                if (!ineqConstraints.isEmpty()) {
                    @SuppressWarnings("unchecked")
                    ToDoubleFunction<double[]>[] arr = ineqConstraints.toArray(new ToDoubleFunction[0]);
                    problem.inequalityConstraints(arr);
                }
                if (!eqConstraints.isEmpty()) {
                    @SuppressWarnings("unchecked")
                    ToDoubleFunction<double[]>[] arr = eqConstraints.toArray(new ToDoubleFunction[0]);
                    problem.equalityConstraints(arr);
                }
                return problem.solve();
            } else {
                // Use L-BFGS-B for unconstrained or bound-constrained optimization
                LBFGSBProblem problem = LBFGSBProblem.create();
                if (func != null) {
                    problem.objective(func);
                } else {
                    problem.objective(univariate);
                }
                if (x0 != null) problem.initialPoint(x0);
                if (bounds != null) problem.bounds(bounds);
                return problem.solve();
            }
        }
    }

    /**
     * Builder for least-squares optimization (always uses TRF).
     */
    public static final class LeastSquaresBuilder {
        private final BiConsumer<double[], double[]> residualsFn;
        private final Multivariate multivariate;
        private final int m;
        private Bound[] bounds;
        private double[] x0;

        private LeastSquaresBuilder(BiConsumer<double[], double[]> residualsFn, Multivariate multivariate, int m) {
            this.residualsFn = residualsFn;
            this.multivariate = multivariate;
            this.m = m;
        }

        /** Sets variable bounds. */
        public LeastSquaresBuilder bounds(Bound... bounds) {
            this.bounds = bounds;
            return this;
        }

        /** Sets the initial point. */
        public LeastSquaresBuilder startingFrom(double... x0) {
            this.x0 = x0;
            return this;
        }

        /**
         * Runs the least-squares optimization using TRF.
         *
         * @return optimization result
         */
        public OptimizationResult run() {
            TRFProblem problem = TRFProblem.create();
            if (residualsFn != null) {
                problem.residuals(residualsFn, m);
            } else {
                problem.objective(multivariate);
            }
            if (x0 != null) problem.initialPoint(x0);
            if (bounds != null) problem.bounds(bounds);
            return problem.solve();
        }
    }
}
