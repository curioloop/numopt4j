/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

import com.curioloop.numopt4j.optim.root.RootFinder;

import java.util.function.BiConsumer;
import java.util.function.DoubleUnaryOperator;

/**
 * Unified root-finding entry point, parallel to {@link Minimize}.
 *
 * <h2>One-dimensional usage (Brentq)</h2>
 * <pre>{@code
 * OptimizationResult r = FindRoot.scalar(x -> Math.sin(x))
 *     .bracket(3.0, 4.0)
 *     .solve();
 * double root = r.getRoot();
 * }</pre>
 *
 * <h2>Multi-dimensional usage</h2>
 * <pre>{@code
 * OptimizationResult r = FindRoot.equations((x, f) -> {
 *         f[0] = x[0]*x[0] - 2;
 *         f[1] = x[1] - x[0];
 *     }, 2)
 *     .initialPoint(1.0, 1.0)
 *     .solve();
 * double[] solution = r.getSolution();
 * }</pre>
 */
public final class FindRoot {

    private FindRoot() {}

    /**
     * Creates a {@link RootFinder} pre-configured with a scalar function for Brentq.
     *
     * @param f scalar function whose root is sought
     * @return configured {@link RootFinder} builder
     */
    public static RootFinder scalar(DoubleUnaryOperator f) {
        return RootFinder.create().function(f);
    }

    /**
     * Creates a {@link RootFinder} pre-configured with a system of equations.
     * Use {@link RootFinder#jacobian(NumericalJacobian)} to override the default {@link NumericalJacobian#FORWARD}.
     *
     * @param fn system of equations F(x) = 0
     * @param n  number of equations / unknowns
     * @return configured {@link RootFinder} builder
     */
    public static RootFinder equations(BiConsumer<double[], double[]> fn, int n) {
        return RootFinder.create().equations(fn, n);
    }
}
