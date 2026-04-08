/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode;

/**
 * ODE system interface for initial value problems (IVP) of the form:
 * <pre>
 *   dy/dt = f(t, y),   y(t₀) = y₀
 * </pre>
 *
 * <p>The outer interface {@code ODE} carries both the right-hand side function and an optional
 * analytic Jacobian.  When only the RHS is needed (explicit RK methods), use the lightweight
 * {@link Equation} functional interface instead.</p>
 *
 * <p>Implicit solvers (BDF, Radau) accept either form:
 * <ul>
 *   <li>Pass an {@link Equation} — the Jacobian is approximated numerically via adaptive
 *       forward differences (scipy {@code num_jac} algorithm).</li>
 *   <li>Pass a full {@code ODE} — the analytic Jacobian is used directly, which is faster
 *       and more accurate for stiff problems.</li>
 * </ul>
 * </p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Equation-only (lambda-friendly)
 * ODE.Equation eq = (t, y, dydt) -> {
 *     dydt[0] = -y[0];
 * };
 *
 * // Full ODE with analytic Jacobian (for stiff problems)
 * ODE stiff = (t, y, dydt, jac) -> {
 *     dydt[0] = -1000 * y[0];
 *     if (jac != null) jac[0] = -1000;   // ∂f₀/∂y₀
 * };
 * }</pre>
 *
 * @see ODEIntegral
 */
public interface ODE {

    /**
     * Supported IVP solver methods.
     *
     * <ul>
     *   <li>{@link #RK23} — Bogacki-Shampine 3(2) explicit RK, suitable for non-stiff problems.</li>
     *   <li>{@link #RK45} — Dormand-Prince 5(4) explicit RK (default), good general-purpose choice.</li>
     *   <li>{@link #DOP853} — Hairer's 8(5,3) explicit RK, high accuracy for smooth problems.</li>
     *   <li>{@link #BDF} — Backward Differentiation Formula (variable order 1–5), for stiff problems.</li>
     *   <li>{@link #Radau} — Radau IIA implicit RK (5th order, 3 stages), for stiff problems.</li>
     * </ul>
     */
    enum Method {
        RK23, RK45, DOP853, BDF, Radau
    }

    /**
     * Lightweight RHS-only interface, suitable for non-stiff problems and lambda expressions.
     *
     * <p>Computes {@code dydt = f(t, y)} in-place.</p>
     */
    @FunctionalInterface
    interface Equation {
        /**
         * Evaluates the ODE right-hand side.
         *
         * @param t    current time
         * @param y    current state vector, length n (read-only)
         * @param dydt output derivative vector, length n (written in-place)
         */
        void evaluate(double t, double[] y, double[] dydt);
    }

    /**
     * Evaluates the ODE right-hand side and optionally the Jacobian.
     *
     * <p>When {@code jac != null}, fills both {@code dydt} and {@code jac}.
     * When {@code jac == null}, only {@code dydt} needs to be computed.</p>
     *
     * @param t    current time
     * @param y    current state vector, length n (read-only)
     * @param dydt output derivative vector, length n (written in-place)
     * @param jac  row-major n×n Jacobian buffer, {@code jac[i*n+j] = ∂fᵢ/∂yⱼ};
     *             may be {@code null} when only the RHS is needed
     */
    void evaluate(double t, double[] y, double[] dydt, double[] jac);
}
