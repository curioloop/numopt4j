package com.curioloop.numopt4j.optim.root;

/**
 * Algorithm selection enumeration for root-finding.
 *
 * <p>Used with {@link RootFinder#method(RootMethod)} to explicitly choose
 * the underlying solver. If not set, the solver is selected automatically
 * based on the configured inputs.</p>
 */
public enum RootMethod {

    /**
     * One-dimensional bracketing method (Brent's method).
     *
     * <p>Robust, derivative-free algorithm that guarantees convergence when
     * a valid bracket {@code [a, b]} with {@code f(a)*f(b) <= 0} is provided.
     * Mirrors {@code scipy.optimize.brentq}.</p>
     */
    BRENTQ,

    /**
     * Multi-dimensional Powell Hybrid method (MINPACK {@code hybrd}).
     *
     * <p>Trust-region method combining QR decomposition with rank-1 Broyden
     * updates. Default solver for multi-dimensional systems.
     * Mirrors {@code scipy.optimize.root(method='hybr')}.</p>
     */
    HYBR,

    /**
     * Multi-dimensional quasi-Newton method (Good Broyden / {@code broyden1}).
     *
     * <p>Jacobian-free iterative solver that maintains a low-cost rank-1
     * approximation of the inverse Jacobian. Suitable when the Jacobian is
     * expensive or unavailable.
     * Mirrors {@code scipy.optimize.root(method='broyden1')}.</p>
     */
    BROYDEN
}
