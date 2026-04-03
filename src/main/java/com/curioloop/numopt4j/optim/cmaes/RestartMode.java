/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

/**
 * Restart mode configuration for CMA-ES.
 *
 * <p>Use {@code null} for no restart (single run).
 * Use {@link #ipop(int, int)} or {@link #bipop(int)} for multi-start strategies.</p>
 *
 * <pre>{@code
 * // No restart (default)
 * Minimizer.cmaes().objective(fn).initialPoint(x0).solve();
 *
 * // IPOP: lambda doubles each restart
 * Minimizer.cmaes().objective(fn).initialPoint(x0)
 *     .restart(RestartMode.ipop(9, 2))
 *     .solve();
 *
 * // BIPOP: alternates large and small population regimes
 * Minimizer.cmaes().objective(fn).initialPoint(x0)
 *     .restart(RestartMode.bipop(9))
 *     .solve();
 * }</pre>
 */
public final class RestartMode {

    /** IPOP or BIPOP. */
    public enum Type { IPOP, BIPOP }

    /** Strategy type. */
    public final Type type;

    /** Maximum number of restarts. */
    public final int maxRestarts;

    /**
     * Population size multiplier per restart (IPOP only).
     * lambda_{k+1} = lambda_k * popSizeMultiplier.
     */
    public final int popSizeMultiplier;

    private RestartMode(Type type, int maxRestarts, int popSizeMultiplier) {
        this.type = type;
        this.maxRestarts = maxRestarts;
        this.popSizeMultiplier = popSizeMultiplier;
    }

    /**
     * IPOP restart: multiply lambda by {@code popSizeMultiplier} on each restart.
     *
     * @param maxRestarts       maximum number of restarts (≥ 0)
     * @param popSizeMultiplier lambda multiplier per restart (≥ 2)
     */
    public static RestartMode ipop(int maxRestarts, int popSizeMultiplier) {
        if (maxRestarts < 0)
            throw new IllegalArgumentException("maxRestarts must be >= 0, got " + maxRestarts);
        if (popSizeMultiplier < 2)
            throw new IllegalArgumentException("popSizeMultiplier must be >= 2, got " + popSizeMultiplier);
        return new RestartMode(Type.IPOP, maxRestarts, popSizeMultiplier);
    }

    /**
     * BIPOP restart: alternates between large-population and small-population regimes.
     * Large population doubles each restart; small population is randomly chosen.
     *
     * @param maxRestarts maximum number of restarts (≥ 0)
     */
    public static RestartMode bipop(int maxRestarts) {
        if (maxRestarts < 0)
            throw new IllegalArgumentException("maxRestarts must be >= 0, got " + maxRestarts);
        return new RestartMode(Type.BIPOP, maxRestarts, 2);
    }

    @Override
    public String toString() {
        if (type == Type.IPOP)
            return "IPOP(maxRestarts=" + maxRestarts + ", popSizeMultiplier=" + popSizeMultiplier + ")";
        return "BIPOP(maxRestarts=" + maxRestarts + ")";
    }
}
