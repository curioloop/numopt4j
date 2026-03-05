package com.curioloop.numopt4j.optim.root;

/**
 * Unified workspace handle for multi-dimensional root-finding solvers.
 *
 * <p>Holds a {@link HYBRWorkspace} and a {@link BroydenWorkspace} so that
 * {@link RootFinder} can allocate once and dispatch to either solver without
 * re-allocating memory between calls.</p>
 *
 * <p>Each solver's workspace is allocated lazily on first use.</p>
 */
public final class RootWorkspace {

    private final int n;
    private HYBRWorkspace   hybr;
    private BroydenWorkspace broyden;

    public RootWorkspace(int n) {
        if (n < 1) throw new IllegalArgumentException("Workspace dimension must be >= 1, got: " + n);
        this.n = n;
    }

    public boolean isCompatible(int n) { return this.n == n; }

    /** Returns (allocating if necessary) the HYBR workspace. */
    public HYBRWorkspace hybr() {
        if (hybr == null) hybr = new HYBRWorkspace(n);
        return hybr;
    }

    /** Returns (allocating if necessary) the Broyden workspace. */
    public BroydenWorkspace broyden() {
        if (broyden == null) broyden = new BroydenWorkspace(n);
        return broyden;
    }

    /** Resets whichever sub-workspaces have been allocated. */
    public void reset() {
        if (hybr    != null) hybr.reset();
        if (broyden != null) broyden.reset();
    }
}
