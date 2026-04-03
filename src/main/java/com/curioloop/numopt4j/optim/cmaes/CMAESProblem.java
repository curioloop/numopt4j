/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;
import java.util.Objects;

import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;
import com.curioloop.numopt4j.optim.Univariate;
import java.util.Random;

/**
 * Fluent configuration builder for the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
 *
 * <p>CMA-ES is a derivative-free, stochastic global optimizer for non-convex, noisy, and
 * multimodal continuous optimization problems.  It adapts a full (or diagonal) covariance
 * matrix each generation to learn the local problem geometry.</p>
 *
 * <h2>Quick Start</h2>
 * <pre>{@code
 * // Minimise sphere function in 5 dimensions
 * Optimization r = Minimizer.cmaes()
 *     .objective((x, n) -> { double s = 0; for (double v : x) s += v*v; return s; })
 *     .initialPoint(new double[5])
 *     .solve();
 * }</pre>
 *
 * <h2>Update Modes</h2>
 * <ul>
 *   <li>{@link UpdateMode#ACTIVE_CMA}  — full C with Active CMA (default; best on ill-conditioned problems)</li>
 *   <li>{@link UpdateMode#CLASSIC_CMA} — full C without Active CMA (classic CMA-ES)</li>
 *   <li>{@link UpdateMode#SEP_CMA}     — diagonal C only; O(n) per iteration, skips O(n²) allocations</li>
 * </ul>
 *
 * <h2>Restart Strategies</h2>
 * <ul>
 *   <li>{@code null} (default) — single run, no restart</li>
 *   <li>{@link RestartMode#ipop(int, int)} — IPOP: multiply λ by {@code popSizeMultiplier} each restart</li>
 *   <li>{@link RestartMode#bipop(int)}     — BIPOP: alternate large/small population regimes</li>
 * </ul>
 *
 * <h2>Evaluation Budget</h2>
 * <p>Exactly one of the following should be set (last one wins):</p>
 * <ul>
 *   <li>{@link #maxEvaluations(int)} — absolute limit on objective function calls</li>
 *   <li>{@link #maxGenerations(int)} — limit in generations; resolved lazily as {@code n × λ}</li>
 * </ul>
 * <p>Default: {@code λ × 1000} evaluations.</p>
 *
 * <h2>Workspace Reuse</h2>
 * <pre>{@code
 * CMAESProblem p = Minimizer.cmaes()
 *     .objective(fn)
 *     .initialPoint(x0);
 * CMAESWorkspace ws = CMAESProblem.workspace();
 * for (double[] pt : points) {
 *     Optimization r = p.initialPoint(pt).solve(ws);
 * }
 * }</pre>
 *
 * <h2>Default Parameter Values</h2>
 * <table border="1">
 *   <tr><th>Parameter</th><th>Default</th><th>Notes</th></tr>
 *   <tr><td>sigma</td><td>0.3</td><td>initial step size σ₀</td></tr>
 *   <tr><td>populationSize</td><td>4 + ⌊3·ln(n)⌋</td><td>λ, auto-computed from dimension</td></tr>
 *   <tr><td>maxIterations</td><td>1000</td><td>per-run generation limit</td></tr>
 *   <tr><td>maxEvaluations</td><td>λ × 1000</td><td>total function evaluation budget</td></tr>
 *   <tr><td>covarianceMode</td><td>ACTIVE_CMA</td><td>full C with Active CMA</td></tr>
 *   <tr><td>restart</td><td>null</td><td>no restart</td></tr>
 *   <tr><td>maxResample</td><td>0</td><td>no resampling for infeasible points</td></tr>
 *   <tr><td>stopFitness</td><td>−∞</td><td>disabled</td></tr>
 *   <tr><td>parameterTolerance</td><td>1e-11</td><td>step-size convergence threshold</td></tr>
 *   <tr><td>functionTolerance</td><td>1e-12</td><td>fitness history range threshold</td></tr>
 *   <tr><td>sigmaUpperBound</td><td>1e3</td><td>σ divergence guard</td></tr>
 * </table>
 *
 * @see Minimizer#cmaes()
 * @see UpdateMode
 * @see RestartMode
 * @see CMAESWorkspace
 */
public final class CMAESProblem
        extends Minimizer<Univariate.Objective, CMAESWorkspace, CMAESProblem> {

    double sigma0 = 0.3;
    private int lambda = 0;                // 0 = auto: 4 + floor(3*ln(n))
    int maxIterations = 1000;
    int maxEvaluations = 0;               // 0 = auto: lambda * 1000
    int maxGenerations = 0;               // 0 = not set; if set: maxEvaluations = maxGenerations * lambda
    UpdateMode updateMode = UpdateMode.ACTIVE_CMA;
    int maxResample = 0;
    private RestartMode restartMode = null;
    double stopFitness = Double.NEGATIVE_INFINITY;
    double parameterTolerance = 1e-11;
    double functionTolerance = 1e-12;
    double sigmaUpperBound = 1e3;
    private Random rng = new Random();

    public CMAESProblem() {}

    // ── Getters ───────────────────────────────────────────────────────────

    public double sigma0()               { return sigma0; }
    public int lambdaConfig()            { return lambda; }
    public int maxIterations()           { return maxIterations; }
    public UpdateMode updateMode() { return updateMode; }
    public int maxResample()             { return maxResample; }
    public RestartMode restartMode()     { return restartMode; }
    public double stopFitness()          { return stopFitness; }
    public double parameterTolerance()   { return parameterTolerance; }
    public double functionTolerance()    { return functionTolerance; }
    public double sigmaUpperBound()      { return sigmaUpperBound; }

    /** Effective lambda (auto-computed if not set). */
    public int effectiveLambda() {
        return (lambda > 0) ? lambda : (4 + (int) Math.floor(3.0 * Math.log(dimension)));
    }

    /** Effective maxEvaluations (auto-computed if not set). */
    public int effectiveMaxEvaluations() {
        if (maxEvaluations > 0) return maxEvaluations;
        if (maxGenerations > 0) return maxGenerations * effectiveLambda();
        return effectiveLambda() * 1000;
    }


    // ── Fluent setters ────────────────────────────────────────────────────

    /** Sets the objective function to minimise (no gradient required). */
    public CMAESProblem objective(Univariate.Objective f) {
        Objects.requireNonNull(f, "objective must not be null");
        this.objective = f;
        return this;
    }

    /** Sets the initial step size σ₀ (must be positive and finite; default 0.3). */
    public CMAESProblem sigma(double s) {
        if (s <= 0 || !Double.isFinite(s))
            throw new IllegalArgumentException("sigma must be positive and finite, got " + s);
        this.sigma0 = s;
        return this;
    }

    /** Sets the population size λ (default: 4 + ⌊3·ln(n)⌋). */
    public CMAESProblem populationSize(int lam) {
        if (lam <= 0) throw new IllegalArgumentException("populationSize must be positive, got " + lam);
        this.lambda = lam;
        return this;
    }

    /** Sets the per-run generation limit (default 1000). */
    public CMAESProblem maxIterations(int v) {
        if (v <= 0) throw new IllegalArgumentException("maxIterations must be positive, got " + v);
        this.maxIterations = v;
        return this;
    }

    /** Sets the absolute limit on objective function calls (default: λ × 1000). */
    public CMAESProblem maxEvaluations(int v) {
        if (v <= 0) throw new IllegalArgumentException("maxEvaluations must be positive, got " + v);
        this.maxEvaluations = v;
        return this;
    }

    /**
     * Sets the maximum number of generations (iterations).
     * Equivalent to {@code maxEvaluations(n * effectiveLambda())}, but evaluated lazily
     * so it correctly accounts for the population size regardless of call order.
     * Mutually exclusive with {@link #maxEvaluations(int)}: the last one set wins.
     *
     * @param n maximum number of generations (must be &gt; 0)
     */
    public CMAESProblem maxGenerations(int n) {
        if (n <= 0) throw new IllegalArgumentException("maxGenerations must be positive, got " + n);
        this.maxGenerations = n;
        return this;
    }

    /**
     * Sets the covariance matrix update mode (default: {@link UpdateMode#ACTIVE_CMA}).
     * Determines whether to use full or diagonal covariance and whether Active CMA is enabled.
     */
    public CMAESProblem updateMode(UpdateMode mode) {
        Objects.requireNonNull(mode, "updateMode must not be null");
        this.updateMode = mode;
        return this;
    }

    /**
     * Sets the maximum number of resampling attempts for infeasible candidates (default 0).
     * Only effective when {@link #bounds(com.curioloop.numopt4j.optim.Bound...)} is set.
     * When 0, infeasible candidates are kept and penalised by {@code evaluateFitness}.
     */
    public CMAESProblem maxResample(int v) {
        if (v < 0) throw new IllegalArgumentException("maxResample must be >= 0, got " + v);
        this.maxResample = v;
        return this;
    }

    /**
     * Sets the restart mode (default: {@code null} = no restart).
     * Use {@link RestartMode#ipop(int, int)} or {@link RestartMode#bipop(int)}.
     */
    public CMAESProblem restartMode(RestartMode mode) {
        this.restartMode = mode;  // null = no restart
        return this;
    }

    /** Stop when best fitness &lt; {@code v} (default: −∞, disabled). */
    public CMAESProblem stopFitness(double v) {
        this.stopFitness = v;
        return this;
    }

    /** Step-size convergence threshold: stop when σ·max(|p_c_i|, √C_ii) &lt; parameterTolerance·σ₀ (default 1e-11). */
    public CMAESProblem parameterTolerance(double v) {
        if (v <= 0) throw new IllegalArgumentException("parameterTolerance must be positive, got " + v);
        this.parameterTolerance = v;
        return this;
    }

    /** Fitness history range threshold: stop when max(history) − min(history) &lt; functionTolerance (default 1e-12). */
    public CMAESProblem functionTolerance(double v) {
        if (v <= 0) throw new IllegalArgumentException("functionTolerance must be positive, got " + v);
        this.functionTolerance = v;
        return this;
    }

    /** σ divergence guard: stop when σ·D_i &gt; sigmaUpperBound·σ₀ for any i (default 1e3). */
    public CMAESProblem sigmaUpperBound(double v) {
        if (v <= 0) throw new IllegalArgumentException("sigmaUpperBound must be positive, got " + v);
        this.sigmaUpperBound = v;
        return this;
    }

    /** Sets the random number generator (default: {@code new Random()}). */
    public CMAESProblem random(Random r) {
        Objects.requireNonNull(r, "random must not be null");
        this.rng = r;
        return this;
    }


    // ── Validation ────────────────────────────────────────────────────────

    private void validate() {
        requireObjective();
        requireInitialPoint();
        if (sigma0 <= 0)
            throw new IllegalArgumentException("sigma must be positive, got " + sigma0);
    }

    // ── Problem interface ─────────────────────────────────────────────────

    /**
     * Creates a new {@link CMAESWorkspace} for use with {@link #solve(CMAESWorkspace)}.
     * Memory is allocated lazily on the first {@code solve()} call.
     */
    public static CMAESWorkspace workspace() {
        return new CMAESWorkspace();
    }

    @Override
    public Optimization solve(CMAESWorkspace workspace) {
        validate();
        int lam = effectiveLambda();
        int resolvedMaxEval = effectiveMaxEvaluations();

        CMAESWorkspace ws = resolveWorkspace(workspace, CMAESWorkspace::new);
        ws.ensure(dimension, lam, updateMode.separable);

        // Build config snapshot with resolved maxEvaluations
        CMAESProblem cfg = snapshot(resolvedMaxEval);

        if (restartMode == null) return solveOnce(initialPoint, ws, cfg);
        switch (restartMode.type) {
            case IPOP:  return solveIPOP(ws, cfg);
            case BIPOP: return solveBIPOP(ws, cfg);
            default:    return solveOnce(initialPoint, ws, cfg);
        }
    }

    /** Creates a config snapshot with a fixed maxEvaluations. */
    CMAESProblem snapshot(int fixedMaxEval) {
        CMAESProblem c = new CMAESProblem();
        c.objective = this.objective;
        c.initialPoint = this.initialPoint;
        c.dimension = this.dimension;
        c.bounds = this.bounds;
        c.sigma0 = this.sigma0;
        c.lambda = this.lambda;
        c.maxIterations = this.maxIterations;
        c.maxEvaluations = fixedMaxEval;
        c.maxGenerations = 0; // already resolved into fixedMaxEval
        c.updateMode = this.updateMode;
        c.maxResample = this.maxResample;
        c.restartMode = this.restartMode;
        c.stopFitness = this.stopFitness;
        c.parameterTolerance = this.parameterTolerance;
        c.functionTolerance = this.functionTolerance;
        c.sigmaUpperBound = this.sigmaUpperBound;
        c.rng = this.rng;
        return c;
    }

    /** Single run (NONE mode). */
    private Optimization solveOnce(double[] x0, CMAESWorkspace ws, CMAESProblem cfg) {
        return CMAESCore.optimize(x0, objective, bounds, ws, cfg, rng);
    }

    /** IPOP restart strategy. */
    private Optimization solveIPOP(CMAESWorkspace ws, CMAESProblem cfg) {
        int currentLambda = effectiveLambda();
        double[] bestX = initialPoint.clone();
        double bestFitness = Double.MAX_VALUE;
        int totalEvals = 0;
        int totalIters = 0;
        Optimization.Status lastStatus = Optimization.Status.MAX_ITERATIONS_REACHED;

        for (int restart = 0; restart <= restartMode.maxRestarts; restart++) {
            int remainingEval = cfg.maxEvaluations - totalEvals;
            if (remainingEval <= 0) break;

            ws.ensure(dimension, currentLambda, cfg.updateMode.separable);

            CMAESProblem runCfg = cfg.snapshot(remainingEval);
            runCfg.lambda = currentLambda;

            Optimization result = CMAESCore.optimize(initialPoint, objective, bounds, ws, runCfg, rng);
            totalEvals += result.getEvaluations();
            totalIters += result.getIterations();
            lastStatus = result.getStatus();

            if (result.getCost() < bestFitness) {
                bestFitness = result.getCost();
                bestX = result.getSolution().clone();
            }

            if (totalEvals >= cfg.maxEvaluations) break;

            currentLambda *= restartMode.popSizeMultiplier;
        }

        return new Optimization(Double.NaN, bestX, bestFitness, lastStatus, totalIters, totalEvals);
    }

    /** BIPOP restart strategy. */
    private Optimization solveBIPOP(CMAESWorkspace ws, CMAESProblem cfg) {
        int lambdaDefault = effectiveLambda();
        int lambdaLarge = lambdaDefault;

        double[] bestX = initialPoint.clone();
        double bestFitness = Double.MAX_VALUE;
        int totalEvals = 0;
        int totalIters = 0;
        Optimization.Status lastStatus = Optimization.Status.MAX_ITERATIONS_REACHED;

        int largeEvals = 0;
        int smallEvals = 0;
        boolean firstRun = true;

        for (int restart = 0; restart <= restartMode.maxRestarts; restart++) {
            if (totalEvals >= cfg.maxEvaluations) break;

            int currentLambda;
            double currentSigma;

            boolean useLarge = firstRun || (largeEvals <= smallEvals);
            firstRun = false;

            if (useLarge) {
                if (restart > 0) lambdaLarge *= 2;
                currentLambda = lambdaLarge;
                currentSigma = sigma0;
            } else {
                double u = rng.nextDouble();
                currentLambda = Math.max(lambdaDefault,
                    (int) Math.floor(lambdaLarge * u * u));
                currentSigma = sigma0 * Math.pow(10.0, -2.0 * rng.nextDouble());
            }

            ws.ensure(dimension, currentLambda, cfg.updateMode.separable);

            int remainingEval = cfg.maxEvaluations - totalEvals;
            CMAESProblem runCfg = cfg.snapshot(remainingEval);
            runCfg.lambda = currentLambda;
            runCfg.sigma0 = currentSigma;

            Optimization result = CMAESCore.optimize(initialPoint, objective, bounds, ws, runCfg, rng);
            int runEvals = result.getEvaluations();
            totalEvals += runEvals;
            totalIters += result.getIterations();
            lastStatus = result.getStatus();

            if (useLarge) largeEvals += runEvals;
            else          smallEvals += runEvals;

            if (result.getCost() < bestFitness) {
                bestFitness = result.getCost();
                bestX = result.getSolution().clone();
            }

            if (totalEvals >= cfg.maxEvaluations) break;
        }

        return new Optimization(Double.NaN, bestX, bestFitness, lastStatus, totalIters, totalEvals);
    }
}
