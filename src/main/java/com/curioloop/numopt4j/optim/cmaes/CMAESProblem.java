/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;

import java.util.Random;
import java.util.function.ToDoubleFunction;

/**
 * Fluent API for the CMA-ES optimizer.
 *
 * <p>Supports standard CMA-ES, sep-CMA-ES (diagonal mode), and IPOP/BIPOP restart strategies.</p>
 *
 * <h2>Basic Usage</h2>
 * <pre>{@code
 * Optimization result = Minimizer.cmaes()
 *     .objective(x -> { double s = 0; for (double v : x) s += v*v; return s; })
 *     .initialPoint(1.0, 1.0, 1.0)
 *     .solve();
 * }</pre>
 *
 * @see Minimizer#cmaes()
 */
public final class CMAESProblem
        extends Minimizer<ToDoubleFunction<double[]>, CMAESWorkspace, CMAESProblem> {

    double sigma0 = 0.3;
    private int lambda = 0;                // 0 = auto: 4 + floor(3*ln(n))
    int maxIterations = 1000;
    int maxEvaluations = 0;               // 0 = auto: lambda * 1000
    boolean diagonalOnly = false;
    boolean isActiveCMA = true;  // Active CMA enabled by default (pycma default)
    int checkFeasibleCount = 0;
    private RestartStrategy restartMode = RestartStrategy.NONE;
    private int maxRestarts = 9;
    private int incPopSize = 2;
    double stopFitness = Double.NEGATIVE_INFINITY;
    double tolX = 1e-11;
    double tolFun = 1e-12;
    double tolUpSigma = 1e3;
    private Random rng = new Random();

    public CMAESProblem() {}

    // ── Getters ───────────────────────────────────────────────────────────

    public double sigma0()               { return sigma0; }
    public int lambdaConfig()            { return lambda; }
    public int maxIterations()           { return maxIterations; }
    public boolean diagonalOnly()        { return diagonalOnly; }
    public boolean isActiveCMA()         { return isActiveCMA; }
    public int checkFeasibleCount()      { return checkFeasibleCount; }
    public RestartStrategy restartMode() { return restartMode; }
    public int maxRestarts()             { return maxRestarts; }
    public int incPopSize()              { return incPopSize; }
    public double stopFitness()          { return stopFitness; }
    public double tolX()                 { return tolX; }
    public double tolFun()               { return tolFun; }
    public double tolUpSigma()           { return tolUpSigma; }

    /** Effective lambda (auto-computed if not set). */
    public int effectiveLambda() {
        return (lambda > 0) ? lambda : (4 + (int) Math.floor(3.0 * Math.log(dimension)));
    }

    /** Effective maxEvaluations (auto-computed if not set). */
    public int effectiveMaxEvaluations() {
        return (maxEvaluations > 0) ? maxEvaluations : effectiveLambda() * 1000;
    }


    // ── Fluent setters ────────────────────────────────────────────────────

    public CMAESProblem objective(ToDoubleFunction<double[]> f) {
        if (f == null) throw new IllegalArgumentException("objective must not be null");
        this.objective = f;
        return this;
    }

    public CMAESProblem sigma(double s) {
        if (s <= 0 || !Double.isFinite(s))
            throw new IllegalArgumentException("sigma must be positive and finite, got " + s);
        this.sigma0 = s;
        return this;
    }

    public CMAESProblem populationSize(int lam) {
        if (lam <= 0) throw new IllegalArgumentException("populationSize must be positive, got " + lam);
        this.lambda = lam;
        return this;
    }

    public CMAESProblem maxIterations(int v) {
        if (v <= 0) throw new IllegalArgumentException("maxIterations must be positive, got " + v);
        this.maxIterations = v;
        return this;
    }

    public CMAESProblem maxEvaluations(int v) {
        if (v <= 0) throw new IllegalArgumentException("maxEvaluations must be positive, got " + v);
        this.maxEvaluations = v;
        return this;
    }

    public CMAESProblem diagonalOnly(boolean v) {
        this.diagonalOnly = v;
        return this;
    }

    public CMAESProblem activeCMA(boolean v) {
        this.isActiveCMA = v;
        return this;
    }

    public CMAESProblem checkFeasibleCount(int v) {
        if (v < 0) throw new IllegalArgumentException("checkFeasibleCount must be >= 0, got " + v);
        this.checkFeasibleCount = v;
        return this;
    }

    public CMAESProblem restartMode(RestartStrategy mode) {
        if (mode == null) throw new IllegalArgumentException("restartMode must not be null");
        this.restartMode = mode;
        return this;
    }

    public CMAESProblem maxRestarts(int v) {
        if (v < 0) throw new IllegalArgumentException("maxRestarts must be >= 0, got " + v);
        this.maxRestarts = v;
        return this;
    }

    public CMAESProblem incPopSize(int v) {
        if (v < 2) throw new IllegalArgumentException("incPopSize must be >= 2, got " + v);
        this.incPopSize = v;
        return this;
    }

    public CMAESProblem stopFitness(double v) {
        this.stopFitness = v;
        return this;
    }

    public CMAESProblem tolX(double v) {
        if (v <= 0) throw new IllegalArgumentException("tolX must be positive, got " + v);
        this.tolX = v;
        return this;
    }

    public CMAESProblem tolFun(double v) {
        if (v <= 0) throw new IllegalArgumentException("tolFun must be positive, got " + v);
        this.tolFun = v;
        return this;
    }

    public CMAESProblem tolUpSigma(double v) {
        if (v <= 0) throw new IllegalArgumentException("tolUpSigma must be positive, got " + v);
        this.tolUpSigma = v;
        return this;
    }

    public CMAESProblem random(Random r) {
        if (r == null) throw new IllegalArgumentException("random must not be null");
        this.rng = r;
        return this;
    }


    // ── Validation ────────────────────────────────────────────────────────

    private void validate() {
        if (objective == null)
            throw new IllegalStateException("objective is required. Call .objective(fn) before .solve().");
        if (initialPoint == null || initialPoint.length == 0)
            throw new IllegalStateException("initialPoint is required. Call .initialPoint(x0) before .solve().");
        for (int i = 0; i < initialPoint.length; i++) {
            double v = initialPoint[i];
            if (Double.isNaN(v) || Double.isInfinite(v))
                throw new IllegalArgumentException(
                    "initialPoint[" + i + "] is " + v + ". All initial values must be finite.");
        }
        if (sigma0 <= 0)
            throw new IllegalArgumentException("sigma must be positive, got " + sigma0);
    }

    // ── Problem interface ─────────────────────────────────────────────────

    @Override
    public CMAESWorkspace alloc() {
        validate();
        int lam = effectiveLambda();
        if (workspace == null || workspace.n != dimension || workspace.lambda != lam) {
            workspace = new CMAESWorkspace(dimension, lam, diagonalOnly);
        }
        return workspace;
    }

    @Override
    public Optimization solve(CMAESWorkspace workspace) {
        validate();
        int lam = effectiveLambda();
        int resolvedMaxEval = effectiveMaxEvaluations();

        // Resolve workspace
        CMAESWorkspace ws = workspace;
        if (ws == null) {
            ws = this.workspace;
            if (ws == null || ws.n != dimension || ws.lambda != lam) {
                ws = new CMAESWorkspace(dimension, lam, diagonalOnly);
                this.workspace = ws;
            }
        } else {
            if (ws.n != dimension || ws.lambda != lam) {
                throw new IllegalArgumentException(
                    "workspace dimension mismatch: workspace(n=" + ws.n + ", lambda=" + ws.lambda
                    + ") vs problem(n=" + dimension + ", lambda=" + lam + ")");
            }
        }

        // Build config snapshot with resolved maxEvaluations
        CMAESProblem cfg = snapshot(resolvedMaxEval);

        switch (restartMode) {
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
        c.diagonalOnly = this.diagonalOnly;
        c.isActiveCMA = this.isActiveCMA;
        c.checkFeasibleCount = this.checkFeasibleCount;
        c.restartMode = this.restartMode;
        c.maxRestarts = this.maxRestarts;
        c.incPopSize = this.incPopSize;
        c.stopFitness = this.stopFitness;
        c.tolX = this.tolX;
        c.tolFun = this.tolFun;
        c.tolUpSigma = this.tolUpSigma;
        c.rng = this.rng;
        return c;
    }

    /** Single run (NONE mode). */
    private Optimization solveOnce(double[] x0, CMAESWorkspace ws, CMAESProblem cfg) {
        ws.reset();
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

        for (int restart = 0; restart <= maxRestarts; restart++) {
            int remainingEval = cfg.maxEvaluations - totalEvals;
            if (remainingEval <= 0) break;

            CMAESWorkspace runWs = (ws.lambda == currentLambda && ws.n == dimension)
                ? ws : new CMAESWorkspace(dimension, currentLambda, cfg.diagonalOnly);

            CMAESProblem runCfg = cfg.snapshot(remainingEval);
            runCfg.lambda = currentLambda;

            Optimization result = CMAESCore.optimize(initialPoint, objective, bounds, runWs, runCfg, rng);
            totalEvals += result.getEvaluations();
            totalIters += result.getIterations();
            lastStatus = result.getStatus();

            if (result.getCost() < bestFitness) {
                bestFitness = result.getCost();
                bestX = result.getSolution().clone();
            }

            if (totalEvals >= cfg.maxEvaluations) break;

            // Increase lambda for next restart
            currentLambda *= incPopSize;
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

        for (int restart = 0; restart <= maxRestarts; restart++) {
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

            CMAESWorkspace runWs = (ws.lambda == currentLambda && ws.n == dimension)
                ? ws : new CMAESWorkspace(dimension, currentLambda, cfg.diagonalOnly);

            int remainingEval = cfg.maxEvaluations - totalEvals;
            CMAESProblem runCfg = cfg.snapshot(remainingEval);
            runCfg.lambda = currentLambda;
            runCfg.sigma0 = currentSigma;

            Optimization result = CMAESCore.optimize(initialPoint, objective, bounds, runWs, runCfg, rng);
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
