/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.subplex;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;

import org.junit.jupiter.api.Test;

import java.util.function.ToDoubleFunction;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * NLopt standard test functions ported from testfuncs.c.
 * All tests use center-of-bounds starting point (matching NLopt's -c flag)
 * with xtol_abs=1e-6, ftol_abs=1e-6, maxeval=500000.
 *
 * <p>NLopt reference eval counts (SBPLX, center start, same tolerances)
 * are noted in comments for comparison.</p>
 */
class NLoptTestFunctionsTest {

    private static final double PI2 = 2 * Math.PI;
    private static final double PI3 = 3 * Math.PI;
    private static final double PI4 = 4 * Math.PI;

    private static double sqr(double x) { return x * x; }

    /** Solve from center of bounds — matches NLopt testopt -c */
    private static Optimization solve(ToDoubleFunction<double[]> f,
                                      double[] lb, double[] ub, int maxEval) {
        int n = lb.length;
        double[] x0 = new double[n];
        Bound[] bounds = new Bound[n];
        for (int i = 0; i < n; i++) {
            x0[i] = 0.5 * (lb[i] + ub[i]);
            bounds[i] = Bound.between(lb[i], ub[i]);
        }
        return Minimizer.subplex()
                .objective(f).initialPoint(x0).bounds(bounds)
                .functionTolerance(1e-6).parameterTolerance(1e-6)
                .maxEvaluations(maxEval).solve();
    }

    // ── 1D ────────────────────────────────────────────────────────────────

    /** NLopt: 47 evals, cost=-0.765 */
    @Test
    void oscillating1D() {
        Optimization r = solve(x -> {
            double y = x[0] - 1.23456789;
            return sqr(y * 0.1) - Math.cos(y - 2 * Math.sin(3 * y));
        }, new double[]{-5}, new double[]{5}, 2000);
        assertThat(r.getCost()).isLessThan(0.0);
    }

    // ── 2D ────────────────────────────────────────────────────────────────

    /** NLopt: 175 evals, cost=1.1e-7 */
    @Test
    void rosenbrock() {
        Optimization r = solve(
                x -> 100 * sqr(x[1] - x[0] * x[0]) + sqr(1 - x[0]),
                new double[]{-2, -2}, new double[]{2, 2}, 5000);
        assertThat(r.getCost()).isCloseTo(0.0, within(1e-3));
    }

    /** NLopt: 73 evals, cost=1.23 (local) */
    @Test
    void mccormic() {
        Optimization r = solve(
                x -> Math.sin(x[0] + x[1]) + sqr(x[0] - x[1]) - 1.5 * x[0] + 2.5 * x[1] + 1,
                new double[]{-1.5, -3}, new double[]{4, 4}, 5000);
        assertThat(r.getCost()).isLessThan(2.0);
    }

    /** NLopt: 114 evals, cost=3.0 — multimodal, may find local min */
    @Test
    void goldsteinPrice() {
        Optimization r = solve(x -> {
            double x0 = x[0], x1 = x[1];
            double a1 = x0+x1+1, a12 = sqr(a1);
            double a2 = 19-14*x0+3*x0*x0-14*x1+6*x0*x1+3*x1*x1;
            double b1 = 2*x0-3*x1, b12 = sqr(b1);
            double b2 = 18-32*x0+12*x0*x0+48*x1-36*x0*x1+27*x1*x1;
            return (1+a12*a2)*(30+b12*b2);
        }, new double[]{-2, -2}, new double[]{2, 2}, 5000);
        assertThat(r.isSuccessful()).isTrue();
    }

    /** NLopt: 99 evals, cost=-1.032 */
    @Test
    void sixHumpCamel() {
        Optimization r = solve(
                x -> 4*sqr(x[0])-2.1*Math.pow(x[0],4)+Math.pow(x[0],6)/3.0
                        +x[0]*x[1]-4*sqr(x[1])+4*Math.pow(x[1],4),
                new double[]{-5, -5}, new double[]{5, 5}, 5000);
        assertThat(r.getCost()).isCloseTo(-1.031628453, within(1e-2));
    }

    /** NLopt: 81 evals, cost=0.102 (local) */
    @Test
    void branin() {
        Optimization r = solve(x -> {
            double a = 1-2*x[1]+0.05*Math.sin(PI4*x[1])-x[0];
            double b = x[1]-0.5*Math.sin(PI2*x[0]);
            return sqr(a)+sqr(b);
        }, new double[]{-10, -10}, new double[]{10, 10}, 5000);
        assertThat(r.getCost()).isLessThan(1.0);
    }

    /** NLopt: 95 evals, cost=-5.26 (local) */
    @Test
    void shubert() {
        Optimization r = solve(x -> {
            double f = 0;
            for (int j = 1; j <= 5; j++)
                for (int i = 0; i < 2; i++)
                    f -= j * Math.sin((j+1)*x[i]+j);
            return f;
        }, new double[]{-10, -10}, new double[]{10, 10}, 10000);
        assertThat(r.getCost()).isLessThan(0.0);
    }

    /** NLopt: 136 evals, cost=-176.54 */
    @Test
    void hansen() {
        Optimization r = solve(x -> {
            double a = 0, b = 0;
            for (int i = 1; i <= 5; i++) a += i*Math.cos((i-1)*x[0]+i);
            for (int i = 1; i <= 5; i++) b += i*Math.cos((i+1)*x[1]+i);
            return a * b;
        }, new double[]{-10, -10}, new double[]{10, 10}, 10000);
        assertThat(r.isSuccessful()).isTrue(); // multimodal, may find local min
    }

    // ── 3D ────────────────────────────────────────────────────────────────

    /** NLopt: 140 evals, cost=3.0e-4 */
    @Test
    void boxBetts() {
        Optimization r = solve(x -> {
            double f = 0;
            for (int i = 1; i <= 10; i++) {
                double e0 = Math.exp(-0.1*i*x[0]), e1 = Math.exp(-0.1*i*x[1]);
                double e2 = Math.exp(-0.1*i)-Math.exp(-1.0*i);
                f += sqr(e0-e1-e2*x[2]);
            }
            return f;
        }, new double[]{0.9, 9, 0.9}, new double[]{1.2, 11.2, 1.2}, 10000);
        assertThat(r.getCost()).isCloseTo(0.0, within(1e-2));
    }

    // ── 4D ────────────────────────────────────────────────────────────────

    /** NLopt: 177 evals, cost=-2.68 (local) */
    @Test
    void shekelM5() {
        Optimization r = solve(shekelFunc(5),
                new double[]{0,0,0,0}, new double[]{10,10,10,10}, 20000);
        assertThat(r.isSuccessful()).isTrue(); // multimodal
    }

    /** NLopt: 177 evals, cost=-2.75 (local) */
    @Test
    void shekelM7() {
        Optimization r = solve(shekelFunc(7),
                new double[]{0,0,0,0}, new double[]{10,10,10,10}, 20000);
        assertThat(r.isSuccessful()).isTrue(); // multimodal
    }

    /** NLopt: 176 evals, cost=-2.87 (local) */
    @Test
    void shekelM10() {
        Optimization r = solve(shekelFunc(10),
                new double[]{0,0,0,0}, new double[]{10,10,10,10}, 20000);
        assertThat(r.isSuccessful()).isTrue(); // multimodal
    }

    /** NLopt: 275 evals, cost=-20.50 */
    @Test
    void levyN4() {
        Optimization r = solve(levyFunc(),
                new double[]{-10,-10,-10,-10}, new double[]{10,10,10,10}, 20000);
        assertThat(r.isSuccessful()).isTrue(); // multimodal
    }

    /** NLopt: 90 evals, cost=1.0 */
    @Test
    void corner4D() {
        Optimization r = solve(x -> {
            double u = x[0]+x[1]*x[2]*Math.sin(2*x[3]);
            double v = x[0]+2*Math.sin(u);
            return 1+v*v+0.1*(x[1]+x[2]+x[3]);
        }, new double[]{0,0,0,0}, new double[]{1,1,1,1}, 10000);
        assertThat(r.getCost()).isCloseTo(1.0, within(0.1));
    }

    /** NLopt: 1080 evals, cost=-141.28 */
    @Test
    void side4D() {
        Optimization r = solve(x -> {
            double x0=+0.4977*x[0]-0.3153*x[1]-0.5066*x[2]-0.4391*x[3];
            double x1=-0.3153*x[0]+0.3248*x[1]-0.4382*x[2]-0.4096*x[3];
            double x2=-0.5066*x[0]-0.4382*x[1]+0.3807*x[2]-0.4543*x[3];
            double x3=-0.4391*x[0]-0.4096*x[1]-0.4543*x[2]+0.5667*x[3];
            return -1.0/(x0*x0+0.01)-1.0/(x1*x1+0.04)-1.0/(x2*x2+0.09)-1.0/(x3*x3+0.16);
        }, new double[]{0.1,-1,-1,-1}, new double[]{1,1,1,1}, 20000);
        assertThat(r.getCost()).isLessThan(-100.0);
    }

    // ── 5-7D ──────────────────────────────────────────────────────────────

    /** NLopt: 261 evals, cost=-10.50 (local) */
    @Test
    void levyN5() {
        Optimization r = solve(levyFunc(),
                new double[]{-5,-5,-5,-5,-5}, new double[]{5,5,5,5,5}, 30000);
        assertThat(r.isSuccessful()).isTrue(); // multimodal
    }

    /** NLopt: 447 evals, cost=-11.50 */
    @Test
    void levyN7() {
        Optimization r = solve(levyFunc(),
                new double[]{-5,-5,-5,-5,-5,-5,-5}, new double[]{5,5,5,5,5,5,5}, 50000);
        assertThat(r.isSuccessful()).isTrue(); // multimodal
    }

    // ── 10D ───────────────────────────────────────────────────────────────

    /** NLopt: 540 evals, cost=-45.78 */
    @Test
    void paviani() {
        double[] lb = new double[10], ub = new double[10];
        for (int i = 0; i < 10; i++) { lb[i] = 2.001; ub[i] = 9.999; }
        Optimization r = solve(x -> {
            double f = 0, prod = 1;
            for (int i = 0; i < 10; i++) {
                f += sqr(Math.log(x[i]-2))+sqr(Math.log(10-x[i]));
                prod *= x[i];
            }
            return f - Math.pow(prod, 0.2);
        }, lb, ub, 100000);
        assertThat(r.getCost()).isCloseTo(-45.7784697, within(1.0));
    }

    /** NLopt: 605 evals, cost=0.034 */
    @Test
    void griewank() {
        double[] lb = new double[10], ub = new double[10];
        for (int i = 0; i < 10; i++) { lb[i] = -500; ub[i] = 600; }
        Optimization r = solve(x -> {
            double f = 1, p = 1;
            for (int i = 0; i < x.length; i++) {
                f += sqr(x[i])*0.00025;
                p *= Math.cos(x[i]/Math.sqrt(i+1.0));
            }
            return f - p;
        }, lb, ub, 100000);
        assertThat(r.getCost()).isLessThan(1.0);
    }

    /** NLopt: 610 evals, cost=1.0 */
    @Test
    void convexCosh() {
        Optimization r = solve(x -> {
            double f = 1;
            for (int i = 0; i < x.length; i++) f *= Math.cosh((x[i]-i)*(i+1));
            return f;
        }, new double[]{-1,0,0,0,0,0,0,0,0,0}, new double[]{2,3,6,7,8,10,11,13,14,16}, 100000);
        assertThat(r.getCost()).isCloseTo(1.0, within(1e-2));
    }

    // ── 30D ───────────────────────────────────────────────────────────────

    /** NLopt: 19791 evals, cost=23.3 */
    @Test
    void generalizedRosenbrock30D() {
        double[] lb = new double[30], ub = new double[30];
        for (int i = 0; i < 30; i++) { lb[i] = -30; ub[i] = 30; }
        Optimization r = solve(x -> {
            double f = 0;
            for (int i = 0; i < 29; i++) f += 100*sqr(x[i+1]-x[i]*x[i])+sqr(1-x[i]);
            return f;
        }, lb, ub, 500000);
        assertThat(r.getCost()).isLessThan(100.0);
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static ToDoubleFunction<double[]> shekelFunc(int m) {
        final double[][] A = {{4,4,4,4},{1,1,1,1},{8,8,8,8},{6,6,6,6},{3,7,3,7},
                {2,9,2,9},{5,5,3,3},{8,1,8,1},{6,2,6,2},{7,3.6,7,3.6}};
        final double[] c = {.1,.2,.2,.4,.4,.6,.3,.7,.5,.5};
        return x -> {
            double f = 0;
            for (int i = 0; i < m; i++)
                f -= 1.0/(c[i]+sqr(x[0]-A[i][0])+sqr(x[1]-A[i][1])+sqr(x[2]-A[i][2])+sqr(x[3]-A[i][3]));
            return f;
        };
    }

    private static ToDoubleFunction<double[]> levyFunc() {
        return x -> {
            int n = x.length;
            double a = x[n-1]-1, b = 1+sqr(Math.sin(PI2*x[n-1]));
            double f = sqr(Math.sin(PI3*x[0]))+a*b;
            for (int i = 0; i < n-1; i++) {
                double ai = x[i]-1, bi = 1+sqr(Math.sin(PI3*x[i+1]));
                f += sqr(ai)*bi;
            }
            return f;
        };
    }
}
