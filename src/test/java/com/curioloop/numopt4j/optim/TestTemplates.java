/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

/**
 * Collection of commonly-used test function templates and helpers.
 *
 * <p>This class centralizes repeated Univariate function factories and small
 * test helpers found across the test-suite. Tests are not modified to use
 * these templates by this change; the file simply provides a single place
 * to look up standard test functions and their explicit formulas.</p>
 */
public final class TestTemplates {

    private TestTemplates() { }

    /**
     * Helper: create an initial point of length n with every component = value.
     */
    public static double[] createInitialPoint(int n, double value) {
        double[] point = new double[n];
        for (int i = 0; i < n; i++) point[i] = value;
        return point;
    }

    /**
     * Helper: AssertJ Offset for tolerances.
     */
    public static org.assertj.core.data.Offset<Double> within(double tolerance) {
        return org.assertj.core.data.Offset.offset(tolerance);
    }

    // ------------------------------------------------------------------
    // Standard objective / constraint templates
    // ------------------------------------------------------------------

    /**
     * Quadratic (sphere) function:
     *   f(x) = \sum_i x_i^2
     * Gradient: g_i = 2 * x_i
     */
    public static Univariate quadratic() {
        return (x, g) -> {
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                f += x[i] * x[i];
            }
            if (g != null) {
                for (int i = 0; i < x.length; i++) g[i] = 2.0 * x[i];
            }
            return f;
        };
    }

    /**
     * Shifted quadratic:
     *   f(x) = \sum_i (x_i - shift)^2
     * Gradient: g_i = 2 * (x_i - shift)
     */
    public static Univariate shiftedQuadratic(double shift) {
        return (x, g) -> {
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                double d = x[i] - shift;
                f += d * d;
            }
            if (g != null) {
                for (int i = 0; i < x.length; i++) g[i] = 2.0 * (x[i] - shift);
            }
            return f;
        };
    }

    /**
     * General quadratic with target vector t:
     *   f(x) = \sum_i (x_i - t_i)^2
     * Gradient: g_i = 2 * (x_i - t_i)
     */
    public static Univariate quadraticWithTarget(double[] target) {
        final double[] t = target == null ? new double[0] : target.clone();
        return (x, g) -> {
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                double ti = i < t.length ? t[i] : 0.0;
                double d = x[i] - ti;
                f += d * d;
                if (g != null) g[i] = 2.0 * d;
            }
            return f;
        };
    }

    /**
     * Scaled quadratic:
     *   f(x) = \sum_i scale_i * x_i^2
     * Gradient: g_i = 2 * scale_i * x_i
     */
    public static Univariate scaledQuadratic(final double[] scale) {
        final double[] s = scale == null ? new double[0] : scale.clone();
        return (x, g) -> {
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                double si = i < s.length ? s[i] : 1.0;
                f += si * x[i] * x[i];
                if (g != null) g[i] = 2.0 * si * x[i];
            }
            return f;
        };
    }

    /**
     * Slow quadratic:
     *   f(x) = \sum_i x_i^2
     * Gradient: g_i = 2 * x_i
     */
    public static Univariate slowQuadratic(long sleep) {
        return (x, g) -> {
            try {
                Thread.sleep(sleep);  // Sleep 2ms per evaluation
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                f += x[i] * x[i];
            }
            if (g != null) {
                for (int i = 0; i < x.length; i++) g[i] = 2.0 * x[i];
            }
            return f;
        };
    }

    /**
     * Rosenbrock function (classic):
     *   f(x) = \sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
     * Gradient computed analytically (standard expression).
     */
    public static Univariate rosenbrock() {
        return (x, g) -> {
            double f = 0.0;
            for (int i = 0; i < x.length - 1; i++) {
                double t1 = x[i + 1] - x[i] * x[i];
                double t2 = 1.0 - x[i];
                f += 100.0 * t1 * t1 + t2 * t2;
            }
            if (g != null) {
                // first component
                g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
                for (int i = 1; i < x.length - 1; i++) {
                    g[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                         - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
                         - 2.0 * (1.0 - x[i]);
                }
                if (x.length >= 2) {
                    g[x.length - 1] = 200.0 * (x[x.length - 1] - x[x.length - 2] * x[x.length - 2]);
                }
            }
            return f;
        };
    }

    /**
     * Sum constraint (equality/inequality helper):
     *   c(x) = \sum_i x_i - target
     * Gradient: g_i = 1
     */
    public static Univariate sumConstraint(double target) {
        return (x, g) -> {
            double sum = 0.0;
            for (int i = 0; i < x.length; i++) sum += x[i];
            if (g != null) {
                for (int i = 0; i < x.length; i++) g[i] = 1.0;
            }
            return sum - target;
        };
    }

    /**
     * General linear constraint with coefficients c and right-hand side rhs:
     *   c(x) = \sum_i coeffs_i * x_i - rhs
     * Gradient: g_i = coeffs_i
     */
    public static Univariate linearConstraint(final double[] coeffs, final double rhs) {
        final double[] a = coeffs == null ? new double[0] : coeffs.clone();
        return (x, g) -> {
            double sum = 0.0;
            for (int i = 0; i < x.length; i++) {
                double ai = i < a.length ? a[i] : 0.0;
                sum += ai * x[i];
                if (g != null) g[i] = ai;
            }
            return sum - rhs;
        };
    }

    /**
     * Inequality constraint concentrating on a single index (used in many tests):
     *   c(x) = x[idx] + bound  (interpreted as c(x) >= 0)
     * Gradient: g_j = 1 if j == idx else 0
     */
    public static Univariate inequalityAtIndex(final int idx, final double bound) {
        return (x, g) -> {
            if (g != null) {
                for (int j = 0; j < x.length; j++) g[j] = (j == idx) ? 1.0 : 0.0;
            }
            return x[idx] + bound;
        };
    }

    // NOTE: `inequalityIndexMinusK` was removed. Use `inequalityAtIndex(idx, bound)` where
    // `inequalityAtIndex(idx, bound) == x[idx] + bound`, which corresponds to the
    // previous `inequalityIndexMinusK(idx, k)` when called as `inequalityAtIndex(idx, -k)`.

    /**
     * Unit-ball / circle constraint (general n-dim):
     *   c(x) = radius^2 - \sum_i x_i^2  (>= 0 means inside ball)
     * Gradient: g_i = -2 * x_i
     */
    public static Univariate unitBallConstraint(final double radius) {
        final double r2 = radius * radius;
        return (x, g) -> {
            double sum = 0.0;
            for (int i = 0; i < x.length; i++) sum += x[i] * x[i];
            if (g != null) {
                for (int i = 0; i < x.length; i++) g[i] = -2.0 * x[i];
            }
            return r2 - sum;
        };
    }

    // ------------------------------------------------------------------
    // A few named 2D benchmark functions (present in tests)
    // ------------------------------------------------------------------

    /**
     * Beale function (2D):
     *   f(x,y) = (1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2
     */
    public static Univariate beale2D() {
        return (x, g) -> {
            double x1 = x[0], x2 = x[1];
            double t1 = 1.5 - x1 + x1 * x2;
            double t2 = 2.25 - x1 + x1 * x2 * x2;
            double t3 = 2.625 - x1 + x1 * x2 * x2 * x2;
            double f = t1 * t1 + t2 * t2 + t3 * t3;
            if (g != null) {
                g[0] = 2 * t1 * (-1 + x2) + 2 * t2 * (-1 + x2 * x2) + 2 * t3 * (-1 + x2 * x2 * x2);
                g[1] = 2 * t1 * x1 + 2 * t2 * (2 * x1 * x2) + 2 * t3 * (3 * x1 * x2 * x2);
            }
            return f;
        };
    }

    /**
     * Booth function (2D):
     *   f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
     */
    public static Univariate booth2D() {
        return (x, g) -> {
            double x1 = x[0], x2 = x[1];
            double t1 = x1 + 2 * x2 - 7;
            double t2 = 2 * x1 + x2 - 5;
            double f = t1 * t1 + t2 * t2;
            if (g != null) {
                g[0] = 2 * t1 + 4 * t2;
                g[1] = 4 * t1 + 2 * t2;
            }
            return f;
        };
    }

    /**
     * Matyas function (2D):
     *   f(x,y) = 0.26*(x^2 + y^2) - 0.48*x*y
     */
    public static Univariate matyas2D() {
        return (x, g) -> {
            double x1 = x[0], x2 = x[1];
            double f = 0.26 * (x1 * x1 + x2 * x2) - 0.48 * x1 * x2;
            if (g != null) {
                g[0] = 0.52 * x1 - 0.48 * x2;
                g[1] = 0.52 * x2 - 0.48 * x1;
            }
            return f;
        };
    }
}
