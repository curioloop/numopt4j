package com.curioloop.numopt4j.quad.de;

import com.curioloop.numopt4j.quad.Quadrature;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Boost-style double-exponential quadrature rules.
 *
 * <p>Each rule rewrites the original integral as an integral over the whole real line
 * and then applies a nested trapezoidal rule on dyadic meshes {@code h_n = 2^(-n)}.
 * With the transformed kernel {@code G(t)}, the refinement performed throughout this
 * file is
 * <pre>
 * I_n     = h_n * sum_k G(k h_n),
 * I_n     = 0.5 * I_(n-1) + h_n * sum_k G((2k+1) h_n),
 * error_n = |I_n - I_(n-1)|.
 * </pre>
 * The internal {@code l1} estimate is the companion quadrature estimate of
 * {@code integral |G(t)| dt}, so convergence checks use
 * {@code error_n <= tolerance * l1_n}.</p>
 *
 * <p>The three DE maps implemented here are</p>
 * <pre>
 * tanh-sinh: z(t) = tanh((pi/2) sinh(t))
 *            z'(t)= (pi/2) cosh(t) / cosh((pi/2) sinh(t))^2
 *
 * exp-sinh:  x(t) = exp((pi/2) sinh(t))
 *            x'(t)= (pi/2) cosh(t) x(t)
 *
 * sinh-sinh: x(t) = sinh((pi/2) sinh(t))
 *            x'(t)= (pi/2) cosh(t) cosh((pi/2) sinh(t))
 * </pre>
 *
 * <p>The public entry points mirror Boost's three rules: {@code tanh_sinh} for finite
 * intervals and mapped infinite ranges, {@code exp_sinh} for half-infinite domains,
 * and {@code sinh_sinh} for the whole real line.</p>
 */
public final class DoubleExponentialCore {

    private static final double BOOST_EPSILON = Math.ulp(1.0);
    private static final double ROOT_EPSILON = Math.sqrt(BOOST_EPSILON);
    private static final double BOOST_MIN_VALUE = Double.MIN_NORMAL;
    private static final double HALF_PI = 0.5 * Math.PI;
    private static final double TWO_DIV_PI = 2.0 / Math.PI;
    private static final double WHOLE_LINE_LIMIT = 4.0 * Math.sqrt(BOOST_MIN_VALUE);
    private static final double TANH_SINH_T_MAX = DoubleExponentialsTables.TANH_SINH_T_MAX;
    private static final int TANH_SINH_INITIAL_ROW_LENGTH = (int) TANH_SINH_T_MAX;
    private static final double TANH_SINH_INITIAL_STEP = TANH_SINH_T_MAX / TANH_SINH_INITIAL_ROW_LENGTH;
    private static final double[] EMPTY_DOUBLE_ARRAY = {};
    private static final int[] EMPTY_INT_ARRAY = {};

    public static final double DEFAULT_TOLERANCE = ROOT_EPSILON;
    public static final int DEFAULT_TANH_SINH_REFINEMENTS = 15;
    public static final int DEFAULT_HALF_LINE_REFINEMENTS = 9;
    public static final double DEFAULT_MIN_COMPLEMENT = 4.0 * BOOST_MIN_VALUE;

    private static Quadrature quadrature(double estimate, double error, double l1, int levels, double tolerance) {
        Quadrature.Status status = error <= Math.abs(tolerance * l1)
            ? Quadrature.Status.CONVERGED
            : Quadrature.Status.MAX_REFINEMENTS_REACHED;
        return new Quadrature(estimate, error, status, levels, 0);
    }

    private static Quadrature scale(Quadrature quadrature, double factor) {
        double magnitude = Math.abs(factor);
        return new Quadrature(
            quadrature.getValue() * factor,
            quadrature.getEstimatedError() * magnitude,
            quadrature.getStatus(),
            quadrature.getIterations(),
            quadrature.getEvaluations()
        );
    }

    private static final class DoubleArrayBuilder {
        private double[] values;
        private int size;

        private DoubleArrayBuilder(int initialCapacity) {
            values = new double[Math.max(4, initialCapacity)];
        }

        private void add(double value) {
            ensureCapacity(size + 1);
            values[size++] = value;
        }

        private double[] toArray() {
            return Arrays.copyOf(values, size);
        }

        private int size() {
            return size;
        }

        private void ensureCapacity(int capacity) {
            if (capacity <= values.length) {
                return;
            }
            int nextCapacity = values.length;
            while (nextCapacity < capacity) {
                nextCapacity *= 2;
            }
            values = Arrays.copyOf(values, nextCapacity);
        }
    }

    private DoubleExponentialCore() {}

    /**
     * Convenience overload of
     * {@link #tanhSinh(DoubleUnaryOperator, double, double, double, int, double)}
     * using the library defaults.
     */
    public static Quadrature tanhSinh(DoubleUnaryOperator function, double lower, double upper) {
        return tanhSinh(
            function,
            lower,
            upper,
            DEFAULT_TOLERANCE,
            DEFAULT_TANH_SINH_REFINEMENTS,
            DEFAULT_MIN_COMPLEMENT
        );
    }

    /**
     * Convenience overload of
     * {@link #tanhSinh(DoubleUnaryOperator, double, double, double, int, double)}
     * using the default minimum endpoint complement.
     */
    public static Quadrature tanhSinh(
        DoubleUnaryOperator function,
        double lower,
        double upper,
        double tolerance,
        int maxRefinements
    ) {
        return tanhSinh(function, lower, upper, tolerance, maxRefinements, DEFAULT_MIN_COMPLEMENT);
    }

    /**
     * Integrates a finite interval or a rationally mapped infinite interval with tanh-sinh.
     *
     * <p>For a finite interval {@code [a,b]}, the code uses</p>
     * <pre>
     * c = (a+b)/2, h = (b-a)/2,
     * x(t)  = c + h tanh((pi/2) sinh(t)),
     * dx/dt = h (pi/2) cosh(t) / cosh((pi/2) sinh(t))^2,
     * I     = integral_{-inf}^{+inf} f(x(t)) dx/dt dt.
     * </pre>
     *
     * <p>Infinite domains are first mapped to {@code z in (-1,1)} and then fed to the same
     * tanh-sinh kernel:</p>
     * <pre>
     * [a,+inf):    x = a - 1 + 2 / (1 + z)
     * (-inf,b]:    x = b + 1 - 2 / (1 + z)
     * (-inf,+inf): x = z / (1 - z^2).
     * </pre>
     * The implementation evaluates these maps in algebraically equivalent forms near
     * {@code z = +/-1} so that the mapped abscissa and endpoint distance remain accurate in
     * double precision.
     *
     * <p>{@code minComplement} is the minimum retained value of {@code 1 - |z|} in the
     * tanh-sinh tables. Nodes whose endpoint distance is smaller than that threshold are
     * pruned because the outer map can no longer resolve them stably.</p>
     */
    public static Quadrature tanhSinh(
        DoubleUnaryOperator function,
        double lower,
        double upper,
        double tolerance,
        int maxRefinements,
        double minComplement
    ) {
        return tanhSinh(function, lower, upper, tolerance, maxRefinements, minComplement, null);
    }

    /**
     * Workspace-aware tanh-sinh overload.
     *
     * <p>Pass a reusable {@link DEPool} to retain the refined DE rows between calls with the
     * same rule family.</p>
     */
    public static Quadrature tanhSinh(
        DoubleUnaryOperator function,
        double lower,
        double upper,
        double tolerance,
        int maxRefinements,
        double minComplement,
        DEPool workspace
    ) {
        Objects.requireNonNull(function, "function cannot be null");
        validateTolerance(tolerance);
        validateRefinements(maxRefinements);
        DEPool pool = workspace != null ? workspace : new DEPool();
        double tableMinComplement = sanitizeMinComplement(minComplement);
        DETable table = initTanhSinhTable(maxRefinements, tableMinComplement, pool);

        if (Double.isNaN(lower) || Double.isNaN(upper)) {
            throw new IllegalArgumentException("The domain of integration is not sensible; please check the bounds.");
        }

        if (lower <= -Double.MAX_VALUE && upper >= Double.MAX_VALUE) {
            DoubleBinaryOperator transformed = (t, tc) -> {
                double tSquared = t * t;
                double inverse;
                if (t > 0.5) {
                    inverse = 1.0 / ((2.0 - tc) * tc);
                } else if (t < -0.5) {
                    inverse = 1.0 / ((2.0 + tc) * -tc);
                } else {
                    inverse = 1.0 / (1.0 - tSquared);
                }
                return function.applyAsDouble(t * inverse) * (1.0 + tSquared) * inverse * inverse;
            };
            return integrateTanhSinh(transformed, table, pool, WHOLE_LINE_LIMIT, WHOLE_LINE_LIMIT, tolerance);
        }

        if (Double.isFinite(lower) && upper >= Double.MAX_VALUE) {
            DoubleBinaryOperator transformed = (t, tc) -> {
                double z = t > -0.5 ? 1.0 / (t + 1.0) : -1.0 / tc;
                double argument = t < 0.5 ? 2.0 * z + lower - 1.0 : lower + tc / (2.0 - tc);
                return function.applyAsDouble(argument) * z * z;
            };
            return scale(integrateTanhSinh(transformed, table, pool, WHOLE_LINE_LIMIT, BOOST_MIN_VALUE, tolerance), 2.0);
        }

        if (Double.isFinite(upper) && lower <= -Double.MAX_VALUE) {
            DoubleBinaryOperator transformed = (t, tc) -> {
                double z = t > -0.5 ? 1.0 / (t + 1.0) : -1.0 / tc;
                double argument = t < 0.5 ? 2.0 * z - 1.0 : tc / (2.0 - tc);
                return function.applyAsDouble(upper - argument) * z * z;
            };
            return scale(integrateTanhSinh(transformed, table, pool, WHOLE_LINE_LIMIT, BOOST_MIN_VALUE, tolerance), 2.0);
        }

        if (Double.isFinite(lower) && Double.isFinite(upper)) {
            if (lower == upper) {
                return new Quadrature(0.0, 0.0, Quadrature.Status.CONVERGED, 0, 0);
            }
            if (upper < lower) {
                return scale(tanhSinh(function, upper, lower, tolerance, maxRefinements, minComplement, pool), -1.0);
            }
            double average = 0.5 * (lower + upper);
            double halfWidth = 0.5 * (upper - lower);
            double averageOverDiffMinusOne = lower / halfWidth;
            double averageOverDiffPlusOne = upper / halfWidth;
            boolean haveSmallLeft = Math.abs(lower) < 0.5;
            boolean haveSmallRight = Math.abs(upper) < 0.5;

            double leftMinComplement = Math.nextUp(averageOverDiffMinusOne) - averageOverDiffMinusOne;
            double minComplementLimit = Math.max(BOOST_MIN_VALUE, Math.nextUp(BOOST_MIN_VALUE / halfWidth));
            if (leftMinComplement < minComplementLimit) {
                leftMinComplement = minComplementLimit;
            }
            double rightMinComplement = averageOverDiffPlusOne - Math.nextDown(averageOverDiffPlusOne);
            if (rightMinComplement < minComplementLimit) {
                rightMinComplement = minComplementLimit;
            }

            DoubleBinaryOperator transformed = (z, zc) -> {
                double position;
                if (z < -0.5) {
                    if (haveSmallLeft) {
                        return function.applyAsDouble(halfWidth * (averageOverDiffMinusOne - zc));
                    }
                    position = lower - halfWidth * zc;
                } else if (z > 0.5) {
                    if (haveSmallRight) {
                        return function.applyAsDouble(halfWidth * (averageOverDiffPlusOne - zc));
                    }
                    position = upper - halfWidth * zc;
                } else {
                    position = average + halfWidth * z;
                }
                return function.applyAsDouble(position);
            };

            return scale(
                integrateTanhSinh(
                    transformed,
                    table,
                    pool,
                    leftMinComplement,
                    rightMinComplement,
                    tolerance
                ),
                halfWidth
            );
        }

        throw new IllegalArgumentException("The domain of integration is not sensible; please check the bounds.");
    }

    /**
     * Convenience overload of
     * {@link #tanhSinh(DoubleBinaryOperator, double, double, double, int, double)}
     * using the library defaults.
     */
    public static Quadrature tanhSinh(DoubleBinaryOperator function, double lower, double upper) {
        return tanhSinh(
            function,
            lower,
            upper,
            DEFAULT_TOLERANCE,
            DEFAULT_TANH_SINH_REFINEMENTS,
            DEFAULT_MIN_COMPLEMENT
        );
    }

    /**
     * Tanh-sinh variant whose callback receives both the physical abscissa and the nearer
     * endpoint distance.
     *
     * <p>On a finite interval {@code [a,b]}, the callback is evaluated as {@code g(x, c)} with</p>
     * <pre>
     * c = min(x - a, b - x).
     * </pre>
     * This lets singular models such as {@code c^alpha}, {@code log(c)}, or mixed
     * {@code x/c}-dependent factors be written without subtractive cancellation near an
     * endpoint. The quadrature rule and dyadic refinement are otherwise identical to the
     * unary tanh-sinh overload.</p>
     */
    public static Quadrature tanhSinh(
        DoubleBinaryOperator function,
        double lower,
        double upper,
        double tolerance,
        int maxRefinements,
        double minComplement
    ) {
        return tanhSinh(function, lower, upper, tolerance, maxRefinements, minComplement, null);
    }

    /**
     * Workspace-aware complement-callback tanh-sinh overload.
     *
     * <p>Pass a reusable {@link DEPool} to reuse previously generated refined rows.</p>
     */
    public static Quadrature tanhSinh(
        DoubleBinaryOperator function,
        double lower,
        double upper,
        double tolerance,
        int maxRefinements,
        double minComplement,
        DEPool workspace
    ) {
        Objects.requireNonNull(function, "function cannot be null");
        validateTolerance(tolerance);
        validateRefinements(maxRefinements);
        if (!Double.isFinite(lower) || !Double.isFinite(upper) || upper <= lower) {
            throw new IllegalArgumentException(
                "Arguments to tanhSinh are in the wrong order; integration over [a,b] must have b > a with finite bounds."
            );
        }

        DEPool pool = workspace != null ? workspace : new DEPool();
        double tableMinComplement = sanitizeMinComplement(minComplement);
        DETable table = initTanhSinhTable(maxRefinements, tableMinComplement, pool);
        double halfWidth = 0.5 * (upper - lower);
        DoubleBinaryOperator transformed = (z, zc) -> {
            if (z < 0.0) {
                return function.applyAsDouble(
                    0.5 * (lower - upper) * zc + lower,
                    0.5 * (upper - lower) * zc
                );
            }
            return function.applyAsDouble(
                0.5 * (lower - upper) * zc + upper,
                0.5 * (upper - lower) * zc
            );
        };
        return scale(integrateTanhSinh(transformed, table, pool, tableMinComplement, tableMinComplement, tolerance), halfWidth);
    }

    /**
     * Convenience overload of {@link #expSinh(DoubleUnaryOperator, double, double, double, int)}
     * using the library defaults.
     */
    public static Quadrature expSinh(DoubleUnaryOperator function, double lower, double upper) {
        return expSinh(function, lower, upper, DEFAULT_TOLERANCE, DEFAULT_HALF_LINE_REFINEMENTS);
    }

    /**
     * Integrates a half-line integral with the exp-sinh transformation.
     *
     * <pre>
     * x(t)  = exp((pi/2) sinh(t)),
     * dx/dt = (pi/2) cosh(t) x(t).
     *
     * [a,+inf): I = integral_{-inf}^{+inf} f(a + x(t)) dx/dt dt
     * (-inf,b]: I = integral_{-inf}^{+inf} f(b - x(t)) dx/dt dt.
     * </pre>
     *
     * <p>The transformed kernel decays double-exponentially when the original integrand
     * decays sufficiently fast on the half-line.</p>
     */
    public static Quadrature expSinh(
        DoubleUnaryOperator function,
        double lower,
        double upper,
        double tolerance,
        int maxRefinements
    ) {
        return expSinh(function, lower, upper, tolerance, maxRefinements, null);
    }

    /**
     * Workspace-aware exp-sinh overload.
     *
     * <p>Pass a reusable {@link DEPool} to retain the generated half-line refinement rows.</p>
     */
    public static Quadrature expSinh(
        DoubleUnaryOperator function,
        double lower,
        double upper,
        double tolerance,
        int maxRefinements,
        DEPool workspace
    ) {
        Objects.requireNonNull(function, "function cannot be null");
        validateTolerance(tolerance);
        validateRefinements(maxRefinements);
        if (Double.isNaN(lower) || Double.isNaN(upper)) {
            throw new IllegalArgumentException("NaN supplied as one limit of integration.");
        }
        DEPool pool = workspace != null ? workspace : new DEPool();
        DETable table = initExpSinhTable(maxRefinements, pool);
        if (Double.isFinite(lower) && upper >= Double.MAX_VALUE) {
            if (lower == 0.0) {
                return integrateExpSinh(function, table, pool, tolerance);
            }
            return integrateExpSinh(t -> function.applyAsDouble(t + lower), table, pool, tolerance);
        }
        if (Double.isFinite(upper) && lower <= -Double.MAX_VALUE) {
            return integrateExpSinh(t -> function.applyAsDouble(upper - t), table, pool, tolerance);
        }
        if (lower <= -Double.MAX_VALUE && upper >= Double.MAX_VALUE) {
            throw new IllegalArgumentException(
                "Use sinhSinh for integration over the whole real line; expSinh is for half-infinite integrals."
            );
        }
        throw new IllegalArgumentException(
            "Use tanhSinh for integration over finite domains; expSinh is for half-infinite integrals."
        );
    }

    /**
     * Convenience overload of {@link #sinhSinh(DoubleUnaryOperator, double, int)} using the
     * library defaults.
     */
    public static Quadrature sinhSinh(DoubleUnaryOperator function) {
        return sinhSinh(function, DEFAULT_TOLERANCE, DEFAULT_HALF_LINE_REFINEMENTS);
    }

    /**
     * Integrates a whole-line integral with the sinh-sinh transformation.
     *
     * <pre>
     * x(t)  = sinh((pi/2) sinh(t)),
     * dx/dt = (pi/2) cosh(t) cosh((pi/2) sinh(t)),
     * I     = integral_{-inf}^{+inf} f(x(t)) dx/dt dt.
     * </pre>
     *
     * <p>This rule is intended for rapidly decaying whole-line integrands, so the
     * implementation checks heuristically that {@code f(x)} tends to zero at both infinities.</p>
     */
    public static Quadrature sinhSinh(DoubleUnaryOperator function, double tolerance, int maxRefinements) {
        return sinhSinh(function, tolerance, maxRefinements, null);
    }

    /**
     * Workspace-aware sinh-sinh overload.
     *
     * <p>Pass a reusable {@link DEPool} to retain the generated whole-line refinement rows.</p>
     */
    public static Quadrature sinhSinh(DoubleUnaryOperator function, double tolerance, int maxRefinements, DEPool workspace) {
        Objects.requireNonNull(function, "function cannot be null");
        validateTolerance(tolerance);
        validateRefinements(maxRefinements);
        DEPool pool = workspace != null ? workspace : new DEPool();
        return integrateSinhSinh(function, initSinhSinhTable(maxRefinements, pool), pool, tolerance);
    }

    /**
     * Evaluates the exp-sinh transformed kernel with nested dyadic trapezoidal refinement.
     *
     * <p>With {@code G(t) = f(x(t)) x'(t)} and {@code h_n = 2^(-n)}, each new row contributes
     * only the odd nodes that were absent from the previous row:</p>
     * <pre>
     * I_n = 0.5 * I_(n-1) + h_n * sum_k G((2k+1) h_n).
     * </pre>
     * The first two rows also define a numerically active abscissa window so later refinements
     * can skip tails that are already zero at machine precision.
     */
    private static Quadrature integrateExpSinh(DoubleUnaryOperator function, DETable table, DEPool pool, double tolerance) {
        DETable base = pool.opts.baseTable();
        int rowCount = activeRowCount(base, table, pool);
        double minAbscissa = 0.0;
        double maxAbscissa = Double.MAX_VALUE;
        int row0Start = table.rowStart(0);
        int row0End = row0Start + table.rowLength(0);
        double estimate0 = 0.0;
        double l10 = 0.0;
        for (int offset = row0Start; offset < row0End; offset++) {
            double x = DETable.abscissa(base, table, offset);
            double y = requireFinite(function.applyAsDouble(x), "expSinh");
            double last = estimate0;
            double weight = DETable.weight(base, table, offset);
            estimate0 += y * weight;
            l10 += Math.abs(y) * weight;
            if (last == estimate0 && Math.abs(estimate0) != 0.0) {
                maxAbscissa = x;
                break;
            }
        }

        double estimate1 = estimate0;
        double l11 = l10;
        boolean haveFirstIndex = false;
        int firstIndex = 0;
        int row1Start = table.rowStart(1);
        int row1End = row1Start + table.rowLength(1);
        for (int offset = row1Start; offset < row1End && DETable.abscissa(base, table, offset) < maxAbscissa; offset++) {
            double x = DETable.abscissa(base, table, offset);
            double weight = DETable.weight(base, table, offset);
            double y = requireFinite(function.applyAsDouble(x), "expSinh");
            double last = estimate1;
            estimate1 += y * weight;
            l11 += Math.abs(y) * weight;
            if (!haveFirstIndex && last == estimate1) {
                if (offset < row1End - 1 && DETable.abscissa(base, table, offset + 1) > maxAbscissa) {
                    haveFirstIndex = true;
                } else {
                    minAbscissa = x;
                    firstIndex = offset - row1Start;
                }
            } else {
                haveFirstIndex = true;
            }
        }

        if (estimate0 == 0.0) {
            minAbscissa = 0.0;
            maxAbscissa = Double.MAX_VALUE;
        }

        estimate1 *= 0.5;
        l11 *= 0.5;
        double error = Math.abs(estimate0 - estimate1);
        int level = 2;
        for (; level < rowCount; level++) {
            estimate0 = estimate1;
            l10 = l11;
            estimate1 = 0.5 * estimate0;
            l11 = 0.5 * l10;
            double step = Math.scalb(1.0, -level);
            double sum = 0.0;
            double absum = 0.0;
            int rowStart = table.rowStart(level);
            int rowEnd = rowStart + table.rowLength(level);

            firstIndex = firstIndex == 0 ? 0 : 2 * firstIndex - 1;
            int offset = rowStart + firstIndex;
            while (offset < rowEnd && DETable.abscissa(base, table, offset) < minAbscissa) {
                offset++;
            }
            for (; offset < rowEnd && DETable.abscissa(base, table, offset) < maxAbscissa; offset++) {
                double x = DETable.abscissa(base, table, offset);
                double weight = DETable.weight(base, table, offset);
                double y = requireFinite(function.applyAsDouble(x), "expSinh");
                sum += y * weight;
                absum += Math.abs(y) * weight;
            }

            estimate1 += sum * step;
            l11 += absum * step;
            error = Math.abs(estimate0 - estimate1);
            if (!Double.isFinite(l11)) {
                throw new ArithmeticException(
                    "The expSinh quadrature evaluated the function at a singular point; please ensure the integrand is finite over its domain."
                );
            }
            if (error <= tolerance * l11) {
                break;
            }
        }
        return quadrature(estimate1, error, l11, level, tolerance);
    }

    /**
     * Evaluates the odd whole-line sinh-sinh kernel symmetrically about the origin.
     *
     * <p>With {@code G(t) = f(x(t)) x'(t)} and the odd map {@code x(-t) = -x(t)}, the refined
     * trapezoidal sequence is</p>
     * <pre>
     * I_n = h_n [ G(0) + sum_{k>=1} ( G(k h_n) + G(-k h_n) ) ],
     * error_n = |I_n - I_(n-1)|.
     * </pre>
     * The tail scan terminates once two consecutive transformed terms are below the machine
     * scale implied by the current {@code l1} estimate.
     */
    private static Quadrature integrateSinhSinh(DoubleUnaryOperator function, DETable table, DEPool pool, double tolerance) {
        DETable base = pool.opts.baseTable();
        int rowCount = activeRowCount(base, table, pool);
        double yMax = function.applyAsDouble(Double.MAX_VALUE);
        if (!Double.isFinite(yMax) || Math.abs(yMax) > BOOST_EPSILON) {
            throw new IllegalArgumentException(
                "The function does not appear to decay to zero at +infinity; sinhSinh requires a rapidly decaying whole-line integrand."
            );
        }
        double yMin = function.applyAsDouble(-Double.MAX_VALUE);
        if (!Double.isFinite(yMin) || Math.abs(yMin) > BOOST_EPSILON) {
            throw new IllegalArgumentException(
                "The function does not appear to decay to zero at -infinity; sinhSinh requires a rapidly decaying whole-line integrand."
            );
        }

        double estimate0 = requireFinite(function.applyAsDouble(0.0), "sinhSinh") * HALF_PI;
        double l10 = Math.abs(estimate0);
        int row0Start = table.rowStart(0);
        int row0End = row0Start + table.rowLength(0);
        for (int offset = row0Start; offset < row0End; offset++) {
            double x = DETable.abscissa(base, table, offset);
            double weight = DETable.weight(base, table, offset);
            double yp = requireFinite(function.applyAsDouble(x), "sinhSinh");
            double ym = requireFinite(function.applyAsDouble(-x), "sinhSinh");
            estimate0 += (yp + ym) * weight;
            l10 += (Math.abs(yp) + Math.abs(ym)) * weight;
        }

        double estimate1 = estimate0;
        double l11 = l10;
        int row1Start = table.rowStart(1);
        int row1End = row1Start + table.rowLength(1);
        for (int offset = row1Start; offset < row1End; offset++) {
            double x = DETable.abscissa(base, table, offset);
            double weight = DETable.weight(base, table, offset);
            double yp = requireFinite(function.applyAsDouble(x), "sinhSinh");
            double ym = requireFinite(function.applyAsDouble(-x), "sinhSinh");
            estimate1 += (yp + ym) * weight;
            l11 += (Math.abs(yp) + Math.abs(ym)) * weight;
        }

        estimate1 *= 0.5;
        l11 *= 0.5;
        double error = Math.abs(estimate0 - estimate1);
        int level = 2;
        for (; level < rowCount; level++) {
            estimate0 = estimate1;
            l10 = l11;
            estimate1 = 0.5 * estimate0;
            l11 = 0.5 * l10;
            double step = Math.scalb(1.0, -level);
            double sum = 0.0;
            double absum = 0.0;
            double previousAbsTerm = 1.0;
            double epsilon = BOOST_EPSILON * l11;
            int rowStart = table.rowStart(level);
            int rowEnd = rowStart + table.rowLength(level);
            for (int offset = rowStart; offset < rowEnd; offset++) {
                double x = DETable.abscissa(base, table, offset);
                double weight = DETable.weight(base, table, offset);
                double yp = requireFinite(function.applyAsDouble(x), "sinhSinh");
                double ym = requireFinite(function.applyAsDouble(-x), "sinhSinh");
                sum += (yp + ym) * weight;
                double absTerm = (Math.abs(yp) + Math.abs(ym)) * weight;
                absum += absTerm;
                if (x > 100.0 && absTerm < epsilon && previousAbsTerm < epsilon) {
                    break;
                }
                previousAbsTerm = absTerm;
            }

            estimate1 += sum * step;
            l11 += absum * step;
            error = Math.abs(estimate0 - estimate1);
            if (!Double.isFinite(l11)) {
                throw new ArithmeticException(
                    "The sinhSinh quadrature encountered a singular point inside the domain."
                );
            }
            if (error <= tolerance * l11) {
                break;
            }
        }
        return quadrature(estimate1, error, l11, level, tolerance);
    }

    /**
     * Evaluates the tanh-sinh transformed kernel with endpoint-complement bookkeeping.
     *
     * <p>Let</p>
     * <pre>
     * z(t)  = tanh((pi/2) sinh(t)),
     * zc(t) = 1 - |z(t)|,
     * w(t)  = (pi/2) cosh(t) / cosh((pi/2) sinh(t))^2.
     * </pre>
     * This method applies the dyadic trapezoidal rule to the transformed kernel
     * <pre>
     * H(t) = function(z(t), zc(t)) w(t).
     * </pre>
     * Near {@code |z| = 1} the tables store {@code zc(t)} instead of {@code z(t)} itself,
     * because the endpoint complement is the stable quantity needed to reconstruct physical
     * distances such as {@code x-a} or {@code b-x}.
     *
     * <p>{@code leftMinComplement} and {@code rightMinComplement} are the minimum physical
     * endpoint distances retained on each side. They stop refinement once the outer domain
     * mapping can no longer distinguish new nodes from the endpoint in double precision.</p>
     */
    private static Quadrature integrateTanhSinh(
        DoubleBinaryOperator function,
        DETable table,
        DEPool pool,
        double leftMinComplement,
        double rightMinComplement,
        double tolerance
    ) {
        DETable base = pool.opts.baseTable();
        int rowCount = activeRowCount(base, table, pool);
        double h = TANH_SINH_INITIAL_STEP;
        int row0Start = table.rowStart(0);
        int row0Length = table.rowLength(0);

        double estimate0 = HALF_PI * function.applyAsDouble(0.0, 1.0);
        double l10 = Math.abs(estimate0);

        int maxLeftPosition = row0Length - 1;
        int maxRightPosition = maxLeftPosition;
        while (maxLeftPosition > 1 && Math.abs(DETable.abscissa(base, table, row0Start + maxLeftPosition)) < leftMinComplement) {
            maxLeftPosition--;
        }
        while (maxRightPosition > 1 && Math.abs(DETable.abscissa(base, table, row0Start + maxRightPosition)) < rightMinComplement) {
            maxRightPosition--;
        }

        double yp = function.applyAsDouble(
            -1.0 - DETable.abscissa(base, table, row0Start + maxLeftPosition),
            DETable.abscissa(base, table, row0Start + maxLeftPosition)
        );
        double ym = function.applyAsDouble(
            1.0 + DETable.abscissa(base, table, row0Start + maxRightPosition),
            -DETable.abscissa(base, table, row0Start + maxRightPosition)
        );
        double tailTolerance = Math.max(BOOST_EPSILON, tolerance * tolerance);

        while (maxLeftPosition > 0) {
            if (Double.isFinite(yp)) {
                break;
            }
            maxLeftPosition--;
            yp = function.applyAsDouble(
                -1.0 - DETable.abscissa(base, table, row0Start + maxLeftPosition),
                DETable.abscissa(base, table, row0Start + maxLeftPosition)
            );
        }
        while (maxLeftPosition > 1) {
            if (Math.abs(yp * DETable.weight(base, table, row0Start + maxLeftPosition)) > Math.abs(l10 * tailTolerance)) {
                break;
            }
            maxLeftPosition--;
            yp = function.applyAsDouble(
                -1.0 - DETable.abscissa(base, table, row0Start + maxLeftPosition),
                DETable.abscissa(base, table, row0Start + maxLeftPosition)
            );
        }

        while (maxRightPosition > 0) {
            if (Double.isFinite(ym)) {
                break;
            }
            maxRightPosition--;
            ym = function.applyAsDouble(
                1.0 + DETable.abscissa(base, table, row0Start + maxRightPosition),
                -DETable.abscissa(base, table, row0Start + maxRightPosition)
            );
        }
        while (maxRightPosition > 1) {
            if (Math.abs(ym * DETable.weight(base, table, row0Start + maxRightPosition)) > Math.abs(l10 * tailTolerance)) {
                break;
            }
            maxRightPosition--;
            ym = function.applyAsDouble(
                1.0 + DETable.abscissa(base, table, row0Start + maxRightPosition),
                -DETable.abscissa(base, table, row0Start + maxRightPosition)
            );
        }

        if (maxLeftPosition == 0 || maxRightPosition == 0) {
            throw new ArithmeticException(
                "The tanhSinh quadrature found the function to be non-finite everywhere near the endpoints."
            );
        }

        estimate0 += yp * DETable.weight(base, table, row0Start + maxLeftPosition)
            + ym * DETable.weight(base, table, row0Start + maxRightPosition);
        l10 += Math.abs(yp * DETable.weight(base, table, row0Start + maxLeftPosition))
            + Math.abs(ym * DETable.weight(base, table, row0Start + maxRightPosition));

        for (int index = 1; index < row0Length; index++) {
            if (index >= maxRightPosition && index >= maxLeftPosition) {
                break;
            }
            double x = DETable.abscissa(base, table, row0Start + index);
            double xc = x;
            double weight = DETable.weight(base, table, row0Start + index);
            if (x < 0.0) {
                x = 1.0 + xc;
            } else {
                xc = x - 1.0;
            }
            yp = index < maxRightPosition ? function.applyAsDouble(x, -xc) : 0.0;
            ym = index < maxLeftPosition ? function.applyAsDouble(-x, xc) : 0.0;
            estimate0 += (yp + ym) * weight;
            l10 += (Math.abs(yp) + Math.abs(ym)) * weight;
        }

        int level = 1;
        double estimate1 = estimate0;
        double l11 = l10;
        double error = 0.0;
        int thrashCount = 0;

        while (level < 4 || level < rowCount) {
            estimate0 = estimate1;
            l10 = l11;

            estimate1 = 0.5 * estimate0;
            l11 = 0.5 * l10;
            h *= 0.5;
            double sum = 0.0;
            double absum = 0.0;
            double endpointError;

            int rowStart = table.rowStart(level);
            int rowLength = table.rowLength(level);
            int firstComplementIndex = DETable.firstComplement(base, table, level);

            int maxLeftIndex = maxLeftPosition - 1;
            maxLeftPosition *= 2;
            int maxRightIndex = maxRightPosition - 1;
            maxRightPosition *= 2;

            if (rowLength > maxLeftIndex + 1 && Math.abs(DETable.abscissa(base, table, rowStart + maxLeftIndex + 1)) > leftMinComplement) {
                maxLeftPosition++;
                maxLeftIndex++;
            }
            if (rowLength > maxRightIndex + 1 && Math.abs(DETable.abscissa(base, table, rowStart + maxRightIndex + 1)) > rightMinComplement) {
                maxRightPosition++;
                maxRightIndex++;
            }

            do {
                yp = function.applyAsDouble(
                    -1.0 - DETable.abscissa(base, table, rowStart + maxLeftIndex),
                    DETable.abscissa(base, table, rowStart + maxLeftIndex)
                );
                if (Double.isFinite(yp)) {
                    break;
                }
                if (maxLeftPosition <= 2) {
                    throw new ArithmeticException(
                        "The tanhSinh quadrature found the function to be non-finite everywhere near the left endpoint."
                    );
                }
                maxLeftPosition -= 2;
                maxLeftIndex--;
            } while (DETable.abscissa(base, table, rowStart + maxLeftIndex) < 0.0);

            boolean truncateLeft = Math.abs(l11 * tailTolerance) > Math.abs(yp * DETable.weight(base, table, rowStart + maxLeftIndex));

            do {
                ym = function.applyAsDouble(
                    1.0 + DETable.abscissa(base, table, rowStart + maxRightIndex),
                    -DETable.abscissa(base, table, rowStart + maxRightIndex)
                );
                if (Double.isFinite(ym)) {
                    break;
                }
                if (maxRightPosition <= 2) {
                    throw new ArithmeticException(
                        "The tanhSinh quadrature found the function to be non-finite everywhere near the right endpoint."
                    );
                }
                maxRightIndex--;
                maxRightPosition -= 2;
            } while (DETable.abscissa(base, table, rowStart + maxRightIndex) < 0.0);

            boolean truncateRight = Math.abs(l11 * tailTolerance) > Math.abs(ym * DETable.weight(base, table, rowStart + maxRightIndex));

            sum += yp * DETable.weight(base, table, rowStart + maxLeftIndex)
                + ym * DETable.weight(base, table, rowStart + maxRightIndex);
            absum += Math.abs(yp * DETable.weight(base, table, rowStart + maxLeftIndex))
                + Math.abs(ym * DETable.weight(base, table, rowStart + maxRightIndex));
            endpointError = absum;

            for (int index = 0; index < rowLength; index++) {
                if (index >= maxLeftIndex && index >= maxRightIndex) {
                    break;
                }

                double x = DETable.abscissa(base, table, rowStart + index);
                double xc = x;
                double weight = DETable.weight(base, table, rowStart + index);
                if (index >= firstComplementIndex) {
                    x = 1.0 + xc;
                } else {
                    xc = x - 1.0;
                }

                yp = index >= maxRightIndex ? 0.0 : function.applyAsDouble(x, -xc);
                ym = index >= maxLeftIndex ? 0.0 : function.applyAsDouble(-x, xc);
                sum += (yp + ym) * weight;
                absum += (Math.abs(yp) + Math.abs(ym)) * weight;
            }

            estimate1 += sum * h;
            l11 += absum * h;

            level++;
            double lastError = error;
            error = Math.abs(estimate0 - estimate1);

            if (!Double.isFinite(estimate1)) {
                throw new ArithmeticException(
                    "The tanhSinh quadrature evaluated the function at a singular point inside the domain."
                );
            }

            if (error * 1.5 > lastError && level > 4) {
                boolean terminate = false;
                if ((++thrashCount > 1) && lastError < 1.0e-3) {
                    terminate = true;
                } else if (thrashCount > 2) {
                    terminate = true;
                } else if (lastError < ROOT_EPSILON) {
                    terminate = true;
                } else if (Math.abs(endpointError / sum) > error) {
                    terminate = true;
                }

                if (terminate) {
                    estimate1 = estimate0;
                    l11 = l10;
                    level--;
                    error = lastError + endpointError;
                    break;
                }
            }

            if (error <= Math.abs(tolerance * l11) && level >= 4) {
                if (maxLeftIndex > 0
                    && maxLeftIndex < rowLength - 1
                    && Math.abs(DETable.abscissa(base, table, rowStart + maxLeftIndex + 1)) > leftMinComplement) {
                    yp = function.applyAsDouble(
                        -1.0 - DETable.abscissa(base, table, rowStart + maxLeftIndex),
                        DETable.abscissa(base, table, rowStart + maxLeftIndex)
                    ) * DETable.weight(base, table, rowStart + maxLeftIndex);
                    ym = function.applyAsDouble(
                        -1.0 - DETable.abscissa(base, table, rowStart + maxLeftIndex - 1),
                        DETable.abscissa(base, table, rowStart + maxLeftIndex - 1)
                    ) * DETable.weight(base, table, rowStart + maxLeftIndex - 1);
                    if (Math.abs(yp) > Math.abs(ym)) {
                        throw new ArithmeticException(
                            "The tanhSinh quadrature narrowed the left endpoint, but the integral is still increasing at that boundary."
                        );
                    }
                }
                if (maxRightIndex > 0
                    && maxRightIndex < rowLength - 1
                    && Math.abs(DETable.abscissa(base, table, rowStart + maxRightIndex + 1)) > rightMinComplement) {
                    yp = function.applyAsDouble(
                        1.0 + DETable.abscissa(base, table, rowStart + maxRightIndex),
                        -DETable.abscissa(base, table, rowStart + maxRightIndex)
                    ) * DETable.weight(base, table, rowStart + maxRightIndex);
                    ym = function.applyAsDouble(
                        1.0 + DETable.abscissa(base, table, rowStart + maxRightIndex - 1),
                        -DETable.abscissa(base, table, rowStart + maxRightIndex - 1)
                    ) * DETable.weight(base, table, rowStart + maxRightIndex - 1);
                    if (Math.abs(yp) > Math.abs(ym)) {
                        throw new ArithmeticException(
                            "The tanhSinh quadrature narrowed the right endpoint, but the integral is still increasing at that boundary."
                        );
                    }
                }
                error += endpointError;
                break;
            }

            if (truncateLeft && maxLeftPosition > 1) {
                maxLeftPosition--;
            }
            if (truncateRight && maxRightPosition > 1) {
                maxRightPosition--;
            }
        }

        return quadrature(estimate1, error, l11, level, tolerance);
    }

    private static DETable initExpSinhTable(int maxRefinements, DEPool pool) {
        DETable base = pool.select(DEOpts.EXP_SINH);
        int precomputedRows = base.rowCount();
        int rowCount = requestedRows(maxRefinements, precomputedRows);
        if (rowCount <= precomputedRows) {
            return base;
        }
        if (pool.refine != null && pool.refine.rowCount() >= rowCount) {
            pool.refineRows = rowCount;
            return pool.refine;
        }

        int[] rowData = Arrays.copyOf(DoubleExponentialsTables.EXP_SINH_ROWS, rowCount << 1);
        int split = DoubleExponentialsTables.EXP_SINH_ABSCISSAS.length;
        DoubleArrayBuilder extendedAbscissas = new DoubleArrayBuilder(128);
        DoubleArrayBuilder extendedWeights = new DoubleArrayBuilder(128);

        double tMin = DoubleExponentialsTables.EXP_SINH_T_MIN;
        double tMax = Math.log(2.0 * TWO_DIV_PI * Math.log(2.0 * TWO_DIV_PI * Math.sqrt(Double.MAX_VALUE)));
        for (int row = precomputedRows; row < rowCount; row++) {
            double h = Math.scalb(1.0, -row);
            int extensionStart = extendedAbscissas.size();
            int rowStart = split + extensionStart;
            double step = 2.0 * h;
            double start = tMin + h;
            for (double argument = start; argument + h < tMax; argument += step) {
                double x = Math.exp(HALF_PI * Math.sinh(argument));
                extendedAbscissas.add(x);
                extendedWeights.add(Math.cosh(argument) * HALF_PI * x);
            }
            setRow(rowData, row, rowStart, extendedAbscissas.size() - extensionStart);
        }
        DETable refine = new DETable(
            extendedAbscissas.toArray(),
            extendedWeights.toArray(),
            rowData
        );
        pool.refine = refine;
        pool.refineRows = rowCount;
        return refine;
    }

    private static DETable initSinhSinhTable(int maxRefinements, DEPool pool) {
        DETable base = pool.select(DEOpts.SINH_SINH);
        int precomputedRows = base.rowCount();
        int rowCount = requestedRows(maxRefinements, precomputedRows);
        if (rowCount <= precomputedRows) {
            return base;
        }
        if (pool.refine != null && pool.refine.rowCount() >= rowCount) {
            pool.refineRows = rowCount;
            return pool.refine;
        }

        int[] rowData = Arrays.copyOf(DoubleExponentialsTables.SINH_SINH_ROWS, rowCount << 1);
        int split = DoubleExponentialsTables.SINH_SINH_ABSCISSAS.length;
        DoubleArrayBuilder extendedAbscissas = new DoubleArrayBuilder(128);
        DoubleArrayBuilder extendedWeights = new DoubleArrayBuilder(128);

        double tMax = DoubleExponentialsTables.SINH_SINH_T_MAX;
        for (int row = precomputedRows; row < rowCount; row++) {
            double h = Math.scalb(1.0, -row);
            int extensionStart = extendedAbscissas.size();
            int rowStart = split + extensionStart;
            double step = 2.0 * h;
            for (double argument = h; argument < tMax; argument += step) {
                double tmp = HALF_PI * Math.sinh(argument);
                double x = Math.sinh(tmp);
                extendedAbscissas.add(x);
                extendedWeights.add(Math.cosh(argument) * HALF_PI * Math.cosh(tmp));
            }
            setRow(rowData, row, rowStart, extendedAbscissas.size() - extensionStart);
        }
        DETable refine = new DETable(
            extendedAbscissas.toArray(),
            extendedWeights.toArray(),
            rowData
        );
        pool.refine = refine;
        pool.refineRows = rowCount;
        return refine;
    }

    private static DETable initTanhSinhTable(int maxRefinements, double minComplement, DEPool pool) {
        DETable base = pool.select(DEOpts.TANH_SINH);
        int precomputedRows = base.rowCount();
        int rowCount = requestedRows(maxRefinements, precomputedRows);
        if (rowCount <= precomputedRows && minComplement <= DEFAULT_MIN_COMPLEMENT) {
            return base;
        }
        DETable rawRefine = pool.refine;
        if (rowCount > precomputedRows && (rawRefine == null || rawRefine.rowCount() < rowCount)) {
            rawRefine = buildTanhSinhRefineTable(rowCount, base, precomputedRows);
            pool.refine = rawRefine;
        }
        if (rowCount > precomputedRows) {
            pool.refineRows = rowCount;
        }
        if (rowCount > precomputedRows && minComplement <= DEFAULT_MIN_COMPLEMENT) {
            return rawRefine;
        }
        return viewTanhSinhTable(pool, rawRefine, rowCount, minComplement);
    }

    private static DETable buildTanhSinhRefineTable(int rowCount, DETable base, int precomputedRows) {
        if (rowCount <= precomputedRows) {
            return base;
        }

        int[] rowData = Arrays.copyOf(DoubleExponentialsTables.TANH_SINH_ROWS, rowCount << 1);
        int[] firstComplements = new int[rowCount - precomputedRows];
        double[] extendedAbscissasArray;
        double[] extendedWeightsArray;
        int split = DoubleExponentialsTables.TANH_SINH_ABSCISSAS.length;
        DoubleArrayBuilder extendedAbscissas = new DoubleArrayBuilder(256);
        DoubleArrayBuilder extendedWeights = new DoubleArrayBuilder(256);
        double tCrossover = tFromAbscissaComplement(0.5);
        double h = Math.scalb(1.0, -(precomputedRows - 1));
        for (int row = precomputedRows; row < rowCount; row++) {
            h *= 0.5;
            int extensionStart = extendedAbscissas.size();
            int rowStart = split + extensionStart;
            int firstComplement = 0;
            for (double position = h; position < TANH_SINH_T_MAX; position += 2.0 * h) {
                // Once z(t) is too close to 1, store -zc(t) instead of z(t) so the caller
                // can recover endpoint distances without subtracting nearly equal numbers.
                if (position < tCrossover) {
                    firstComplement++;
                    extendedAbscissas.add(abscissaAtT(position));
                } else {
                    extendedAbscissas.add(-abscissaComplementAtT(position));
                }
                extendedWeights.add(weightAtT(position));
            }
            setRow(rowData, row, rowStart, extendedAbscissas.size() - extensionStart);
            firstComplements[row - precomputedRows] = firstComplement;
        }
        extendedAbscissasArray = extendedAbscissas.toArray();
        extendedWeightsArray = extendedWeights.toArray();

        return new DETable(extendedAbscissasArray, extendedWeightsArray, rowData, firstComplements);
    }

    private static DETable viewTanhSinhTable(
        DEPool pool,
        DETable rawRefine,
        int rowCount,
        double minComplement
    ) {
        DETable base = pool.opts.baseTable();
        int precomputedRows = base.rowCount();
        boolean needsExtendedRows = rowCount > precomputedRows;
        boolean needsPrunedRows = minComplement > DEFAULT_MIN_COMPLEMENT;
        if (!needsExtendedRows && !needsPrunedRows) {
            return base;
        }

        int[] rowData = Arrays.copyOf(needsExtendedRows ? rawRefine.rows : base.rows, rowCount << 1);
        int split = base.abscissas.length;
        double[] extendedAbscissas = needsExtendedRows ? rawRefine.abscissas : EMPTY_DOUBLE_ARRAY;
        double[] extendedWeights = needsExtendedRows ? rawRefine.weights : EMPTY_DOUBLE_ARRAY;
        int[] firstComplements = needsExtendedRows
            ? rawRefine.complements
            : EMPTY_INT_ARRAY;
        if (needsPrunedRows) {
            pruneToMinComplement(base.abscissas, extendedAbscissas, split, rowData, minComplement);
        }
        return new DETable(extendedAbscissas, extendedWeights, rowData, firstComplements);
    }

    private static int requestedRows(int maxRefinements, int precomputedRows) {
        return Math.max(precomputedRows, maxRefinements + 1);
    }

    private static int activeRowCount(DETable base, DETable table, DEPool pool) {
        if (table == base) {
            return base.rowCount();
        }
        if (table == pool.refine) {
            return pool.refineRows;
        }
        return table.rowCount();
    }

    private static void setRow(int[] rows, int row, int rowStart, int rowLength) {
        int index = row << 1;
        rows[index] = rowStart;
        rows[index + 1] = rowLength;
    }

    private static void pruneToMinComplement(
        double[] abscissas,
        double[] extendedAbscissas,
        int split,
        int[] rows,
        double minComplement
    ) {
        for (int row = 0; row < (rows.length >>> 1); row++) {
            int rowStart = rows[row << 1];
            int rowLength = rows[(row << 1) + 1];
            int keep = rowLength;
            for (int index = 0; index < rowLength; index++) {
                if (Math.abs(segmentedValue(abscissas, extendedAbscissas, split, rowStart + index)) <= minComplement) {
                    keep = index;
                    break;
                }
            }
            rows[(row << 1) + 1] = keep;
        }
    }

    private static double segmentedValue(double[] values, double[] extendedValues, int split, int index) {
        return index < split ? values[index] : extendedValues[index - split];
    }

    // Tanh-sinh abscissa z(t) = tanh((pi/2) sinh(t)).
    private static double abscissaAtT(double t) {
        return Math.tanh(HALF_PI * Math.sinh(t));
    }

    // Tanh-sinh Jacobian z'(t) = (pi/2) cosh(t) / cosh((pi/2) sinh(t))^2.
    private static double weightAtT(double t) {
        double cosh = Math.cosh(HALF_PI * Math.sinh(t));
        return HALF_PI * Math.cosh(t) / (cosh * cosh);
    }

    // Stable endpoint complement zc(t) = 1 - z(t) = 1 / (exp(u) cosh(u)), u = (pi/2) sinh(t).
    private static double abscissaComplementAtT(double t) {
        double u = HALF_PI * Math.sinh(t);
        return 1.0 / (Math.exp(u) * Math.cosh(u));
    }

    // Inverts the tanh-sinh endpoint-complement relation zc = 1 - tanh((pi/2) sinh(t)).
    private static double tFromAbscissaComplement(double complement) {
        double logTerm = Math.log(2.0 - complement) - Math.log(complement);
        return Math.log((Math.sqrt(logTerm * logTerm + Math.PI * Math.PI) + logTerm) / Math.PI);
    }

    private static double sanitizeMinComplement(double minComplement) {
        if (Double.isNaN(minComplement) || minComplement < 0.0 || minComplement >= 1.0) {
            throw new IllegalArgumentException("minComplement must be in [0, 1): " + minComplement);
        }
        return Math.max(minComplement, DEFAULT_MIN_COMPLEMENT);
    }

    private static void validateTolerance(double tolerance) {
        if (!(tolerance > 0.0) || Double.isNaN(tolerance)) {
            throw new IllegalArgumentException("tolerance must be positive: " + tolerance);
        }
        if (Math.abs(tolerance) > 1.0) {
            throw new IllegalArgumentException(
                "The tolerance is unusually large; did you confuse it with a domain bound? tolerance=" + tolerance
            );
        }
    }

    private static void validateRefinements(int maxRefinements) {
        if (maxRefinements < 0) {
            throw new IllegalArgumentException("maxRefinements must be non-negative: " + maxRefinements);
        }
    }

    private static double requireFinite(double value, String rule) {
        if (!Double.isFinite(value)) {
            throw new ArithmeticException(rule + " integrand returned a non-finite value.");
        }
        return value;
    }
}