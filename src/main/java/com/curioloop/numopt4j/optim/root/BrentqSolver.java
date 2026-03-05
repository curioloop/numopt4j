package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.OptimizationResult;
import com.curioloop.numopt4j.optim.OptimizationStatus;

import java.util.function.DoubleUnaryOperator;

/**
 * One-dimensional root finder using Brent's method.
 *
 * <p>Strictly mirrors the loop logic of {@code scipy/optimize/Zeros/brentq.c}.
 * The algorithm combines bisection, secant, and inverse quadratic interpolation
 * to achieve superlinear convergence while guaranteeing the bracket is maintained.</p>
 *
 * <p>This class is package-private; use {@link RootFinder} for the public API.</p>
 */
final class BrentqSolver {

    /** Default absolute tolerance (matches scipy brentq default). */
    static final double DEFAULT_XTOL = 2e-12;

    /** Default relative tolerance: 4 * machine epsilon (matches scipy brentq default). */
    static final double DEFAULT_RTOL = 4 * Math.ulp(1.0);  // 4 * DBL_EPSILON

    /** Default maximum number of iterations. */
    static final int DEFAULT_MAXITER = 100;

    // Prevent instantiation
    private BrentqSolver() {}

    /**
     * Finds a root of {@code f} in the bracket {@code [xa, xb]}.
     *
     * <p>The algorithm is a strict port of {@code brentq.c} from SciPy:</p>
     * <ol>
     *   <li>If {@code f(xa) == 0} or {@code f(xb) == 0}, the endpoint is returned immediately.</li>
     *   <li>If {@code f(xa) * f(xb) > 0}, an {@link IllegalArgumentException} is thrown.</li>
     *   <li>Each iteration chooses between inverse quadratic interpolation, secant step,
     *       or bisection, accepting the interpolation only when it is safe.</li>
     *   <li>If {@code f(x)} returns {@code NaN} or {@code Infinity} at any point,
     *       {@link OptimizationStatus#ABNORMAL_TERMINATION} is returned.</li>
     * </ol>
     *
     * @param f       the scalar function whose root is sought
     * @param xa      left endpoint of the bracket
     * @param xb      right endpoint of the bracket
     * @param xtol    absolute tolerance (must be &gt;= 0)
     * @param rtol    relative tolerance (must be &gt;= 0)
     * @param maxiter maximum number of iterations
     * @return an {@link OptimizationResult} describing the outcome
     * @throws IllegalArgumentException if {@code f(xa) * f(xb) > 0}
     */
    static OptimizationResult solve(
            DoubleUnaryOperator f,
            double xa, double xb,
            double xtol, double rtol,
            int maxiter) {

        double xpre = xa, xcur = xb;
        double xblk = 0, fblk = 0, spre = 0, scur = 0;

        double fpre = f.applyAsDouble(xpre);
        double fcur = f.applyAsDouble(xcur);

        // Handle endpoints that are already roots
        if (fpre == 0) {
            return new OptimizationResult(xpre, null, 0, OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED, 0);
        }
        if (fcur == 0) {
            return new OptimizationResult(xcur, null, 0, OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED, 0);
        }

        // Bracket condition check
        if (fpre * fcur > 0) {
            throw new IllegalArgumentException(
                    "f(a) and f(b) must have opposite signs: f(" + xa + ")=" + fpre +
                    ", f(" + xb + ")=" + fcur);
        }

        // Check for NaN/Inf in initial evaluations
        if (!Double.isFinite(fpre) || !Double.isFinite(fcur)) {
            return new OptimizationResult(xcur, null, Math.abs(fcur), OptimizationStatus.ABNORMAL_TERMINATION, 0);
        }

        for (int i = 1; i <= maxiter; i++) {

            // scipy: fpre != 0 && fcur != 0 && signbit(fpre) != signbit(fcur)
            // Equivalent to fpre*fcur < 0 when both are finite and non-zero
            if (fpre != 0 && fcur != 0 && ((fpre < 0) != (fcur < 0))) {
                xblk = xpre;
                fblk = fpre;
                spre = scur = xcur - xpre;
            }

            if (Math.abs(fblk) < Math.abs(fcur)) {
                xpre = xcur;  xcur = xblk;  xblk = xpre;
                fpre = fcur;  fcur = fblk;  fblk = fpre;
            }

            double delta = (xtol + rtol * Math.abs(xcur)) / 2.0;
            double sbis  = (xblk - xcur) / 2.0;

            // Convergence check
            if (fcur == 0 || Math.abs(sbis) < delta) {
                return new OptimizationResult(xcur, null, Math.abs(fcur), OptimizationStatus.COEFFICIENT_TOLERANCE_REACHED, i);
            }

            // Step selection
            if (Math.abs(spre) > delta && Math.abs(fcur) < Math.abs(fpre)) {
                double stry;
                if (xpre == xblk) {
                    // Secant step (interpolate)
                    stry = -fcur * (xcur - xpre) / (fcur - fpre);
                } else {
                    // Inverse quadratic interpolation (extrapolate)
                    double dpre = (fpre - fcur) / (xpre - xcur);
                    double dblk = (fblk - fcur) / (xblk - xcur);
                    stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
                }

                if (2.0 * Math.abs(stry) < Math.min(Math.abs(spre), 3.0 * Math.abs(sbis) - delta)) {
                    spre = scur;
                    scur = stry;    // Accept interpolation step
                } else {
                    spre = sbis;
                    scur = sbis;    // Fall back to bisection
                }
            } else {
                spre = sbis;
                scur = sbis;        // Bisection
            }

            xpre = xcur;
            fpre = fcur;
            xcur += (Math.abs(scur) > delta) ? scur : (sbis > 0 ? delta : -delta);

            fcur = f.applyAsDouble(xcur);

            // Check for NaN/Inf after function evaluation
            if (!Double.isFinite(fcur)) {
                return new OptimizationResult(xcur, null, Math.abs(fcur), OptimizationStatus.ABNORMAL_TERMINATION, i);
            }
        }

        return new OptimizationResult(xcur, null, Math.abs(fcur), OptimizationStatus.MAX_ITERATIONS_REACHED, maxiter);
    }
}
