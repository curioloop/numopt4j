/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.adapt;

import com.curioloop.numopt4j.quad.Quadrature;

import java.util.function.DoubleUnaryOperator;

/**
 * Adaptive Gauss-Kronrod (GK15) quadrature with a max-heap for interval selection.
 *
 * <p>Algorithm: adaptive subdivision of [a,b] using the 15-point Gauss-Kronrod rule.
 * At each step the interval with the largest local error is bisected.
 * The global estimate is updated with Kahan compensated summation.</p>
 *
 * <p>GK15 rule on [a,b]:
 *   I ≈ (b−a)/2 · Σ wᵢ·f(cᵢ)
 * where cᵢ = (a+b)/2 + (b−a)/2·xᵢ are the mapped Kronrod nodes.
 * The embedded G7 rule uses the 7 even-indexed Kronrod nodes (indices 1,3,5 and centre).</p>
 *
 * <p>Error estimate: |I_K15 − I_G7| · (b−a)/2</p>
 *
 * <p>Convergence criterion: totalError ≤ max(absTol, relTol·|totalEstimate|)</p>
 *
 * <p>The GK15 abscissae and weights are taken from the standard QUADPACK tables
 * (Piessens et al., "QUADPACK", Springer 1983, Appendix).</p>
 *
 * <p>The integral estimate is maintained with Kahan compensated summation to
 * reduce floating-point drift across many subdivisions.</p>
 *
 * <p>Interval selection uses a binary max-heap keyed on local error, giving
 * O(log n) per iteration instead of O(n) linear scan.</p>
 *
 * <p>{@link #gk15} writes value and error directly into the arena and returns an
 * {@code int} encoding both the evaluation count and success flag:
 * a negative return value means success (finite), positive means failure (non-finite);
 * the absolute value is the number of function evaluations performed.</p>
 */
final class AdaptiveCore {

    // Gauss 7-point weights (for the 4 positive abscissae + centre)
    private static final double[] WG = {
            0.129484966168869693270611432679082,
            0.279705391489276667901467771423780,
            0.381830050505118944950369775488975,
            0.417959183673469387755102040816327
    };

    // Kronrod 15-point abscissae (positive half, including 0)
    private static final double[] XGK = {
            0.991455371120812639206854697526329,
            0.949107912342758524526189684047851,
            0.864864423359769072789712788640926,
            0.741531185599394439863864773280788,
            0.586087235467691130294144838258730,
            0.405845151377397166906606412076961,
            0.207784955007898467600689403773245,
            0.000000000000000000000000000000000
    };

    // Kronrod 15-point weights (positive half, including centre)
    private static final double[] WGK = {
            0.022935322010529224963732008058970,
            0.063092092629978553290700663189204,
            0.104790010322250183839876322541518,
            0.140653259715525918745189590510238,
            0.169004726639267902826583426598550,
            0.190350578064785409913256402421014,
            0.204432940075298892414161999234649,
            0.209482141084727828012999174891714
    };

    private AdaptiveCore() {}

    static Quadrature integrate(DoubleUnaryOperator f, double min, double max,
                                double absTol, double relTol,
                                int maxSubdivisions, int maxEvaluations,
                                AdaptivePool workspace) {
        AdaptivePool pool = workspace.ensure(maxSubdivisions);
        double[] arena = pool.arena();
        int[] heap = pool.heap();
        int leftOffset = pool.intervalLeftOffset();
        int rightOffset = pool.intervalRightOffset();
        int estimateOffset = pool.intervalEstimateOffset();
        int errorOffset = pool.intervalErrorOffset();

        // Two reusable buffers: buf[0] = value, buf[1] = error.
        // gk15 returns: negative → finite (success), positive → non-finite (failure);
        // absolute value = evaluations performed.
        double[] left  = new double[2];
        double[] right = new double[2];

        int r0 = gk15(f, min, max, left);
        if (r0 > 0) {
            return new Quadrature(Double.NaN, Double.NaN,
                    Quadrature.Status.ABNORMAL_TERMINATION, 0, r0);
        }

        arena[leftOffset]     = min;
        arena[rightOffset]    = max;
        arena[estimateOffset] = left[0];
        arena[errorOffset]    = left[1];

        heap[0] = 0;
        int heapSize = 1;

        int count = 1;
        int iterations = 0;
        int evaluations = -r0;
        double totalEstimate = left[0];
        double totalError    = left[1];
        double estimateComp  = 0.0; // Kahan compensation

        while (totalError > tolerance(absTol, relTol, totalEstimate)) {
            if (count >= maxSubdivisions) {
                return new Quadrature(totalEstimate, totalError,
                        Quadrature.Status.MAX_SUBDIVISIONS_REACHED, iterations, evaluations);
            }
            if (evaluations + 30 > maxEvaluations) {
                return new Quadrature(totalEstimate, totalError,
                        Quadrature.Status.MAX_EVALUATIONS_REACHED, iterations, evaluations);
            }

            int split = heapPop(heap, --heapSize, arena, errorOffset);
            double lo  = arena[leftOffset  + split];
            double hi  = arena[rightOffset + split];
            double mid = 0.5 * (lo + hi);
            if (!(lo < mid && mid < hi)) {
                return new Quadrature(totalEstimate, totalError,
                        Quadrature.Status.ROUND_OFF_DETECTED, iterations, evaluations);
            }

            double oldEstimate = arena[estimateOffset + split];
            double oldError    = arena[errorOffset    + split];

            int rL = gk15(f, lo, mid, left);
            int rR = gk15(f, mid, hi, right);
            evaluations += Math.abs(rL) + Math.abs(rR);
            if (rL > 0 || rR > 0) {
                return new Quadrature(Double.NaN, Double.NaN,
                        Quadrature.Status.ABNORMAL_TERMINATION, iterations, evaluations);
            }

            // Kahan compensated update: (left + right) − old
            double y = (left[0] + right[0] - oldEstimate) - estimateComp;
            double t = totalEstimate + y;
            estimateComp = (t - totalEstimate) - y;
            totalEstimate = t;
            totalError += left[1] + right[1] - oldError;

            // Reuse split slot for left child, append right child at next free slot
            int newIdx = count;
            arena[leftOffset  + split]  = lo;   arena[rightOffset  + split]  = mid;
            arena[estimateOffset + split] = left[0]; arena[errorOffset + split] = left[1];
            heapPush(heap, heapSize++, split, arena, errorOffset);

            arena[leftOffset  + newIdx] = mid;  arena[rightOffset  + newIdx] = hi;
            arena[estimateOffset + newIdx] = right[0]; arena[errorOffset + newIdx] = right[1];
            heapPush(heap, heapSize++, newIdx, arena, errorOffset);

            count++;
            iterations++;
        }

        return new Quadrature(totalEstimate, totalError,
                Quadrature.Status.CONVERGED, iterations, evaluations);
    }

    // -----------------------------------------------------------------------
    // Binary max-heap helpers (keyed on arena[errorOffset + index])
    // -----------------------------------------------------------------------

    /** Push interval index {@code idx} onto the heap and sift up. */
    private static void heapPush(int[] heap, int size, int idx,
                                  double[] arena, int errorOffset) {
        heap[size] = idx;
        siftUp(heap, size, arena, errorOffset);
    }

    /**
     * Pop the interval with the maximum error from the heap.
     * {@code size} must be the new heap size after removal (i.e. old size - 1).
     */
    private static int heapPop(int[] heap, int size,
                                double[] arena, int errorOffset) {
        int top = heap[0];
        heap[0] = heap[size];
        siftDown(heap, 0, size, arena, errorOffset);
        return top;
    }

    private static void siftUp(int[] heap, int pos, double[] arena, int errorOffset) {
        while (pos > 0) {
            int parent = (pos - 1) >>> 1;
            if (arena[errorOffset + heap[parent]] >= arena[errorOffset + heap[pos]]) break;
            int tmp = heap[parent]; heap[parent] = heap[pos]; heap[pos] = tmp;
            pos = parent;
        }
    }

    private static void siftDown(int[] heap, int pos, int size,
                                  double[] arena, int errorOffset) {
        while (true) {
            int left = (pos << 1) + 1;
            if (left >= size) break;
            int right = left + 1;
            int largest = (right < size && arena[errorOffset + heap[right]] > arena[errorOffset + heap[left]])
                    ? right : left;
            if (arena[errorOffset + heap[pos]] >= arena[errorOffset + heap[largest]]) break;
            int tmp = heap[pos]; heap[pos] = heap[largest]; heap[largest] = tmp;
            pos = largest;
        }
    }

    private static double tolerance(double absTol, double relTol, double estimate) {
        return Math.max(absTol, relTol * Math.abs(estimate));
    }

    /**
     * Gauss-Kronrod 15-point rule on [min, max].
     *
     * <p>Affine map: x = c + h·t,  c = (min+max)/2,  h = (max−min)/2,  t ∈ [−1,1]
     * Approximation: ∫_{min}^{max} f(x) dx ≈ h · Σ_{i} wᵢ·f(c + h·xᵢ)
     * where xᵢ are the Kronrod nodes (symmetric about 0, stored as positive half + 0)
     * and wᵢ are the corresponding Kronrod weights.</p>
     *
     * <p>The embedded G7 estimate uses nodes at Kronrod indices 1,3,5 (positive half)
     * plus the centre (index 7 in WGK / index 3 in WG).</p>
     *
     * <p>Error estimate: |I_K15 − I_G7| · h</p>
     *
     * <p>Writes {@code out[0] = value} and {@code out[1] = error}.</p>
     *
     * <p>Returns: negative → finite result (success), positive → non-finite (failure);
     * {@code abs(return)} = evaluations performed.</p>
     */
    private static int gk15(DoubleUnaryOperator f, double min, double max, double[] out) {
        double center = 0.5 * (min + max);
        double halfLength = 0.5 * (max - min);

        double fc = f.applyAsDouble(center);
        if (!Double.isFinite(fc)) {
            out[0] = out[1] = Double.NaN;
            return 1; // failure, 1 evaluation
        }

        double resGauss = WG[3] * fc;
        double resKronrod = WGK[7] * fc;

        int evaluations = 1;
        for (int i = 0; i < 7; i++) {
            double abscissa = halfLength * XGK[i];
            double f1 = f.applyAsDouble(center - abscissa);
            double f2 = f.applyAsDouble(center + abscissa);
            evaluations += 2;
            if (!Double.isFinite(f1) || !Double.isFinite(f2)) {
                out[0] = out[1] = Double.NaN;
                return evaluations; // failure
            }

            double sum = f1 + f2;
            resKronrod += WGK[i] * sum;
            if (i == 1) {
                resGauss += WG[0] * sum;
            } else if (i == 3) {
                resGauss += WG[1] * sum;
            } else if (i == 5) {
                resGauss += WG[2] * sum;
            }
        }

        out[0] = resKronrod * halfLength;
        out[1] = Math.abs(resKronrod - resGauss) * halfLength;
        return -evaluations; // success
    }
}
