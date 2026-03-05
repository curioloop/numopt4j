package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.Bound;

/**
 * Test helper to convert primitive bound arrays used in legacy tests
 * into {@link com.curioloop.numopt4j.optim.Bound}[] instances.
 */
final class TestBounds implements LBFGSBConstants {

    private TestBounds() {}

    static Bound[] toBounds(int n, double[] lower, double[] upper, int[] boundType) {
        if (lower == null && upper == null && boundType == null) {
            return null;
        }

        Bound[] bounds = new Bound[n];
        for (int i = 0; i < n; i++) {
            int bt = (boundType != null && i < boundType.length) ? boundType[i] : BOUND_NONE;
            double lv = (lower != null && i < lower.length) ? lower[i] : Double.NaN;
            double uv = (upper != null && i < upper.length) ? upper[i] : Double.NaN;

            switch (bt) {
                case BOUND_LOWER:
                    bounds[i] = Double.isNaN(lv) ? null : Bound.atLeast(lv);
                    break;
                case BOUND_UPPER:
                    bounds[i] = Double.isNaN(uv) ? null : Bound.atMost(uv);
                    break;
                case BOUND_BOTH:
                    if (!Double.isNaN(lv) && !Double.isNaN(uv)) {
                        bounds[i] = Bound.between(lv, uv);
                    } else if (!Double.isNaN(lv)) {
                        bounds[i] = Bound.atLeast(lv);
                    } else if (!Double.isNaN(uv)) {
                        bounds[i] = Bound.atMost(uv);
                    } else {
                        bounds[i] = null;
                    }
                    break;
                case BOUND_NONE:
                default:
                    bounds[i] = null;
            }
        }

        return bounds;
    }
}
