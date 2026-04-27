package com.curioloop.numopt4j.quad.de;

import com.curioloop.numopt4j.quad.Quadrature;
import com.curioloop.numopt4j.quad.special.EndpointOpts;
import com.curioloop.numopt4j.quad.special.EndpointSingularIntegral;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DoubleExponentialCoreTest {

    @Test
    void tanhSinhIntegratesSmoothPolynomial() {
        double value = DoubleExponentialCore.tanhSinh(x -> x * x, 0.0, 2.0, 1.0e-10, 15, 4.0 * Double.MIN_VALUE)
            .getValue();
        assertEquals(8.0 / 3.0, value, 1.0e-10);
    }

    @Test
    void tanhSinhHandlesEndpointSingularity() {
        double value = DoubleExponentialCore.tanhSinh(
            x -> 1.0 / Math.sqrt(x),
            0.0,
            1.0,
            1.0e-12,
            20,
            4.0 * Double.MIN_VALUE
        ).getValue();
        assertEquals(2.0, value, 2.0e-8);
    }

    @Test
    void tanhSinhHandlesSemiInfiniteInterval() {
        double value = DoubleExponentialCore.tanhSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            15,
            4.0 * Double.MIN_VALUE
        ).getValue();
        assertEquals(1.0, value, 1.0e-10);
    }

    @Test
    void tanhSinhHandlesWholeRealLine() {
        double value = DoubleExponentialCore.tanhSinh(
            x -> Math.exp(-x * x),
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            15,
            4.0 * Double.MIN_VALUE
        ).getValue();
        assertEquals(Math.sqrt(Math.PI), value, 1.0e-10);
    }

    @Test
    void tanhSinhSupportsComplementAwareIntegrands() {
        double value = DoubleExponentialCore.tanhSinh(
            (x, complement) -> 1.0 / Math.sqrt(Math.abs(complement)),
            0.0,
            1.0,
            1.0e-10,
            15,
            4.0 * Double.MIN_VALUE
        ).getValue();
        assertEquals(2.0 * Math.sqrt(2.0), value, 1.0e-8);
    }

    @Test
    void tanhSinhClampsSubnormalMinComplementToBoostFloor() {
        Quadrature clamped = DoubleExponentialCore.tanhSinh(
            (x, complement) -> 1.0 / Math.sqrt(Math.abs(complement)),
            0.0,
            1.0,
            1.0e-10,
            15,
            4.0 * Double.MIN_VALUE
        );
        Quadrature boostFloor = DoubleExponentialCore.tanhSinh(
            (x, complement) -> 1.0 / Math.sqrt(Math.abs(complement)),
            0.0,
            1.0,
            1.0e-10,
            15,
            DoubleExponentialCore.DEFAULT_MIN_COMPLEMENT
        );
        assertEquals(boostFloor.getValue(), clamped.getValue(), 0.0);
        assertEquals(boostFloor.getEstimatedError(), clamped.getEstimatedError(), 0.0);
        assertEquals(boostFloor.getIterations(), clamped.getIterations());
        assertEquals(boostFloor.getStatus(), clamped.getStatus());
    }

    @Test
    void expSinhIntegratesSemiInfiniteExponentialTail() {
        double value = DoubleExponentialCore.expSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            9
        ).getValue();
        assertEquals(1.0, value, 1.0e-10);
    }

    @Test
    void expSinhRetainsBoostBakedRowsWhenRequestedRefinementsAreZero() {
        double value = DoubleExponentialCore.expSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            0
        ).getValue();
        assertEquals(1.0, value, 1.0e-10);
    }

    @Test
    void expSinhReusesRefineTableFromPool() {
        DEPool pool = new DEPool();

        DoubleExponentialCore.expSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            9,
            pool
        );
        DETable cached = pool.refine;

        DoubleExponentialCore.expSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            9,
            pool
        );

        assertSame(cached, pool.refine);
    }

    @Test
    void expSinhUsesPoolRefineRowsToLimitCachedRefineDepth() {
        DEPool pool = new DEPool();
        int baseRows = DEOpts.EXP_SINH.baseTable().rowCount();
        int largerMaxRefinements = baseRows + 2;
        int smallerMaxRefinements = baseRows;

        DoubleExponentialCore.expSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            largerMaxRefinements,
            pool
        );
        DETable cached = pool.refine;
        int cachedRows = cached.rowCount();

        DoubleExponentialCore.expSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            smallerMaxRefinements,
            pool
        );

        assertSame(cached, pool.refine);
        assertEquals(baseRows + 1, pool.refineRows);
        assertEquals(cachedRows, pool.refine.rowCount());
    }

    @Test
    void expSinhRejectsFiniteIntervals() {
        assertThrows(
            IllegalArgumentException.class,
            () -> DoubleExponentialCore.expSinh(x -> x, 0.0, 1.0, 1.0e-8, 9)
        );
    }

    @Test
    void sinhSinhIntegratesGaussianOverWholeRealLine() {
        double value = DoubleExponentialCore.sinhSinh(x -> Math.exp(-x * x), 1.0e-10, 9).getValue();
        assertEquals(Math.sqrt(Math.PI), value, 1.0e-10);
    }

    @Test
    void sinhSinhRetainsBoostBakedRowsWhenRequestedRefinementsAreZero() {
        double value = DoubleExponentialCore.sinhSinh(x -> Math.exp(-x * x), 1.0e-10, 0).getValue();
        assertEquals(Math.sqrt(Math.PI), value, 1.0e-10);
    }

    @Test
    void tanhSinhRetainsBoostBakedRowsWhenRequestedRefinementsAreZero() {
        double value = DoubleExponentialCore.tanhSinh(
            x -> x * x,
            0.0,
            2.0,
            1.0e-10,
            0,
            4.0 * Double.MIN_VALUE
        ).getValue();
        assertEquals(8.0 / 3.0, value, 1.0e-10);
    }

    @Test
    void builderUsesExternalPoolForTanhSinhReuse() {
        DEPool pool = new DEPool();
        DoubleExponentialIntegral integral = new DoubleExponentialIntegral(DEOpts.TANH_SINH)
            .function(x -> x * x)
            .bounds(0.0, 2.0)
            .tolerance(1.0e-10)
            .maxRefinements(12);

        integral.integrate(pool);
        DETable cached = pool.refine;
        integral.integrate(pool);

        assertNotNull(cached);
        assertSame(cached, pool.refine);
    }

    @Test
    void defaultTanhSinhRefineIsSharedAcrossFreshPools() {
        DEPool first = new DEPool();
        DoubleExponentialCore.tanhSinh(
            x -> x * x,
            0.0,
            1.0,
            1.0e-10,
            DoubleExponentialCore.DEFAULT_TANH_SINH_REFINEMENTS,
            DoubleExponentialCore.DEFAULT_MIN_COMPLEMENT,
            first
        );
        DETable cached = first.refine;

        DEPool second = new DEPool();
        DoubleExponentialCore.tanhSinh(
            x -> x * x,
            0.0,
            1.0,
            1.0e-10,
            DoubleExponentialCore.DEFAULT_TANH_SINH_REFINEMENTS,
            DoubleExponentialCore.DEFAULT_MIN_COMPLEMENT,
            second
        );

        assertNotNull(cached);
        assertSame(cached, second.refine);
    }

    @Test
    void defaultExpSinhRefineIsSharedAcrossFreshPools() {
        DEPool first = new DEPool();
        DoubleExponentialCore.expSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            DoubleExponentialCore.DEFAULT_HALF_LINE_REFINEMENTS,
            first
        );
        DETable cached = first.refine;

        DEPool second = new DEPool();
        DoubleExponentialCore.expSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            DoubleExponentialCore.DEFAULT_HALF_LINE_REFINEMENTS,
            second
        );

        assertNotNull(cached);
        assertSame(cached, second.refine);
    }

    @Test
    void endpointLogarithmicBuilderUsesExplicitDeWorkspace() {
        EndpointSingularIntegral integral = new EndpointSingularIntegral(EndpointOpts.LOG_LEFT)
            .function(x -> 1.0)
            .bounds(0.0, 1.0)
            .exponents(0.0, 0.0)
            .tolerances(1.0e-10, 1.0e-10);
        DEPool pool = new DEPool();

        Quadrature first = integral.integrateLogarithmic(pool);
        DETable cached = pool.refine;
        Quadrature second = integral.integrateLogarithmic(pool);

        assertNotNull(cached);
        assertSame(cached, pool.refine);
        assertEquals(first.getValue(), second.getValue(), 0.0);
        assertEquals(first.getEstimatedError(), second.getEstimatedError(), 0.0);
        assertEquals(first.getStatus(), second.getStatus());
    }

    @Test
    void poolSwitchesCachedRuleWhenRuleFamilyChanges() {
        DEPool pool = new DEPool();

        DoubleExponentialCore.expSinh(
            x -> Math.exp(-x),
            0.0,
            Double.POSITIVE_INFINITY,
            1.0e-10,
            9,
            pool
        );
        DETable expCached = pool.refine;
        assertSame(DEOpts.EXP_SINH, pool.opts);

        DoubleExponentialIntegral integral = new DoubleExponentialIntegral(DEOpts.TANH_SINH)
            .function(x -> x * x)
            .bounds(0.0, 2.0)
            .tolerance(1.0e-10)
            .maxRefinements(12);

        integral.integrate(pool);

        assertSame(DEOpts.TANH_SINH, pool.opts);
        assertNotSame(expCached, pool.refine);
    }
}