/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import java.lang.invoke.*;

/**
 * Fused Multiply-Add (FMA) operator wrapped via LambdaMetafactory.
 *
 * <h3>FMA Advantages</h3>
 * <ul>
 *   <li><b>Higher Precision</b>: FMA computes (a × b + c) with only one rounding at the end,
 *       whereas separate multiply and add operations round twice, reducing accumulated error.</li>
 *   <li><b>Better Performance</b>: Modern CPUs (x86 FMA3/FMA4, ARM NEON) have dedicated FMA
 *       instructions that execute multiply-add in a single cycle.</li>
 *   <li><b>IEEE 754-2008 Compliant</b>: Math.fma guarantees correctly-rounded results per
 *       the IEEE 754-2008 standard.</li>
 * </ul>
 *
 * <h3>Why LambdaMetafactory?</h3>
 * <p>Using LambdaMetafactory to wrap Math.fma allows the JIT compiler to inline the call
 * directly, achieving performance equivalent to a direct method call. This avoids the
 * overhead of MethodHandle.invokeExact() which may not inline as effectively.</p>
 *
 * <h3>Fallback Behavior</h3>
 * <p>If Math.fma is not available (pre-Java 9), falls back to {@code a * b + c} which
 * has slightly lower precision but compatible behavior.</p>
 */
@FunctionalInterface
public interface FMA {

    /**
     * Computes fused multiply-add: a × b + c
     *
     * @param a first multiplicand
     * @param b second multiplicand
     * @param c addend
     * @return a × b + c (single rounding)
     */
    double fma(double a, double b, double c);

    /**
     * Singleton FMA instance, created via LambdaMetafactory for optimal JIT inlining.
     */
    FMA INSTANCE = createInstance();

     static FMA createInstance() {
        try {
            MethodHandles.Lookup lookup = MethodHandles.lookup();
            MethodHandle fmaHandle = lookup.findStatic(
                    Math.class, "fma",
                    MethodType.methodType(double.class, double.class, double.class, double.class)
            );
            CallSite site = LambdaMetafactory.metafactory(
                    lookup,
                    "fma",
                    MethodType.methodType(FMA.class),
                    MethodType.methodType(double.class, double.class, double.class, double.class),
                    fmaHandle,
                    fmaHandle.type()
            );
            return (FMA) site.getTarget().invokeExact();
        } catch (Throwable t) {
            // Fallback: a * b + c (two roundings, slightly less precise)
            return (a, b, c) -> a * b + c;
        }
    }

    static double op(double a, double b, double c) {
        return INSTANCE.fma(a, b, c);
    }
    
}
