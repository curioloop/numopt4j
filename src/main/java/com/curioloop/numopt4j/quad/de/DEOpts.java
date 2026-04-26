/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.de;

/**
 * Rule selection for double-exponential quadrature.
 *
 * <ul>
 *   <li>{@link #TANH_SINH} — {@code z(t) = tanh((pi/2) sinh(t))} for finite intervals and
 *       rationally mapped infinite bounds</li>
 *   <li>{@link #EXP_SINH} — {@code x(t) = exp((pi/2) sinh(t))} for half-infinite intervals</li>
 *   <li>{@link #SINH_SINH} — {@code x(t) = sinh((pi/2) sinh(t))} for the whole real line</li>
 * </ul>
 */
public enum DEOpts {
    TANH_SINH(new DETable(
        DoubleExponentialsTables.TANH_SINH_ABSCISSAS,
        DoubleExponentialsTables.TANH_SINH_WEIGHTS,
        DoubleExponentialsTables.TANH_SINH_ROWS,
        DoubleExponentialsTables.TANH_SINH_FIRST_COMPLEMENTS
    )),
    EXP_SINH(new DETable(
        DoubleExponentialsTables.EXP_SINH_ABSCISSAS,
        DoubleExponentialsTables.EXP_SINH_WEIGHTS,
        DoubleExponentialsTables.EXP_SINH_ROWS
    )),
    SINH_SINH(new DETable(
        DoubleExponentialsTables.SINH_SINH_ABSCISSAS,
        DoubleExponentialsTables.SINH_SINH_WEIGHTS,
        DoubleExponentialsTables.SINH_SINH_ROWS
    ));

    private final DETable baseTable;

    DEOpts(DETable baseTable) {
        this.baseTable = baseTable;
    }

    DETable baseTable() {
        return baseTable;
    }
}