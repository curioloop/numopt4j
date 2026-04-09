/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode;

/**
 * Supported ODE initial value problem solver methods.
 *
 * <ul>
 *   <li>{@link #RK23} — Bogacki-Shampine 3(2) explicit RK, suitable for non-stiff problems.</li>
 *   <li>{@link #RK45} — Dormand-Prince 5(4) explicit RK (default), good general-purpose choice.</li>
 *   <li>{@link #DOP853} — Hairer's 8(5,3) explicit RK, high accuracy for smooth problems.</li>
 *   <li>{@link #BDF} — Backward Differentiation Formula (variable order 1–5), for stiff problems.</li>
 *   <li>{@link #Radau} — Radau IIA implicit RK (5th order, 3 stages), for stiff problems.</li>
 * </ul>
 */
public enum IVPMethod {
    RK23, RK45, DOP853, BDF, Radau
}
