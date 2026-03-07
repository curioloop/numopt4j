/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg;

public interface Decomposition {

    /**
     * A lightweight matrix view carrying dimension metadata alongside the underlying data array.
     *
     * <p>When {@code complex=false}, {@code data} is a compact row-major array of length {@code m*n}.
     * When {@code complex=true}, {@code data} is an interleaved array of length {@code m*n*2},
     * where each element occupies two consecutive doubles (real part, imaginary part).
     */
    final class Matrix {

        public final int m, n;
        public final boolean complex;
        public final double[] data;

        public Matrix(int m, int n, boolean complex, double[] data) {
            int minLen = complex ? m * n * 2 : m * n;
            if (data.length < minLen)
                throw new IllegalArgumentException(
                    "data.length=" + data.length + " < required " + minLen);
            this.m = m;
            this.n = n;
            this.complex = complex;
            this.data = data;
        }

        /** Returns the real element at (row, col). Only valid when {@code complex=false}. */
        public double get(int row, int col) {
            return data[row * n + col];
        }

        /** Returns the real part of the complex element at (row, col). Only valid when {@code complex=true}. */
        public double real(int row, int col) {
            return data[(row * n + col) * 2];
        }

        /** Returns the imaginary part of the complex element at (row, col). Only valid when {@code complex=true}. */
        public double imag(int row, int col) {
            return data[(row * n + col) * 2 + 1];
        }
    }

    boolean ok();

    Workspace work();

    class Workspace {

        protected double[] work;
        protected int[] iwork;
        protected boolean[] bwork;

        protected Workspace() {}

        public double[] work() {
            return work;
        }

        public int[] iwork() {
            return iwork;
        }

        public boolean[] bwork() {
            return bwork;
        }

        public Workspace ensureWork(int minSize) {
            if (work == null || work.length < minSize) {
                work = new double[minSize];
            }
            return this;
        }

        public Workspace ensureIwork(int minSize) {
            if (iwork == null || iwork.length < minSize) {
                iwork = new int[minSize];
            }
            return this;
        }

        public Workspace ensureBwork(int minSize) {
            if (bwork == null || bwork.length < minSize) {
                bwork = new boolean[minSize];
            }
            return this;
        }
    }
}
