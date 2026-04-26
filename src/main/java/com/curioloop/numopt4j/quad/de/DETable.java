package com.curioloop.numopt4j.quad.de;

/** Segmented DE table payload with optional complement metadata. */
final class DETable {

    private static final int[] EMPTY_INT_ARRAY = {};

    final double[] abscissas;
    final double[] weights;
    final int[] rows;
    final int[] complements;

    DETable(double[] abscissas, double[] weights, int[] rows) {
        this(abscissas, weights, rows, EMPTY_INT_ARRAY);
    }

    DETable(double[] abscissas, double[] weights, int[] rows, int[] complements) {
        this.abscissas = abscissas;
        this.weights = weights;
        this.rows = rows;
        this.complements = complements;
    }

    int rowCount() {
        return rows.length >>> 1;
    }

    int rowStart(int row) {
        return rows[row << 1];
    }

    int rowLength(int row) {
        return rows[(row << 1) + 1];
    }

    static double abscissa(DETable base, DETable table, int index) {
        int split = base.abscissas.length;
        return index < split ? base.abscissas[index] : table.abscissas[index - split];
    }

    static double weight(DETable base, DETable table, int index) {
        int split = base.weights.length;
        return index < split ? base.weights[index] : table.weights[index - split];
    }

    static int firstComplement(DETable base, DETable table, int row) {
        int baseRows = base.rowCount();
        if (row < baseRows) {
            return base.complements[row];
        }
        return table.complements[row - baseRows];
    }
}