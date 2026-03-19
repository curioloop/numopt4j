/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DTREXC reorders the real Schur factorization of a real matrix
 * A = Q * T * Q**T, so that the diagonal block of T with row index
 * {@code ifst} is moved to row {@code ilst}.
 *
 * <p>The reordering is performed by a sequence of orthogonal similarity
 * transformations via {@link Dlaexc}. Optionally, the orthogonal matrix
 * Q is updated.
 *
 * <p>On output, {@code work[0]} contains the final position of the moved
 * block (ifst after reordering) and {@code work[1]} contains the final
 * ilst position.
 *
 * <p>Reference: LAPACK Working Note, DTREXC.
 *
 * @see Dlaexc
 */
interface Dtrexc {

    static boolean dtrexc(boolean wantq, int n,
                          double[] t, int tOff, int ldt,
                          double[] q, int qOff, int ldq,
                          int ifst, int ilst, double[] work, int wOff) {
        if (n == 0) {
            work[wOff]     = ifst;
            work[wOff + 1] = ilst;
            return true;
        }

        if (n == 1) {
            work[wOff]     = ifst;
            work[wOff + 1] = ilst;
            return true;
        }

        int ifstWork = ifst;
        if (ifstWork > 0 && t[tOff + ifstWork * ldt + ifstWork - 1] != 0) {
            ifstWork--;
        }
        int nbf = 1;
        if (ifstWork + 1 < n && t[tOff + (ifstWork + 1) * ldt + ifstWork] != 0) {
            nbf = 2;
        }

        int ilstWork = ilst;
        if (ilstWork > 0 && t[tOff + ilstWork * ldt + ilstWork - 1] != 0) {
            ilstWork--;
        }
        int nbl = 1;
        if (ilstWork + 1 < n && t[tOff + (ilstWork + 1) * ldt + ilstWork] != 0) {
            nbl = 2;
        }

        boolean ok = true;

        if (ifstWork == ilstWork) {
            work[wOff]     = ifstWork;
            work[wOff + 1] = ilstWork;
            return true;
        }

        if (ifstWork < ilstWork) {
            if (nbf == 2 && nbl == 1) ilstWork--;
            if (nbf == 1 && nbl == 2) ilstWork++;

            int here = ifstWork;
            while (here < ilstWork) {
                if (nbf == 1 || nbf == 2) {
                    int nbnext = 1;
                    if (here + nbf + 1 < n && t[tOff + (here + nbf + 1) * ldt + here + nbf] != 0) {
                        nbnext = 2;
                    }

                    ok = Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here, nbf, nbnext, work, wOff);
                    if (!ok) {
                        work[wOff]     = ifstWork;
                        work[wOff + 1] = here;
                        return false;
                    }

                    here += nbnext;

                    if (nbf == 2 && t[tOff + (here + 1) * ldt + here] == 0) {
                        nbf = 3;
                    }
                    continue;
                }

                int nbnext = 1;
                if (here + 3 < n && t[tOff + (here + 3) * ldt + here + 2] != 0) {
                    nbnext = 2;
                }

                ok = Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here + 1, 1, nbnext, work, wOff);
                if (!ok) {
                    work[wOff]     = ifstWork;
                    work[wOff + 1] = here;
                    return false;
                }

                if (nbnext == 1) {
                    Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here, 1, nbnext, work, wOff);
                    here++;
                    continue;
                }

                if (t[tOff + (here + 2) * ldt + here + 1] == 0) {
                    nbnext = 1;
                }

                if (nbnext == 2) {
                    ok = Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here, 1, nbnext, work, wOff);
                    if (!ok) {
                        work[wOff]     = ifstWork;
                        work[wOff + 1] = here;
                        return false;
                    }
                } else {
                    Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here, 1, 1, work, wOff);
                    Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here + 1, 1, 1, work, wOff);
                }

                here += 2;
            }

            work[wOff]     = ifstWork;
            work[wOff + 1] = here;
            return true;

        } else {
            int here = ifstWork;
            while (here > ilstWork) {
                int nbnext = 1;
                if (here >= 2 && t[tOff + (here - 1) * ldt + here - 2] != 0) {
                    nbnext = 2;
                }

                if (nbf == 1 || nbf == 2) {
                    ok = Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here - nbnext, nbnext, nbf, work, wOff);
                    if (!ok) {
                        work[wOff]     = ifstWork;
                        work[wOff + 1] = here;
                        return false;
                    }

                    here -= nbnext;

                    if (nbf == 2 && t[tOff + (here + 1) * ldt + here] == 0) {
                        nbf = 3;
                    }
                    continue;
                }

                ok = Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here - nbnext, nbnext, 1, work, wOff);
                if (!ok) {
                    work[wOff]     = ifstWork;
                    work[wOff + 1] = here;
                    return false;
                }

                if (nbnext == 1) {
                    Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here, nbnext, 1, work, wOff);
                    here--;
                    continue;
                }

                if (t[tOff + here * ldt + here - 1] == 0) {
                    nbnext = 1;
                }

                if (nbnext == 2) {
                    ok = Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here - 1, 2, 1, work, wOff);
                    if (!ok) {
                        work[wOff]     = ifstWork;
                        work[wOff + 1] = here;
                        return false;
                    }
                } else {
                    Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here, 1, 1, work, wOff);
                    Dlaexc.dlaexc(wantq, n, t, tOff, ldt, q, qOff, ldq, here - 1, 1, 1, work, wOff);
                }

                here -= 2;
            }

            work[wOff]     = ifstWork;
            work[wOff + 1] = here;
            return true;
        }
    }
}
