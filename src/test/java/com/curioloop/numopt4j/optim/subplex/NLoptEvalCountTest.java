/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.subplex;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;
import com.curioloop.numopt4j.optim.Univariate;

/**
 * Compares evaluation counts with NLopt SBPLX (center start, xtol=ftol=1e-6).
 */
public class NLoptEvalCountTest {

    private static double sqr(double x) { return x * x; }
    private static final double PI2 = 2*Math.PI, PI3 = 3*Math.PI, PI4 = 4*Math.PI;

    private static Bound[] box(double[] lb, double[] ub) {
        Bound[] b = new Bound[lb.length];
        for (int i = 0; i < lb.length; i++) b[i] = Bound.between(lb[i], ub[i]);
        return b;
    }

    private static double[] mid(double[] lb, double[] ub) {
        double[] x = new double[lb.length];
        for (int i = 0; i < lb.length; i++) x[i] = 0.5*(lb[i]+ub[i]);
        return x;
    }

    private static void test(String name, Univariate.Objective f,
                             double[] lb, double[] ub, int maxEval,
                             int nloptEvals, double nloptCost) {
        double[] x0 = mid(lb, ub);
        Bound[] bounds = box(lb, ub);
        Optimization r = Minimizer.subplex()
                .objective(f).initialPoint(x0).bounds(bounds)
                .functionTolerance(1e-6).parameterTolerance(1e-6)
                .maxEvaluations(maxEval).solve();
        String cmp = r.getEvaluations() <= nloptEvals ? " OK " : "SLOW";
        System.out.printf("%-25s %2d  %6d / %-6d %-14.6e %-14.6e [%s]%n",
                name, x0.length, r.getEvaluations(), nloptEvals,
                r.getCost(), nloptCost, cmp);
    }

    public static void main(String[] args) {
        System.out.printf("%-25s %2s  %14s  %-14s %-14s%n",
                "Function", "N", "Java / NLopt", "Java cost", "NLopt cost");
        for (int i = 0; i < 95; i++) System.out.print("-");
        System.out.println();

        // 1D
        test("1D Oscillating", (x, n) -> { double y=x[0]-1.23456789; return sqr(y*0.1)-Math.cos(y-2*Math.sin(3*y)); },
                new double[]{-5}, new double[]{5}, 2000, 47, -0.765342);

        // 2D
        test("Rosenbrock", (x, n) -> 100*sqr(x[1]-x[0]*x[0])+sqr(1-x[0]),
                new double[]{-2,-2}, new double[]{2,2}, 5000, 175, 1.07607e-7);

        test("McCormic", (x, n) -> Math.sin(x[0]+x[1])+sqr(x[0]-x[1])-1.5*x[0]+2.5*x[1]+1,
                new double[]{-1.5,-3}, new double[]{4,4}, 5000, 73, 1.22837);

        test("Goldstein-Price", (x, n) -> {
            double a1=x[0]+x[1]+1,a2=19-14*x[0]+3*x[0]*x[0]-14*x[1]+6*x[0]*x[1]+3*x[1]*x[1];
            double b1=2*x[0]-3*x[1],b2=18-32*x[0]+12*x[0]*x[0]+48*x[1]-36*x[0]*x[1]+27*x[1]*x[1];
            return (1+a1*a1*a2)*(30+b1*b1*b2);
        }, new double[]{-2,-2}, new double[]{2,2}, 5000, 114, 3.0);

        test("Six-hump Camel", (x, n) -> 4*sqr(x[0])-2.1*Math.pow(x[0],4)+Math.pow(x[0],6)/3.0+x[0]*x[1]-4*sqr(x[1])+4*Math.pow(x[1],4),
                new double[]{-5,-5}, new double[]{5,5}, 5000, 99, -1.03163);

        test("Branin", (x, n) -> { double a=1-2*x[1]+0.05*Math.sin(PI4*x[1])-x[0],b=x[1]-0.5*Math.sin(PI2*x[0]); return sqr(a)+sqr(b); },
                new double[]{-10,-10}, new double[]{10,10}, 5000, 81, 0.102164);

        test("Shubert", (x, n) -> { double f=0; for(int j=1;j<=5;j++) for(int i=0;i<2;i++) f-=j*Math.sin((j+1)*x[i]+j); return f; },
                new double[]{-10,-10}, new double[]{10,10}, 10000, 95, -5.2611);

        test("Hansen", (x, n) -> { double a=0,b=0; for(int i=1;i<=5;i++) a+=i*Math.cos((i-1)*x[0]+i); for(int i=1;i<=5;i++) b+=i*Math.cos((i+1)*x[1]+i); return a*b; },
                new double[]{-10,-10}, new double[]{10,10}, 10000, 136, -176.542);

        // 3D
        test("Box-Betts", (x, n) -> { double f=0; for(int i=1;i<=10;i++){double e0=Math.exp(-0.1*i*x[0]),e1=Math.exp(-0.1*i*x[1]),e2=Math.exp(-0.1*i)-Math.exp(-1.0*i);f+=sqr(e0-e1-e2*x[2]);}return f; },
                new double[]{0.9,9,0.9}, new double[]{1.2,11.2,1.2}, 10000, 140, 0.000296913);

        // 4D
        test("Shekel m=5", shekelFunc(5), new double[]{0,0,0,0}, new double[]{10,10,10,10}, 20000, 177, -2.68286);
        test("Shekel m=7", shekelFunc(7), new double[]{0,0,0,0}, new double[]{10,10,10,10}, 20000, 177, -2.75193);
        test("Shekel m=10", shekelFunc(10), new double[]{0,0,0,0}, new double[]{10,10,10,10}, 20000, 176, -2.87114);

        test("Levy n=4", levyFunc(), new double[]{-10,-10,-10,-10}, new double[]{10,10,10,10}, 20000, 275, -20.5025);

        test("Corner 4D", (x, n) -> { double u=x[0]+x[1]*x[2]*Math.sin(2*x[3]),v=x[0]+2*Math.sin(u); return 1+v*v+0.1*(x[1]+x[2]+x[3]); },
                new double[]{0,0,0,0}, new double[]{1,1,1,1}, 10000, 90, 1.0);

        test("Side 4D", (x, n) -> {
            double x0=+0.4977*x[0]-0.3153*x[1]-0.5066*x[2]-0.4391*x[3],x1=-0.3153*x[0]+0.3248*x[1]-0.4382*x[2]-0.4096*x[3];
            double x2=-0.5066*x[0]-0.4382*x[1]+0.3807*x[2]-0.4543*x[3],x3=-0.4391*x[0]-0.4096*x[1]-0.4543*x[2]+0.5667*x[3];
            return -1.0/(x0*x0+0.01)-1.0/(x1*x1+0.04)-1.0/(x2*x2+0.09)-1.0/(x3*x3+0.16);
        }, new double[]{0.1,-1,-1,-1}, new double[]{1,1,1,1}, 20000, 1080, -141.284);

        // 5-7D
        test("Levy n=5", levyFunc(), new double[]{-5,-5,-5,-5,-5}, new double[]{5,5,5,5,5}, 30000, 261, -10.5048);
        test("Levy n=7", levyFunc(), new double[]{-5,-5,-5,-5,-5,-5,-5}, new double[]{5,5,5,5,5,5,5}, 50000, 447, -11.5044);

        // 10D
        double[] p10l=new double[10],p10u=new double[10]; for(int i=0;i<10;i++){p10l[i]=2.001;p10u[i]=9.999;}
        test("Paviani", (x, n) -> { double f=0,p=1; for(int i=0;i<10;i++){f+=sqr(Math.log(x[i]-2))+sqr(Math.log(10-x[i]));p*=x[i];}return f-Math.pow(p,0.2); },
                p10l, p10u, 100000, 540, -45.7785);

        double[] g10l=new double[10],g10u=new double[10]; for(int i=0;i<10;i++){g10l[i]=-500;g10u[i]=600;}
        test("Griewank", (x, n) -> { double f=1,p=1; for(int i=0;i<x.length;i++){f+=sqr(x[i])*0.00025;p*=Math.cos(x[i]/Math.sqrt(i+1.0));}return f-p; },
                g10l, g10u, 100000, 605, 0.0344921);

        test("Convex Cosh", (x, n) -> { double f=1; for(int i=0;i<x.length;i++) f*=Math.cosh((x[i]-i)*(i+1)); return f; },
                new double[]{-1,0,0,0,0,0,0,0,0,0}, new double[]{2,3,6,7,8,10,11,13,14,16}, 100000, 610, 1.0);

        // 30D
        double[] r30l=new double[30],r30u=new double[30]; for(int i=0;i<30;i++){r30l[i]=-30;r30u[i]=30;}
        test("Gen.Rosenbrock 30D", (x, n) -> { double f=0; for(int i=0;i<29;i++) f+=100*sqr(x[i+1]-x[i]*x[i])+sqr(1-x[i]); return f; },
                r30l, r30u, 500000, 19791, 23.277);
    }

    private static Univariate.Objective shekelFunc(int m) {
        final double[][] A={{4,4,4,4},{1,1,1,1},{8,8,8,8},{6,6,6,6},{3,7,3,7},{2,9,2,9},{5,5,3,3},{8,1,8,1},{6,2,6,2},{7,3.6,7,3.6}};
        final double[] c={.1,.2,.2,.4,.4,.6,.3,.7,.5,.5};
        return (x, n) -> { double f=0; for(int i=0;i<m;i++) f-=1.0/(c[i]+sqr(x[0]-A[i][0])+sqr(x[1]-A[i][1])+sqr(x[2]-A[i][2])+sqr(x[3]-A[i][3])); return f; };
    }

    private static Univariate.Objective levyFunc() {
        return (x, n) -> { int len=x.length; double a=x[len-1]-1,b=1+sqr(Math.sin(PI2*x[len-1])); double f=sqr(Math.sin(PI3*x[0]))+a*b;
            for(int i=0;i<len-1;i++){double ai=x[i]-1,bi=1+sqr(Math.sin(PI3*x[i+1]));f+=sqr(ai)*bi;} return f; };
    }
}
