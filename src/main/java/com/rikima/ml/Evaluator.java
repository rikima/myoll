package com.rikima.ml;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by mrikitoku on 15/08/31.
 */
public class Evaluator {
    List<Integer> expected;
    List<Integer> actual;
    private int pp = 0;
    private int pn = 0;
    private int np = 0;
    private int nn = 0;

    public Evaluator() {
        this.expected = new ArrayList<Integer>();
        this.actual   = new ArrayList<Integer>();
    }

    public void setResult(int expectedY, int actualY) {
        this.expected.add(expectedY);
        this.actual.add(actualY);
    }

    public void printResult() {
        int size = this.expected.size();
        for (int i = 0; i < size; ++i) {
            int ey = this.expected.get(i);
            int ay = this.actual.get(i);

            if (ey * ay > 0) {
                if (ey > 0) {
                    pp++;
                } else {
                    nn++;
                }
            } else {
                if (ey > 0) {
                    np++;
                } else {
                    pn++;
                }
            }
        }

        System.out.println(String.format("PP: %d PN: %d NP: %d NN: %d", this.pp, this.pn, this.np, this.nn));
        double acc = (pp) / (double)(pp + np);
        System.out.println(String.format("acc: %f", acc));

        double prec = (pp) / (double)(pp + pn);
        System.out.println(String.format("prec: %f", prec));

        double f1 = 2.0 * acc * prec / (acc + prec);
        System.out.println(String.format("F1: %f", f1));
    }
}
