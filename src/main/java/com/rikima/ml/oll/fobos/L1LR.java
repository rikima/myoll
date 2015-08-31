package com.rikima.ml.oll.fobos;

import com.rikima.ml.Feature;
import com.rikima.ml.LabeledFeatureVector;

public class L1LR extends L1Fobos {
    public static boolean debug =true;

    // constructor -----------
    public L1LR(double c, double bias) {
        super(c, bias);
    }
    
    // methods ---------------
    protected void update(LabeledFeatureVector fv, int round) {
        double eta = get_eta(round);
        double g_coef = 2.0 * (1.0 - prob(fv));
        for (Feature f : fv) {
            float v = (float)(f.val() * fv.y() * g_coef * eta);
            super.wv.increment(f.id(), v);
        }
    }
    
    private double prob(LabeledFeatureVector fv) {
        double p = 1.0 / (1.0 + Math.exp(-2.0 * getMargin(fv)));
        return p;
    }
}
