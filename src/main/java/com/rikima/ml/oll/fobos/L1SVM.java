package com.rikima.ml.oll.fobos;

//import com.rikima.ml.*;
import com.rikima.ml.Feature;
import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.Model;

public class L1SVM extends L1Fobos {
    public static boolean debug =true;

    // constructor -----------
    public L1SVM(double c, double bias) {
        super(c, bias);
    }
    
    public L1SVM(Model m, double c, double b) {
        super(m, c, b);
    }

    // methods ---------------
    protected void update(LabeledFeatureVector fv, int round) {
        if (getMargin(fv) < 1.0) {
            double eta = get_eta(round);
            assert eta != Double.NaN;
            for (Feature f : fv) {
                double v = f.val() * fv.y() * eta;
                super.wv.increment(f.id(), v);
            }
        }
    }
}
