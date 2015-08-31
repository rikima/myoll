package com.rikima.ml.oll.pa;

import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.oll.OLLTrainer;

// passive agressive-I
public class PA1Trainer extends OLLTrainer {
    //fields -------
    private int exampleN = 0;

    // constructor ---
    public PA1Trainer(double c, double bias) {
    	super(c, bias);
    }

    // mehtods -----

    public void trainExample(LabeledFeatureVector fv, int round) {
        double score = super.getMargin(fv);
        double alpha = fv.y() * Math.min(C, (1.0 - score) / fv.getNorm());
        
        if (debug) {
            this.status.score = score;
            this.status.alpha = alpha;
            
            System.out.println("---");
            System.out.println("trainExample#" + exampleN);
            System.out.println(this.status.toString());
        }
        
        if (score <= 1.0){
            super.update(fv, fv.y() * Math.min(C, (1.0 - score) / fv.getNorm()));
        }
        exampleN++;
    }
}
