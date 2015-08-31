package com.rikima.ml.oll.pa;

import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.oll.OLLTrainer;

// passive agressive
public class PATrainer extends OLLTrainer {
    // fields ---------
    private int exampleN = 0;;
    
    // constructors ---
    public PATrainer(double c, double bias) {
        super(c, bias);
    }
    
    // methods --------
    public void trainExample(LabeledFeatureVector fv, int round) {
        double score = fv.y() * super.wv.score(fv) + bias;
        double alpha = fv.y() * (1.0 - score) / fv.getNorm();
        
        if (debug) {
            this.status.score = score;
            this.status.alpha = alpha;
            
            System.out.println("---");
            System.out.println("trainExample#" + exampleN);
            System.out.println(this.status.toString());
            System.out.println("norm=" + fv.getNorm());
        }

        if (score <= 1.0){
            super.update(fv, fv.y() * (1.0 - score) / fv.getNorm());
        }
        exampleN++;
    }
}
