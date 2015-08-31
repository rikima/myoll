package com.rikima.ml.oll.pa;

import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.oll.OLLTrainer;

// passive agressive-II
public class PA2Trainer extends OLLTrainer {
    // fields ---
    private int exampleN;

    // constructors --

    public PA2Trainer(double c, double bias) {
        super(c, bias);
	}

    // methods ---
    public void trainExample(LabeledFeatureVector fv, int round) {
        double score = super.getMargin(fv);
        double alpha = fv.y() * (1.0 - score) / (fv.getNorm() + 1.0 / 2.0 / C);

        if (debug) {
            this.status.score = score;
            this.status.alpha = alpha;

            System.out.println("---");
            System.out.println("trainExample#" + exampleN);
            System.out.println(this.status.toString());
        }

        if (score <= 1.0){
            super.update(fv, fv.y() * (1.0 - score) / (fv.getNorm() + 1.0 / 2.0 / C));
        }
        exampleN++;
    }
}
