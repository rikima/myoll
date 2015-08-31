package com.rikima.ml.oll.cw;

import java.util.ArrayList;

import com.rikima.ml.*;
import com.rikima.ml.oll.OLLTrainer;

// Confidence-Weighted
public class CWTrainer extends OLLTrainer {
    static boolean debug = false;
    
    class Status {
        public double gamma;
        public double score;
        public double var;
        public double b;

        public String toString() {
            return String.format("score=%f\nvar=%f\nb=%f\ngamma=%f\n", this.score, this.var, this.b, this.gamma);
        }
    }
    
    // fields ---------------
    private ArrayList<Double> cov;
    private double covb;
    //private double b;
    public int exampleN;
    protected Status status = new Status();

    // constructor ----------
    public CWTrainer(double c, double bias) {
        super(c, bias);
        this.cov = new ArrayList<Double>();
    }

    // methods --------------
    public void trainExample(LabeledFeatureVector fv, int round) {
        double score = super.getMargin(fv);
        double var = getVariance(fv);

        double b     = 1.0 + 2.0 * C * score;
        double gamma = (-b + Math.sqrt(b * b - 8.0 * C * (score - C * var))) / (4.0 * C * var);

        if (debug) {
            this.status.score = score;
            this.status.var = var;
            this.status.b = b;
            this.status.gamma = gamma;
        
            this.status.score = score;
            this.status.var = var;
            this.status.b = b;
            this.status.gamma = gamma;

            System.out.println("---");
            System.out.println("trainExample#" + exampleN);
            System.out.println(this.status.toString());

        }

        if (gamma > 0){
            this.update(fv, gamma);
        }
        this.exampleN++;

    }

    // private methods --------
    private double getVariance(FeatureVector fv) {
        double ret = 0.0;
        for (Feature f : fv) {
            if (this.cov.size() <= f.index()) {
                ret += 1.0 * f.val() * f.val(); // assume cov[fv[i].first=1
            } else {
                ret += this.cov.get(f.index()) * f.val() * f.val();
            }
        }
        return ret;
    }
    
    protected void update(LabeledFeatureVector fv, double alpha) {
        for (Feature f : fv) {
            int prevSize = cov.size();
            if (cov.size() <= f.index()+1) {
                for (int j = prevSize; j < f.index()+1; ++j) {
                    cov.add(1.0);
                }
            }
            
            double prev_w = super.wv.getOrCreate(f.id());
            super.wv.set(f.id(), prev_w +  f.val() * alpha * fv.y() * cov.get(f.index()) ) ;

            cov.set(f.index(), 1.0 / (1.0 / cov.get(f.index()) + 2.0 * alpha * C * f.val()  * f.val()));
        }

        //b += alpha * fv.y() * covb * bias;
        this.covb = 1.0 / (1.0 / covb + 2.0 * alpha * C * bias * bias);
    }
}
