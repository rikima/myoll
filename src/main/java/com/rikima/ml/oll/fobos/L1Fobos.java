package com.rikima.ml.oll.fobos;

import java.util.Iterator;
import java.util.Map;

import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.MapWeightVector;
import com.rikima.ml.Model;
import com.rikima.ml.oll.OLLTrainer;

abstract public class L1Fobos extends OLLTrainer {
    static boolean debug = false;

    // fields ---------------
    protected double lambda = Double.NaN;
    
    int unit = 100;
    
    // constructor ----------
    public L1Fobos(double c, double bias) {
        super(c, bias);
        this.wv = new MapWeightVector();
        this.lambda = c;
    }
    
    public L1Fobos(Model m, double c, double b) {
        this(c,b);
        this.wv = m.weightVector();
    }

    // methods --------------
    protected double get_eta(int round) {
        return 1.0 / (1.0 + (double)round / exampleN);
    }
    
    abstract protected void update(LabeledFeatureVector fv, int round);
    
    public void trainExample(LabeledFeatureVector fv, int round) {
        if (debug) {
            double m = getMargin(fv);
            System.out.println("pre margin:" + m);
        }
        update(fv, round);
        //if (this.iter > 0 && this.iter % unit == 0) {
        //	l1regularize(round);
        //}
        if (debug) {
            double m = getMargin(fv);
            System.out.println("post margin:" + m);
        }
    }
    
    @Override
    public void regularize(int round) {
        this.l1regularize(round);
    }
    
    // private methods --------
    protected void l1regularize(int round) {
        double lambda_hat = get_eta(round) * lambda;
        
        Iterator<Map.Entry<Integer, Double>> witer = wv.getWeights().entrySet().iterator();
        while (witer.hasNext()) {
            Map.Entry<Integer, Double> me = witer.next();
            
            int wid = me.getKey();
            double w_i = me.getValue();
            
            double abs_w_i = Math.abs(w_i);
            
            double new_w_i = 0.0;
            if ((abs_w_i - lambda_hat) > 0.0) {
                new_w_i = abs_w_i - lambda_hat;
                if (w_i < 0.0) {
                    new_w_i *= -1;
                }
            }
            
            this.wv.set(wid, (float)new_w_i);
        }
        
        this.wv.slim();
    }
}
