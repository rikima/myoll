package com.rikima.ml.oll;

import java.io.*;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.rikima.ml.*;

abstract public class OLLTrainer {
    public static boolean debug = false;
    public static final int EXAMPLE_N = 1000;

    public class Status {
        public double score;
        public double alpha;
        public double b;
        public String toString() {
            return String.format("score=%f\nalpha%f\nb=%f\n", this.score, this.alpha, this.b);
        }
    }
    
    // fields ----------
    protected int maxRound;

    public int exampleN;
    public MapWeightVector wv;

    protected double bias;
    protected double C;

    public Status status;

    // constructors ----

    public OLLTrainer(double c, double bias) {
        this.exampleN = EXAMPLE_N;
        this.C = c;
        this.bias = bias;
        this.wv = new MapWeightVector();

        this.status = new Status();
    }

    public OLLTrainer(Model m, double c2, double bias2) {
        this(c2, bias2);
        this.wv = m.weightVector();
    }

    // methods ---------
    // abstract public OLLClassifier getClassifier();
    
    public void train(List<LabeledFeatureVector> traindata, int maxRound) {
        this.maxRound = maxRound;
        this.exampleN = traindata.size();
        int round = 0;
        while (round++ < maxRound) {

            for (LabeledFeatureVector fv : traindata) {
                trainExample(fv, round);
            }
            
            regularize(round);
        }
        
        int pp = 0, pn = 0, np = 0, nn = 0;
        for (LabeledFeatureVector fv : traindata) {
            
            double m = getMargin(fv);
            
            if (fv.y() > 0) {
                if (m > 0) {
                    pp++;
                } else {
                    pn++;
                }
            }
            else {
                if (m > 0) {
                    nn++;
                } else {
                    np++;
                }
            }
        }
        
        System.out.println("---");
        System.out.println("# closed test");
        System.out.println("pp pn np nn :" + pp + " " + pn + " " + np + " " + nn);
        
        this.wv.slim();
        System.out.println("dim=" + this.wv.size());
    }
    
    
    public abstract void trainExample(LabeledFeatureVector fv, int round);
    
    public void regularize(int round) {
    }
    
    protected void update(FeatureVector fv, double alpha) {
        for (Feature f : fv) {
            wv.increment(f.id(), f.val() * alpha);
        }
        //b += alpha * bias;
    }
    
    public double getMargin(LabeledFeatureVector fv) {
        return fv.y() * this.wv.score(fv) + this.bias;
    }
    
    public Model getModel() {
        Model m = new Model(this.wv, this.bias);
        return m;
    }

    public OLLClassifier getClassifier() {
        return new OLLClassifier(this.getModel());
    }

    /*
    public void outputWeightVector(String fname, MapWeightVector weightVector) throws IOException {
        System.out.println("output weight vector to " + fname + " ...");
        
        BufferedWriter bw 
            = new BufferedWriter( new OutputStreamWriter( new FileOutputStream(fname), System.getProperty("file.encoding")));
        
        try {
            JSONObject jo = new JSONObject();
            jo.put("count", weightVector.size());
            jo.put("c", this.C);
            
            JSONArray ja = new JSONArray();
            Set<Map.Entry<Integer, Double>> eset = weightVector.getWeights().entrySet();
            for (Iterator<Map.Entry<Integer, Double>> iter = eset.iterator();iter.hasNext();) {
                Map.Entry<Integer, Double> me = iter.next();
                int wid = me.getKey();
                double val = me.getValue();
                
                if (Math.abs(val) > 1.0e-10) {
                    JSONObject fjo = new JSONObject();
                    fjo.put("rep", wid);
                    fjo.put("val", val);
                    
                    ja.put(fjo);
                }
            }
            
            jo.put("weights", ja);
            
            bw.write(jo.toString());
            bw.close();
            
            System.out.println(" .done");
        }
        catch (JSONException e) {
            e.printStackTrace();
        }
    }
    */
}
