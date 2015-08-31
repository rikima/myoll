package com.rikima.ml.oll;

import com.rikima.ml.Feature;
import com.rikima.ml.FeatureVector;
import com.rikima.ml.Model;

import java.util.ArrayList;

public class OLLClassifier {
    public static final int POSITIVE = 1;
    public static final int NEGATIVE = -1;
    
    // fields ----------
    private Model model;
    
    private ArrayList<Feature> positiveHits;
    private ArrayList<Feature> negativeHits;
    
    // constructor -----
    public OLLClassifier(Model model) {
        this.model = model;
        
        this.positiveHits = new ArrayList<Feature>();
        this.negativeHits = new ArrayList<Feature>();
    }

    // methods ---------
    public int classify(FeatureVector fv) {
        double s = this.model.score(fv);
        if (s > 0) {
            return POSITIVE;
        } else {
            return NEGATIVE;
        }
    }
    
    public void clear() {
        this.positiveHits.clear();
        this.negativeHits.clear();
    }
    
    public double score(FeatureVector fv) {
        return this.model.score(fv);
    }
    
    public double bias() {
        return this.model.bias();
    }
    
    public ArrayList<Feature> getPositiveHitInfos() {
    	return this.positiveHits;
    }
    
    public ArrayList<Feature> getNegativeHitInfos() {
    	return this.negativeHits;
    }

    /*
	public double scoreWithInfo(FeatureVector fv) {
		return this.model.scoreWithInfo(fv, this.positiveHits, this.negativeHits);
	}
    */
	
    /*
    public int classifyWithInfo(FeatureVector fv) {
    double s = this.model.scoreWithInfo(fv, this.positiveHits, this.negativeHits);
    if (s >= 0) {
    return POSITIVE;
    }
    else {
    return NEGATIVE;
    }
    }
    */
}
