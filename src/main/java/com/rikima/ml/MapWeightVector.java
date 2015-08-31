package com.rikima.ml;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

//import cern.colt.map.OpenIntDoubleHashMap;

public class MapWeightVector implements Serializable {
    private static final long serialVersionUID = -2334953763714111002L;
    // fields -------
    //private OpenIntDoubleHashMap wetights2;
    private Map<Integer, Double> weights;
    
    // constructor ---
    public MapWeightVector() {
        this.weights   = new HashMap<Integer, Double>();
        //this.wetights2 = new OpenIntDoubleHashMap();
    }

    // methods -------
    public int size() {
        return this.weights.size();
    }
    
    public void slim() {
        this.slim(1.0e-10);
    }
    
    public void slim(double epsilon) {
        HashSet<Integer> buf = new HashSet<Integer>();
        Set<Map.Entry<Integer, Double>> eset = this.weights.entrySet();
        
        for (Iterator<Map.Entry<Integer, Double>> iter = eset.iterator();iter.hasNext();) {
            Map.Entry<Integer, Double> me = iter.next();
            int wid = me.getKey();
            double val = me.getValue();
            if (Math.abs(val) < epsilon) {	
                buf.add(wid);
            }
        }
        // remove
        for (Iterator<Integer> iter = buf.iterator();iter.hasNext();) {
            this.weights.remove(iter.next());
        }
    }

    public void increment(int id, double val) {
        if (this.weights.containsKey(id)) {
            this.weights.put(id, this.weights.get(id) + val);
        } else {
            this.weights.put(id, val);
        }
    }

    public void set(int id, double val) {
        this.weights.put(id, val);
    }

    public double getOrCreate(int id) {
        if (!this.weights.containsKey(id)) {
            this.weights.put(id, 0.0);
        }
        return this.weights.get(id);
    }

    public double get(int id) {
        if (!this.weights.containsKey(id)) {
            return 0;
        }
        return this.weights.get(id);
    }

    public void remove(int id) {
        this.weights.remove(id);
    }

    public double score(FeatureVector fv) {
        double ret = 0;
        for (Feature f : fv) {
            ret += this.get(f.id()) * f.val();
        }
        return ret;
    }

    public Map<Integer, Double> getWeights() {
    	return this.weights;
    }

    public Map<String, Double> translateForJson() {
        Map<String, Double> weights = new HashMap<String, Double>();
        for (Iterator<Entry<Integer, Double>> iter = this.weights.entrySet().iterator(); iter.hasNext(); ) {
            Entry<Integer, Double> e = iter.next();
            int fid = e.getKey();
            double val = e.getValue();
            // そのまま素性化
            String feature = Integer.toString(fid);
            weights.put(feature, val);
        }
        return weights;
    }
}
