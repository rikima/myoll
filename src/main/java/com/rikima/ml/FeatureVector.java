package com.rikima.ml;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.SortedSet;

public class FeatureVector implements Iterable<Feature> {
    // fields ---------------------------
    protected List<Feature> features;
    
    // constructors ---------------------
    public FeatureVector(SortedSet<Feature> src) {
        this.features = new ArrayList<Feature>();
        this.features.addAll(src);
    }
    
    // methods -------------------------
    public int size() {
        return this.features.size();
    }
    
    public Feature getFeature(int index) {
    	return this.features.get(index);
    }
    
    public double dot(FeatureVector fv) {
        double v = 0;
        int idx1 = 0;
        int idx2 = 0;
        while (idx1 < this.size() && idx2 < fv.size()) {
            Feature f1 = this.features.get(idx1);
            Feature f2 = fv.getFeature(idx2);
            
            if (f1.id == f2.id) {
                v += f1.val() * f2.val();
                idx1++;
                idx2++;
            } else if (f1.id > f2.id) {
                idx2++;
            } else {
                idx1++;
            }
        }
        return v;
    }

    public double getNorm() {
        double s = 0;
        for (Feature f : this.features) {
            s += f.val * f.val;
        }
        return s;
    }
    
    public Iterator<Feature> iterator() {
        return this.features.iterator();
    }

    public boolean check() {
        int prev = 0;
        for (Feature f : this.features) {
            if (prev >= f.id) {
                return false;
            }
            prev = f.id;
        }
        return true;
    }
    
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Feature f : this.features) {
            sb.append(" " + f.toString());
        }
        return sb.toString();
    }
}
