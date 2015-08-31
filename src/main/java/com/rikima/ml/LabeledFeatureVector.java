package com.rikima.ml;

import java.util.SortedSet;

public class LabeledFeatureVector extends FeatureVector {
    private int y;
    
	public LabeledFeatureVector(int y, SortedSet<Feature> src) {
        super(src);
        this.y = y;
    }
    
    public void setY(int y) {
        assert y == 1 || y == -1;
        this.y = y;
    }
    
    public int y() {
        return this.y;
    }
    
    public boolean check() {
        if (!super.check()) {
            return false;
        }
        return this.y == 1 || this.y == -1;
    }
}
