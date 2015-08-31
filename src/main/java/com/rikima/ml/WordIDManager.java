package com.rikima.ml;

import java.util.HashMap;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

//import com.google.common.collect.BiMap;
//import com.google.common.collect.HashBiMap;

public class WordIDManager {
    public static final String DELIMITER = ":";
    public static final String ZERO_FEATURE = "0";

    // fields ----------------
    private static WordIDManager self;

    private Map<String, Integer> word2ids;
    //private BiMap<String, Integer> biword2ids;

    private int id = 0;

    // constructors ----------
    private WordIDManager() {
        this.id = 0;
        this.word2ids = new HashMap<String, Integer>();
        //   this.biword2ids = HashBiMap.create();
    }

    public static WordIDManager getInstance() {
        if (self == null) {
            self = new WordIDManager();
        }
        
        if (self.word2ids == null) {
            self.reset();
        }
        return self;
    }

    // methods ---------------
    public void reset() {
        this.word2ids = new HashMap<String, Integer>();
       // this.biword2ids = HashBiMap.create();
    }

    public int count() {
		return this.word2ids.size();
	}

    public void set(String surface, int wordId) {
        this.word2ids.put(surface, wordId);
        if (this.id < wordId) {
        	this.id = wordId;
        }
    }

    /*
    public String featureById(int fid) {
        //return this.biword2ids.inverse().get(fid);
        return null;
    }
    */

    public int idByFeature(String surface) {
        if (!this.word2ids.containsKey(surface)) {
            ++id;
            this.word2ids.put(surface, id);
           // this.biword2ids.put(surface, id);
        }
        return this.word2ids.get(surface);
    }

    public LabeledFeatureVector createViaPureSvmformat(String line) throws Exception {
        assert line.trim().length() > 0;
        String[] ss = line.trim().split("\\s");
        
        int y = 0;
        String ys = ss[0];
        if ("+1".equals(ys)) {
            ys = "1";
        }
        y = Integer.parseInt(ys);
        assert y == 1 || y == -1 || y == 0;
        
        SortedSet<Feature> tmp = new TreeSet<Feature>();
        int index = -1;
        for (String s : ss) {
            if (index++ < 0) {
                continue;
            }
            
            String[] ss2 = s.split(":");
            int fid = Integer.parseInt(ss2[0].trim());
            double val = Double.parseDouble(ss2[1]);
            
            Feature f = new Feature(fid, val);
            tmp.add(f);
        }
        
        LabeledFeatureVector fv = new LabeledFeatureVector(y, tmp);
        return fv;
    }

    public LabeledFeatureVector createLabeledFeatureVector(String line) throws Exception {
        assert line.trim().length() > 0;
        String[] ss = line.trim().split(" ");

        int y = 0;
        String ys = ss[0];
        if ("+1".equals(ys)) {
            ys = "1";
        }
        y = Integer.parseInt(ys);
        assert y == 1 || y == -1 || y == 0;

        SortedSet<Feature> tmp = new TreeSet<Feature>();
        int index = -1;
        for (String s : ss) {
            if (index++ < 0) {
                continue;
            }
            
            String[] ss2 = s.split(":");
            int fid = this.idByFeature(ss2[0].trim());
            double val = Double.parseDouble(ss2[1]);

            Feature f = new Feature(fid, val);
            tmp.add(f);
        }

        LabeledFeatureVector fv = new LabeledFeatureVector(y, tmp);
        return fv;
    }

    public FeatureVector createFeatureVector(String line) throws Exception {
        assert line.trim().length() > 0;
        String[] ss = line.trim().split(" ");
        SortedSet<Feature> tmp = new TreeSet<Feature>();
        int index = -1;
        for (String s : ss) {
            if (index++ < 0) {
                continue;
            }
            
            String[] ss2 = s.split(":");
            int fid = this.idByFeature(ss2[0].trim());
            double val = Double.parseDouble(ss2[1]);

            Feature f = new Feature(fid, val);
            tmp.add(f);
        }

        FeatureVector fv = new FeatureVector(tmp);
        return fv;
    }
}
