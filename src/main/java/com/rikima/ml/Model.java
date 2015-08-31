package com.rikima.ml;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import com.fasterxml.jackson.annotation.JsonAutoDetect.Visibility;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import com.rikima.ml.oll.OLLClassifier;

public class Model {
    public static final String BIAS_KEY   = "bias";
    public static final String WEIGHT_KEY = "weights";

    @JsonProperty(value="weight_vector")
    protected MapWeightVector wv;
    protected double bias;

    // constructor ----
    public Model(MapWeightVector wv, double b) {
        this.wv   = wv;
        this.bias = b;
    }

    public Model() {
        this.wv   = new MapWeightVector();
        this.bias = 0;
    }

    // methods --------
    public double score(FeatureVector fv) {
        return this.wv.score(fv) + this.bias;
    }
    public int featureDimension() {
        return this.wv.size();
    }
    public double bias() {
        return this.bias;
    }

    public MapWeightVector weightVector() {
    	return this.wv;
    }

    public OLLClassifier createClassifier() {
        return new OLLClassifier(this);
    }

    public String toJson() throws JsonProcessingException {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.setVisibility(PropertyAccessor.FIELD, Visibility.ANY);
        Map<String, Object> kv = new HashMap<String, Object>();
        kv.put(BIAS_KEY, this.bias);
        kv.put(WEIGHT_KEY, this.weightVector().translateForJson());
        String json = objectMapper.writeValueAsString(kv);
        return json;
    }
    
    public static Model construct(String json) throws JsonParseException, JsonMappingException, IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.setVisibility(PropertyAccessor.FIELD, Visibility.ANY);
        
        Model m = new Model();
        if (json != null && json.length() > 0) {
            Map<String, Object> kv = objectMapper.readValue(json, Map.class);
            m.bias = (Double) kv.get(BIAS_KEY);
            Map<String, Double> weights = (Map<String, Double>) kv.get(WEIGHT_KEY);
            for (Iterator<Entry<String, Double>> iter = weights.entrySet().iterator(); iter.hasNext();) {
                Entry<String, Double> e = iter.next();
                String feature =  e.getKey();
                double val = e.getValue();
                //int fid = WordIDManager.getInstance().idByFeature(feature);
                int fid = Integer.parseInt(feature);
                m.wv.set(fid, val);
            }
        }
        return m;
    }
}
