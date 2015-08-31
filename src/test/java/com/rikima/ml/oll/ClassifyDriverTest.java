package com.rikima.ml.oll;

import com.rikima.ml.LabeledFeatureVector;
import org.junit.Test;

import java.io.File;
import java.util.List;

import static org.junit.Assert.*;
import static junit.framework.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Created by mrikitoku on 15/08/31.
 */
public class ClassifyDriverTest {

    @Test
    public void testClassify() throws Exception {
        String input      = "./data/adult_filter.svmdata";
        String model_json = "./data/adult_filter.svmdata.model.json";
        double bias = 0.0;
        List<LabeledFeatureVector> data = ClassifyDriver.loadData(new File(input));
        ClassifyDriver.run(data, model_json, bias);
    }
}