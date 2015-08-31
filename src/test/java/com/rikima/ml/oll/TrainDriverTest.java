package com.rikima.ml.oll;

import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.Model;
import org.junit.Test;

import java.io.File;
import java.util.List;

import static org.junit.Assert.assertNotNull;

/**
 * Created by mrikitoku on 15/08/31.
 */
public class TrainDriverTest {

    @Test
    public void testTrain() throws Exception {
        String input      = "./data/adult_filter.svmdata";
        String model_json = "./data/adult_filter.svmdata.model.json";
        OLLTrainer trainer = OLLTrainerFactory.create(OLLTrainerFactory.Algorithm.L1SVM, 1.0, 0);
        assertNotNull(trainer);

        List<LabeledFeatureVector> trainData = TrainDriver.loadTrainData(new File(input));
        trainer.train(trainData, 10);
        Model m = trainer.getModel();
        assertNotNull(m);
        String json = m.toJson();
        assertNotNull(json);
    }
}