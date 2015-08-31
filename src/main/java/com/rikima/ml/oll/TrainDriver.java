package com.rikima.ml.oll;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.Model;
import com.rikima.ml.WordIDManager;
import com.rikima.ml.oll.OLLTrainerFactory.Algorithm;
import com.rikima.utils.IoUtil;
import com.rikima.utils.LineIterator;

/**
 * Created by mrikitoku on 15/08/31.
 */
public class TrainDriver {

    public static Model run(List<LabeledFeatureVector> trainData, Algorithm alg, double c, double bias, int tryCount) throws Exception {
        OLLTrainer trainer = OLLTrainerFactory.create(alg, c, bias);
        trainer.train(trainData, tryCount);
        return trainer.getModel();
    }

    public static List<LabeledFeatureVector> loadTrainData(File input) throws Exception {
        ArrayList<LabeledFeatureVector> trainData = new ArrayList<LabeledFeatureVector>();
        LineIterator iter = IoUtil.iterator(input);
        while (iter.hasNext()) {
            String line = iter.next();
            LabeledFeatureVector lfv = WordIDManager.getInstance().createViaPureSvmformat(line);
            trainData.add(lfv);
        }
        // shuffle
        Collections.shuffle(trainData);
        return trainData;
    }

    static void printUsage() {
        System.out.println("com.rikima.ml.TrainDriver -i [input] -a [PA|PA1|PA2|CW|SCW|L1SVM|L1LR] -c [c] -b [bias] -t [try_count] -m [model path]");
    }

    public static void main(String[] args) {
        Algorithm alg = null;
        double c = 1.0;
        double bias = 0.0;
        String input = null;
        String model_json = null;
        int tryCount = 5;
        List<LabeledFeatureVector> trainData = null;
        for (int i = 0; i < args.length; ++i) {
            String a = args[i];
            if (a.equals("-a") || a.equals("--alg")) {
                alg = Algorithm.valueOf(args[++i]);
            } else if (a.equals("-i") || a.equals("--input")) {
                input = args[++i];
                } else if (a.equals("-c")) {
                c = Double.parseDouble(args[++i]);
            } else if (a.equals("-b") || a.equals("--bias")) {
                bias = Double.parseDouble(args[++i]);
            } else if (a.equals("-t") || a.equals("--try_count")) {
                tryCount = Integer.parseInt(args[++i]);
            } else if (a.equals("-m") || a.equals("--model_json")) {
                model_json = args[++i];
            }
        }
        try {
            model_json = String.format("%s.%s.model.json", alg, input);

            trainData = loadTrainData(new File(input));
            Model m = run(trainData, alg, c, bias, tryCount);
            String json = m.toJson();
            IoUtil.outputAsText(model_json, json);
        } catch (Exception e) {
            //e.printStackTrace();
            printUsage();
        }
    }
}
