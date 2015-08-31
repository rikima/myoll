package com.rikima.ml.oll;

import com.rikima.ml.Evaluator;
import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.Model;
import com.rikima.ml.WordIDManager;
import com.rikima.utils.IoUtil;
import com.rikima.utils.LineIterator;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static com.rikima.ml.oll.OLLTrainerFactory.Algorithm;

/**
 * Created by mrikitoku on 15/08/31.
 */
public class ClassifyDriver {

    public static void run(List<LabeledFeatureVector> data, String model_json, double bias) throws Exception {
        String json = IoUtil.read(new File(model_json));
        Model m = Model.construct(json);
        OLLClassifier classifier = new OLLClassifier(m);
        Evaluator eval = new Evaluator();
        for (LabeledFeatureVector lfv : data) {
            int y = classifier.classify(lfv);
            eval.setResult(lfv.y(), y);
        }

        eval.printResult();
    }

    public static List<LabeledFeatureVector> loadData(File input) throws Exception {
        ArrayList<LabeledFeatureVector> data = new ArrayList<LabeledFeatureVector>();
        LineIterator iter = IoUtil.iterator(input);
        while (iter.hasNext()) {
            String line = iter.next();
            LabeledFeatureVector lfv = WordIDManager.getInstance().createViaPureSvmformat(line);
            data.add(lfv);
        }
        return data;
    }

    static void printUsage() {
        System.out.println("com.rikima.ml.oll.ClassifyDriver -i [input] -m [model path]");
    }

    public static void main(String[] args) {
        Algorithm alg = null;
        String input = null;
        String model_json = null;
        List<LabeledFeatureVector> data = null;
        double bias = 0.0;
        for (int i = 0; i < args.length; ++i) {
            String a = args[i];
            if (a.equals("-i") || a.equals("--input")) {
                input = args[++i];
            } else if (a.equals("-b") || a.equals("--bias")) {
                bias = Double.parseDouble(args[++i]);
            } else if (a.equals("-m") || a.equals("--model_json")) {
                model_json = args[++i];
            }
        }

        try {
            data = loadData(new File(input));
            run(data, model_json, bias);
        } catch (Exception e) {
            //e.printStackTrace();
            printUsage();
        }
    }
}
