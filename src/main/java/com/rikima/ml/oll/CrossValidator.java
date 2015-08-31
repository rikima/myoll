package com.rikima.ml.oll;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.rikima.ml.Evaluator;
import com.rikima.ml.LabeledFeatureVector;

/**
 * Created by mrikitoku on 15/09/01.
 */
public class CrossValidator {
    private int maxRound;
    private int nfold;
    private Evaluator eval;
    private List<LabeledFeatureVector> data;

    public CrossValidator(int nfold, List<LabeledFeatureVector> data, int maxRound) {
        this.maxRound = maxRound;
        this.nfold = nfold;
        Collections.shuffle(data);
        this.data = data;
        this.eval = new Evaluator();
    }

    public void run(OLLTrainer trainer) {
        List<LabeledFeatureVector> trainData = new ArrayList<LabeledFeatureVector>();
        List<LabeledFeatureVector> testData = new ArrayList<LabeledFeatureVector>();
        for (int fold = 0;fold < nfold;++fold) {
            trainData.clear();
            testData.clear();
            for (int i = 0; i < data.size(); ++i) {
                LabeledFeatureVector lfv = data.get(i);
                if (i % nfold == fold) {
                    testData.add(lfv);
                } else {
                    trainData.add(lfv);
                }
            }
            // train
            trainer.train(trainData, maxRound);
            OLLClassifier classifier = trainer.getClassifier();
            // classify
            for (LabeledFeatureVector lfv : testData) {
                int y = classifier.classify(lfv);
                this.eval.setResult(lfv.y(), y);
            }
        }
        this.eval.printResult();
    }

    static void printUsage() {
        System.out.println("com.rikima.ml.TrainDriver -i [input] -a [PA|PA1|PA2|CW|SCW|L1SVM|L1LR] -c [c] -b [bias] -t [try_count] -n [nfold]");
    }

    public static void main(String[] args) {
        OLLTrainerFactory.Algorithm alg = null;
        double c = 1.0;
        String input = null;
        double bias = 0.0;
        int maxRound = 1;
        int nfold = 10;
        List<LabeledFeatureVector> data = null;
        for (int i = 0; i < args.length; ++i) {
            String a = args[i];
            if (a.equals("-a") || a.equals("--arg")) {
                alg = OLLTrainerFactory.Algorithm.valueOf(args[++i]);
            } else if (a.equals("-i") || a.equals("--input")) {
                input = args[++i];
            } else if (a.equals("-c")) {
                c = Double.parseDouble(args[++i]);
            } else if (a.equals("-t") || a.equals("--try_count")) {
                maxRound = Integer.parseInt(args[++i]);
            } else if (a.equals("-n") || a.equals("--nfold")) {
                nfold = Integer.parseInt(args[++i]);
            }
        }

        try {

            data = TrainDriver.loadTrainData(new File(input));
            OLLTrainer trainer = OLLTrainerFactory.create(alg, c, bias);
            CrossValidator cv = new CrossValidator(nfold, data, maxRound);
            cv.run(trainer);
        } catch (Exception e) {
            //e.printStackTrace();
            printUsage();
        }
    }
}
