package com.rikima.ml.oll;

import com.rikima.ml.Model;
import com.rikima.ml.oll.cw.CWTrainer;
import com.rikima.ml.oll.fobos.L1LR;
import com.rikima.ml.oll.fobos.L1SVM;
import com.rikima.ml.oll.pa.PA1Trainer;
import com.rikima.ml.oll.pa.PA2Trainer;
import com.rikima.ml.oll.pa.PATrainer;
import com.rikima.ml.oll.scw.SCWTrainer;

public class OLLTrainerFactory {
	public enum Algorithm {AP, PA, PA1, PA2, CW, SCW, L1SVM, L1LR};
    
    public static OLLTrainer create(Algorithm a, double c, double b) {
        switch (a) {
            case PA:
                return new PATrainer(c, b);
            case PA1:
                return new PA1Trainer(c, b);
            case PA2:
                return new PA2Trainer(c, b);
            case CW:
                return new CWTrainer(c, b);
            case L1SVM:
                return new L1SVM(c, b);
            case L1LR:
                return new L1LR(c, b);
            case SCW:
                return new SCWTrainer(c, b);
            default:
                return null;
        }
    }

    public static OLLTrainer create(Model m , Algorithm a, double c, double b) {
        switch (a) {
            case SCW:
                return new SCWTrainer(m, c, b);
            case L1SVM:
                return new L1SVM(m, c, b);
            default:
                return null;
        }
    }
}
