package com.rikima.ml.oll.scw;

import java.util.ArrayList;

import com.rikima.ml.Feature;
import com.rikima.ml.FeatureVector;
import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.Model;
import com.rikima.ml.oll.OLLTrainer;

// Soft Confidence-Weighted
// http://icml.cc/2012/papers/86.pdf
public class SCWTrainer extends OLLTrainer {
    static boolean debug = false;
    enum Type {Type1, Type2};

    class Status {
        public double alpha;
        public double beta;
        public double margin;
        public double var;
        
        public String toString() {
            return String.format(" margin=%f\n var=%f\n alpha=%f\n beta=%f\n", this.margin, this.var, this.alpha, this.beta);
        }
    }

    // fields ---------------
    private Type type = Type.Type2;

    private ArrayList<Double> cov;
    private double phi = 1.0;

    public int exampleN;

    protected Status status = new Status();

    // constructor ----------
    public SCWTrainer(double c, double bias) {
        super(c, bias);
        this.cov = new ArrayList<Double>();
    }

    public SCWTrainer(Model m, double c, double bias) {
        super(m, c, bias);
        this.cov = new ArrayList<Double>();
    }

    // methods --------------
    public void trainExample(LabeledFeatureVector fv, int round) {
        double margin = super.getMargin(fv);
        double var    = getVariance(fv);
        
        double alpha = getAlpha(margin, var);
        if (alpha> 0){
            double beta  = getBeta(var, alpha);
            this.update(fv, alpha, beta);
            
            status.alpha = alpha;
            status.beta = beta;
            status.margin = margin;
            status.var = var;
            
            System.out.println(status);
        }
        this.exampleN++;
    }
    
    // private methods --------
    private double getAlpha(double margin, double cov) {
    	switch (this.type) { 
        case Type1:
            return this.getAlpha1(margin, cov);
        case Type2:
            return this.getAlpha2(margin, cov);
        default:
            return Double.NaN;
        }
    }
    
    private double getAlpha1(double margin, double cov) {
        double psi   = 1.0 + phi * phi / 2.0;
        double zeta  = 1 + phi * phi;
        double phi2 = phi * phi;
        double phi4 = phi2 * phi2;
        /*
        αt = min{C, max{0,
        	1
        	(−mt ψ +
        	υt ζ
        	mt 2
        	φ4
        	+ υt φ2 ζ)}}
         */
        double alpha = (-margin * psi + Math.sqrt(margin * margin * phi4 / 4.0 + cov * phi2 * zeta)) / (cov * zeta);
        if (alpha <= 0.0) {
            return 0.0;
        } else if (alpha >= C) {
            return C;
        } 
        return alpha;
    }
    
    private double getAlpha2(double margin, double v) {
        double n = v + 1.0 / (2.0 * C);
        double gamma = phi * Math.sqrt(phi * phi * margin * margin * v * v + 4 * n *v * (n+v*phi * phi));
        double alpha = - (2.0 * margin * n + phi * phi * margin * v) + gamma;
        alpha /= (2.0 * (n * n + n * v * phi * phi));
        if (alpha <= 0.0) {
            return 0.0;
        }
        return alpha;
    }
    
    private double getBeta(double var, double alpha) {
        double u = (-alpha * var * phi + Math.sqrt(alpha * alpha * var * var * phi * phi + 4.0 * var));
        u = u * u / 4.0;

        double beta = alpha * phi / (Math.sqrt(u) + var * alpha * phi);
        return beta;
    }
    
    private double getVariance(FeatureVector fv) {
        double ret = 0.0;
        for (Feature f : fv) {
            if (this.cov.size() <= f.index()) {
                ret += 1.0 * f.val() * f.val(); // assume cov[fv[i].first=1
            } else {
                ret += this.cov.get(f.index()) * f.val() * f.val();
            }
        }
        return ret;
    }
        
    private void update(LabeledFeatureVector fv, double alpha, double beta) {
        for (Feature f : fv) {
            int prevSize = cov.size();
            if (cov.size() <= f.index()+1) {
                for (int j = prevSize; j < f.index()+1; ++j) {
                    cov.add(1.0);
                }
            }

            double prev_w = super.wv.getOrCreate(f.id());
            // μ t+1 = μ t +αt yt Σt xt 
            super.wv.set(f.id(), prev_w  + alpha * fv.y() * f.val() * cov.get(f.index()));

            // Σt+1 = Σt −βt Σt xt T xt Σt
            double prev_cov = cov.get(f.index());
            cov.set(f.index(),  prev_cov - beta * f.val() * f.val() * prev_cov * prev_cov);
        }
    }
}
