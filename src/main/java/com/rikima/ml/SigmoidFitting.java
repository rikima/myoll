package com.rikima.ml;

/**
 * Created by mrikitoku on 14/09/29.
 */
public class SigmoidFitting {

    /**
     * Platt's binary SVM Probablistic Output: an improvement from Lin et al.
     */
    public static void sigmoid_train(int l, double[] dec_values, double[] labels, double[] probAB) {
        double A, B;
        double prior1=0, prior0 = 0;
        int i;

        for (i = 0;i < l;i++) {
            if (labels[i] > 0) {
                prior1 += 1;
            } else {
                prior0 += 1;
            }
        }

        int max_iter = 100;       // Maximal number of iterations
        double min_step = 1e-10;  // Minimal step taken in line search
        double sigma = 1e-12;     // For numerically strict PD of Hessian
        double eps = 1e-5;
        double hiTarget = (prior1+1.0)/(prior1+2.0);
        double loTarget = 1/(prior0+2.0);
        double[] t = new double[l];
        double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
        double newA,newB,newf,d1,d2;
        int iter;

        // Initial Point and Initial Fun Value
        A = 0.0;
        B = Math.log((prior0+1.0)/(prior1+1.0));
        double fval = 0.0;

        for (i = 0;i < l;i++) {
            if (labels[i]>0) {
                t[i]=hiTarget;
            } else {
                t[i]=loTarget;
            }
            fApB = dec_values[i]*A+B;
            if (fApB>=0) {
                fval += t[i]*fApB + Math.log(1+Math.exp(-fApB));
            } else {
                fval += (t[i] - 1)*fApB +Math.log(1+Math.exp(fApB));
            }
        }

        for (iter=0;iter<max_iter;iter++) {
            // Update Gradient and Hessian (use H' = H + sigma I)
            h11 = sigma; // numerically ensures strict PD
            h22 = sigma;
            h21 = 0.0;g1=0.0;g2=0.0;
            for (i = 0;i < l;i++) {
                fApB = dec_values[i]*A+B;
                if (fApB >= 0) {
                    p=Math.exp(-fApB)/(1.0+Math.exp(-fApB));
                    q=1.0/(1.0+Math.exp(-fApB));
                } else {
                    p=1.0/(1.0+Math.exp(fApB));
                    q=Math.exp(fApB)/(1.0+Math.exp(fApB));
                }
                d2 = p*q;
                h11 += dec_values[i] * dec_values[i] * d2;
                h22 += d2;
                h21 += dec_values[i] * d2;
                d1 = t[i]-p;
                g1 += dec_values[i] * d1;
                g2 += d1;
            }

            // Stopping Criteria
            if (Math.abs(g1) < eps && Math.abs(g2) < eps) {
                break;
            }

            // Finding Newton direction: -inv(H') * g
            det = h11 * h22 - h21 * h21;
            dA = -(h22 * g1 - h21 * g2) / det;
            dB = -(-h21 * g1+ h11 * g2) / det;
            gd = g1 * dA + g2 * dB;

            stepsize = 1;           // Line Search
            while (stepsize >= min_step) {
                newA = A + stepsize * dA;
                newB = B + stepsize * dB;

                // New function value
                newf = 0.0;
                for (i=0;i<l;i++) {
                    fApB = dec_values[i]*newA+newB;
                    if (fApB >= 0) {
                        fApB = dec_values[i]*newA+newB;
                    }
                    if (fApB >= 0) {
                        newf += t[i]*fApB + Math.log(1+Math.exp(-fApB));
                    } else {
                        newf += (t[i] - 1)*fApB +Math.log(1+Math.exp(fApB));
                    }
                }
                // Check sufficient decrease
                if (newf < fval + 0.0001 * stepsize * gd) {
                    A=newA;B=newB;fval=newf;
                    break;
                } else
                    stepsize = stepsize / 2.0;
            }

            if (stepsize < min_step) {
                //svm.info("Line search fails in two-class probability estimates\n");
                break;
            }
        }

        if (iter >= max_iter) {
            //svm.info("Reaching maximal iterations in two-class probability estimates\n");
        }
        probAB[0]=A;probAB[1]=B;
    }

    public static double sigmoid_predict(double decision_value, double A, double B) {
        double fApB = decision_value * A + B;
        if (fApB >= 0) {
            return Math.exp(-fApB) / (1.0+Math.exp(-fApB));
        } else {
            return 1.0 / (1+Math.exp(fApB)) ;
        }
    }
}
