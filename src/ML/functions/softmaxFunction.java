package ML.functions;

import ML.Globals;
import np.*;

public class softmaxFunction implements Function {
    @Override
    public NdArray call(NdArray x, boolean d) {
        NdArray res = np.exp(x);
        NdArray out = np.zeros(x.shape[0], x.shape[1]);
        double[] sum = new double[res.shape[0]];

        for (int i = 0; i < res.shape[0]; i++) {
            sum[i] = 0;
            for (int j = 0; j < res.shape[1]; j++) {
                sum[i] += res.get(i,j);
            }
        }

        for (int i = 0; i < res.shape[0]; i++) {
            for (int j = 0; j < res.shape[1]; j++) {
                out.set(i, j, res.get(i, j) / sum[i]);
            }
        }

//        if (d) {
//            return np.multiply(out, np.subtract(1, out));
//        }
        return out;

    }
}
