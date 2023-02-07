package ML.functions;

import np.*;

public class sigmoidFunction implements Function {
    public NdArray call(NdArray x, boolean d) {
        if (d) {
            return np.divide(np.exp(np.multiply(x, -1)), np.power(np.add(np.exp(np.multiply(x, -1)), 1), 2));
        }
        return np.divide(1, np.add(np.exp(np.multiply(x, -1)), 1));
    }
}
