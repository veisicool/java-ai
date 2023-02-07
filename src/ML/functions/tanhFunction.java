package ML.functions;

import np.*;

public class tanhFunction implements Function {
    public NdArray call(NdArray x, boolean d) {
        if (d) {
            return np.divide(1, np.power(np.cosh(x), 2));
        }
        return np.tanh(x);
    }
}
