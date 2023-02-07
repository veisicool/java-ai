package ML.functions;

import np.NdArray;

public interface Function {
    NdArray call(NdArray x, boolean d);
}
