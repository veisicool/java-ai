package ML.layers;

import np.*;

import static ML.Globals.BATCH_SIZE;

public class Input extends Layer{
    public Input(int size) {
        super(size);
    }

    public NdArray propagate(NdArray input, NdArray weights) {
        layer = input;
        return np.dot(layer, weights);
    }

    public NdArray backPropagate(NdArray error, NdArray weights) {
        return np.multiply(np.dot(error, weights.T()), layer);
    }

    public NdArray getStep(NdArray error) {
        return np.dot(layer.T(), error);

    }
}
