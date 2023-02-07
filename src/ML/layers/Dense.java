package ML.layers;

import ML.Globals;
import ML.activations.Activation;
import np.NdArray;
import np.np;

import static ML.Globals.*;

public class Dense extends Layer{
    NdArray activated;
    NdArray derivative;
    Activation activation;

    public Dense(int size, Activation activation) {
        super(size);

        activated = np.zeros(BATCH_SIZE, size);
        derivative = np.zeros(BATCH_SIZE, size);

        this.activation = activation;
    }

    public NdArray backPropagate(NdArray error, NdArray weights) {
        return np.multiply(np.dot(error, weights.T()), derivative);
    }

    public NdArray propagate(NdArray weights) {
        return np.dot(activated, weights);
    }

    public void set(NdArray layer) {
        this.layer = layer;
        activated = activation.act(layer, false);
        derivative = activation.act(layer, true);
    }

    public NdArray getStep(NdArray error) {
        return np.dot(activated.T(), error);
    }
}
