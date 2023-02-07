package ML.layers;

import ML.activations.Activation;
import ML.losses.Loss;
import np.*;

import static ML.Globals.BATCH_SIZE;

public class Output extends Layer {
    public NdArray activated;
    public NdArray derivative;
    public Activation activation;

    public Output(int size, Activation activation) {
        super(size);

        activated = np.zeros(BATCH_SIZE, size);
        derivative = np.zeros(BATCH_SIZE, size);

        this.activation = activation;
    }

    public NdArray backPropagate(Loss loss, NdArray label) {
        return np.multiply(loss.call(activated, label, true), derivative);
    }

    public void set(NdArray layer) {
        this.layer = layer;
        activated = activation.act(layer, false);
        derivative = activation.act(layer, true);
    }
}
