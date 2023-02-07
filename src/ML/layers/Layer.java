package ML.layers;

import np.*;

import static ML.Globals.BATCH_SIZE;

public class Layer {
    public NdArray layer;
    public int size;

    public Layer(int size) {
        layer = np.zeros(BATCH_SIZE, size);
        this.size = size;
    }
}
