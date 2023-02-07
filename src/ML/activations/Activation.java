package ML.activations;

import ML.functions.*;
import np.NdArray;
import org.jetbrains.annotations.NotNull;

public class Activation {
    Function function;
    double bound;

    public Activation(Function function, double bound) {
        this.function = function;
        this.bound = bound;
    }

    public NdArray act(@NotNull NdArray layer, boolean derivative) {
        for (int i = 0; i < layer.shape[0]; i++) {
            for (int j = 0; j < layer.shape[1]; j++) {
                if (layer.get(i, j) > bound) {
                    layer.set(i, j, bound);
                }
                if (layer.get(i, j) < -bound) {
                    layer.set(i, j, -bound);
                }
            }
        }
        return function.call(layer, derivative);
    }
}
