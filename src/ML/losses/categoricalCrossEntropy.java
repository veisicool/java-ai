package ML.losses;

import np.*;

public class categoricalCrossEntropy implements Loss {
    public NdArray call(NdArray x, NdArray y, boolean derivative) {
        for (int i = 0; i < x.shape[0]; i++) {
            for (int j = 0; j < x.shape[1]; j++) {
                if (x.get(i, j) < 0.00001) {
                    x.set(i, j, 0.00001);
                }
                if (x.get(i, j) > 1 - 0.00001) {
                    x.set(i, j, 1-0.00001);
                }
            }
        }
        NdArray out = np.zeros(x.shape[0], x.shape[1]);
        if (derivative) {
            for (int i = 0; i < y.shape[0]; i++) {
                for (int j = 0; j < y.shape[1]; j++) {
                    if (y.get(i, j) == 1) {
                        out.set(i, j, -1 / x.get(i, j));
                    } else {
                        out.set(i, j, 1 / (1 - x.get(i, j)));
                    }
                }
            }
            return out;
        }
        for (int i = 0; i < y.shape[0]; i++) {
            for (int j = 0; j < y.shape[1]; j++) {
                if (y.get(i, j) == 1) {
                    out.set(i, j, -Math.log(x.get(i, j)));
                } else {
                    out.set(i, j, Math.log(1 - x.get(i, j)));
                }
            }
        }
        return out;
    }
}
