package ML.losses;

import np.NdArray;
import np.np;

public class mse implements Loss {
    public NdArray call(NdArray x, NdArray y, boolean derivative) {
        if (derivative) {
            return np.multiply(np.subtract(x, y), 2);
        }
        return np.power(np.subtract(x,y),2);
    }
}
