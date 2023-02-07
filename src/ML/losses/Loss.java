package ML.losses;

import np.NdArray;

public interface Loss {
    NdArray call(NdArray x, NdArray y,boolean derivative);
}
