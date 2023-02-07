package np;

public class NdArray {
    private double[][] arr;
    public int[] shape;

    public NdArray(double[][] arr) {
        this.arr = arr;
        shape = new int[2];
        shape[0] = arr.length;
        shape[1] = arr[0].length;
    }

    public NdArray T() {
        double[][] trans = new double[shape[1]][shape[0]];
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                trans[j][i] = arr[i][j];
            }
        }
        return new NdArray(trans);
    }

    public double[] flatten() {
        double[] out = new double[shape[0] * shape[1]];
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                out[(i * shape[1]) + j] = arr[i][j];
            }
        }
        return out;
    }

    public double[] get(int i) {
        return arr[i];
    }

    public double get(int i, int j) {
        return arr[i][j];
    }

    public NdArray getRange(int i1, int i2) {
        if (i1 < 0) {
            i1 = arr.length + i1;
        }

        if (i2 < 0) {
            i2 = arr.length + i2;
        }


        double[][] out = new double[i2 - i1][shape[1]];

        for (int i = i1; i < i2; i++) {
            for (int j = 0; j < shape[1]; j++) {
                out[i - i1][j] = arr[i][j];
            }
        }
        return new NdArray(out);
    }

    public NdArray getRange(int i1, int i2, int j1, int j2) {
        if (i1 < 0) {
            i1 = arr.length + i1;
        }

        if (i2 < 0) {
            i2 = arr.length + i2;
        }

        if (j1 < 0) {
            j1 = arr.length + j1;
        }

        if (j2 < 0) {
            j2 = arr.length + j2;
        }

        double[][] out = new double[i2 - i1][j2 - j1];

        for (int i = i1; i < i2; i++) {
            for (int j = j1; j < j2; j++) {
                out[i][j] = arr[i][j];
            }
        }
        return new NdArray(out);
    }

    public void set(int i, int j, double val) {
        arr[i][j] = val;
    }

    public String toString() {
        String out = "[";
        for (int i = 0; i < shape[0]; i++) {
            if (i > 0) {
                out += " ";
            }
            out += ("[");
            for (int j = 0; j < shape[1]; j++) {
                out += " " + arr[i][j];
            }
            out += " ]";
            if (i < shape[0] - 1) {
                out += "\n";
            }
        }
        return out + "]";
    }
}
