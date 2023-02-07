package np;

import java.security.InvalidParameterException;

public class Numpy {
    // array creation methods
    public static NdArray array(double[][] a) {
        return new NdArray(a);
    }

    public static NdArray full(int dim0, int dim1, double n) {
        double[][] out = new double[dim0][dim1];

        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                out[i][j] = n;
            }
        }
        return new NdArray(out);
    }

    public static NdArray zeros(int dim0, int dim1) {
        return full(dim0, dim1, 0);
    }

    public static NdArray ones(int dim0, int dim1) {
        return full(dim0, dim1, 0);
    }

    public static NdArray rand(int dim0, int dim1) {
        double[][] out = new double[dim0][dim1];

        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                out[i][j] = Math.random();
            }
        }
        return new NdArray(out);
    }

    // dot function
    private static double dotVector(double[] a, double[] b) {
        double out = 0;

        for (int i = 0; i < a.length; i++) {
            out += a[i] * b[i];
        }


        return out;
    }

    public static NdArray dot(NdArray a, NdArray b) throws InvalidParameterException {
        if (a.shape[1] != b.shape[0]) {
            throw new InvalidParameterException("\narrays dimensions not compatible:\n    cannot broadcast together shapes ( (n, " + a.shape[1] + "), (" + b.shape[0] + ", n) )");
        }

        double[][] out = new double[a.shape[0]][b.shape[1]];

        for (int i = 0; i < a.shape[0]; i++) {
            for (int j = 0; j < b.shape[1]; j++) {
                out[i][j] = dotVector(a.get(i), b.T().get(j));
            }
        }
        return new NdArray(out);
    }

    // outer
    public static NdArray outer(NdArray a, NdArray b) {
        double[] flatA = a.flatten();
        double[] flatB = b.flatten();

        double[][] out = new double[flatA.length][flatB.length];

        for (int i = 0; i < flatA.length; i++) {
            for (int j = 0; j < flatB.length; j++) {
                out[i][j] = flatA[i] * flatB[j];
            }
        }
        return new NdArray(out);
    }

    // array - number operators
    public static NdArray operator(NdArray a, double b, Operator func) {
        double[][] out = new double[a.shape[0]][a.shape[1]];

        for (int i = 0; i < a.shape[0]; i++) {
            for (int j = 0; j < a.shape[1]; j++) {
                out[i][j] = func.call(a.get(i,j), b);
            }
        }
        return new NdArray(out);
    }

    public static NdArray add(NdArray a, double b) {
        class add implements Operator {
            public double call(double a, double b) {
                return a + b;
            }
        }
        return operator(a, b, new add());
    }

    public static NdArray subtract(NdArray a, double b) {
        class subtract implements Operator {
            public double call(double a, double b) {
                return a - b;
            }
        }
        return operator(a, b, new subtract());
    }

    public static NdArray multiply(NdArray a, double b) {
        class multiply implements Operator {
            public double call(double a, double b) {
                return a * b;
            }
        }
        return operator(a, b, new multiply());
    }

    public static NdArray divide(NdArray a, double b) {
        class divide implements Operator {
            public double call(double a, double b) {
                return a / b;
            }
        }
        return operator(a, b, new divide());
    }

    // array - array operators
    public static NdArray operator(NdArray a, NdArray b, Operator func) throws InvalidParameterException {
        if (a.shape[0] != b.shape[0]) {
            throw new InvalidParameterException("\narrays dimensions not compatible:\n dimension 0:\n cannot broadcast together shapes ( " + a.shape[0] + ", " + b.shape[0] + " )");
        }

        if (a.shape[1] != b.shape[1]) {
            throw new InvalidParameterException("\narrays dimensions not compatible:\n dimension 1:\n cannot broadcast together shapes ( " + a.shape[1] + ", " + b.shape[1] + " )");
        }

        double[][] out = new double[a.shape[0]][a.shape[1]];

        for (int i = 0; i < a.shape[0]; i++) {
            for (int j = 0; j < a.shape[1]; j++) {
                out[i][j] = func.call(a.get(i,j), b.get(i,j));
            }
        }
        return new NdArray(out);
    }

    public static NdArray add(NdArray a, NdArray b) {
        class add implements Operator {
            public double call(double a, double b) {
                return a + b;
            }
        }
        return operator(a, b, new add());
    }

    public static NdArray subtract(NdArray a, NdArray b) {
        class subtract implements Operator {
            public double call(double a, double b) {
                return a - b;
            }
        }
        return operator(a, b, new subtract());
    }

    public static NdArray multiply(NdArray a, NdArray b) {
        class multiply implements Operator {
            public double call(double a, double b) {
                return a * b;
            }
        }
        return operator(a, b, new multiply());
    }

    public static NdArray divide(NdArray a, NdArray b) {
        class divide implements Operator {
            public double call(double a, double b) {
                return a / b;
            }
        }
        return operator(a, b, new divide());
    }
}
