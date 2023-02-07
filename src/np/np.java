package np;

import ML.Globals;

import java.security.InvalidParameterException;

public class np {
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
                out[i][j] = Math.random() * 2 - 1;
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
            throw new InvalidParameterException("\narrays dimensions not compatible:\n    cannot broadcast together shapes ( ( " + a.shape[0] + ", " + a.shape[1] + "), (" + b.shape[0] + ", " + b.shape[1] + " ) )");
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
    public static NdArray outer(double[] a, double[] b) {
        double[][] out = new double[a.length][b.length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                out[i][j] = a[i] * b[j];
            }
        }
        return new NdArray(out);
    }

    public static NdArray outer(NdArray a, NdArray b) {
        return outer(a.flatten(), b.flatten());
    }

    // single operators
    private static class exp implements SingleOperator {
        public double call(double a) {
            return Math.exp(a);
        }
    }

    private static class log implements SingleOperator {
        public double call(double a) {
            return Math.log(a);
        }
    }

    private static class cosh implements SingleOperator {
        public double call(double a) {
            return Math.cosh(a);
        }
    }

    private static class tanh implements SingleOperator {
        public double call(double a) {
            return Math.tanh(a);
        }
    }


    // single array operations
    public static NdArray operator(NdArray a, SingleOperator func) {
        double[][] out = new double[a.shape[0]][a.shape[1]];

        for (int i = 0; i < a.shape[0]; i++) {
            for (int j = 0; j < a.shape[1]; j++) {
                out[i][j] = func.call(a.get(i,j));
            }
        }
        return new NdArray(out);
    }

    public static NdArray exp(NdArray a) {
        return operator(a, new exp());
    }

    public static NdArray log(NdArray a) {
        return operator(a, new log());
    }

    public static NdArray cosh(NdArray a) {
        return operator(a, new cosh());
    }

    public static NdArray tanh(NdArray a) {
        return operator(a, new tanh());
    }


    // operators
    private static class add implements Operator {
        public double call(double a, double b) {
            return a + b;
        }
    }

    private static class subtract implements Operator {
        public double call(double a, double b) {
            return a - b;
        }
    }

    private static class multiply implements Operator {
        public double call(double a, double b) {
            return a * b;
        }
    }

    private static class divide implements Operator {
        public double call(double a, double b) {
            return a / b;
        }
    }

    private static class power implements Operator {
        public double call(double a, double b) {
            return Math.pow(a, b);
        }
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
        return operator(a, b, new add());
    }

    public static NdArray subtract(NdArray a, double b) {
        return operator(a, b, new subtract());
    }

    public static NdArray multiply(NdArray a, double b) {
        return operator(a, b, new multiply());
    }

    public static NdArray divide(NdArray a, double b) {
        return operator(a, b, new divide());
    }

    public static NdArray power(NdArray a, double b) {
        return operator(a, b, new power());
    }

    // number - array operators
    public static NdArray operator(double a, NdArray b, Operator func) {
        double[][] out = new double[b.shape[0]][b.shape[1]];

        for (int i = 0; i < b.shape[0]; i++) {
            for (int j = 0; j < b.shape[1]; j++) {
                out[i][j] = func.call(a, b.get(i,j));
            }
        }
        return new NdArray(out);
    }

    public static NdArray add(double a, NdArray b) {
        return operator(a, b, new add());
    }

    public static NdArray subtract(double a, NdArray b) {
        return operator(a, b, new subtract());
    }

    public static NdArray multiply(double a, NdArray b) {
        return operator(a, b, new multiply());
    }

    public static NdArray divide(double a, NdArray b) {
        return operator(a, b, new divide());
    }

    public static NdArray power(double a, NdArray b) {
        return operator(a, b, new power());
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
        return operator(a, b, new add());
    }

    public static NdArray subtract(NdArray a, NdArray b) {
        return operator(a, b, new subtract());
    }

    public static NdArray multiply(NdArray a, NdArray b) {
        return operator(a, b, new multiply());
    }

    public static NdArray divide(NdArray a, NdArray b) {
        return operator(a, b, new divide());
    }

    public static NdArray power(NdArray a, NdArray b) {
        return operator(a, b, new divide());
    }

    // argmax
    public static int[] argmax(NdArray a) {
        int[] out = new int[a.shape[0]];

        for (int i = 0; i < a.shape[0]; i++) {
            double max = a.get(i, 0);
            out[i] = 0;
            for (int j = 0; j < a.shape[1]; j++) {
                if (a.get(i, j) < max) {
                    max = a.get(i, j);
                    out[i] = j;
                } else {
                    Globals.println("a: " + a.get(i, j) + " max: " + max);
                }
            }
        }

        return out;
    }



    // displaying
    public static String vecToString(int[] a) {
        String out = "[";
        for (Object o : a) {
            out += " " + o;
        }
        out += " ]";
        return out;
    }

    public static String vecToString(double[] a) {
        String out = "[";
        for (Object o : a) {
            out += " " + o;
        }
        out += " ]";
        return out;
    }
}
