package ML.activations;

import ML.functions.*;


public class activations {
    public static Activation sigmoid = new Activation(new sigmoidFunction(), 100);
    public static Activation tanh = new Activation(new tanhFunction(), 100);
    public static Activation softmax = new Activation(new softmaxFunction(), 100);
}
