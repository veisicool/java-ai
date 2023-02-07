package ML;

import ML.layers.*;
import ML.losses.Loss;
import np.*;

public class Model {
    public Input input;
    public Dense[] layers;
    public Output output;
    public NdArray[] weights;
    Loss loss;

    public Model(int inputShape) {
        layers = new Dense[0];
        input = new Input(inputShape);
        weights = new NdArray[0];
    }

    public void add(Dense layer) {
        Dense[] newLayers = new Dense[layers.length + 1];
        for (int i = 0; i < newLayers.length - 1; i++) {
            newLayers[i] = layers[i];
        }
        newLayers[newLayers.length - 1] = layer;
        layers = newLayers;

        NdArray[] newWeights = new NdArray[weights.length + 1];
        for (int i = 0; i < newWeights.length - 1; i++) {
            newWeights[i] = weights[i];
        }
        if (layers.length == 1) {
            newWeights[newWeights.length - 1] = np.rand(input.size, layer.size);
        } else {
            newWeights[newWeights.length - 1] = np.rand(layers[layers.length - 2].size, layer.size);
        }
        weights = newWeights;


    }

    public void finish(Output layer) {
        NdArray[] newWeights = new NdArray[weights.length + 1];
        for (int i = 0; i < newWeights.length - 1; i++) {
            newWeights[i] = weights[i];
        }
        newWeights[newWeights.length - 1] = np.rand(layers[layers.length - 1].size, layer.size);
        weights = newWeights;

        output = layer;
    }

    public void compile(Loss loss) {
        this.loss = loss;
    }

    public NdArray predict(NdArray input) {
        //Globals.println(weights[0]);
        layers[0].set(this.input.propagate(input, weights[0]));

        for (int i = 1; i < layers.length; i++) {
            layers[i].set(layers[i-1].propagate(weights[i]));
        }
        output.set(layers[layers.length-1].propagate(weights[weights.length - 1]));

        return output.activated;
    }

    public void fit(NdArray x, NdArray y, double learning_rate, int epochs) {
        int batch_size = Globals.BATCH_SIZE;

        for (int e = 0; e < epochs; e++) {
            int prev_batch_index = 0;
            int batches_completed = 0;

            for (int batch_index = batch_size; batch_index < x.shape[0]; batch_index += batch_size) {
                NdArray data = x.getRange(prev_batch_index, batch_index);
                NdArray label = y.getRange(prev_batch_index, batch_index);

                predict(data);

                NdArray error = output.backPropagate(loss, label);

                for (int i = weights.length - 1; i > 0; i--) {
                    NdArray step = layers[i - 1].getStep(error);
                    //Globals.println(step + "epoch: " + e + " i: " + i);

                    weights[i] = np.subtract(weights[i], np.multiply(step, learning_rate));

                    error = layers[i - 1].backPropagate(error, weights[i]);

                }
                NdArray step = input.getStep(error);
                weights[0] = np.subtract(weights[0], np.multiply(step, learning_rate));


                prev_batch_index = batch_index;
                batches_completed ++;
            }
        }
    }
}
