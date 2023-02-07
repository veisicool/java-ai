import ML.Globals;
import ML.Model;
import ML.activations.*;
import ML.layers.Dense;
import ML.layers.Output;
import ML.losses.categoricalCrossEntropy;
import ML.losses.mse;
import np.*;

import java.io. * ;
import java.util.Scanner;

public class Main {
    static Scanner in = new Scanner(System.in);
    public static void main(String[] args) throws FileNotFoundException {
        Scanner sc = new Scanner(new File("/Users/veisicool/projects/java-ai/src/data/mnist_train.csv"));

        sc.useDelimiter(",|\\n");

        NdArray data = np.zeros(10000, 28*28);
        NdArray label = np.zeros(10000, 10);

        //setting comma as delimiter pattern
        for (int i = 0; i < 10000; i++) {
            label.set(i, Integer.parseInt(sc.next()), 1);
            for (int j = 0; j < 28*28; j++) {
                data.set(i,j, (double)(Integer.parseInt(sc.next())) / 255);
            }
        }
        sc.close();


        Model model = new Model(28*28);
        model.add(new Dense(16, activations.sigmoid));
        model.add(new Dense(8, activations.sigmoid));
        model.finish(new Output(10, activations.softmax));

        model.compile(new mse());

        model.fit(data, label, -0.1, 1);

        //model.predict(data);
        Globals.println(model.layers[1].layer.toString());

        Globals.println(model.predict(data.getRange(0,10)).toString());
        Globals.println(model.predict(data.getRange(0,10)).toString());
    }
}