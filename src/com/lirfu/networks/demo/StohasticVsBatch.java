package com.lirfu.networks.demo;

import com.lirfu.graphicslib.functions.Linear;
import com.lirfu.graphicslib.functions.Sigmoid;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.graphicslib.vector.Vector;
import com.lirfu.lirfugraph.*;
import com.lirfu.lirfugraph.Row;
import com.lirfu.lirfugraph.VerticalContainer;
import com.lirfu.lirfugraph.graphs.MultiLinearGraph;
import com.lirfu.networks.*;
import com.lirfu.networks.descendmethods.DescendMethod;
import com.lirfu.networks.descendmethods.MomentumDescend;
import com.lirfu.networks.initializers.RandomInitializer;
import com.lirfu.networks.initializers.WeightInitializer;
import com.lirfu.networks.layers.FullyConnectedLayer;
import com.lirfu.networks.layers.InputLayer;

public class StohasticVsBatch {
    static Point2D bounds = new Point2D(0, Math.PI);

    public static void main(String[] args) {
        int numberOfPoints = 50;

        /* Define data. */
        IMatrix[] inputs = new IMatrix[numberOfPoints];
        IMatrix[] outputs = new IMatrix[numberOfPoints];

        double input;
        for (int i = 0; i < numberOfPoints; i++) {
            input = i * (bounds.y - bounds.x) / numberOfPoints + bounds.x;
            inputs[i] = new Matrix(new Vector(input));
            outputs[i] = new Matrix(new Vector(func(input)));
        }

        WeightInitializer initializer = new RandomInitializer(-1, 1);
//        DescendMethod descendMethod = new VanillaGradientDescend();
        DescendMethod descendMethod = new MomentumDescend(0.9);

        /* Build the network. */
        Network netBatc = new Network(
                new InputLayer(1),
                new FullyConnectedLayer(1, 5, new Sigmoid(), descendMethod, initializer),
                new FullyConnectedLayer(5, 1, new Linear(), descendMethod, initializer)
        );
        Network netStoh = new Network(netBatc);
        Network netMB = new Network(netBatc);

        System.out.println("INITIAL:");
        System.out.println("  - Batch:");
        System.out.println(netBatc);
        System.out.println("  - Stochastic:");
        System.out.println(netStoh);
        System.out.println("  - Minibatch:");
        System.out.println(netMB);

        /* Collect training results. */

        MultiLinearGraph errorsGraph = new MultiLinearGraph("Total error", 3, "Stochastic", "Batch", "Minibatch");

        SeparatedData data = DataSeparator.simpleData(inputs, outputs);
        SeparatedData[] dataB = DataSeparator.toBatch(data);
        SeparatedData[] dataS = DataSeparator.toStocastic(data);
        SeparatedData[] dataMB = DataSeparator.toMiniBatch(data, inputs.length / 4, true);

        int iteration = 0;
        double errorB, errorS, errorsMB;
        double lr = 0.001;

        errorS = netStoh.calculateError(data.getTestInputs(), data.getTestOutputs());
        errorB = netBatc.calculateError(data.getTestInputs(), data.getTestOutputs());
        errorsMB = netMB.calculateError(data.getTestInputs(), data.getTestOutputs());
        System.out.println("Iteration " + iteration + " has errors:   " + errorS + "   " + errorB + "   " + errorsMB);

        while (iteration < 100_000) {
            errorS = netStoh.backpropagate(lr, dataS);
            errorB = netBatc.backpropagate(lr, dataB);
            errorsMB = netMB.backpropagate(lr, dataMB);
            if (iteration++ % 1000 == 0) {
                errorsGraph.add(errorS, errorB, errorsMB);
                System.out.println("Iteration " + iteration + " has errors:   " + errorS + "   " + errorB + "   " + errorsMB);
            }
        }

        System.out.println("RESULTS:");
        System.out.println("  - Batch:");
        System.out.println(netBatc);
        System.out.println("  - Stochastic:");
        System.out.println(netStoh);
        System.out.println("  - Minibatch:");
        System.out.println(netMB);

        /* Display the final results. */

        MultiLinearGraph finalResults = new MultiLinearGraph("Final results", 4, "Stochastic", "Batch", "Minibatch", "Data");
        finalResults.setShowDots(false);
        for (double x = bounds.x; x <= bounds.y; x += 0.1)
            finalResults.add(
                    netStoh.getOutput(new Matrix(new Vector(x))).get(0, 0),
                    netBatc.getOutput(new Matrix(new Vector(x))).get(0, 0),
                    netMB.getOutput(new Matrix(new Vector(x))).get(0, 0),
                    func(x)
            );

        new Window(new VerticalContainer(
                new Row(errorsGraph), new Row(finalResults)
        ), true, true);
    }

    private static double func(double input) {
        return 4 * Math.sin(1.5 * input) + 5;
//        return (input - 1) * (input - 1);
    }
}
