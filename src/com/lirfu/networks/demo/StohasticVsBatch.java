package com.lirfu.networks.demo;

import com.lirfu.graphicslib.functions.Linear;
import com.lirfu.graphicslib.functions.Sigmoid;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.graphicslib.vector.Vector;
import com.lirfu.lirfugraph.*;
import com.lirfu.networks.*;
import com.lirfu.networks.initializers.RandomInitializer;
import com.lirfu.networks.initializers.WeightInitializer;
import com.lirfu.networks.layers.FullyConnectedLayer;
import com.lirfu.networks.layers.InputLayer;

public class StohasticVsBatch {
    public static void main(String[] args) {
        int numberOfPoints = 40;

        /* Define data. */
        IMatrix[] inputs = new IMatrix[numberOfPoints];
        IMatrix[] outputs = new IMatrix[numberOfPoints];

        double input;
        for (int i = 0; i < numberOfPoints; i++) {
            input = i * 1.24 * Math.PI / numberOfPoints;
            inputs[i] = new Matrix(new Vector(input));
            outputs[i] = new Matrix(new Vector(4 * Math.sin(2 * input) + 5));
        }

        WeightInitializer initializer = new RandomInitializer(-1,1);

        /* Build the network. */
        Network netStoh = new Network(
                new InputLayer(1),
                new FullyConnectedLayer(1, 6, new Sigmoid(), initializer),
                new FullyConnectedLayer(6, 1, new Linear(), initializer)
        );
        Network netBatc = new Network(netStoh);

        /* Collect training results. */

        MultiLinearGraph errorsGraph = new MultiLinearGraph("Total error", 2, "Stochastic", "Batch");

        SeparatedData[] dataB = DataSeparator.toBatch(DataSeparator.simpleData(inputs, outputs));
        SeparatedData[] dataS = DataSeparator.toStocastic(DataSeparator.simpleData(inputs, outputs));
        int iteration = 0;
        double errorB, errorS;
        double lr = 7e-5;
        while (iteration < 1_000_000) {
            errorS = netStoh.backpropagate(1e-3, dataS);
            errorB = netBatc.backpropagate(lr, dataB);
            if (iteration++ % 1000 == 0) {
                errorsGraph.add(errorS, errorB);
                System.out.println("Iteration " + iteration + " has errors:   " + errorS + "   " + errorB);
            }
        }

        /* Display the final results. */

        MultiLinearGraph finalResults = new MultiLinearGraph("Final results", 3, "Stochastic", "Batch", "Data");
        for (int index = 0; index < inputs.length; index++)
            finalResults.add(netStoh.getOutput(inputs[index]).get(0, 0), netBatc.getOutput(inputs[index]).get(0, 0), outputs[index].get(0, 0));

        new Window(new VerticalContainer(
                new Row(errorsGraph), new Row(finalResults)
        ), true, true);
    }
}
