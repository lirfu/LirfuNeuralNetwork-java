package com.lirfu.networks.demo;

import com.lirfu.graphicslib.functions.Linear;
import com.lirfu.lirfugraph.*;
import com.lirfu.lirfugraph.Window;
import com.lirfu.networks.DataSeparator;
import com.lirfu.networks.Network;
import com.lirfu.graphicslib.IncompatibleOperandException;
import com.lirfu.graphicslib.functions.Sigmoid;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.graphicslib.vector.Vector;
import com.lirfu.networks.layers.FullyConnectedLayer;
import com.lirfu.networks.layers.InputLayer;

import java.awt.*;

/**
 * Created by lirfu on 08.08.17..
 */
public class Demo {
    public static void main(String[] args) throws IncompatibleOperandException {
        int numberOfPoints = 50;

        /* Define data. */
        IMatrix[] inputs = new IMatrix[numberOfPoints];
        IMatrix[] outputs = new IMatrix[numberOfPoints];

        double input;
        for (int i = 0; i < numberOfPoints; i++) {
            input = i * 2. * Math.PI / numberOfPoints;
            inputs[i] = new Matrix(new Vector(input));
            outputs[i] = new Matrix(new Vector(4 * Math.sin(2 * input) + 2));
        }

        /* Build the network. */
        Network net = new Network(
                3e-3,
                new InputLayer(1),
                new FullyConnectedLayer(1, 6, new Sigmoid()),
                new FullyConnectedLayer(6, 1, new Linear())
        );

        /* Collect training results. */

        LinearGraph errorsGraph = new LinearGraph("Total error");

        int inputIndex = 3;
        DualLinearGraph resultsGraph = new DualLinearGraph("Results (" + inputIndex + ")");

        int iteration = 0;
        double error, result;
        while ((error = net.backpropagate(DataSeparator.separateData(inputs, outputs, 0.8))) > 5e-4) {
            if (iteration++ % 1000 == 0) {
                result = net.getOutput(inputs[inputIndex]).get(0, 0);
                errorsGraph.add(error);
                resultsGraph.add(result, outputs[inputIndex].get(0, 0));
                System.out.println("Iteration " + iteration + " has error: " + error);
            }
        }

        /* Display the final results. */

        System.out.println("Weights:\n" + net.toString());

        DualLinearGraph finalResults = new DualLinearGraph("Final results");
        for (int index = 0; index < inputs.length; index++)
            finalResults.add(net.getOutput(inputs[index]).get(0, 0), outputs[index].get(0, 0));

        Window window = new Window(new VerticalContainer(
                new Row(errorsGraph), new Row(resultsGraph), new Row(finalResults)
        ), true);
        window.setSize(new Dimension(600, 800));
        window.setVisibility(true);

    }
}
