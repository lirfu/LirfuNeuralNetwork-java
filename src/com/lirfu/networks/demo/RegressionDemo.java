package com.lirfu.networks.demo;

import com.lirfu.graphicslib.functions.Linear;
import com.lirfu.lirfugraph.Row;
import com.lirfu.lirfugraph.VerticalContainer;
import com.lirfu.lirfugraph.Window;
import com.lirfu.lirfugraph.graphs.DualLinearGraph;
import com.lirfu.lirfugraph.graphs.LinearGraph;
import com.lirfu.networks.*;
import com.lirfu.graphicslib.IncompatibleOperandException;
import com.lirfu.graphicslib.functions.Sigmoid;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;
import com.lirfu.networks.descendmethods.DescendMethod;
import com.lirfu.networks.descendmethods.VanillaGradientDescend;
import com.lirfu.networks.initializers.RandomInitializer;
import com.lirfu.networks.initializers.WeightInitializer;
import com.lirfu.networks.layers.FullyConnectedLayer;
import com.lirfu.networks.layers.InputLayer;

/**
 * Created by lirfu on 08.08.17..
 */
public class RegressionDemo {
    public static void main(String[] args) throws IncompatibleOperandException {
        int numberOfPoints = 50;

        /* Define data. */
        IMatrix[] inputs = new IMatrix[numberOfPoints];
        IMatrix[] outputs = new IMatrix[numberOfPoints];

        double input;
        for (int i = 0; i < numberOfPoints; i++) {
            input = i * 2. * Math.PI / numberOfPoints;
            inputs[i] = new Matrix(1,1,input);
            outputs[i] = new Matrix(1,1,4 * Math.sin(2 * input) + 5);
        }

        WeightInitializer initializer = new RandomInitializer(-2, 2);
        DescendMethod descendMethod = new VanillaGradientDescend();
//        DescendMethod descendMethod = new MomentumDescend(0.9);

        /* Build the network. */
        Network net = new Network(
                new InputLayer(1),
                new FullyConnectedLayer(1, 6, new Sigmoid(), descendMethod, initializer),
                new FullyConnectedLayer(6, 1, new Linear(), descendMethod, initializer)
        );

        /* Collect training results. */

        LinearGraph errorsGraph = new LinearGraph("Total error");

        int inputIndex = 3;
        DualLinearGraph resultsGraph = new DualLinearGraph("Results for input #" + inputIndex);

        SeparatedData[] data = DataSeparator.toBatch(DataSeparator.separateData(inputs, outputs, 0.8));
        int iteration = 0;
        double error, result;
        while ((error = net.backpropagate(1e-3, data)) > 1e-1) {
            if (iteration++ % 1000 == 0) {
                result = net.getOutput(inputs[inputIndex]).get(0, 0);
                errorsGraph.add(error);
                resultsGraph.add(result, outputs[inputIndex].get(0, 0));
                System.out.println("Iteration " + iteration + " has error: " + error);
            }
        }
        result = net.getOutput(inputs[inputIndex]).get(0, 0);
        errorsGraph.add(error);
        resultsGraph.add(result, outputs[inputIndex].get(0, 0));
        System.out.println("Iteration " + iteration + " has error: " + error);

        /* Display the final results. */

        System.out.println("Weights:\n" + net.toString());

        DualLinearGraph finalResults = new DualLinearGraph("Final results");
        for (int index = 0; index < inputs.length; index++)
            finalResults.add(net.getOutput(inputs[index]).get(0, 0), outputs[index].get(0, 0));

        new Window(new VerticalContainer(
                new Row(errorsGraph), new Row(resultsGraph), new Row(finalResults)
        ), true, true);
    }
}
