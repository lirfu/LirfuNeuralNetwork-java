package com.lirfu.networks.layers;

import com.lirfu.graphicslib.functions.DerivativeFunction;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.graphicslib.matrix.Matrix;

import java.util.Random;

/**
 * Created by lirfu on 08.08.17..
 */
public class FullyConnectedLayer extends InnerLayer {
    public FullyConnectedLayer(int inputSize, int outputSize, DerivativeFunction function) {
        super(inputSize, outputSize, function);

        biases = new Matrix(1, outputSize);
        weights = new Matrix(inputSize, outputSize);

        // Init weight values
        Random rand = new Random();
        for (int c = 0; c < weights.getColsCount(); c++) {
            for (int r = 0; r < weights.getRowsCount(); r++)
                weights.set(r, c, nextRandom(rand, -1, 1));

            biases.set(0, c, nextRandom(rand, -1, 1));
        }
    }

    public void forwardPass(Layer leftLayer) {
        net = leftLayer.getOutput().nMultiply(weights).add(biases);
        output = net.nApplyFunction(function.getFunction());
    }

    public IMatrix backwardPass(IMatrix outputDifferences, IMatrix leftOutputs, double learningRate) {
        // Layer differences
        IMatrix differences = outputDifferences.nHadamardProduct(net.nApplyFunction(function.getDerivative()).nTranspose(false));

        // Update the differences for the next iteration
        outputDifferences = weights.nMultiply(differences);

        // Update weight and bias values
        weights.add(differences.nMultiply(leftOutputs).scalarMultiply(learningRate).nTranspose(false));
        biases.add(differences.scalarMultiply(learningRate).nTranspose(false));

        return outputDifferences;
    }
}
