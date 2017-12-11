package com.lirfu.networks;

import com.lirfu.graphicslib.IncompatibleOperandException;
import com.lirfu.graphicslib.matrix.IMatrix;
import com.lirfu.networks.layers.InnerLayer;
import com.lirfu.networks.layers.InputLayer;
import com.lirfu.networks.layers.Layer;

/**
 * Created by lirfu on 08.08.17..
 */
public class Network {
    private double learningRate;
    private InputLayer inputLayer;
    private InnerLayer[] hiddenLayers;

    public Network(double learningRate, InputLayer inputLayer, InnerLayer... hiddenLayers) {
        this.learningRate = learningRate;
        this.hiddenLayers = hiddenLayers;
        this.inputLayer = inputLayer;
    }

    /**
     * Calculates the output of the neural network for the given input.
     * @param input Matrix of inputs.
     * @return Matrix of outputs.
     * @throws IncompatibleOperandException if input-weights matrix dimensions between layers mismatch.
     */
    public IMatrix getOutput(IMatrix input) throws IncompatibleOperandException {
        Layer lastLayer = inputLayer;
        ((InputLayer) lastLayer).setOutput(input);

        InnerLayer currentLayer;
        for (InnerLayer hiddenLayer : hiddenLayers) {
            currentLayer = hiddenLayer;
            currentLayer.forwardPass(lastLayer);
            lastLayer = currentLayer;
        }

        return lastLayer.getOutput();
    }

    public double backpropagate(SeparatedData data) throws IncompatibleOperandException {
        return backpropagate(data.getTrainingInputs(), data.getTrainingOutputs(), data.getTestInputs(), data.getTestOutputs());
    }

    public double backpropagate(IMatrix[] trainingInputs, IMatrix[] trainingOutputs, IMatrix[] testInputs, IMatrix[] testOutputs) throws IncompatibleOperandException {
        for (int index = 0; index < trainingInputs.length; index++) {
            IMatrix input = trainingInputs[index];
            IMatrix targetOutput = trainingOutputs[index];

            // Forward pass to get output
            IMatrix guessedOutput = getOutput(input);

            // Calculate the output difference
            IMatrix outDiff = targetOutput.nSub(guessedOutput).nTranspose(false);

            // Iterate through the rest of the layers (backwards)
            InnerLayer currentLayer;
            Layer leftLayer;
            for (int i = hiddenLayers.length - 1; i >= 0; i--) {
                currentLayer = hiddenLayers[i];
                if (i == 0)
                    leftLayer = inputLayer;
                else
                    leftLayer = hiddenLayers[i - 1];

//                // Layer differences
//                differences = outDiff.nHadamardProduct(currentLayer.getNet().nApplyFunction(currentLayer.getFunction().getDerivative()).nTranspose(false));
//
//                // Update the differences for the next iteration
//                outDiff = currentLayer.getWeights().nMultiply(differences);

                // Update weights
//                currentLayer.getWeights().add(differences.nMultiply(leftLayer.getOutput()).scalarMultiply(learningRate).nTranspose(false));
//                currentLayer.getBiases().add(differences.scalarMultiply(learningRate).nTranspose(false));

                outDiff = currentLayer.backwardPass(outDiff, leftLayer.getOutput(), learningRate);
            }
        }

        /*
        Use test inputs to calculate the final error.
        */
        double totalError = 0;

        for (int i = 0; i < testInputs.length; i++) {
            // Forward pass to get output
            IMatrix guessedOutput = getOutput(testInputs[i]);

            // Calculate the output difference
            IMatrix outDiff = testOutputs[i].nSub(guessedOutput).nTranspose(false);

            // Calculate the total output error
            for (int r = 0; r < outDiff.getRowsCount(); r++)
                for (int c = 0; c < outDiff.getColsCount(); c++)
                    totalError += outDiff.get(r, c) * outDiff.get(r, c);
        }

        return totalError * 0.5 / testInputs.length;
    }

    @Override
    public String toString() {
        String s = "";

        for (int i = 1; i < hiddenLayers.length; i++)
            s += "\tLayer " + i + ":\n" + hiddenLayers[i].getWeights().toString(4) + "\n";

        return s;
    }
}
